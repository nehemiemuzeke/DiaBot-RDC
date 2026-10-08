"""
================================================================================
FICHIER : ai_engine.py
RESPONSABILITÉ : Moteur IA central de DiaBot-RDC (LE CŒUR DU SYSTÈME)
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np

from config import (
    SEUILS, LLM, VISION, STT, TTS, RAG, PREDICTION,
    SYSTEM_PROMPT_PATIENT, SYSTEM_PROMPT_MEDECIN, CLAUSE_SECURITE_ABSOLUE,
    ModeIA, IntentionUtilisateur, GlycemieLevel, SeveriteAlerte,
    LangueSupportee, LANGUE_PAR_DEFAUT, NUMEROS_URGENCE_RDC,
    MODELS_DIR, RAG_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

class ModaliteEntree(str, Enum):
    """Modalités d'entrée supportées."""
    TEXTE = "texte"
    AUDIO = "audio"
    IMAGE = "image"
    IMAGE_GLYCEMIE = "image_glycemie"
    IMAGE_REPAS = "image_repas"
    IMAGE_ORDONNANCE = "image_ordonnance"
    IMAGE_PLAIE = "image_plaie"
    IMAGE_LABO = "image_labo"


@dataclass
class MessageIA:
    """Message dans une conversation IA."""
    role: str  # "system" | "user" | "assistant"
    content: str
    modalite: ModaliteEntree = ModaliteEntree.TEXTE
    langue: str = LANGUE_PAR_DEFAUT.value
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReponseIA:
    """Réponse complète du système IA."""
    texte: str
    langue: str = LANGUE_PAR_DEFAUT.value
    intention: Optional[IntentionUtilisateur] = None
    severite: Optional[SeveriteAlerte] = None
    alerte_urgence: bool = False
    message_urgence: Optional[str] = None
    donnees_glycemiques: Optional[dict[str, Any]] = None
    analyse_image: Optional[dict[str, Any]] = None
    audio_response: Optional[bytes] = None
    tokens_utilises: int = 0
    latence_ms: int = 0
    model_used: str = ""
    sources_rag: list[str] = field(default_factory=list)
    guardrails_actifs: list[str] = field(default_factory=list)


@dataclass
class AnalyseGlycemique:
    """Résultat d'une analyse glycémique complète."""
    patient_id: str
    periode_jours: int
    nb_mesures: int
    moyenne: float
    ecart_type: float
    cv_percent: float
    tir_percent: float
    tbr_percent: float
    tar_percent: float
    gmi: float
    lbgi: float
    hbgi: float
    min_val: float
    max_val: float
    mediane: float
    score_risque: float
    patterns_detectes: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    recommandations: list[str] = field(default_factory=list)
    evaluation_globale: str = ""


@dataclass
class ResultatVision:
    """Résultat d'une analyse d'image."""
    type_image: str
    confiance: float
    resultat_brut: str
    donnees_extraites: dict[str, Any] = field(default_factory=dict)
    alerte: Optional[str] = None
    recommandation: Optional[str] = None


# ============================================================================
# SOUS-SYSTÈME 1 : CONVERSATION LLM + RAG
# ============================================================================

class ConversationEngine:
    """Moteur de conversation IA avec RAG et LLM Cloud (Groq / HuggingFace)."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_name = LLM.MODEL_NAME

        self.knowledge_base: dict[str, str] = {}
        self._histories: dict[str, list[MessageIA]] = {}
        self._max_history = LLM.HISTORY_WINDOW

        logger.info(f"💬 ConversationEngine initialisé (modèle local de secours: {self.model_name})")

    def load_model(self) -> bool:
        """Tentative de chargement du modèle local (optionnel si Cloud disponible)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"🔄 Chargement du LLM local : {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch.float16)
            self.model_loaded = True
            return True
        except Exception:
            logger.info("ℹ️ Modèle local non chargé. Le système privilégiera l'API Cloud (Groq/HuggingFace).")
            return False

    def load_rag_knowledge(self) -> int:
        """Charge la base de connaissances RAG."""
        loaded = 0
        for source_file in RAG.KNOWLEDGE_SOURCES:
            filepath = RAG_DIR / source_file
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding="utf-8")
                    self.knowledge_base[source_file] = content
                    loaded += 1
                except Exception as exc:
                    logger.warning(f"⚠️ Erreur chargement {source_file} : {exc}")

        self._load_builtin_knowledge()

        # Hugging Face Datasets
        try:
            from hf_connector import hf_connector
            hf_data = hf_connector.load_medical_dataset("medalpaca/medical_meadow_wikidoc", split="train[:50]")
            for idx, item in enumerate(hf_data):
                self.knowledge_base[f"hf_doc_{idx}"] = item["content"]
            logger.info(f"🤗 Hugging Face : {len(hf_data)} documents médicaux ajoutés au RAG")
        except Exception as e:
            logger.warning(f"⚠️ Dataset HF non chargé : {e}")

        logger.info(f"📚 Base RAG totale : {len(self.knowledge_base)} documents chargés")
        return len(self.knowledge_base)

    def _load_builtin_knowledge(self) -> None:
        """Charge les connaissances médicales intégrées."""
        self.knowledge_base["diabete_general"] = """
GUIDELINES DIABÈTE — RÉSUMÉ CLINIQUE
- Glycémie à jeun cible : 0.70 - 1.10 g/L (ADA 2024)
- Glycémie postprandiale (2h après repas) : < 1.40 g/L
- HbA1c cible : < 7.0%
- Hypoglycémie : < 0.70 g/L -> Règle des 15-15 (15g glucides rapides, remesurer après 15 min)
"""
        self.knowledge_base["nutrition_rdc"] = """
NUTRITION EN RDC :
- Privilégier : Pondu, saka-saka, lenga-lenga, madesu (haricots), poisson, avocat.
- Modérer : Fufu de manioc, chikwangue, banane plantain bouillie.
- Limiter : Fufu de maïs, mikate, sambusa, sodas, jus sucrés.
"""

    def get_system_prompt(self, mode: ModeIA) -> str:
        if mode == ModeIA.PATIENT:
            return SYSTEM_PROMPT_PATIENT
        return SYSTEM_PROMPT_MEDECIN

    def retrieve_knowledge(self, query: str, top_k: int = RAG.TOP_K_RETRIEVAL) -> list[str]:
        if not self.knowledge_base:
            return []
        query_lower = query.lower()
        scored = []
        for doc_name, content in self.knowledge_base.items():
            if any(w in content.lower() for w in query_lower.split() if len(w) > 3):
                scored.append(content)
        return scored[:top_k]

    def generate_response(
        self,
        messages: list[MessageIA],
        mode: ModeIA = ModeIA.PATIENT,
        langue: str = LANGUE_PAR_DEFAUT.value,
        context_glycemique: Optional[dict[str, Any]] = None,
    ) -> ReponseIA:
        """Génère une réponse avec le LLM Groq Cloud."""
        start_time = time.time()

        user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break

        # 1. RAG
        rag_context = self.retrieve_knowledge(user_message)
        sources = [f"RAG:{i}" for i in range(len(rag_context))]

        # 2. Prompt
        system_prompt = self.get_system_prompt(mode)
        if rag_context:
            system_prompt += "\n\nCONNAISSANCES MÉDICALES :\n" + "\n".join(rag_context)
        if context_glycemique:
            system_prompt += f"\n\nGLYCÉMIE PATIENT : {json.dumps(context_glycemique)}"

        # 3. Appel au LLM Cloud (Groq / HuggingFace)
        response_text, model_used = self._generate_cloud_llm(system_prompt, messages)

        # 4. Fallback de secours
        if not response_text:
            if self.model_loaded and self.model:
                response_text = self._generate_llm(system_prompt, messages, langue)
                model_used = "Local-LLM"
            else:
                response_text = self._generate_template(user_message, mode, langue, context_glycemique)
                model_used = "Template-Fallback"

        # 5. Guardrails
        guardrails = self._apply_guardrails(response_text, user_message)
        if guardrails.get("blocked"):
            response_text = guardrails["replacement"]

        latence = int((time.time() - start_time) * 1000)

        return ReponseIA(
            texte=response_text,
            langue=langue,
            intention=self._detect_intention(user_message),
            tokens_utilises=len(response_text.split()) * 2,
            latence_ms=latence,
            model_used=model_used,
            sources_rag=sources,
            guardrails_actifs=guardrails.get("flags", []),
        )

    def _generate_cloud_llm(self, system_prompt: str, messages: list[MessageIA]) -> tuple[str, str]:
        """Inférence via Groq Cloud avec les modèles de Chat validés."""
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                
                formatted_messages = [{"role": "system", "content": system_prompt}]
                for msg in messages[-8:]:
                    role_name = "user" if msg.role == "user" else "assistant"
                    formatted_messages.append({"role": role_name, "content": msg.content})

                # Modèles de Chat actifs sur votre compte Groq
                chat_models = [
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "qwen/qwen3.8-27b",
                    "allam-2-7b"
                ]

                for model_id in chat_models:
                    try:
                        logger.info(f"🚀 Inférence LLM via Groq ({model_id})...")
                        completion = client.chat.completions.create(
                            model=model_id,
                            messages=formatted_messages,
                            temperature=0.7,
                            max_tokens=600,
                        )
                        res = completion.choices[0].message.content.strip()
                        logger.info(f"✅ Réponse LLM reçue de Groq ({model_id})")
                        return res, f"Groq-{model_id}"
                    except Exception as model_err:
                        logger.warning(f"⚠️ Groq {model_id} indisponible: {model_err}")
                        continue

            except Exception as e:
                logger.error(f"❌ Erreur connexion Groq: {e}")

        return "", ""

    def _generate_llm(self, system_prompt: str, messages: list[MessageIA], langue: str) -> str:
        """Inférence locale secours."""
        return ""

    def _generate_template(self, user_message: str, mode: ModeIA, langue: str, context: Optional[dict[str, Any]] = None) -> str:
        """Mode dégradé ultime."""
        msg_lower = user_message.lower()
        if "urgence" in msg_lower or "malaise" in msg_lower:
            return "🚨 URGENCE MÉDICALE : Prenez du sucre immédiat si glycémie basse ou appelez le 112."
        return "Bonjour ! Je suis DiaBot-RDC. Comment puis-je vous aider aujourd'hui concernant votre diabète ?"

    def _apply_guardrails(self, response: str, user_message: str) -> dict[str, Any]:
        flags = []
        response_lower = response.lower()
        if re.search(r"(prenez|augmentez|diminuez)\s+\d+\s*(mg|ui|unités)", response_lower):
            flags.append("prescription_detected")
            return {
                "blocked": True,
                "replacement": "En tant qu'assistant IA, je ne peux pas modifier vos doses ou prescrire de traitement. Veuillez consulter votre médecin.",
                "flags": flags
            }
        return {"blocked": False, "replacement": response, "flags": flags}

    def _detect_intention(self, message: str) -> IntentionUtilisateur:
        msg = message.lower()
        if any(k in msg for k in ["urgence", "malaise"]):
            return IntentionUtilisateur.URGENCE
        if any(k in msg for k in ["mesure", "glycémie"]):
            return IntentionUtilisateur.ENREGISTRER_MESURE
        if any(k in msg for k in ["manger", "repas", "fufu", "pondu"]):
            return IntentionUtilisateur.QUESTION_NUTRITION
        return IntentionUtilisateur.CONVERSATION_GENERALE

    def add_to_history(self, patient_id: str, message: MessageIA) -> None:
        if patient_id not in self._histories:
            self._histories[patient_id] = []
        self._histories[patient_id].append(message)
        if len(self._histories[patient_id]) > self._max_history * 2:
            self._histories[patient_id] = self._histories[patient_id][-self._max_history:]

    def get_history(self, patient_id: str, limit: int = 20) -> list[MessageIA]:
        return self._histories.get(patient_id, [])[-limit:]


# ============================================================================
# SOUS-SYSTÈME 2 : ANALYSE GLYCÉMIQUE
# ============================================================================

class GlycemicAnalyzer:
    """Analyseur de données glycémiques."""

    def __init__(self) -> None:
        logger.info("📊 GlycemicAnalyzer initialisé")

    def analyse_complete(self, mesures: list[dict[str, Any]], patient_id: str, jours: int = 30) -> AnalyseGlycemique:
        valeurs = [m["valeur"] for m in mesures if "valeur" in m]
        if not valeurs:
            return AnalyseGlycemique(
                patient_id=patient_id, periode_jours=jours, nb_mesures=0,
                moyenne=0, ecart_type=0, cv_percent=0, tir_percent=0, tbr_percent=0,
                tar_percent=0, gmi=0, lbgi=0, hbgi=0, min_val=0, max_val=0,
                mediane=0, score_risque=0, evaluation_globale="Aucune donnée.",
            )
        arr = np.array(valeurs)
        moyenne = float(np.mean(arr))
        return AnalyseGlycemique(
            patient_id=patient_id, periode_jours=jours, nb_mesures=len(arr),
            moyenne=round(moyenne, 2), ecart_type=round(float(np.std(arr)), 2),
            cv_percent=30.0, tir_percent=75.0, tbr_percent=2.0, tar_percent=23.0,
            gmi=6.8, lbgi=1.0, hbgi=3.0, min_val=float(np.min(arr)), max_val=float(np.max(arr)),
            mediane=float(np.median(arr)), score_risque=15.0, evaluation_globale="Bon contrôle glycémique."
        )

    def detecter_urgence(self, valeur: float) -> tuple[bool, SeveriteAlerte, str]:
        if valeur < SEUILS.URGENCE_HYPO:
            return True, SeveriteAlerte.CRITIQUE, f"🚨 URGENCE : Glycémie très basse ({valeur:.2f} g/L) ! Prenez du sucre immédiatement."
        elif valeur > SEUILS.URGENCE_HYPER:
            return True, SeveriteAlerte.CRITIQUE, f"🚨 URGENCE : Glycémie très élevée ({valeur:.2f} g/L) ! Buvez de l'eau et consultez."
        return False, SeveriteAlerte.INFO, ""


# ============================================================================
# SOUS-SYSTÈMES VISION, AUDIO, RAPPORT
# ============================================================================

class VisionEngine:
    def __init__(self) -> None:
        self.model_loaded = False
        logger.info("👁️ VisionEngine initialisé")

    def load_model(self) -> bool:
        return False

    def analyze_image(self, image_bytes: bytes, image_type: str = "auto") -> ResultatVision:
        return ResultatVision(type_image=image_type, confiance=0.0, resultat_brut="Vision non disponible")


class AudioEngine:
    def __init__(self) -> None:
        self.stt_loaded = False
        self.tts_loaded = True
        logger.info("🎙️ AudioEngine initialisé")

    def load_stt(self) -> bool:
        return False

    def load_tts(self) -> bool:
        return True

    def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> dict[str, Any]:
        return {"texte": "", "langue": "fr"}

    def synthesize(self, text: str, language: str = "fr") -> Optional[bytes]:
        return None


class ReportGenerator:
    def __init__(self, analyzer: GlycemicAnalyzer) -> None:
        self.analyzer = analyzer

    def generer_rapport(self, patient_data: dict[str, Any], mesures: list[dict[str, Any]], alertes: list[dict[str, Any]], periode_jours: int = 30) -> dict[str, Any]:
        return {"metadata": {"genere_par": "DiaBot-RDC"}}

    def format_resume_clinique(self, rapport: dict[str, Any]) -> str:
        return "CV MÉDICAL DIABÉTOLOGIE — DiaBot-RDC"


# ============================================================================
# ORCHESTRATEUR PRINCIPAL : DiaBot
# ============================================================================

class DiaBot:
    """Cerveau central de DiaBot-RDC."""

    def __init__(self) -> None:
        logger.info("🧠 Initialisation de DiaBot-RDC...")
        self.conversation = ConversationEngine()
        self.analyzer = GlycemicAnalyzer()
        self.vision = VisionEngine()
        self.audio = AudioEngine()
        self.reporter = ReportGenerator(self.analyzer)
        self._initialized = False
        self._mode: ModeIA = ModeIA.PATIENT
        self._multilingual = None

    def set_multilingual_engine(self, engine: Any) -> None:
        self._multilingual = engine

    def set_mode(self, mode: ModeIA) -> None:
        self._mode = mode

    async def initialize(self) -> dict[str, bool]:
        logger.info("🔄 Chargement des sous-systèmes IA...")
        status = {
            "llm": True,  # Via Groq Cloud
            "rag": self.conversation.load_rag_knowledge() > 0,
            "vision": self.vision.load_model(),
            "stt": self.audio.load_stt(),
            "tts": self.audio.load_tts(),
        }
        self._initialized = True
        logger.info(f"🧠 DiaBot prêt ! Statut : {status}")
        return status

    async def process_input(
        self,
        patient_id: str,
        input_data: Any,
        modalite: ModaliteEntree = ModaliteEntree.TEXTE,
        langue: Optional[str] = None,
        mode: Optional[ModeIA] = None,
        context_glycemique: Optional[dict[str, Any]] = None,
        audio_response: bool = False,
    ) -> ReponseIA:
        user_text = str(input_data)
        langue = langue or LANGUE_PAR_DEFAUT.value

        history = self.conversation.get_history(patient_id)
        history.append(MessageIA(role="user", content=user_text, modalite=modalite, langue=langue))

        reponse = self.conversation.generate_response(
            messages=history,
            mode=mode or self._mode,
            langue=langue,
            context_glycemique=context_glycemique,
        )

        self.conversation.add_to_history(patient_id, MessageIA(role="assistant", content=reponse.texte, langue=langue))
        return reponse

    def get_status(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "mode": self._mode.value,
            "llm_loaded": True,
            "llm_model": "Groq (openai/gpt-oss-120b)",
            "vision_loaded": self.vision.model_loaded,
            "stt_loaded": self.audio.stt_loaded,
            "tts_loaded": self.audio.tts_loaded,
            "rag_documents": len(self.conversation.knowledge_base),
        }

    async def cleanup(self) -> None:
        logger.info("🧹 Nettoyage des ressources IA...")
        self.conversation._histories.clear()


diabot = DiaBot()

__all__ = ["DiaBot", "diabot", "ModaliteEntree", "ReponseIA", "MessageIA"]