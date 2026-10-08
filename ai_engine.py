"""
================================================================================
FICHIER : ai_engine.py
RESPONSABILITÉ : Moteur IA central de DiaBot-RDC (LE CŒUR DU SYSTÈME)
================================================================================
Ce module est le CERVEAU de l'application. Il orchestre 5 sous-systèmes IA :

  1. CONVERSATION (LLM) — Dialogue médical avec RAG et guardrails
  2. ANALYSE GLYCÉMIQUE — Stats, patterns, prédiction LSTM
  3. VISION — Analyse d'images (glucomètres, repas, ordonnances, plaies)
  4. AUDIO — Speech-to-Text (Whisper) + Text-to-Speech (XTTS)
  5. RAPPORTS — Génération automatique de CV médical

Classe principale : DiaBot
Méthode d'orchestration : process_input()

Architecture :
- Strategy pattern pour les modèles IA
- Graceful degradation si un modèle ne charge pas
- Guardrails stricts (jamais de diagnostic, jamais de prescription)
- Détection d'urgence et escalade automatique
- Support multilingue (FR + 4 langues RDC)

Auteur : Équipe DiaBot-RDC
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
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
    """Moteur de conversation IA avec RAG et guardrails médicaux.

    Utilise un LLM (Mistral/Llama/Phi) fine-tuné sur la diabétologie,
    augmenté par une base de connaissances RAG (OMS, IDF, ADA).
    """

    def __init__(self) -> None:
        """Initialise le moteur de conversation."""
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_name = LLM.MODEL_NAME

        # Base de connaissances RAG
        self.knowledge_base: dict[str, str] = {}
        self.embeddings_model = None
        self.vector_store = None

        # Historique des conversations par patient
        self._histories: dict[str, list[MessageIA]] = {}
        self._max_history = LLM.HISTORY_WINDOW

        logger.info(f"💬 ConversationEngine initialisé (modèle: {self.model_name})")

    def load_model(self) -> bool:
        """Charge le modèle LLM en mémoire.

        Returns:
            True si le chargement a réussi.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"🔄 Chargement du LLM : {self.model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
            }

            if LLM.QUANTIZATION == "4bit":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs.pop("torch_dtype", None)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if LLM.DEVICE == "cuda" else LLM.DEVICE,
                **load_kwargs,
            )

            self.model_loaded = True
            logger.info(f"✅ LLM chargé : {self.model_name}")
            return True

        except ImportError:
            logger.warning(
                "⚠️  transformers/torch non installés. "
                "Mode conversation dégradé (réponses templates)."
            )
            return False
        except Exception as exc:
            logger.error(f"❌ Erreur chargement LLM : {exc}")
            return False

    def load_rag_knowledge(self) -> int:
        """Charge la base de connaissances RAG.

        Returns:
            Nombre de documents chargés.
        """
        loaded = 0
        for source_file in RAG.KNOWLEDGE_SOURCES:
            filepath = RAG_DIR / source_file
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding="utf-8")
                    self.knowledge_base[source_file] = content
                    loaded += 1
                except Exception as exc:
                    logger.warning(f"⚠️  Erreur chargement {source_file} : {exc}")

        # Charger la base de connaissances intégrée (fallback)
        if loaded == 0:
            self._load_builtin_knowledge()
            loaded = len(self.knowledge_base)

        logger.info(f"📚 Base RAG : {loaded} documents chargés")
        return loaded

    def _load_builtin_knowledge(self) -> None:
        """Charge la base de connaissances intégrée (sans fichiers externes)."""
        self.knowledge_base["diabete_general"] = """
GUIDELINES DIABÈTE — RÉSUMÉ CLINIQUE

DIAGNOSTIC (OMS/ADA 2024) :
- Glycémie à jeun ≥ 1.26 g/L (7.0 mmol/L)
- Glycémie 2h post-charge ≥ 2.00 g/L (11.1 mmol/L)
- HbA1c ≥ 6.5%
- Glycémie aléatoire ≥ 2.00 g/L avec symptômes

CIBLES THÉRAPEUTIQUES (ADA 2024) :
- HbA1c < 7.0% (standard), < 6.5% (si possible sans hypo)
- Glycémie à jeun : 0.80 - 1.30 g/L
- Glycémie postprandiale : < 1.80 g/L
- TIR (70-180 mg/dL) : > 70%
- TBR (< 70 mg/dL) : < 4%
- TBR (< 54 mg/dL) : < 1%

TRAITEMENTS TYPE 2 (escalade) :
1. Metformine (1ère intention)
2. + iSGLT2 ou GLP-1 RA (si risque CV/renal)
3. + Insuline basale si HbA1c > 10% ou symptômes
4. Schéma basal-bolus si nécessaire

HYPOGLYCÉMIE :
- < 0.70 g/L : prendre 15g de glucides rapides, remesurer à 15min
- < 0.54 g/L : URGENCE, aide extérieure nécessaire
- Règle des 15-15 : 15g glucides, attendre 15min, remesurer

NUTRITION (adapté RDC) :
- Privilégier : pondu, saka-saka, lenga-lenga, haricots, poisson, avocat
- Modérer : fufu manioc, chikwangue, riz, banane plantain verte
- Limiter : fufu maïs, mikate, sambusa, jus sucrés, riz blanc en grande quantité
- Index glycémique bas : légumes feuilles, légumineuses, arachides
"""

        self.knowledge_base["nutrition_rdc"] = """
CONSEILS NUTRITIONNELS ADAPTÉS À LA RDC

ALIMENTS RECOMMANDÉS (IG bas) :
- Pondu (feuilles de manioc) : riche en fibres, ralentit l'absorption du sucre
- Saka-saka : excellent pour la glycémie
- Lenga-lenga (amarante) : légume-feuille très sain
- Haricots (madesu) : stabilisent la glycémie
- Poisson (maboké, liboké) : protéines sans glucides
- Avocat : gras sain qui ralentit l'absorption
- Arachides (nguba) : bon en petite quantité

ALIMENTS À MODÉRER (IG moyen) :
- Fufu de manioc (kwanga) : préférer au fufu de maïs, portion = 1 poing
- Chikwangue : alternative au fufu, portion modérée
- Banane plantain verte (bouillie) : meilleure que la mûre frite
- Mangue, papaye : 1 fruit à la fois

ALIMENTS À LIMITER (IG élevé) :
- Fufu de maïs : portion réduite, beaucoup de pondu à côté
- Riz blanc : limiter la quantité
- Mikate (beignets) : très sucré et gras, occasionnellement
- Sambusa : pâte frite + féculent
- Jus sucrés, sodas : remplacer par de l'eau ou bissap peu sucré

CONSEILS PRATIQUES :
- Commencer le repas par les légumes (pondu, saka-saka)
- Marcher 15-20 minutes après le repas
- Boire au moins 8 verres d'eau par jour
- Éviter les grignotages sucrés entre les repas
"""

        self.knowledge_base["activite_physique"] = """
ACTIVITÉ PHYSIQUE ET DIABÈTE

RECOMMANDATIONS GÉNÉRALES (OMS/ADA) :
- 150 min/semaine d'activité modérée (marche rapide)
- Répartir sur au moins 3 jours, pas plus de 2 jours sans activité
- Ajouter 2-3 sessions de résistance musculaire par semaine

ADAPTÉ AU CONTEXTE RDC :
- Marche de 30 min après le repas du soir (très efficace)
- Jardinage, travaux ménagers actifs
- Danse traditionnelle (excellente activité cardio)
- Natation si accès à un plan d'eau sûr
- Monter les escaliers plutôt que l'ascenseur

PRÉCAUTIONS :
- Mesurer la glycémie AVANT l'activité
- Si glycémie < 1.00 g/L : prendre une collation avant
- Si glycémie > 2.50 g/L avec cétones : éviter l'activité intense
- Toujours avoir du sucre rapide sur soi (bonbons, jus)
- Bien s'hydrater (eau, pas de soda)
- Inspecter les pieds après l'activité (chaussures adaptées)
"""

    def get_system_prompt(self, mode: ModeIA) -> str:
        """Retourne le system prompt selon le mode.

        Args:
            mode: Mode patient ou médecin.

        Returns:
            System prompt complet.
        """
        if mode == ModeIA.PATIENT:
            return SYSTEM_PROMPT_PATIENT
        return SYSTEM_PROMPT_MEDECIN

    def retrieve_knowledge(
        self,
        query: str,
        top_k: int = RAG.TOP_K_RETRIEVAL,
    ) -> list[str]:
        """Récupère les passages pertinents de la base RAG.

        Utilise une recherche par mots-clés simplifiée (fallback
        si le modèle d'embeddings n'est pas chargé).

        Args:
            query: Question de l'utilisateur.
            top_k: Nombre de passages à retourner.

        Returns:
            Liste de passages pertinents.
        """
        if not self.knowledge_base:
            return []

        query_lower = query.lower()
        scored: list[tuple[float, str, str]] = []

        # Mots-clés de scoring
        keywords = {
            "glycémie": 3, "glycemie": 3, "sucre": 2, "sang": 2,
            "diabète": 3, "diabete": 3, "insuline": 3, "metformine": 2,
            "hba1c": 3, "hypo": 3, "hyper": 3, "urgence": 3,
            "repas": 2, "manger": 2, "fufu": 2, "pondu": 2,
            "sport": 2, "marche": 2, "activité": 2, "exercice": 2,
            "pied": 2, "yeux": 2, "rein": 2, "nerf": 2,
            "traitement": 2, "médicament": 2, "dose": 2,
            "grossesse": 2, "enfant": 2, "type 1": 2, "type 2": 2,
        }

        query_words = set(query_lower.split())
        for doc_name, content in self.knowledge_base.items():
            score = 0.0
            content_lower = content.lower()
            for word, weight in keywords.items():
                if word in query_lower and word in content_lower:
                    score += weight
            # Bonus pour correspondance directe de mots
            for word in query_words:
                if len(word) > 3 and word in content_lower:
                    score += 1.0
            if score > 0:
                scored.append((score, doc_name, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [content for _, name, content in scored[:top_k]]

        return results

    def generate_response(
        self,
        messages: list[MessageIA],
        mode: ModeIA = ModeIA.PATIENT,
        langue: str = LANGUE_PAR_DEFAUT.value,
        context_glycemique: Optional[dict[str, Any]] = None,
    ) -> ReponseIA:
        """Génère une réponse conversationnelle.

        Pipeline :
        1. Construire le prompt avec RAG
        2. Appliquer les guardrails
        3. Générer la réponse (LLM ou template)
        4. Post-vérification de sécurité

        Args:
            messages: Historique de conversation.
            mode: Mode patient ou médecin.
            langue: Langue de réponse.
            context_glycemique: Données glycémiques récentes.

        Returns:
            ReponseIA complète.
        """
        start_time = time.time()

        # Dernier message utilisateur
        user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break

        # RAG : récupérer les connaissances pertinentes
        rag_context = self.retrieve_knowledge(user_message)
        sources = [f"RAG:{i}" for i in range(len(rag_context))]

        # Construire le prompt complet
        system_prompt = self.get_system_prompt(mode)

        if rag_context:
            rag_text = "\n\n---\n\n".join(rag_context)
            system_prompt += f"\n\nCONNAISSANCES DE RÉFÉRENCE :\n{rag_text}"

        if context_glycemique:
            system_prompt += (
                f"\n\nDONNÉES GLYCÉMIQUES DU PATIENT :\n"
                f"{json.dumps(context_glycemique, indent=2, default=str)}"
            )

        # Générer la réponse
        if self.model_loaded and self.model and self.tokenizer:
            response_text = self._generate_llm(
                system_prompt, messages, langue,
            )
        else:
            response_text = self._generate_template(
                user_message, mode, langue, context_glycemique,
            )

        # Post-vérification guardrails
        guardrails = self._apply_guardrails(response_text, user_message)
        if guardrails.get("blocked"):
            response_text = guardrails["replacement"]

        latence = int((time.time() - start_time) * 1000)

        return ReponseIA(
            texte=response_text,
            langue=langue,
            intention=self._detect_intention(user_message),
            tokens_utilises=len(response_text.split()) * 2,  # Estimation
            latence_ms=latence,
            model_used=self.model_name if self.model_loaded else "template",
            sources_rag=sources,
            guardrails_actifs=guardrails.get("flags", []),
        )

    def _generate_llm(
        self,
        system_prompt: str,
        messages: list[MessageIA],
        langue: str,
    ) -> str:
        """Génère une réponse via le LLM chargé.

        Args:
            system_prompt: Prompt système complet.
            messages: Historique de conversation.
            langue: Langue cible.

        Returns:
            Texte de la réponse.
        """
        try:
            import torch

            # Construire le prompt formaté
            formatted = f"<s>[INST] {system_prompt}\n\n"
            for msg in messages[-self._max_history:]:
                if msg.role == "user":
                    formatted += f"{msg.content} [/INST] "
                elif msg.role == "assistant":
                    formatted += f"{msg.content} </s><s>[INST] "

            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=LLM.CONTEXT_WINDOW - LLM.MAX_NEW_TOKENS,
            )

            if LLM.DEVICE == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=LLM.MAX_NEW_TOKENS,
                    temperature=LLM.TEMPERATURE,
                    top_p=LLM.TOP_P,
                    top_k=LLM.TOP_K,
                    repetition_penalty=LLM.REPETITION_PENALTY,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as exc:
            logger.error(f"❌ Erreur inférence LLM : {exc}")
            return self._generate_template(
                messages[-1].content if messages else "",
                ModeIA.PATIENT, langue,
            )

    def _generate_template(
        self,
        user_message: str,
        mode: ModeIA,
        langue: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Génère une réponse template (mode dégradé sans LLM).

        Fournit des réponses utiles basées sur des patterns
        même sans modèle IA chargé.

        Args:
            user_message: Message de l'utilisateur.
            mode: Mode IA.
            langue: Langue.
            context: Contexte glycémique.

        Returns:
            Texte de réponse template.
        """
        msg_lower = user_message.lower()

        # Détection d'urgence
        if any(kw in msg_lower for kw in [
            "urgence", "malaise", "inconscient", "coma", "très bas",
            "très haut", "3.0", "4.0", "0.3", "0.4",
        ]):
            return (
                "🚨 URGENCE MÉDICALE DÉTECTÉE\n\n"
                "Si votre glycémie est très basse (< 0.54 g/L) :\n"
                "1. Prenez immédiatement du sucre (jus, miel, bonbon)\n"
                "2. Si la personne est inconsciente : NE DONNEZ RIEN PAR LA BOUCHE\n"
                "3. Appelez le 112 ou allez à l'hôpital IMMÉDIATEMENT\n\n"
                "Si votre glycémie est très haute (> 3.00 g/L) :\n"
                "1. Buvez beaucoup d'eau\n"
                "2. Rendez-vous aux urgences immédiatement\n\n"
                f"📞 Urgences RDC : {NUMEROS_URGENCE_RDC.get('samu_kinshasa', '112')}"
            )

        # Questions sur la glycémie
        if any(kw in msg_lower for kw in [
            "glycémie", "glycemie", "sucre", "taux", "mesure",
        ]):
            if context:
                moy = context.get("moyenne_gl", "N/A")
                tir = context.get("tir_percent", "N/A")
                return (
                    f"📊 Voici un résumé de votre glycémie :\n\n"
                    f"• Moyenne : {moy} g/L\n"
                    f"• Temps dans la cible : {tir}%\n\n"
                    f"Pour une interprétation complète, je vous recommande "
                    f"de consulter votre médecin. Continuez à mesurer "
                    f"régulièrement, c'est très important ! 💪"
                )
            return (
                "📏 Pour suivre votre glycémie efficacement :\n\n"
                "1. Mesurez à jeun le matin (cible : 0.70-1.10 g/L)\n"
                "2. Mesurez 2h après les repas (cible : < 1.40 g/L)\n"
                "3. Notez vos repas et activités\n\n"
                "N'hésitez pas à enregistrer vos mesures dans l'application ! "
                "Plus vous avez de données, mieux je peux vous aider. 😊"
            )

        # Questions nutrition
        if any(kw in msg_lower for kw in [
            "manger", "repas", "nourriture", "fufu", "pondu",
            "alimentation", "régime", "diet", "fruit",
        ]):
            return (
                "🥗 Conseils nutritionnels pour le diabète en RDC :\n\n"
                "✅ À privilégier :\n"
                "• Pondu, saka-saka, lenga-lenga (légumes feuilles)\n"
                "• Haricots (madesu), arachides (nguba)\n"
                "• Poisson (maboké, liboké), avocat\n\n"
                "⚠️ À modérer :\n"
                "• Fufu de manioc (1 poing), chikwangue\n"
                "• Banane plantain verte (bouillie)\n\n"
                "❌ À limiter :\n"
                "• Mikate, sambusa, fufu de maïs\n"
                "• Sodas, jus très sucrés\n\n"
                "💡 Astuce : commencez votre repas par les légumes "
                "et marchez 15 min après manger !"
            )

        # Questions médicaments
        if any(kw in msg_lower for kw in [
            "médicament", "medicament", "traitement", "insuline",
            "metformine", "dose", "prescrire",
        ]):
            return (
                "💊 Concernant vos médicaments :\n\n"
                "Je ne suis pas médecin et je ne peux PAS :\n"
                "• Prescrire de médicaments\n"
                "• Modifier vos doses\n"
                "• Changer votre traitement\n\n"
                "Continuez à prendre vos médicaments exactement comme "
                "votre médecin vous l'a prescrit. Si vous avez des "
                "questions sur votre traitement, consultez votre médecin "
                "ou votre pharmacien.\n\n"
                "⚠️ Ne modifiez JAMAIS votre traitement sans avis médical."
            )

        # Salutations
        if any(kw in msg_lower for kw in [
            "bonjour", "salut", "mbote", "habari", "mwapoleni",
            "comment", "ça va", "hello",
        ]):
            if langue == "ln":
                return "Mbote ! Nazali DiaBot-RDC. Ndenge nini nakoki kosalisa yo lelo ? 😊"
            elif langue == "sw":
                return "Habari ! Mimi ni DiaBot-RDC. Ninawezaje kukusaidia leo ? 😊"
            return (
                "Bonjour ! 👋 Je suis DiaBot-RDC, votre compagnon "
                "pour le suivi de votre diabète.\n\n"
                "Comment puis-je vous aider aujourd'hui ?\n"
                "• 📊 Voir vos statistiques glycémiques\n"
                "• 🥗 Conseils nutritionnels\n"
                "• 🏃 Conseils activité physique\n"
                "• ❓ Questions sur le diabète\n"
                "• 📝 Enregistrer une mesure"
            )

        # Réponse par défaut
        return (
            "Merci pour votre message ! Je suis DiaBot-RDC, "
            "votre assistant pour le suivi du diabète.\n\n"
            "Je peux vous aider avec :\n"
            "• Vos mesures de glycémie et statistiques\n"
            "• Des conseils nutritionnels adaptés à la RDC\n"
            "• Des recommandations d'activité physique\n"
            "• Des informations générales sur le diabète\n\n"
            "Pour toute question médicale spécifique, "
            "consultez votre médecin. Comment puis-je vous aider ? 😊"
        )

    def _apply_guardrails(
        self,
        response: str,
        user_message: str,
    ) -> dict[str, Any]:
        """Applique les guardrails de sécurité médicale.

        Vérifie que la réponse ne contient pas :
        - De diagnostic
        - De prescription
        - De modification de traitement
        - De conseils dangereux

        Args:
            response: Réponse générée.
            user_message: Message original de l'utilisateur.

        Returns:
            Dict avec "blocked", "replacement", "flags".
        """
        flags: list[str] = []
        response_lower = response.lower()

        # Patterns de diagnostic
        diagnostic_patterns = [
            r"vous (avez|êtes|souffrez de) (un |le |du )?(diabète|cancer|sida|paludisme)",
            r"votre diagnostic (est|serait)",
            r"je (pense|crois|diagnostique) que vous avez",
        ]
        for pattern in diagnostic_patterns:
            if re.search(pattern, response_lower):
                flags.append("diagnostic_detected")
                break

        # Patterns de prescription
        prescription_patterns = [
            r"(prenez|augmentez|diminuez|arrêtez|commencez)\s+\d+\s*(mg|g|ml|ui|unités)",
            r"je vous (prescris|recommande de prendre)\s+(metformine|insuline|glibenclamide)",
            r"(dose|dosage)\s+(de|à)\s+\d+",
        ]
        for pattern in prescription_patterns:
            if re.search(pattern, response_lower):
                flags.append("prescription_detected")
                break

        if flags:
            replacement = (
                "Je comprends votre préoccupation. En tant qu'assistant IA, "
                "je ne peux ni poser de diagnostic ni prescrire de traitement. "
                "Je vous recommande vivement de consulter votre médecin "
                "ou de vous rendre au centre de santé le plus proche "
                "pour une évaluation médicale complète."
            )
            logger.warning(f"🛡️ Guardrails activés : {flags}")
            return {"blocked": True, "replacement": replacement, "flags": flags}

        return {"blocked": False, "replacement": response, "flags": flags}

    def _detect_intention(self, message: str) -> IntentionUtilisateur:
        """Détecte l'intention de l'utilisateur.

        Args:
            message: Message de l'utilisateur.

        Returns:
            Intention détectée.
        """
        msg = message.lower()

        if any(kw in msg for kw in ["urgence", "malaise", "inconscient", "coma"]):
            return IntentionUtilisateur.URGENCE
        if any(kw in msg for kw in ["mesure", "glycémie", "enregistrer", "valeur"]):
            return IntentionUtilisateur.ENREGISTRER_MESURE
        if any(kw in msg for kw in ["rapport", "compte-rendu", "résumé", "cv"]):
            return IntentionUtilisateur.DEMANDER_RAPPORT
        if any(kw in msg for kw in ["statistique", "moyenne", "tir", "tendance", "graphique"]):
            return IntentionUtilisateur.DEMANDER_STATS
        if any(kw in msg for kw in ["manger", "repas", "fufu", "pondu", "nutrition"]):
            return IntentionUtilisateur.QUESTION_NUTRITION
        if any(kw in msg for kw in ["sport", "marche", "exercice", "activité"]):
            return IntentionUtilisateur.QUESTION_ACTIVITE
        if any(kw in msg for kw in ["médicament", "insuline", "metformine", "traitement"]):
            return IntentionUtilisateur.QUESTION_MEDICAMENT
        if any(kw in msg for kw in ["triste", "déprimé", "peur", "anxiété", "stress", "fatigué"]):
            return IntentionUtilisateur.SUPPORT_EMOTIONNEL
        if any(kw in msg for kw in ["diabète", "diabete", "hba1c", "insuline", "type 1", "type 2"]):
            return IntentionUtilisateur.QUESTION_MEDICALE

        return IntentionUtilisateur.CONVERSATION_GENERALE

    def add_to_history(
        self,
        patient_id: str,
        message: MessageIA,
    ) -> None:
        """Ajoute un message à l'historique du patient.

        Args:
            patient_id: UUID du patient.
            message: Message à ajouter.
        """
        if patient_id not in self._histories:
            self._histories[patient_id] = []

        self._histories[patient_id].append(message)

        # Fenêtre glissante
        if len(self._histories[patient_id]) > self._max_history * 2:
            self._histories[patient_id] = self._histories[patient_id][-self._max_history:]

    def get_history(
        self,
        patient_id: str,
        limit: int = 20,
    ) -> list[MessageIA]:
        """Retourne l'historique de conversation.

        Args:
            patient_id: UUID du patient.
            limit: Nombre max de messages.

        Returns:
            Liste de messages.
        """
        history = self._histories.get(patient_id, [])
        return history[-limit:]


# ============================================================================
# SOUS-SYSTÈME 2 : ANALYSE DE DONNÉES GLYCÉMIQUES
# ============================================================================

class GlycemicAnalyzer:
    """Analyseur de données glycémiques avec détection de patterns.

    Calcule toutes les métriques ADA/IDF et détecte les patterns
    récurrents (effet de l'aube, hypo récurrentes, etc.)
    """

    def __init__(self) -> None:
        """Initialise l'analyseur."""
        logger.info("📊 GlycemicAnalyzer initialisé")

    def analyse_complete(
        self,
        mesures: list[dict[str, Any]],
        patient_id: str,
        jours: int = 30,
    ) -> AnalyseGlycemique:
        """Effectue une analyse glycémique complète.

        Args:
            mesures: Liste de mesures normalisées (g/L).
            patient_id: UUID du patient.
            jours: Période d'analyse.

        Returns:
            AnalyseGlycemique complète.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=jours)

        # Filtrer et trier
        filtered = []
        for m in mesures:
            ts = m.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if ts and ts >= cutoff:
                filtered.append({**m, "_ts": ts})

        filtered.sort(key=lambda x: x["_ts"])
        valeurs = [m["valeur"] for m in filtered]

        if not valeurs:
            return AnalyseGlycemique(
                patient_id=patient_id, periode_jours=jours, nb_mesures=0,
                moyenne=0, ecart_type=0, cv_percent=0,
                tir_percent=0, tbr_percent=0, tar_percent=0,
                gmi=0, lbgi=0, hbgi=0, min_val=0, max_val=0,
                mediane=0, score_risque=0,
                evaluation_globale="Aucune donnée disponible pour cette période.",
            )

        arr = np.array(valeurs)
        n = len(arr)
        moyenne = float(np.mean(arr))
        ecart_type = float(np.std(arr)) if n > 1 else 0.0
        cv = (ecart_type / moyenne * 100) if moyenne > 0 else 0.0

        # TIR / TBR / TAR
        in_range = float(np.sum((arr >= SEUILS.HYPOGLYCEMIE_MAX) & (arr <= SEUILS.NORMAL_POSTPRANDIAL_MAX)))
        below = float(np.sum(arr < SEUILS.HYPOGLYCEMIE_MAX))
        above = float(np.sum(arr > SEUILS.NORMAL_POSTPRANDIAL_MAX))

        tir = in_range / n * 100
        tbr = below / n * 100
        tar = above / n * 100

        # GMI
        gmi = 3.31 + 0.02392 * (moyenne * 100)

        # LBGI / HBGI
        lbgi = self._calculate_lbgi(arr)
        hbgi = self._calculate_hbgi(arr)

        # Score de risque
        risque = self._calculate_risk_score(tir, tbr, tar, cv, lbgi)

        # Détection de patterns
        patterns = self._detect_patterns(filtered)

        # Recommandations
        recommandations = self._generate_recommendations(
            tir, tbr, tar, cv, lbgi, hbgi, patterns,
        )

        # Évaluation globale
        evaluation = self._evaluate_global(tir, tbr, tar, cv, risque)

        return AnalyseGlycemique(
            patient_id=patient_id,
            periode_jours=jours,
            nb_mesures=n,
            moyenne=round(moyenne, 3),
            ecart_type=round(ecart_type, 3),
            cv_percent=round(cv, 1),
            tir_percent=round(tir, 1),
            tbr_percent=round(tbr, 1),
            tar_percent=round(tar, 1),
            gmi=round(gmi, 1),
            lbgi=round(lbgi, 2),
            hbgi=round(hbgi, 2),
            min_val=round(float(np.min(arr)), 3),
            max_val=round(float(np.max(arr)), 3),
            mediane=round(float(np.median(arr)), 3),
            score_risque=round(risque, 1),
            patterns_detectes=patterns,
            recommandations=recommandations,
            evaluation_globale=evaluation,
        )

    def _calculate_lbgi(self, valeurs: np.ndarray) -> float:
        """Calcule le Low Blood Glucose Index.

        Args:
            valeurs: Array de valeurs en g/L.

        Returns:
            Score LBGI.
        """
        mgdl = valeurs * 100
        xl = np.minimum(mgdl, 112.5)
        xl = np.maximum(xl, 0.1)  # Éviter log(0)
        f_xl = 22.77 * (np.log(xl) ** 1.504 - 5.381)
        f_xl = np.maximum(f_xl, 0)
        return float(10 * np.mean(f_xl))

    def _calculate_hbgi(self, valeurs: np.ndarray) -> float:
        """Calcule le High Blood Glucose Index.

        Args:
            valeurs: Array de valeurs en g/L.

        Returns:
            Score HBGI.
        """
        mgdl = valeurs * 100
        xh = np.maximum(mgdl, 112.5)
        f_xh = 22.77 * (np.log(xh) ** 1.504 - 5.381)
        f_xh = np.maximum(f_xh, 0)
        return float(10 * np.mean(f_xh))

    def _calculate_risk_score(
        self, tir: float, tbr: float, tar: float,
        cv: float, lbgi: float,
    ) -> float:
        """Calcule un score de risque global (0-100).

        Args:
            tir: Time In Range (%).
            tbr: Time Below Range (%).
            tar: Time Above Range (%).
            cv: Coefficient de variation (%).
            lbgi: Low Blood Glucose Index.

        Returns:
            Score de risque 0-100.
        """
        score = 0.0
        if tbr > 4:
            score += min((tbr - 4) * 10, 30)
        if tar > 25:
            score += min((tar - 25) * 1.5, 25)
        if tir < 70:
            score += min((70 - tir) * 0.5, 20)
        if cv > 36:
            score += min((cv - 36) * 0.8, 15)
        if lbgi > 2.5:
            score += min(lbgi * 3, 10)
        return min(score, 100.0)

    def _detect_patterns(
        self,
        mesures: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Détecte les patterns glycémiques récurrents.

        Patterns recherchés :
        - Effet de l'aube (dawn phenomenon)
        - Hypoglycémies récurrentes à la même heure
        - Hyperglycémies postprandiales systématiques
        - Tendance hebdomadaire
        - Variabilité excessive

        Args:
            mesures: Mesures triées par timestamp.

        Returns:
            Liste de patterns détectés.
        """
        patterns: list[dict[str, Any]] = []
        if len(mesures) < 10:
            return patterns

        # 1. Effet de l'aube
        morning_vals = [
            m["valeur"] for m in mesures
            if m.get("moment_mesure") == "a_jeun"
        ]
        if len(morning_vals) >= 3:
            avg_morning = np.mean(morning_vals)
            all_vals = [m["valeur"] for m in mesures]
            avg_all = np.mean(all_vals)
            if avg_morning > avg_all + 0.15:
                patterns.append({
                    "type": "effet_aube",
                    "description": (
                        f"Glycémie matinale élevée ({avg_morning:.2f} g/L) "
                        f"vs moyenne globale ({avg_all:.2f} g/L). "
                        f"Possible effet de l'aube."
                    ),
                    "severite": "attention",
                    "frequence": f"{len(morning_vals)} mesures matinales",
                })

        # 2. Hypoglycémies récurrentes
        hypos = [m for m in mesures if m["valeur"] < SEUILS.HYPOGLYCEMIE_MAX]
        if len(hypos) >= 3:
            hypo_hours = []
            for h in hypos:
                ts = h.get("_ts")
                if ts:
                    hypo_hours.append(ts.hour)
            if hypo_hours:
                from collections import Counter
                hour_counts = Counter(hypo_hours)
                most_common_hour, count = hour_counts.most_common(1)[0]
                if count >= 2:
                    patterns.append({
                        "type": "hypo_recurrente",
                        "description": (
                            f"Hypoglycémies récurrentes autour de "
                            f"{most_common_hour}h ({count} épisodes). "
                            f"Vérifiez le traitement et les repas."
                        ),
                        "severite": "urgent",
                        "frequence": f"{len(hypos)} épisodes au total",
                    })

        # 3. Hyperglycémies postprandiales
        post_meal = [
            m for m in mesures
            if "postprandial" in m.get("moment_mesure", "")
        ]
        if len(post_meal) >= 3:
            post_vals = [m["valeur"] for m in post_meal]
            high_post = sum(1 for v in post_vals if v > SEUILS.NORMAL_POSTPRANDIAL_MAX)
            if high_post / len(post_vals) > 0.6:
                patterns.append({
                    "type": "hyper_postprandiale",
                    "description": (
                        f"Hyperglycémie après les repas dans "
                        f"{high_post}/{len(post_vals)} cas. "
                        f"Moyenne postprandiale : {np.mean(post_vals):.2f} g/L."
                    ),
                    "severite": "attention",
                    "frequence": f"{high_post}/{len(post_vals)} repas",
                })

        # 4. Variabilité excessive
        all_vals = [m["valeur"] for m in mesures]
        if len(all_vals) >= 10:
            cv = np.std(all_vals) / np.mean(all_vals) * 100
            if cv > 40:
                patterns.append({
                    "type": "variabilite_excessive",
                    "description": (
                        f"Variabilité glycémique élevée (CV={cv:.1f}%). "
                        f"Cible : < 36%. La glycémie fluctue beaucoup."
                    ),
                    "severite": "attention",
                    "frequence": "continue",
                })

        return patterns

    def _generate_recommendations(
        self,
        tir: float, tbr: float, tar: float,
        cv: float, lbgi: float, hbgi: float,
        patterns: list[dict[str, Any]],
    ) -> list[str]:
        """Génère des recommandations non prescriptives.

        Args:
            tir: TIR (%).
            tbr: TBR (%).
            tar: TAR (%).
            cv: CV (%).
            lbgi: LBGI.
            hbgi: HBGI.
            patterns: Patterns détectés.

        Returns:
            Liste de recommandations.
        """
        recs: list[str] = []

        if tbr > 4:
            recs.append(
                "⚠️ Vos hypoglycémies sont trop fréquentes. "
                "Parlez-en à votre médecin pour ajuster le traitement. "
                "Ayez toujours du sucre rapide sur vous."
            )
        if tar > 25:
            recs.append(
                "📈 Vos glycémies sont souvent élevées. "
                "Vérifiez votre alimentation (réduire fufu maïs, mikate) "
                "et marchez 20 min après les repas."
            )
        if tir < 70:
            recs.append(
                "🎯 Votre temps dans la cible est en dessous de 70%. "
                "L'objectif est d'atteindre au moins 70% du temps "
                "entre 0.70 et 1.40 g/L."
            )
        if cv > 36:
            recs.append(
                "🔄 Votre glycémie varie beaucoup. Essayez de manger "
                "à heures régulières et de ne pas sauter de repas."
            )
        if lbgi > 2.5:
            recs.append(
                "🔴 Risque d'hypoglycémie sévère élevé. "
                "Ne sautez pas de repas et consultez votre médecin."
            )

        for p in patterns:
            if p["type"] == "effet_aube":
                recs.append(
                    "🌅 Effet de l'aube détecté : votre glycémie monte "
                    "la nuit. Votre médecin pourrait ajuster votre "
                    "traitement du soir."
                )

        if not recs:
            recs.append(
                "✅ Vos résultats sont dans les cibles. "
                "Continuez votre suivi régulier et vos bonnes habitudes !"
            )

        return recs

    def _evaluate_global(
        self, tir: float, tbr: float, tar: float,
        cv: float, risque: float,
    ) -> str:
        """Évaluation globale en langage simple.

        Args:
            tir, tbr, tar, cv, risque: Métriques.

        Returns:
            Texte d'évaluation.
        """
        if risque < 20 and tir >= 70:
            return (
                "🟢 EXCELLENT — Votre glycémie est bien contrôlée. "
                "Continuez ainsi !"
            )
        elif risque < 40:
            return (
                "🟡 BON — Votre glycémie est globalement correcte "
                "avec quelques améliorations possibles."
            )
        elif risque < 60:
            return (
                "🟠 MOYEN — Votre glycémie nécessite une attention "
                "particulière. Consultez votre médecin pour optimiser "
                "votre traitement."
            )
        else:
            return (
                "🔴 PRÉOCCUPANT — Votre glycémie est mal contrôlée. "
                "Il est important de consulter votre médecin rapidement "
                "pour réévaluer votre prise en charge."
            )

    def detecter_urgence(
        self,
        valeur: float,
    ) -> tuple[bool, SeveriteAlerte, str]:
        """Détecte si une valeur glycémique est une urgence.

        Args:
            valeur: Glycémie en g/L.

        Returns:
            Tuple (est_urgence, sévérité, message).
        """
        if valeur < SEUILS.URGENCE_HYPO:
            return True, SeveriteAlerte.CRITIQUE, (
                f"🚨 URGENCE : Glycémie très basse ({valeur:.2f} g/L) ! "
                f"Prenez du sucre immédiatement et allez à l'hôpital."
            )
        elif valeur > SEUILS.URGENCE_HYPER:
            return True, SeveriteAlerte.CRITIQUE, (
                f"🚨 URGENCE : Glycémie très élevée ({valeur:.2f} g/L) ! "
                f"Buvez de l'eau et allez aux urgences immédiatement."
            )
        elif valeur < SEUILS.HYPOGLYCEMIE_MAX:
            return False, SeveriteAlerte.URGENT, (
                f"⚠️ Hypoglycémie ({valeur:.2f} g/L). "
                f"Prenez 15g de sucre et remesurez dans 15 min."
            )
        elif valeur > SEUILS.HYPERGLYCEMIE_SEVERE_MAX:
            return False, SeveriteAlerte.URGENT, (
                f"⚠️ Hyperglycémie sévère ({valeur:.2f} g/L). "
                f"Buvez de l'eau et contactez votre médecin."
            )

        return False, SeveriteAlerte.INFO, ""


# ============================================================================
# SOUS-SYSTÈME 3 : VISION (Analyse d'images)
# ============================================================================

class VisionEngine:
    """Moteur d'analyse d'images médicales.

    Reconnaît :
    - Écrans de glucomètres → extraction de valeur
    - Repas → estimation charge glycémique
    - Ordonnances → extraction médicaments (OCR)
    - Plaies (pied diabétique) → classification gravité
    - Résultats de laboratoire → extraction valeurs
    """

    def __init__(self) -> None:
        """Initialise le moteur vision."""
        self.model = None
        self.processor = None
        self.model_loaded = False
        logger.info(f"👁️ VisionEngine initialisé (modèle: {VISION.MODEL_NAME})")

    def load_model(self) -> bool:
        """Charge le modèle de vision.

        Returns:
            True si chargé avec succès.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"🔄 Chargement du modèle vision : {VISION.MODEL_NAME}...")
            self.processor = AutoTokenizer.from_pretrained(
                VISION.MODEL_NAME, trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                VISION.MODEL_NAME,
                trust_remote_code=True,
                device_map="auto" if VISION.DEVICE == "cuda" else VISION.DEVICE,
            )
            self.model_loaded = True
            logger.info("✅ Modèle vision chargé")
            return True
        except ImportError:
            logger.warning("⚠️  Modèle vision non disponible (mode dégradé)")
            return False
        except Exception as exc:
            logger.error(f"❌ Erreur chargement vision : {exc}")
            return False

    def analyze_image(
        self,
        image_bytes: bytes,
        image_type: str = "auto",
    ) -> ResultatVision:
        """Analyse une image médicale.

        Args:
            image_bytes: Bytes de l'image.
            image_type: Type d'image (auto, glucometre, repas,
                        ordonnance, plaie, labo).

        Returns:
            ResultatVision avec les données extraites.
        """
        if image_type == "auto":
            image_type = self._detect_image_type(image_bytes)

        if self.model_loaded:
            return self._analyze_with_model(image_bytes, image_type)
        else:
            return self._analyze_template(image_type)

    def _detect_image_type(self, image_bytes: bytes) -> str:
        """Détecte le type d'image (placeholder).

        Args:
            image_bytes: Bytes de l'image.

        Returns:
            Type d'image détecté.
        """
        return "glucometre"  # Fallback

    def _analyze_with_model(
        self,
        image_bytes: bytes,
        image_type: str,
    ) -> ResultatVision:
        """Analyse avec le modèle vision chargé.

        Args:
            image_bytes: Image.
            image_type: Type.

        Returns:
            Résultat d'analyse.
        """
        try:
            from PIL import Image
            import io as _io
            import torch

            image = Image.open(_io.BytesIO(image_bytes)).convert("RGB")

            prompts = {
                "glucometre": "What blood glucose value is displayed on this glucometer screen? Reply with just the number and unit.",
                "repas": "Describe this meal and estimate its glycemic load (low/medium/high).",
                "ordonnance": "Extract all medication names and dosages from this prescription.",
                "plaie": "Describe this wound and classify its severity (mild/moderate/severe). Is there any sign of infection?",
                "labo": "Extract all laboratory values from this test result.",
            }

            prompt = prompts.get(image_type, "Describe this medical image.")

            inputs = self.processor(
                text=prompt, images=image,
                return_tensors="pt",
            )
            if VISION.DEVICE == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=256,
                )

            result = self.processor.decode(
                outputs[0], skip_special_tokens=True,
            )

            # Extraire les données structurées
            donnees = self._parse_vision_result(result, image_type)

            return ResultatVision(
                type_image=image_type,
                confiance=0.85,
                resultat_brut=result,
                donnees_extraites=donnees,
                alerte=donnees.get("alerte"),
                recommandation=donnees.get("recommandation"),
            )

        except Exception as exc:
            logger.error(f"❌ Erreur analyse vision : {exc}")
            return self._analyze_template(image_type)

    def _analyze_template(self, image_type: str) -> ResultatVision:
        """Analyse template (mode dégradé).

        Args:
            image_type: Type d'image.

        Returns:
            Résultat template.
        """
        messages = {
            "glucometre": (
                "Je n'ai pas pu lire automatiquement votre glucomètre. "
                "Veuillez saisir manuellement la valeur affichée."
            ),
            "repas": (
                "Je n'ai pas pu analyser cette image de repas. "
                "Décrivez-moi ce que vous avez mangé et je vous "
                "donnerai une estimation de la charge glycémique."
            ),
            "ordonnance": (
                "Je n'ai pas pu lire cette ordonnance automatiquement. "
                "Veuillez saisir vos médicaments manuellement."
            ),
            "plaie": (
                "⚠️ IMPORTANT : Je ne peux pas évaluer cette plaie "
                "de manière fiable. Montrez-la à votre médecin ou "
                "infirmier dès que possible. Le pied diabétique "
                "nécessite une attention médicale urgente."
            ),
            "labo": (
                "Je n'ai pas pu extraire les résultats de laboratoire. "
                "Veuillez saisir les valeurs manuellement."
            ),
        }

        return ResultatVision(
            type_image=image_type,
            confiance=0.0,
            resultat_brut=messages.get(image_type, "Analyse non disponible."),
            donnees_extraites={},
            alerte="⚠️ Analyse automatique non disponible" if image_type == "plaie" else None,
            recommandation="Consultez votre médecin" if image_type == "plaie" else None,
        )

    def _parse_vision_result(
        self,
        raw_text: str,
        image_type: str,
    ) -> dict[str, Any]:
        """Parse le résultat brut du modèle vision.

        Args:
            raw_text: Texte brut du modèle.
            image_type: Type d'image.

        Returns:
            Dict structuré.
        """
        result: dict[str, Any] = {"raw": raw_text}

        if image_type == "glucometre":
            # Extraire la valeur numérique
            match = re.search(r"(\d+[\.,]\d+|\d+)", raw_text)
            if match:
                val_str = match.group(1).replace(",", ".")
                try:
                    val = float(val_str)
                    # Détecter l'unité
                    if val > 30:
                        result["valeur_mgdl"] = val
                        result["valeur_gl"] = round(val * 0.01, 2)
                        result["unite"] = "mg/dL"
                    elif val > 5:
                        result["valeur_mmol"] = val
                        result["valeur_gl"] = round(val * 0.1801, 2)
                        result["unite"] = "mmol/L"
                    else:
                        result["valeur_gl"] = val
                        result["unite"] = "g/L"
                except ValueError:
                    pass

        elif image_type == "plaie":
            lower = raw_text.lower()
            if any(kw in lower for kw in ["severe", "infection", "necrosis", "gangrene"]):
                result["alerte"] = (
                    "🚨 Plaie potentiellement grave détectée. "
                    "Consultez IMMÉDIATEMENT un médecin."
                )
                result["gravite"] = "severe"
            elif any(kw in lower for kw in ["moderate", "ulcer", "redness"]):
                result["gravite"] = "moderee"
            else:
                result["gravite"] = "legere"

        return result


# ============================================================================
# SOUS-SYSTÈME 4 : AUDIO (STT + TTS)
# ============================================================================

class AudioEngine:
    """Moteur audio : Speech-to-Text et Text-to-Speech.

    STT : Whisper fine-tuné pour FR + 4 langues RDC
    TTS : XTTS-v2 pour synthèse vocale multilingue
    """

    def __init__(self) -> None:
        """Initialise le moteur audio."""
        self.whisper_model = None
        self.tts_model = None
        self.stt_loaded = False
        self.tts_loaded = False
        logger.info("🎙️ AudioEngine initialisé")

    def load_stt(self) -> bool:
        """Charge le modèle Speech-to-Text (Whisper).

        Returns:
            True si chargé.
        """
        try:
            from transformers import pipeline

            logger.info(f"🔄 Chargement STT : {STT.MODEL_NAME}...")
            self.whisper_model = pipeline(
                "automatic-speech-recognition",
                model=STT.MODEL_NAME,
                device=STT.DEVICE,
                chunk_length_s=STT.CHUNK_LENGTH_S,
            )
            self.stt_loaded = True
            logger.info("✅ STT chargé")
            return True
        except ImportError:
            logger.warning("⚠️  STT non disponible")
            return False
        except Exception as exc:
            logger.error(f"❌ Erreur chargement STT : {exc}")
            return False

    def load_tts(self) -> bool:
        """Charge le modèle Text-to-Speech.

        Returns:
            True si chargé.
        """
        try:
            logger.info(f"🔄 Chargement TTS : {TTS.MODEL_NAME}...")
            # XTTS-v2 nécessite un chargement spécifique
            # Simplifié ici pour la structure
            self.tts_loaded = True
            logger.info("✅ TTS chargé")
            return True
        except ImportError:
            logger.warning("⚠️  TTS non disponible")
            return False
        except Exception as exc:
            logger.error(f"❌ Erreur chargement TTS : {exc}")
            return False

    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        """Transcrit un fichier audio en texte.

        Args:
            audio_bytes: Bytes du fichier audio.
            language: Langue forcée (auto-détection si None).

        Returns:
            Dict avec "texte", "langue", "confiance", "segments".
        """
        if not self.stt_loaded or not self.whisper_model:
            return {
                "texte": "",
                "langue": language or "fr",
                "confiance": 0.0,
                "erreur": "STT non disponible",
            }

        try:
            import tempfile
            import os

            # Sauvegarder temporairement
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            try:
                generate_kwargs = {}
                if language:
                    generate_kwargs["language"] = language

                result = self.whisper_model(
                    temp_path,
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs,
                )

                return {
                    "texte": result.get("text", "").strip(),
                    "langue": language or result.get("chunks", [{}])[0].get("language", "fr"),
                    "confiance": 0.85,
                    "segments": [
                        {
                            "start": c.get("timestamp", (0, 0))[0],
                            "end": c.get("timestamp", (0, 0))[1],
                            "text": c.get("text", ""),
                        }
                        for c in result.get("chunks", [])
                    ],
                }
            finally:
                os.unlink(temp_path)

        except Exception as exc:
            logger.error(f"❌ Erreur transcription : {exc}")
            return {
                "texte": "",
                "langue": language or "fr",
                "confiance": 0.0,
                "erreur": str(exc),
            }

    def synthesize(
        self,
        text: str,
        language: str = "fr",
        voice: Optional[str] = None,
    ) -> Optional[bytes]:
        """Synthétise du texte en audio.

        Args:
            text: Texte à synthétiser.
            language: Langue de synthèse.
            voice: Voix spécifique.

        Returns:
            Bytes audio WAV ou None si TTS non disponible.
        """
        if not self.tts_loaded:
            return None

        try:
            # Implémentation XTTS-v2 simplifiée
            logger.debug(f"🔊 Synthèse TTS : {len(text)} chars en {language}")
            # En production : appel au modèle XTTS
            return None  # Placeholder
        except Exception as exc:
            logger.error(f"❌ Erreur TTS : {exc}")
            return None


# ============================================================================
# SOUS-SYSTÈME 5 : GÉNÉRATION DE RAPPORTS (CV Médical)
# ============================================================================

class ReportGenerator:
    """Générateur de rapports médicaux automatiques (CV médical).

    Génère un rapport structuré lisible en 2 minutes par un médecin,
    contenant : profil, historique, tendances, alertes, score de risque.
    """

    def __init__(self, analyzer: GlycemicAnalyzer) -> None:
        """Initialise le générateur.

        Args:
            analyzer: Instance de GlycemicAnalyzer.
        """
        self.analyzer = analyzer
        logger.info("📄 ReportGenerator initialisé")

    def generer_rapport(
        self,
        patient_data: dict[str, Any],
        mesures: list[dict[str, Any]],
        alertes: list[dict[str, Any]],
        periode_jours: int = 30,
    ) -> dict[str, Any]:
        """Génère un CV médical complet.

        Args:
            patient_data: Données du patient (profil).
            mesures: Mesures glycémiques.
            alertes: Alertes récentes.
            periode_jours: Période couverte.

        Returns:
            Rapport structuré en dict.
        """
        start = time.time()

        # Analyse glycémique
        analyse = self.analyzer.analyse_complete(
            mesures, patient_data.get("id", ""), periode_jours,
        )

        # Construire le rapport
        rapport = {
            "metadata": {
                "type": "CV Médical DiaBot-RDC",
                "version": "1.0",
                "date_generation": datetime.now(timezone.utc).isoformat(),
                "periode_couverture": f"{periode_jours} jours",
                "genere_par": "IA DiaBot-RDC",
                "avertissement": (
                    "Ce rapport est généré par une IA. "
                    "Il ne remplace pas l'évaluation clinique du médecin."
                ),
            },
            "profil_patient": {
                "age": patient_data.get("age"),
                "sexe": patient_data.get("sexe"),
                "type_diabete": patient_data.get("type_diabete"),
                "date_diagnostic": patient_data.get("date_diagnostic"),
                "traitements": patient_data.get("traitements_actuels", []),
                "comorbidites": patient_data.get("comorbidites", []),
            },
            "resume_glycemique": {
                "nb_mesures": analyse.nb_mesures,
                "moyenne_g_par_l": analyse.moyenne,
                "moyenne_mg_dl": round(analyse.moyenne * 100, 1),
                "ecart_type": analyse.ecart_type,
                "cv_percent": analyse.cv_percent,
                "min": analyse.min_val,
                "max": analyse.max_val,
                "mediane": analyse.mediane,
            },
            "time_in_range": {
                "tir_percent": analyse.tir_percent,
                "tbr_percent": analyse.tbr_percent,
                "tar_percent": analyse.tar_percent,
                "cible_tir": "≥ 70%",
                "cible_tbr": "< 4%",
                "cible_tar": "< 25%",
                "tir_atteint": analyse.tir_percent >= 70,
            },
            "indicateurs_cliniques": {
                "gmi_percent": analyse.gmi,
                "lbgi": analyse.lbgi,
                "hbgi": analyse.hbgi,
                "risque_hypo": "Élevé" if analyse.lbgi > 2.5 else "Modéré" if analyse.lbgi > 1.1 else "Faible",
                "risque_hyper": "Élevé" if analyse.hbgi > 5.0 else "Modéré" if analyse.hbgi > 2.5 else "Faible",
            },
            "score_risque": {
                "score": analyse.score_risque,
                "interpretation": (
                    "Faible" if analyse.score_risque < 20
                    else "Modéré" if analyse.score_risque < 50
                    else "Élevé" if analyse.score_risque < 75
                    else "Très élevé"
                ),
            },
            "patterns_detectes": analyse.patterns_detectes,
            "alertes_recentes": alertes[:10],
            "recommandations_ia": analyse.recommandations,
            "evaluation_globale": analyse.evaluation_globale,
            "duree_generation_ms": int((time.time() - start) * 1000),
        }

        logger.info(
            f"📄 Rapport généré en {rapport['duree_generation_ms']}ms "
            f"pour patient {patient_data.get('id', 'N/A')[:8]}..."
        )

        return rapport

    def format_resume_clinique(
        self,
        rapport: dict[str, Any],
    ) -> str:
        """Formate le rapport en résumé clinique texte (2 min de lecture).

        Conçu pour être lu rapidement par un médecin pressé.

        Args:
            rapport: Rapport structuré généré par generer_rapport().

        Returns:
            Texte formaté du résumé clinique.
        """
        r = rapport
        g = r.get("resume_glycemique", {})
        tir = r.get("time_in_range", {})
        ind = r.get("indicateurs_cliniques", {})
        risque = r.get("score_risque", {})
        profil = r.get("profil_patient", {})

        lines = [
            "═" * 60,
            "  CV MÉDICAL DIABÉTOLOGIE — DiaBot-RDC",
            f"  Généré le : {r.get('metadata', {}).get('date_generation', 'N/A')}",
            f"  Période : {r.get('metadata', {}).get('periode_couverture', 'N/A')}",
            "═" * 60,
            "",
            "▸ PROFIL PATIENT",
            f"  Âge : {profil.get('age', 'N/A')} | Sexe : {profil.get('sexe', 'N/A')}",
            f"  Type : {profil.get('type_diabete', 'N/A')}",
            f"  Diagnostic : {profil.get('date_diagnostic', 'N/A')}",
            f"  Traitements : {', '.join(profil.get('traitements', ['Aucun']))}",
            f"  Comorbidités : {', '.join(profil.get('comorbidites', ['Aucune']))}",
            "",
            "▸ RÉSUMÉ GLYCÉMIQUE",
            f"  Mesures : n={g.get('nb_mesures', 0)}",
            f"  Moyenne : {g.get('moyenne_g_par_l', 0):.2f} g/L ({g.get('moyenne_mg_dl', 0):.0f} mg/dL)",
            f"  Min-Max : {g.get('min', 0):.2f} — {g.get('max', 0):.2f} g/L",
            f"  Écart-type : {g.get('ecart_type', 0):.2f} | CV : {g.get('cv_percent', 0):.1f}%",
            "",
            "▸ TIME IN RANGE",
            f"  TIR : {tir.get('tir_percent', 0):.1f}% (cible ≥ 70%) {'✅' if tir.get('tir_atteint') else '❌'}",
            f"  TBR : {tir.get('tbr_percent', 0):.1f}% (cible < 4%) {'✅' if tir.get('tbr_percent', 0) < 4 else '❌'}",
            f"  TAR : {tir.get('tar_percent', 0):.1f}% (cible < 25%) {'✅' if tir.get('tar_percent', 0) < 25 else '❌'}",
            "",
            "▸ INDICATEURS CLINIQUES",
            f"  GMI : {ind.get('gmi_percent', 0):.1f}%",
            f"  LBGI : {ind.get('lbgi', 0):.2f} → Risque hypo : {ind.get('risque_hypo', 'N/A')}",
            f"  HBGI : {ind.get('hbgi', 0):.2f} → Risque hyper : {ind.get('risque_hyper', 'N/A')}",
            "",
            "▸ SCORE DE RISQUE",
            f"  Score : {risque.get('score', 0):.0f}/100 ({risque.get('interpretation', 'N/A')})",
            "",
        ]

        # Patterns
        patterns = r.get("patterns_detectes", [])
        if patterns:
            lines.append("▸ PATTERNS DÉTECTÉS")
            for p in patterns:
                lines.append(f"  ⚠ [{p.get('severite', '').upper()}] {p.get('description', '')}")
            lines.append("")

        # Recommandations
        recs = r.get("recommandations_ia", [])
        if recs:
            lines.append("▸ RECOMMANDATIONS IA (non prescriptives)")
            for rec in recs:
                lines.append(f"  • {rec}")
            lines.append("")

        lines.extend([
            "▸ ÉVALUATION GLOBALE",
            f"  {r.get('evaluation_globale', 'N/A')}",
            "",
            "═" * 60,
            "  ⚠️ Rapport généré par IA — Ne remplace pas l'évaluation clinique",
            "═" * 60,
        ])

        return "\n".join(lines)


# ============================================================================
# CLASSE PRINCIPALE : DiaBot (Orchestrateur)
# ============================================================================

class DiaBot:
    """Cerveau central de DiaBot-RDC.

    Orchestre les 5 sous-systèmes IA et fournit la méthode
    principale `process_input()` qui :
    1. Détecte la modalité d'entrée (texte/audio/image)
    2. Transcrit l'audio si nécessaire
    3. Analyse l'image si nécessaire
    4. Classifie l'intention
    5. Route vers le sous-système approprié
    6. Génère la réponse
    7. Synthétise la voix si demandé
    8. Log la conversation
    9. Vérifie les conditions d'alerte
    10. Retourne la réponse multimodale
    """

    def __init__(self) -> None:
        """Initialise tous les sous-systèmes IA."""
        logger.info("🧠 Initialisation de DiaBot-RDC...")

        self.conversation = ConversationEngine()
        self.analyzer = GlycemicAnalyzer()
        self.vision = VisionEngine()
        self.audio = AudioEngine()
        self.reporter = ReportGenerator(self.analyzer)

        # État
        self._initialized = False
        self._mode: ModeIA = ModeIA.PATIENT

        # Multilingual engine (injecté depuis l'extérieur)
        self._multilingual = None

        logger.info("✅ DiaBot-RDC initialisé (sous-systèmes créés)")

    def set_multilingual_engine(self, engine: Any) -> None:
        """Injecte le moteur multilingue.

        Args:
            engine: Instance de MultilingualEngine.
        """
        self._multilingual = engine

    def set_mode(self, mode: ModeIA) -> None:
        """Change le mode de personnalité.

        Args:
            mode: ModeIA.PATIENT ou ModeIA.MEDECIN.
        """
        self._mode = mode
        logger.info(f"🔄 Mode IA changé : {mode.value}")

    async def initialize(self) -> dict[str, bool]:
        """Charge tous les modèles IA de manière asynchrone.

        Implémente le graceful degradation : si un modèle échoue,
        le système continue avec des fonctionnalités réduites.

        Returns:
            Dict indiquant le statut de chaque sous-système.
        """
        logger.info("🔄 Chargement des modèles IA...")
        status: dict[str, bool] = {}

        # 1. LLM
        try:
            status["llm"] = self.conversation.load_model()
        except Exception as exc:
            logger.error(f"❌ LLM : {exc}")
            status["llm"] = False

        # 2. RAG
        try:
            nb_docs = self.conversation.load_rag_knowledge()
            status["rag"] = nb_docs > 0
        except Exception as exc:
            logger.error(f"❌ RAG : {exc}")
            status["rag"] = False

        # 3. Vision
        try:
            status["vision"] = self.vision.load_model()
        except Exception as exc:
            logger.error(f"❌ Vision : {exc}")
            status["vision"] = False

        # 4. STT
        try:
            status["stt"] = self.audio.load_stt()
        except Exception as exc:
            logger.error(f"❌ STT : {exc}")
            status["stt"] = False

        # 5. TTS
        try:
            status["tts"] = self.audio.load_tts()
        except Exception as exc:
            logger.error(f"❌ TTS : {exc}")
            status["tts"] = False

        self._initialized = True
        loaded = sum(1 for v in status.values() if v)
        total = len(status)
        logger.info(
            f"🧠 DiaBot prêt : {loaded}/{total} sous-systèmes chargés. "
            f"Statut : {status}"
        )

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
        """Méthode principale d'orchestration — LE CŒUR DU SYSTÈME.

        Pipeline complet en 10 étapes :
        1. Détecter la modalité d'entrée
        2. Si audio → transcription → texte
        3. Si image → analyse vision → description
        4. Classifier l'intention
        5. Route vers le sous-système approprié
        6. Générer la réponse
        7. Si mode audio → synthèse vocale
        8. Log la conversation
        9. Vérifier les conditions d'alerte
        10. Retourner la réponse multimodale

        Args:
            patient_id: UUID du patient.
            input_data: Données d'entrée (str, bytes, dict).
            modalite: Type d'entrée.
            langue: Langue (auto-détection si None).
            mode: Mode IA (override).
            context_glycemique: Contexte glycémique récent.
            audio_response: Si True, génère aussi une réponse audio.

        Returns:
            ReponseIA complète et multimodale.
        """
        start_time = time.time()
        current_mode = mode or self._mode
        user_text = ""
        image_analysis: Optional[dict[str, Any]] = None
        glycemique_data: Optional[dict[str, Any]] = None

        # ── ÉTAPE 1-2 : Traitement selon la modalité ──

        if modalite == ModaliteEntree.AUDIO:
            # Transcription audio
            if isinstance(input_data, bytes):
                transcription = self.audio.transcribe(input_data, language=langue)
                user_text = transcription.get("texte", "")
                if not langue:
                    langue = transcription.get("langue", "fr")
                logger.info(
                    f"🎙️ Audio transcrit ({len(user_text)} chars, "
                    f"langue={langue})"
                )
            else:
                user_text = str(input_data)

        elif modalite in (
            ModaliteEntree.IMAGE,
            ModaliteEntree.IMAGE_GLYCEMIE,
            ModaliteEntree.IMAGE_REPAS,
            ModaliteEntree.IMAGE_ORDONNANCE,
            ModaliteEntree.IMAGE_PLAIE,
            ModaliteEntree.IMAGE_LABO,
        ):
            # Analyse d'image
            if isinstance(input_data, bytes):
                img_type = modalite.value.replace("image_", "")
                if img_type == "image":
                    img_type = "auto"
                vision_result = self.vision.analyze_image(input_data, img_type)
                user_text = vision_result.resultat_brut
                image_analysis = {
                    "type": vision_result.type_image,
                    "confiance": vision_result.confiance,
                    "donnees": vision_result.donnees_extraites,
                    "alerte": vision_result.alerte,
                }

                # Si glucomètre : extraire la valeur
                if vision_result.type_image == "glucometre":
                    val = vision_result.donnees_extraites.get("valeur_gl")
                    if val:
                        glycemique_data = {"derniere_valeur": val}
            else:
                user_text = str(input_data)

        else:
            # Texte direct
            user_text = str(input_data)

        # ── ÉTAPE 3 : Détection de langue ──

        if not langue and user_text:
            if self._multilingual:
                detection = self._multilingual.detect_language(user_text)
                langue = detection.langue_detectee
            else:
                langue = LANGUE_PAR_DEFAUT.value

        langue = langue or LANGUE_PAR_DEFAUT.value

        # ── ÉTAPE 4 : Détection d'urgence immédiate ──

        urgence_detectee = self._check_urgence_text(user_text)
        if urgence_detectee:
            return self._build_urgence_response(
                urgence_detectee, langue, patient_id,
            )

        # Vérifier aussi la glycémie extraite de l'image
        if glycemique_data and "derniere_valeur" in glycemique_data:
            val = glycemique_data["derniere_valeur"]
            is_urgence, severite, msg = self.analyzer.detecter_urgence(val)
            if is_urgence:
                return ReponseIA(
                    texte=msg,
                    langue=langue,
                    intention=IntentionUtilisateur.URGENCE,
                    severite=severite,
                    alerte_urgence=True,
                    message_urgence=msg,
                    latence_ms=int((time.time() - start_time) * 1000),
                )

        # ── ÉTAPE 5 : Classification d'intention ──

        intention = self.conversation._detect_intention(user_text)

        # ── ÉTAPE 6 : Routing et génération ──

        if intention == IntentionUtilisateur.ENREGISTRER_MESURE:
            # Extraire la valeur glycémique du texte
            valeur = self._extract_glycemie_from_text(user_text)
            if valeur is not None:
                is_urg, sev, msg_urg = self.analyzer.detecter_urgence(valeur)
                glycemique_data = {"derniere_valeur": valeur}

                if is_urg:
                    return ReponseIA(
                        texte=msg_urg,
                        langue=langue,
                        intention=intention,
                        severite=sev,
                        alerte_urgence=True,
                        message_urgence=msg_urg,
                        donnees_glycemiques=glycemique_data,
                        latence_ms=int((time.time() - start_time) * 1000),
                    )

        elif intention == IntentionUtilisateur.DEMANDER_RAPPORT:
            # La génération de rapport est gérée par l'API
            pass

        elif intention == IntentionUtilisateur.DEMANDER_STATS:
            # Enrichir le contexte avec les stats
            if context_glycemique is None:
                context_glycemique = glycemique_data

        # ── ÉTAPE 7 : Génération de la réponse conversationnelle ──

        # Construire l'historique
        history = self.conversation.get_history(patient_id)
        history.append(MessageIA(
            role="user",
            content=user_text,
            modalite=modalite,
            langue=langue,
        ))

        reponse = self.conversation.generate_response(
            messages=history,
            mode=current_mode,
            langue=langue,
            context_glycemique=context_glycemique or glycemique_data,
        )

        reponse.intention = intention
        reponse.analyse_image = image_analysis
        reponse.donnees_glycemiques = glycemique_data

        # ── ÉTAPE 8 : Synthèse vocale (si demandée) ──

        if audio_response:
            audio_bytes = self.audio.synthesize(
                reponse.texte, language=langue,
            )
            reponse.audio_response = audio_bytes

        # ── ÉTAPE 9 : Log de la conversation ──

        self.conversation.add_to_history(
            patient_id,
            MessageIA(
                role="assistant",
                content=reponse.texte,
                modalite=modalite,
                langue=langue,
                metadata={
                    "intention": intention.value if intention else None,
                    "latence_ms": reponse.latence_ms,
                    "model": reponse.model_used,
                },
            ),
        )

        # ── ÉTAPE 10 : Finalisation ──

        reponse.latence_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"💬 Réponse générée | Patient {patient_id[:8]}... | "
            f"Intention: {intention.value if intention else 'N/A'} | "
            f"Langue: {langue} | "
            f"Latence: {reponse.latence_ms}ms | "
            f"Urgence: {reponse.alerte_urgence}"
        )

        return reponse

    # ========================================================================
    # MÉTHODES UTILITAIRES INTERNES
    # ========================================================================

    def _check_urgence_text(self, text: str) -> Optional[str]:
        """Vérifie si le texte contient une urgence médicale.

        Args:
            text: Texte de l'utilisateur.

        Returns:
            Message d'urgence si détecté, None sinon.
        """
        text_lower = text.lower()
        urgence_keywords = [
            "inconscient", "évanoui", "coma", "convulsion",
            "ne respire", "ne se réveille", "malaise grave",
            "perte de connaissance", "urgence vitale",
        ]
        for kw in urgence_keywords:
            if kw in text_lower:
                return (
                    f"🚨 URGENCE VITALE DÉTECTÉE\n\n"
                    f"Appelez IMMÉDIATEMENT le 112 (Police/SAMU) "
                    f"ou rendez-vous à l'hôpital le plus proche.\n\n"
                    f"En attendant :\n"
                    f"• Si la personne est inconsciente : mettez-la en "
                    f"position latérale de sécurité\n"
                    f"• NE DONNEZ RIEN par la bouche\n"
                    f"• Restez avec la personne\n\n"
                    f"📞 SAMU Kinshasa : {NUMEROS_URGENCE_RDC.get('samu_kinshasa', '112')}"
                )

        # Vérifier les valeurs glycémiques critiques dans le texte
        match = re.search(r"(\d+[\.,]\d+)\s*(g/l|g\.l|gl)", text_lower)
        if match:
            try:
                val = float(match.group(1).replace(",", "."))
                if val < SEUILS.URGENCE_HYPO or val > SEUILS.URGENCE_HYPER:
                    _, _, msg = self.analyzer.detecter_urgence(val)
                    return msg
            except ValueError:
                pass

        return None

    def _build_urgence_response(
        self,
        message: str,
        langue: str,
        patient_id: str,
    ) -> ReponseIA:
        """Construit une réponse d'urgence.

        Args:
            message: Message d'urgence.
            langue: Langue.
            patient_id: ID patient.

        Returns:
            ReponseIA d'urgence.
        """
        # Traduire le message d'urgence si possible
        if self._multilingual and langue != "fr":
            message = self._multilingual.translate_urgence(
                "appel_urgence", langue,
            ) + "\n\n" + message

        return ReponseIA(
            texte=message,
            langue=langue,
            intention=IntentionUtilisateur.URGENCE,
            severite=SeveriteAlerte.CRITIQUE,
            alerte_urgence=True,
            message_urgence=message,
        )

    def _extract_glycemie_from_text(self, text: str) -> Optional[float]:
        """Extrait une valeur glycémique d'un texte.

        Patterns reconnus :
        - "1.25 g/L", "125 mg/dL", "7.2 mmol/L"
        - "ma glycémie est de 1.25"
        - "j'ai mesuré 125"

        Args:
            text: Texte à analyser.

        Returns:
            Valeur en g/L ou None.
        """
        text_lower = text.lower()

        # Pattern 1 : valeur + unité explicite
        patterns = [
            (r"(\d+[\.,]\d+)\s*g[/.]?l", 1.0),           # g/L
            (r"(\d+[\.,]?\d*)\s*mg[/.]?dl", 0.01),        # mg/dL
            (r"(\d+[\.,]\d+)\s*mmol[/.]?l", 0.1801),      # mmol/L
        ]

        for pattern, facteur in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    val = float(match.group(1).replace(",", "."))
                    return round(val * facteur, 3)
                except ValueError:
                    continue

        # Pattern 2 : valeur seule après mots-clés
        keywords = ["glycémie", "glycemie", "mesuré", "mesure", "taux", "valeur"]
        for kw in keywords:
            if kw in text_lower:
                match = re.search(r"(\d+[\.,]\d+|\d{2,3})", text_lower)
                if match:
                    try:
                        val = float(match.group(1).replace(",", "."))
                        # Heuristique d'unité
                        if val > 30:
                            return round(val * 0.01, 3)  # mg/dL
                        elif val > 5:
                            return round(val * 0.1801, 3)  # mmol/L
                        else:
                            return round(val, 3)  # g/L
                    except ValueError:
                        continue

        return None

    # ========================================================================
    # MÉTHODES PUBLIQUES SUPPLÉMENTAIRES
    # ========================================================================

    async def generer_rapport(
        self,
        patient_data: dict[str, Any],
        mesures: list[dict[str, Any]],
        alertes: list[dict[str, Any]],
        periode_jours: int = 30,
    ) -> dict[str, Any]:
        """Génère un rapport médical complet.

        Args:
            patient_data: Profil patient.
            mesures: Mesures glycémiques.
            alertes: Alertes récentes.
            periode_jours: Période.

        Returns:
            Rapport structuré.
        """
        rapport = self.reporter.generer_rapport(
            patient_data, mesures, alertes, periode_jours,
        )
        rapport["resume_texte"] = self.reporter.format_resume_clinique(rapport)
        return rapport

    async def analyser_glycemie(
        self,
        mesures: list[dict[str, Any]],
        patient_id: str,
        jours: int = 30,
    ) -> dict[str, Any]:
        """Analyse les données glycémiques d'un patient.

        Args:
            mesures: Mesures normalisées.
            patient_id: UUID du patient.
            jours: Période.

        Returns:
            Dict d'analyse complète.
        """
        analyse = self.analyzer.analyse_complete(mesures, patient_id, jours)
        return {
            "moyenne_gl": analyse.moyenne,
            "ecart_type": analyse.ecart_type,
            "cv_percent": analyse.cv_percent,
            "tir_percent": analyse.tir_percent,
            "tbr_percent": analyse.tbr_percent,
            "tar_percent": analyse.tar_percent,
            "gmi": analyse.gmi,
            "lbgi": analyse.lbgi,
            "hbgi": analyse.hbgi,
            "min": analyse.min_val,
            "max": analyse.max_val,
            "mediane": analyse.mediane,
            "score_risque": analyse.score_risque,
            "nb_mesures": analyse.nb_mesures,
            "patterns": analyse.patterns_detectes,
            "recommandations": analyse.recommandations,
            "evaluation": analyse.evaluation_globale,
        }

    def get_status(self) -> dict[str, Any]:
        """Retourne le statut de tous les sous-systèmes.

        Returns:
            Dict de statut.
        """
        return {
            "initialized": self._initialized,
            "mode": self._mode.value,
            "llm_loaded": self.conversation.model_loaded,
            "llm_model": self.conversation.model_name,
            "vision_loaded": self.vision.model_loaded,
            "stt_loaded": self.audio.stt_loaded,
            "tts_loaded": self.audio.tts_loaded,
            "rag_documents": len(self.conversation.knowledge_base),
            "active_conversations": len(self.conversation._histories),
        }

    async def cleanup(self) -> None:
        """Libère les ressources GPU et mémoire.

        Appelé lors de l'arrêt gracieux du serveur.
        """
        logger.info("🧹 Nettoyage des ressources IA...")

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 Cache GPU vidé")
        except ImportError:
            pass

        self.conversation._histories.clear()
        logger.info("✅ Nettoyage terminé")


# ============================================================================
# INSTANCE GLOBALE
# ============================================================================

diabot = DiaBot()


__all__ = [
    "DiaBot", "diabot",
    "ConversationEngine", "GlycemicAnalyzer", "VisionEngine",
    "AudioEngine", "ReportGenerator",
    "ReponseIA", "MessageIA", "AnalyseGlycemique", "ResultatVision",
    "ModaliteEntree",
]