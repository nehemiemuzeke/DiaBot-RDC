"""
================================================================================
FICHIER : hf_connector.py
RESPONSABILITÉ : Connecteur Hugging Face pour DiaBot-RDC (Datasets + LLM API)
================================================================================
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from datasets import load_dataset

# Charger les variables d'environnement (.env)
load_dotenv()

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "")

class HuggingFaceConnector:
    """Connecteur officiel aux ressources Hugging Face pour DiaBot-RDC."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or HF_TOKEN
        if not self.token:
            logger.warning("⚠️  HF_TOKEN non trouvé ! L'accès aux modèles restreints/API sera limité.")
        else:
            logger.info("🔑 Token Hugging Face chargé avec succès.")

        # Modèle par défaut sur HF (Inference API ultra-rapide)
        self.default_model = "mistralai/Mistral-7B-Instruct-v0.3"
        self.client = InferenceClient(model=self.default_model, token=self.token) if self.token else None

    # =========================================================================
    # 1. RECUPÉRATION DE DONNÉES (DATASETS) DEPUIS HUGGING FACE
    # =========================================================================
    def load_medical_dataset(self, dataset_name: str = "medalpaca/medical_meadow_wikidoc", split: str = "train[:100]") -> List[Dict[str, str]]:
        """Charge un dataset médical depuis Hugging Face pour enrichir le RAG.

        Args:
            dataset_name: Nom du dataset sur Hugging Face.
            split: Fraction des données à charger (ex: 'train[:100]' pour 100 exemples).

        Returns:
            Liste de connaissances extraites.
        """
        logger.info(f"📥 Chargement du dataset Hugging Face : {dataset_name}...")
        knowledge_base = []
        try:
            # Téléchargement du dataset depuis Hugging Face
            ds = load_dataset(dataset_name, split=split)
            
            for item in ds:
                # Adaptez selon la structure du dataset
                input_text = item.get("input", "") or item.get("question", "")
                output_text = item.get("output", "") or item.get("answer", "")
                
                if input_text and output_text:
                    knowledge_base.append({
                        "source": f"HF:{dataset_name}",
                        "content": f"Question: {input_text}\nRéponse médicale: {output_text}"
                    })

            logger.info(f"✅ {len(knowledge_base)} documents médicaux importés depuis Hugging Face.")
            return knowledge_base

        except Exception as exc:
            logger.error(f"❌ Erreur lors du chargement du dataset HF '{dataset_name}': {exc}")
            return []

    # =========================================================================
    # 2. INFERENCE AVEC LES MEILLEURS LLM SUR HUGGING FACE
    # =========================================================================
    def generate_response(self, system_prompt: str, user_message: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Génère une réponse via l'Inference API de Hugging Face (sans GPU local nécessaire).

        Args:
            system_prompt: Consignes de sécurité et rôle.
            user_message: Message du patient.
            max_tokens: Nombre max de tokens générés.
            temperature: Créativité (basse pour la médecine).

        Returns:
            Réponse générée par le modèle sur les serveurs Hugging Face.
        """
        if not self.client:
            return "⚠️ Connexion Hugging Face non disponible (Token manquant)."

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            logger.info(f"🧠 Envoi de la requête au modèle HF : {self.default_model}...")
            
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return response.choices[0].message.content.strip()

        except Exception as exc:
            logger.error(f"❌ Erreur lors de l'appel à l'API HF : {exc}")
            return "Une erreur est survenue lors de la consultation du modèle IA distant."


# Instance globale
hf_connector = HuggingFaceConnector()