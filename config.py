"""
================================================================================
FICHIER : config.py
RESPONSABILITÉ : Configuration centralisée de DiaBot-RDC
================================================================================
Ce module centralise TOUTES les configurations du système :
- Variables d'environnement et secrets
- Seuils glycémiques médicaux (basés sur ADA/IDF/OMS 2024)
- Configuration des modèles IA (LLM, Vision, STT, TTS, Embeddings)
- Configuration multilingue (FR, Lingala, Swahili, Tshiluba, Kikongo)
- Configuration base de données, API, sécurité, alertes
- Constantes médicales et paramètres cliniques

Auteur : Équipe DiaBot-RDC
Version : 1.0.0
Licence : Propriétaire - Usage médical RDC
================================================================================
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ============================================================================
# CHARGEMENT DE L'ENVIRONNEMENT
# ============================================================================

load_dotenv()

BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
LOGS_DIR: Path = BASE_DIR / "logs"
UPLOADS_DIR: Path = BASE_DIR / "uploads"
EXPORTS_DIR: Path = BASE_DIR / "exports"
TRANSLATIONS_DIR: Path = BASE_DIR / "translations"
RAG_DIR: Path = BASE_DIR / "rag_knowledge_base"

for _dir in (DATA_DIR, MODELS_DIR, LOGS_DIR, UPLOADS_DIR,
             EXPORTS_DIR, TRANSLATIONS_DIR, RAG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ENVIRONNEMENT D'EXÉCUTION
# ============================================================================

class Environment(str, Enum):
    """Environnements d'exécution supportés."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


ENVIRONMENT: Environment = Environment(
    os.getenv("ENVIRONMENT", "development").lower()
)
DEBUG: bool = ENVIRONMENT == Environment.DEVELOPMENT
APP_NAME: str = "DiaBot-RDC"
APP_VERSION: str = "1.0.0"


# ============================================================================
# SEUILS GLYCÉMIQUES MÉDICAUX (en g/L — standard RDC/Francophone)
# Source : ADA 2024, IDF 2023, OMS Guidelines
# ============================================================================

class GlycemieLevel(str, Enum):
    """Classification clinique des niveaux glycémiques."""
    URGENCE_HYPO = "urgence_hypoglycemie"           # < 0.54 g/L
    HYPOGLYCEMIE_SEVERE = "hypoglycemie_severe"     # < 0.54 g/L
    HYPOGLYCEMIE = "hypoglycemie"                   # 0.54 - 0.70
    NORMAL_JEUN = "normal_a_jeun"                   # 0.70 - 1.10
    NORMAL_POSTPRANDIAL = "normal_postprandial"     # < 1.40
    PRE_HYPERGLYCEMIE = "pre_hyperglycemie"         # 1.10 - 1.40
    HYPERGLYCEMIE_MODEREE = "hyperglycemie_moderee" # 1.40 - 2.50
    HYPERGLYCEMIE_SEVERE = "hyperglycemie_severe"   # 2.50 - 3.00
    URGENCE_HYPER = "urgence_hyperglycemie"         # > 3.00 g/L


@dataclass(frozen=True)
class SeuilsGlycemiques:
    """Seuils glycémiques en g/L (grammes par litre)."""
    URGENCE_HYPO: float = 0.54
    HYPOGLYCEMIE_MAX: float = 0.70
    NORMAL_JEUN_MIN: float = 0.70
    NORMAL_JEUN_MAX: float = 1.10
    NORMAL_POSTPRANDIAL_MAX: float = 1.40
    HYPERGLYCEMIE_MODEREE_MAX: float = 2.50
    HYPERGLYCEMIE_SEVERE_MAX: float = 3.00
    URGENCE_HYPER: float = 3.00

    # Cibles TIR (Time In Range)
    TIR_CIBLE_MIN_PERCENT: float = 70.0
    TBR_LIMITE_PERCENT: float = 4.0
    TAR_LIMITE_PERCENT: float = 25.0

    # HbA1c (en %)
    HBA1C_CIBLE_STANDARD: float = 7.0
    HBA1C_CIBLE_STRICTE: float = 6.5
    HBA1C_SEUIL_DIAGNOSTIC: float = 6.5

    # Valeurs aberrantes (détection d'erreurs d'import)
    VALEUR_MIN_PLAUSIBLE: float = 0.20
    VALEUR_MAX_PLAUSIBLE: float = 6.00


SEUILS = SeuilsGlycemiques()


# ============================================================================
# CONVERSIONS D'UNITÉS
# ============================================================================

class UniteGlycemie(str, Enum):
    """Unités de mesure glycémique."""
    G_PAR_L = "g/L"
    MG_PAR_DL = "mg/dL"
    MMOL_PAR_L = "mmol/L"


# Facteurs de conversion vers g/L (unité de référence)
CONVERSION_VERS_G_PAR_L: dict[str, float] = {
    "g/L": 1.0,
    "mg/dL": 0.01,         # mg/dL × 0.01 = g/L
    "mmol/L": 0.1801,      # mmol/L × 18.0182 = mg/dL → × 0.01 = g/L
}


# ============================================================================
# MOMENTS DE MESURE
# ============================================================================

class MomentMesure(str, Enum):
    """Moments de prise de mesure glycémique."""
    A_JEUN = "a_jeun"
    AVANT_REPAS = "avant_repas"
    POSTPRANDIAL_1H = "postprandial_1h"
    POSTPRANDIAL_2H = "postprandial_2h"
    AVANT_COUCHER = "avant_coucher"
    NUIT = "nuit"
    APRES_ACTIVITE = "apres_activite"
    AUTRE = "autre"


# ============================================================================
# BASE DE DONNÉES
# ============================================================================

@dataclass(frozen=True)
class DatabaseConfig:
    """Configuration de la base de données."""
    URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{DATA_DIR}/diabot.db",
    )
    ECHO: bool = DEBUG
    POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    POOL_RECYCLE: int = 3600


DATABASE = DatabaseConfig()


# ============================================================================
# SÉCURITÉ ET CHIFFREMENT
# ============================================================================

@dataclass(frozen=True)
class SecurityConfig:
    """Configuration de la sécurité et du chiffrement."""
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "CHANGE_ME_IN_PRODUCTION_USE_SECRETS_TOKEN_URLSAFE_32",
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    BCRYPT_ROUNDS: int = 12

    AES_KEY: str = os.getenv("AES_ENCRYPTION_KEY", "")
    QR_SIGNING_KEY: str = os.getenv("QR_SIGNING_KEY", "")

    # Rate limiting
    RATE_LIMIT_PATIENT_PER_MIN: int = 100
    RATE_LIMIT_MEDECIN_PER_MIN: int = 200
    RATE_LIMIT_LOGIN_PER_MIN: int = 5

    # 2FA
    OTP_LENGTH: int = 6
    OTP_EXPIRE_MINUTES: int = 5


SECURITY = SecurityConfig()


# ============================================================================
# API / SERVEUR
# ============================================================================

@dataclass(frozen=True)
class APIConfig:
    """Configuration du serveur API."""
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = field(default_factory=lambda: os.getenv(
        "CORS_ORIGINS", "*"
    ).split(","))
    MAX_UPLOAD_SIZE_MB: int = 25
    REQUEST_TIMEOUT_SECONDS: int = 60


API = APIConfig()


# ============================================================================
# CONFIGURATION DES MODÈLES IA
# ============================================================================

@dataclass(frozen=True)
class LLMConfig:
    """Configuration du modèle de langage principal."""
    MODEL_NAME: str = os.getenv(
        "LLM_MODEL_NAME",
        "mistralai/Mistral-7B-Instruct-v0.3",
    )
    MODEL_PATH: str = str(MODELS_DIR / "llm")
    DEVICE: str = os.getenv("LLM_DEVICE", "cuda")  # cuda | cpu | mps
    QUANTIZATION: str = os.getenv("LLM_QUANTIZATION", "4bit")
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.3           # Bas pour cohérence médicale
    TOP_P: float = 0.9
    TOP_K: int = 40
    REPETITION_PENALTY: float = 1.1
    CONTEXT_WINDOW: int = 8192
    HISTORY_WINDOW: int = 20           # Nb de messages gardés en contexte
    USE_VLLM: bool = os.getenv("USE_VLLM", "false").lower() == "true"


@dataclass(frozen=True)
class VisionConfig:
    """Configuration du modèle vision."""
    MODEL_NAME: str = os.getenv("VISION_MODEL", "vikhyatk/moondream2")
    MODEL_PATH: str = str(MODELS_DIR / "vision")
    DEVICE: str = os.getenv("VISION_DEVICE", "cuda")
    MAX_IMAGE_SIZE: int = 1024
    SUPPORTED_FORMATS: tuple[str, ...] = ("jpg", "jpeg", "png", "webp", "bmp")


@dataclass(frozen=True)
class STTConfig:
    """Configuration du Speech-to-Text (Whisper)."""
    MODEL_NAME: str = os.getenv("STT_MODEL", "openai/whisper-large-v3")
    MODEL_PATH: str = str(MODELS_DIR / "whisper")
    DEVICE: str = os.getenv("STT_DEVICE", "cuda")
    LANGUAGE_DETECTION: bool = True
    DEFAULT_LANGUAGE: str = "fr"
    CHUNK_LENGTH_S: int = 30
    SUPPORTED_FORMATS: tuple[str, ...] = ("wav", "mp3", "ogg", "m4a", "flac", "webm")


@dataclass(frozen=True)
class TTSConfig:
    """Configuration du Text-to-Speech."""
    MODEL_NAME: str = os.getenv("TTS_MODEL", "coqui/XTTS-v2")
    MODEL_PATH: str = str(MODELS_DIR / "tts")
    DEVICE: str = os.getenv("TTS_DEVICE", "cuda")
    OUTPUT_FORMAT: str = "wav"
    SAMPLE_RATE: int = 24000
    VOICE_SAMPLES_DIR: str = str(MODELS_DIR / "tts" / "voices")


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Configuration des embeddings pour le RAG."""
    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    MODEL_PATH: str = str(MODELS_DIR / "embeddings")
    DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cuda")
    DIMENSION: int = 1024
    BATCH_SIZE: int = 32


@dataclass(frozen=True)
class RAGConfig:
    """Configuration du système Retrieval-Augmented Generation."""
    VECTOR_STORE: str = "chromadb"
    COLLECTION_NAME: str = "diabot_medical_knowledge"
    PERSIST_DIR: str = str(RAG_DIR / "chroma")
    TOP_K_RETRIEVAL: int = 5
    SCORE_THRESHOLD: float = 0.6
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    KNOWLEDGE_SOURCES: tuple[str, ...] = (
        "oms_diabete.md",
        "idf_guidelines.md",
        "ada_standards.md",
        "pharmacopee_antidiabetiques.md",
        "nutrition_rdc.md",
        "activite_physique.md",
    )


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration du modèle prédictif glycémique."""
    MODEL_NAME: str = "lstm_bidirectional"
    MODEL_PATH: str = str(MODELS_DIR / "prediction" / "glucose_lstm.pt")
    HORIZON_HOURS: tuple[int, ...] = (2, 6, 12, 24)
    MIN_HISTORY_POINTS: int = 48
    CONFIDENCE_INTERVAL: float = 0.95


LLM = LLMConfig()
VISION = VisionConfig()
STT = STTConfig()
TTS = TTSConfig()
EMBEDDINGS = EmbeddingsConfig()
RAG = RAGConfig()
PREDICTION = PredictionConfig()


# ============================================================================
# CONFIGURATION MULTILINGUE
# ============================================================================

class LangueSupportee(str, Enum):
    """Langues supportées par DiaBot-RDC."""
    FRANCAIS = "fr"
    LINGALA = "ln"
    SWAHILI = "sw"
    TSHILUBA = "lu"
    KIKONGO = "kg"


LANGUE_PAR_DEFAUT: LangueSupportee = LangueSupportee.FRANCAIS

LANGUES_INFO: dict[str, dict[str, str]] = {
    "fr": {"nom": "Français", "nom_natif": "Français", "iso": "fra"},
    "ln": {"nom": "Lingala", "nom_natif": "Lingála", "iso": "lin"},
    "sw": {"nom": "Swahili", "nom_natif": "Kiswahili", "iso": "swa"},
    "lu": {"nom": "Tshiluba", "nom_natif": "Tshilubà", "iso": "lua"},
    "kg": {"nom": "Kikongo", "nom_natif": "Kikóngo", "iso": "kon"},
}

TRANSLATION_FILES: dict[str, Path] = {
    lang.value: TRANSLATIONS_DIR / f"{lang.value}.json"
    for lang in LangueSupportee
}


# ============================================================================
# MODES DE PERSONNALITÉ DE L'IA
# ============================================================================

class ModeIA(str, Enum):
    """Modes de personnalité de DiaBot-RDC."""
    PATIENT = "patient"
    MEDECIN = "medecin"


# ============================================================================
# ALERTES
# ============================================================================

class SeveriteAlerte(str, Enum):
    """Niveaux de sévérité des alertes."""
    INFO = "info"
    ATTENTION = "attention"
    URGENT = "urgent"
    CRITIQUE = "critique"


class CanalAlerte(str, Enum):
    """Canaux de diffusion des alertes."""
    IN_APP = "in_app"
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    APPEL_URGENCE = "appel_urgence"


ALERTES_CONFIG: dict[str, dict[str, Any]] = {
    SeveriteAlerte.CRITIQUE.value: {
        "canaux": [CanalAlerte.PUSH, CanalAlerte.SMS, CanalAlerte.IN_APP],
        "destinataires": ["patient", "contact_urgence", "medecin"],
        "retry": 3,
        "cooldown_minutes": 5,
    },
    SeveriteAlerte.URGENT.value: {
        "canaux": [CanalAlerte.PUSH, CanalAlerte.IN_APP],
        "destinataires": ["patient", "medecin"],
        "retry": 2,
        "cooldown_minutes": 15,
    },
    SeveriteAlerte.ATTENTION.value: {
        "canaux": [CanalAlerte.IN_APP, CanalAlerte.PUSH],
        "destinataires": ["patient"],
        "retry": 1,
        "cooldown_minutes": 60,
    },
    SeveriteAlerte.INFO.value: {
        "canaux": [CanalAlerte.IN_APP],
        "destinataires": ["patient"],
        "retry": 0,
        "cooldown_minutes": 240,
    },
}


# ============================================================================
# QR CODE
# ============================================================================

@dataclass(frozen=True)
class QRConfig:
    """Configuration des QR codes dynamiques."""
    DUREES_VALIDITE_MINUTES: tuple[int, ...] = (5, 15, 30, 60)
    DUREE_DEFAUT_MINUTES: int = 15
    MAX_SCANS_PAR_QR: int = 3
    REGENERATION_AUTO_MINUTES: int = 10
    QR_SIZE: int = 400
    QR_BORDER: int = 2
    QR_ERROR_CORRECTION: str = "H"  # Haute correction d'erreur


class NiveauPermissionQR(int, Enum):
    """Niveaux de permissions pour l'accès via QR."""
    RESUME_SEUL = 1
    HISTORIQUE_30J = 2
    HISTORIQUE_COMPLET_RAPPORT = 3
    ACCES_TOTAL = 4


QR = QRConfig()


# ============================================================================
# FRÉQUENCES DE MESURE RECOMMANDÉES (par type de diabète)
# ============================================================================

class TypeDiabete(str, Enum):
    """Types de diabète."""
    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    GESTATIONNEL = "gestationnel"
    MODY = "mody"
    AUTRE = "autre"


FREQUENCES_MESURE_RECOMMANDEES: dict[str, int] = {
    TypeDiabete.TYPE_1.value: 4,         # 4+ par jour
    TypeDiabete.TYPE_2.value: 2,         # 2 par jour (sous insuline)
    TypeDiabete.GESTATIONNEL.value: 4,
    TypeDiabete.MODY.value: 2,
    TypeDiabete.AUTRE.value: 2,
}


# ============================================================================
# URGENCES RDC — Numéros à afficher
# ============================================================================

NUMEROS_URGENCE_RDC: dict[str, str] = {
    "police": "112",
    "pompiers": "118",
    "samu_kinshasa": "+243 81 700 0000",
    "centre_antipoison": "+243 81 000 0000",
}


# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if not DEBUG else "DEBUG")
LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)-25s | "
    "%(filename)s:%(lineno)d | %(message)s"
)
LOG_FILE: Path = LOGS_DIR / "diabot.log"
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 5


def configure_logging() -> logging.Logger:
    """Configure le système de logging global.

    Returns:
        Logger racine configuré.
    """
    from logging.handlers import RotatingFileHandler

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Désactiver les logs trop verbeux de certaines libs
    for noisy in ("urllib3", "httpx", "httpcore", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger


# ============================================================================
# SYSTEM PROMPTS DE L'IA (CLAUSE DE SÉCURITÉ ABSOLUE)
# ============================================================================

CLAUSE_SECURITE_ABSOLUE: str = """
RÈGLES ABSOLUES ET NON NÉGOCIABLES :
1. Je suis DiaBot-RDC, un outil d'aide au suivi du diabète. Je NE SUIS PAS médecin.
2. Je ne pose JAMAIS de diagnostic médical.
3. Je ne prescris JAMAIS de traitement ni de médicament.
4. Je ne modifie JAMAIS une prescription existante.
5. En cas de glycémie critique (< 0.54 g/L ou > 3.00 g/L), je déclenche
   une alerte d'urgence et recommande de consulter IMMÉDIATEMENT.
6. Je redirige toujours vers un professionnel de santé pour toute décision
   thérapeutique.
7. Je respecte strictement la vie privée du patient.
8. En cas de doute, je recommande la consultation médicale.
"""

SYSTEM_PROMPT_PATIENT: str = f"""
Tu es DiaBot-RDC, un assistant conversationnel bienveillant spécialisé dans
l'accompagnement des patients diabétiques en République Démocratique du Congo.

TON STYLE :
- Chaleureux, empathique, encourageant
- Langage simple sans jargon médical
- Utilise des analogies culturelles RDC (alimentation : fufu, pondu, makemba,
  sombe, chikwangue, sambusa, poisson salé, etc.)
- Tu peux parler français, lingala, swahili, tshiluba, kikongo
- Réponds dans la langue de l'utilisateur

{CLAUSE_SECURITE_ABSOLUE}

TA MISSION :
- Aider le patient à mieux comprendre sa glycémie
- L'encourager à bien suivre ses mesures
- Lui donner des conseils généraux (nutrition locale, activité physique)
- Détecter les situations préoccupantes et l'orienter
- L'écouter avec bienveillance
"""

SYSTEM_PROMPT_MEDECIN: str = f"""
Tu es DiaBot-RDC en mode professionnel de santé. Tu assistes un médecin ou
un professionnel de santé dans le suivi d'un patient diabétique.

TON STYLE :
- Technique, précis, structuré
- Terminologie médicale appropriée (HbA1c, TIR, TBR, TAR, GMI, CV%, LBGI, HBGI)
- Références aux guidelines : OMS, IDF, ADA 2024
- Format compte-rendu clinique
- Résumés structurés avec tableaux et statistiques
- Classification des alertes par criticité

{CLAUSE_SECURITE_ABSOLUE}

TA MISSION :
- Fournir des synthèses cliniques structurées
- Mettre en évidence les patterns glycémiques
- Signaler les événements notables et les tendances
- Suggérer des pistes d'investigation (evidence-based)
- NE JAMAIS prescrire ni poser de diagnostic définitif
"""


# ============================================================================
# INTENTIONS DE L'UTILISATEUR (pour routing interne)
# ============================================================================

class IntentionUtilisateur(str, Enum):
    """Intentions détectées dans un message utilisateur."""
    QUESTION_MEDICALE = "question_medicale"
    ENREGISTRER_MESURE = "enregistrer_mesure"
    DEMANDER_RAPPORT = "demander_rapport"
    DEMANDER_STATS = "demander_statistiques"
    CONVERSATION_GENERALE = "conversation_generale"
    URGENCE = "urgence"
    SUPPORT_EMOTIONNEL = "support_emotionnel"
    QUESTION_NUTRITION = "question_nutrition"
    QUESTION_ACTIVITE = "question_activite_physique"
    QUESTION_MEDICAMENT = "question_medicament"


# ============================================================================
# PÉRIODES D'ANALYSE
# ============================================================================

PERIODES_ANALYSE_JOURS: dict[str, int] = {
    "7j": 7,
    "14j": 14,
    "30j": 30,
    "90j": 90,
    "180j": 180,
    "365j": 365,
}


# ============================================================================
# VALIDATION AU DÉMARRAGE
# ============================================================================

def validate_config() -> list[str]:
    """Valide la configuration et retourne une liste d'avertissements/erreurs.

    Returns:
        Liste des messages d'avertissement.
    """
    warnings: list[str] = []

    if SECURITY.SECRET_KEY.startswith("CHANGE_ME"):
        warnings.append(
            "⚠️  SECRET_KEY par défaut détectée - À CHANGER EN PRODUCTION"
        )
    if not SECURITY.AES_KEY and ENVIRONMENT == Environment.PRODUCTION:
        warnings.append("⚠️  AES_ENCRYPTION_KEY non définie en production")
    if not SECURITY.QR_SIGNING_KEY:
        warnings.append("⚠️  QR_SIGNING_KEY non définie (fallback SECRET_KEY)")

    return warnings


def get_all_config() -> dict[str, Any]:
    """Retourne un dictionnaire récapitulatif de la configuration (sans secrets).

    Returns:
        Dictionnaire des configurations publiques.
    """
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "environment": ENVIRONMENT.value,
        "debug": DEBUG,
        "api": {
            "host": API.HOST, "port": API.PORT, "prefix": API.PREFIX,
        },
        "database": {"url_scheme": DATABASE.URL.split(":")[0]},
        "llm_model": LLM.MODEL_NAME,
        "vision_model": VISION.MODEL_NAME,
        "stt_model": STT.MODEL_NAME,
        "tts_model": TTS.MODEL_NAME,
        "embeddings_model": EMBEDDINGS.MODEL_NAME,
        "langues_supportees": [l.value for l in LangueSupportee],
        "seuils_glycemie_g_par_l": {
            "urgence_hypo": SEUILS.URGENCE_HYPO,
            "hypo_max": SEUILS.HYPOGLYCEMIE_MAX,
            "normal_jeun": [SEUILS.NORMAL_JEUN_MIN, SEUILS.NORMAL_JEUN_MAX],
            "postprandial_max": SEUILS.NORMAL_POSTPRANDIAL_MAX,
            "hyper_modere_max": SEUILS.HYPERGLYCEMIE_MODEREE_MAX,
            "urgence_hyper": SEUILS.URGENCE_HYPER,
        },
    }


# ============================================================================
# INITIALISATION DU LOGGING AU CHARGEMENT DU MODULE
# ============================================================================

logger = configure_logging()
logger.info(f"✅ Configuration chargée - {APP_NAME} v{APP_VERSION} [{ENVIRONMENT.value}]")

for warning in validate_config():
    logger.warning(warning)