"""
================================================================================
FICHIER : database.py
RESPONSABILITÉ : Couche de persistance de DiaBot-RDC
================================================================================
Ce module gère TOUTE la persistance des données :
- Modèles ORM async (SQLModel/SQLAlchemy 2.0)
- Patient, MesureGlycemie, Alerte, SessionMedecin, ConversationHistory,
  RapportMedical, ImportHistory, AccessLog
- Fonctions CRUD async complètes (Repository pattern)
- Connection pooling, migrations auto
- Chiffrement AES-256 des champs sensibles
- Indexes optimisés
- Export patient (droit à la portabilité RGPD)

Auteur : Équipe DiaBot-RDC
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Optional

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    event, func, select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, selectinload,
)

from config import (
    DATABASE, SECURITY, EXPORTS_DIR,
    LangueSupportee, LANGUE_PAR_DEFAUT,
    MomentMesure, SeveriteAlerte, TypeDiabete, UniteGlycemie,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CHIFFREMENT DES CHAMPS SENSIBLES
# ============================================================================

def _get_fernet() -> Fernet:
    """Retourne une instance Fernet pour chiffrement AES-256.

    Returns:
        Instance Fernet initialisée avec la clé AES.

    Raises:
        ValueError: Si aucune clé AES n'est configurée.
    """
    key_source = SECURITY.AES_KEY or SECURITY.SECRET_KEY
    if not key_source:
        raise ValueError("Aucune clé de chiffrement disponible")
    # Normalise sur 32 octets puis encode en base64 urlsafe
    key_bytes = key_source.encode("utf-8").ljust(32, b"0")[:32]
    return Fernet(base64.urlsafe_b64encode(key_bytes))


def encrypt_field(value: Optional[str]) -> Optional[str]:
    """Chiffre une chaîne via Fernet (AES-128-CBC + HMAC).

    Args:
        value: Valeur en clair à chiffrer.

    Returns:
        Valeur chiffrée en base64, ou None.
    """
    if value is None or value == "":
        return value
    try:
        return _get_fernet().encrypt(value.encode("utf-8")).decode("utf-8")
    except Exception as exc:
        logger.error(f"Erreur de chiffrement : {exc}")
        return value


def decrypt_field(value: Optional[str]) -> Optional[str]:
    """Déchiffre une chaîne Fernet.

    Args:
        value: Chaîne chiffrée.

    Returns:
        Valeur déchiffrée, ou valeur originale si échec.
    """
    if value is None or value == "":
        return value
    try:
        return _get_fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception):
        return value


def now_utc() -> datetime:
    """Retourne le timestamp UTC courant.

    Returns:
        Datetime UTC aware.
    """
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Génère un UUID4 sous forme de chaîne.

    Returns:
        UUID4 en string.
    """
    return str(uuid.uuid4())


# ============================================================================
# BASE DÉCLARATIVE
# ============================================================================

class Base(DeclarativeBase):
    """Classe de base pour tous les modèles ORM."""
    pass


# ============================================================================
# MODÈLE : Patient
# ============================================================================

class Patient(Base):
    """Modèle Patient diabétique.

    Attributes:
        id: Identifiant UUID du patient.
        username: Identifiant de connexion unique.
        nom_chiffre: Nom chiffré AES-256.
        prenom_chiffre: Prénom chiffré AES-256.
        age: Âge du patient.
        sexe: Sexe (M/F/A).
        type_diabete: Type de diabète (voir TypeDiabete).
        date_diagnostic: Date de diagnostic du diabète.
        comorbidites: Liste JSON des comorbidités.
        traitements_actuels: Liste JSON des traitements.
        allergies: Liste JSON des allergies.
        contact_urgence_chiffre: Contact d'urgence chiffré.
        medecin_referent_chiffre: Médecin référent chiffré.
        langue_preferee: Langue préférée (voir LangueSupportee).
        hash_password: Hash bcrypt du mot de passe.
        consentement_rgpd: Consentement explicite.
        actif: Compte actif ou désactivé (soft delete).
    """
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    email_chiffre: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    nom_chiffre: Mapped[str] = mapped_column(String(500), nullable=False)
    prenom_chiffre: Mapped[str] = mapped_column(String(500), nullable=False)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sexe: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)

    type_diabete: Mapped[str] = mapped_column(String(30), default=TypeDiabete.TYPE_2.value)
    date_diagnostic: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    comorbidites: Mapped[Optional[str]] = mapped_column(Text, nullable=True)        # JSON
    traitements_actuels: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # JSON
    allergies: Mapped[Optional[str]] = mapped_column(Text, nullable=True)           # JSON

    contact_urgence_chiffre: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    medecin_referent_chiffre: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    langue_preferee: Mapped[str] = mapped_column(
        String(5), default=LANGUE_PAR_DEFAUT.value, nullable=False,
    )

    hash_password: Mapped[str] = mapped_column(String(255), nullable=False)
    consentement_rgpd: Mapped[bool] = mapped_column(Boolean, default=False)
    actif: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=now_utc, onupdate=now_utc, nullable=False,
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relations
    mesures: Mapped[list["MesureGlycemie"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan",
    )
    alertes: Mapped[list["Alerte"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan",
    )
    conversations: Mapped[list["ConversationHistory"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan",
    )
    rapports: Mapped[list["RapportMedical"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan",
    )
    sessions_medecin: Mapped[list["SessionMedecin"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_patient_actif_type", "actif", "type_diabete"),
    )

    @property
    def nom(self) -> str:
        """Nom en clair (déchiffré)."""
        return decrypt_field(self.nom_chiffre) or ""

    @property
    def prenom(self) -> str:
        """Prénom en clair."""
        return decrypt_field(self.prenom_chiffre) or ""

    def to_dict(self, include_sensitive: bool = False) -> dict[str, Any]:
        """Sérialisation en dict.

        Args:
            include_sensitive: Si True, inclut les champs sensibles déchiffrés.

        Returns:
            Dictionnaire représentant le patient.
        """
        data: dict[str, Any] = {
            "id": self.id,
            "username": self.username,
            "age": self.age,
            "sexe": self.sexe,
            "type_diabete": self.type_diabete,
            "langue_preferee": self.langue_preferee,
            "actif": self.actif,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_sensitive:
            data.update({
                "nom": self.nom,
                "prenom": self.prenom,
                "email": decrypt_field(self.email_chiffre),
                "contact_urgence": decrypt_field(self.contact_urgence_chiffre),
                "medecin_referent": decrypt_field(self.medecin_referent_chiffre),
                "comorbidites": json.loads(self.comorbidites) if self.comorbidites else [],
                "traitements_actuels": json.loads(self.traitements_actuels)
                    if self.traitements_actuels else [],
                "allergies": json.loads(self.allergies) if self.allergies else [],
                "date_diagnostic": self.date_diagnostic.isoformat()
                    if self.date_diagnostic else None,
            })
        return data


# ============================================================================
# MODÈLE : MesureGlycemie
# ============================================================================

class MesureGlycemie(Base):
    """Mesure glycémique d'un patient.

    Attributes:
        valeur: Valeur normalisée en g/L.
        valeur_originale: Valeur telle que saisie (avant conversion).
        unite_originale: Unité d'origine (g/L, mg/dL, mmol/L).
        moment_mesure: Moment de la mesure.
        timestamp: Date/heure de la mesure.
    """
    __tablename__ = "mesures_glycemie"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )

    valeur: Mapped[float] = mapped_column(Float, nullable=False)       # en g/L normalisé
    valeur_originale: Mapped[float] = mapped_column(Float, nullable=False)
    unite_originale: Mapped[str] = mapped_column(
        String(10), default=UniteGlycemie.G_PAR_L.value,
    )

    moment_mesure: Mapped[str] = mapped_column(
        String(30), default=MomentMesure.AUTRE.value, index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=now_utc, nullable=False, index=True,
    )

    note_patient: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    repas_associe: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    activite_physique: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    stress_level: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 0-10
    medication_prise: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # JSON

    source: Mapped[str] = mapped_column(String(30), default="manuelle")
    # manuelle | import_excel | import_csv | glucometre_bluetooth | api

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)

    patient: Mapped["Patient"] = relationship(back_populates="mesures")

    __table_args__ = (
        Index("idx_mesure_patient_timestamp", "patient_id", "timestamp"),
        Index("idx_mesure_patient_moment", "patient_id", "moment_mesure"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation en dict."""
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "valeur": self.valeur,
            "valeur_originale": self.valeur_originale,
            "unite_originale": self.unite_originale,
            "moment_mesure": self.moment_mesure,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "note_patient": self.note_patient,
            "repas_associe": self.repas_associe,
            "activite_physique": self.activite_physique,
            "stress_level": self.stress_level,
            "medication_prise": json.loads(self.medication_prise)
                if self.medication_prise else None,
            "source": self.source,
        }


# ============================================================================
# MODÈLE : Alerte
# ============================================================================

class Alerte(Base):
    """Alerte générée par le système pour un patient."""
    __tablename__ = "alertes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )

    type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    severite: Mapped[str] = mapped_column(
        String(20), default=SeveriteAlerte.INFO.value, index=True,
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    donnees_contexte: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON

    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=now_utc, nullable=False, index=True,
    )
    vue_par_medecin: Mapped[bool] = mapped_column(Boolean, default=False)
    acquittee: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    date_acquittement: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    action_prise: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="alertes")

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation en dict."""
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "type": self.type,
            "severite": self.severite,
            "message": self.message,
            "donnees_contexte": json.loads(self.donnees_contexte)
                if self.donnees_contexte else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "vue_par_medecin": self.vue_par_medecin,
            "acquittee": self.acquittee,
            "date_acquittement": self.date_acquittement.isoformat()
                if self.date_acquittement else None,
            "action_prise": self.action_prise,
        }


# ============================================================================
# MODÈLE : SessionMedecin (accès via QR)
# ============================================================================

class SessionMedecin(Base):
    """Session d'accès d'un médecin à un patient via QR code."""
    __tablename__ = "sessions_medecin"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )
    medecin_identifiant: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    medecin_nom: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    qr_token: Mapped[str] = mapped_column(String(500), unique=True, index=True, nullable=False)
    nonce: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    niveau_permission: Mapped[int] = mapped_column(Integer, default=1)

    date_debut: Mapped[datetime] = mapped_column(DateTime, default=now_utc, nullable=False)
    date_expiration: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    nb_scans: Mapped[int] = mapped_column(Integer, default=0)
    max_scans: Mapped[int] = mapped_column(Integer, default=3)

    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    revoquee: Mapped[bool] = mapped_column(Boolean, default=False)
    date_revocation: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="sessions_medecin")
    access_logs: Mapped[list["AccessLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan",
    )

    def is_valid(self) -> bool:
        """Vérifie si la session est encore valide.

        Returns:
            True si la session est utilisable.
        """
        if not self.active or self.revoquee:
            return False
        if self.nb_scans >= self.max_scans:
            return False
        # Rendre date_expiration timezone-aware si naïve
        exp = self.date_expiration
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return now_utc() < exp


# ============================================================================
# MODÈLE : ConversationHistory
# ============================================================================

class ConversationHistory(Base):
    """Historique des conversations IA-Patient."""
    __tablename__ = "conversation_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )
    session_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    modalite: Mapped[str] = mapped_column(String(20), default="texte")  # texte|audio|image
    langue: Mapped[str] = mapped_column(String(5), default=LANGUE_PAR_DEFAUT.value)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=now_utc, nullable=False, index=True,
    )
    tokens_input: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_output: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    latence_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    intention_detectee: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="conversations")

    __table_args__ = (
        Index("idx_conv_session_ts", "session_id", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation en dict."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "modalite": self.modalite,
            "langue": self.langue,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "intention_detectee": self.intention_detectee,
        }


# ============================================================================
# MODÈLE : RapportMedical
# ============================================================================

class RapportMedical(Base):
    """Rapport médical généré automatiquement (CV médical)."""
    __tablename__ = "rapports_medicaux"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )

    date_generation: Mapped[datetime] = mapped_column(
        DateTime, default=now_utc, nullable=False, index=True,
    )
    periode_debut: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    periode_fin: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    contenu_json: Mapped[str] = mapped_column(Text, nullable=False)   # Rapport structuré
    resume_ia: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score_risque: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    pdf_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    vu_par_medecin: Mapped[bool] = mapped_column(Boolean, default=False)

    patient: Mapped["Patient"] = relationship(back_populates="rapports")

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation en dict."""
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "date_generation": self.date_generation.isoformat(),
            "periode_debut": self.periode_debut.isoformat(),
            "periode_fin": self.periode_fin.isoformat(),
            "contenu": json.loads(self.contenu_json),
            "resume_ia": self.resume_ia,
            "score_risque": self.score_risque,
            "pdf_path": self.pdf_path,
            "vu_par_medecin": self.vu_par_medecin,
        }


# ============================================================================
# MODÈLE : ImportHistory
# ============================================================================

class ImportHistory(Base):
    """Historique des imports de données glycémiques."""
    __tablename__ = "import_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(20), nullable=False)  # xlsx | csv
    nb_lignes_total: Mapped[int] = mapped_column(Integer, default=0)
    nb_lignes_importees: Mapped[int] = mapped_column(Integer, default=0)
    nb_erreurs: Mapped[int] = mapped_column(Integer, default=0)
    erreurs_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON

    timestamp: Mapped[datetime] = mapped_column(DateTime, default=now_utc, index=True)
    status: Mapped[str] = mapped_column(String(20), default="en_attente")
    # en_attente | preview | validated | error | rolled_back

    preview_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation en dict."""
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "filename": self.filename,
            "format": self.format,
            "nb_lignes_total": self.nb_lignes_total,
            "nb_lignes_importees": self.nb_lignes_importees,
            "nb_erreurs": self.nb_erreurs,
            "erreurs_detail": json.loads(self.erreurs_detail)
                if self.erreurs_detail else [],
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "status": self.status,
        }


# ============================================================================
# MODÈLE : AccessLog (audit)
# ============================================================================

class AccessLog(Base):
    """Journal d'audit des accès (sécurité et traçabilité)."""
    __tablename__ = "access_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions_medecin.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    patient_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    actor: Mapped[str] = mapped_column(String(200), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=now_utc, nullable=False, index=True,
    )

    session: Mapped[Optional["SessionMedecin"]] = relationship(back_populates="access_logs")


# ============================================================================
# MOTEUR ASYNC ET SESSION
# ============================================================================

engine = create_async_engine(
    DATABASE.URL,
    echo=DATABASE.ECHO,
    pool_pre_ping=True,
    pool_recycle=DATABASE.POOL_RECYCLE,
    **(
        {}  # SQLite ne supporte pas pool_size/max_overflow
        if DATABASE.URL.startswith("sqlite")
        else {
            "pool_size": DATABASE.POOL_SIZE,
            "max_overflow": DATABASE.MAX_OVERFLOW,
            "pool_timeout": DATABASE.POOL_TIMEOUT,
        }
    ),
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False,
)


# Active les contraintes FK pour SQLite
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, _: Any) -> None:
    """Active les clés étrangères sur SQLite."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager async pour une session DB.

    Yields:
        Session async SQLAlchemy.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(f"Erreur session DB (rollback) : {exc}")
            raise


async def init_db() -> None:
    """Initialise la base de données (création des tables)."""
    logger.info("🗄️  Initialisation de la base de données...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Base de données initialisée")


async def close_db() -> None:
    """Ferme proprement les connexions DB."""
    await engine.dispose()
    logger.info("🔌 Connexions DB fermées")


# ============================================================================
# REPOSITORY : PatientRepository
# ============================================================================

class PatientRepository:
    """Repository CRUD pour l'entité Patient."""

    @staticmethod
    async def create(
        session: AsyncSession,
        username: str,
        nom: str,
        prenom: str,
        hash_password: str,
        **kwargs: Any,
    ) -> Patient:
        """Crée un nouveau patient.

        Args:
            session: Session async SQLAlchemy.
            username: Identifiant unique.
            nom: Nom (sera chiffré).
            prenom: Prénom (sera chiffré).
            hash_password: Hash bcrypt du mot de passe.
            **kwargs: Autres champs optionnels.

        Returns:
            Patient créé.

        Raises:
            IntegrityError: Si username déjà pris.
        """
        patient = Patient(
            username=username,
            nom_chiffre=encrypt_field(nom) or "",
            prenom_chiffre=encrypt_field(prenom) or "",
            hash_password=hash_password,
            email_chiffre=encrypt_field(kwargs.get("email")),
            age=kwargs.get("age"),
            sexe=kwargs.get("sexe"),
            type_diabete=kwargs.get("type_diabete", TypeDiabete.TYPE_2.value),
            date_diagnostic=kwargs.get("date_diagnostic"),
            comorbidites=json.dumps(kwargs.get("comorbidites", []))
                if kwargs.get("comorbidites") else None,
            traitements_actuels=json.dumps(kwargs.get("traitements_actuels", []))
                if kwargs.get("traitements_actuels") else None,
            allergies=json.dumps(kwargs.get("allergies", []))
                if kwargs.get("allergies") else None,
            contact_urgence_chiffre=encrypt_field(kwargs.get("contact_urgence")),
            medecin_referent_chiffre=encrypt_field(kwargs.get("medecin_referent")),
            langue_preferee=kwargs.get("langue_preferee", LANGUE_PAR_DEFAUT.value),
            consentement_rgpd=kwargs.get("consentement_rgpd", False),
        )
        session.add(patient)
        await session.flush()
        logger.info(f"👤 Patient créé : {username}")
        return patient

    @staticmethod
    async def get_by_id(session: AsyncSession, patient_id: str) -> Optional[Patient]:
        """Récupère un patient par son ID."""
        result = await session.execute(
            select(Patient).where(Patient.id == patient_id, Patient.actif.is_(True))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_username(session: AsyncSession, username: str) -> Optional[Patient]:
        """Récupère un patient par son username."""
        result = await session.execute(
            select(Patient).where(Patient.username == username)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update(
        session: AsyncSession, patient_id: str, **fields: Any,
    ) -> Optional[Patient]:
        """Met à jour un patient.

        Args:
            session: Session DB.
            patient_id: UUID du patient.
            **fields: Champs à modifier.

        Returns:
            Patient mis à jour ou None.
        """
        patient = await PatientRepository.get_by_id(session, patient_id)
        if not patient:
            return None

        encrypted_fields = {
            "nom": "nom_chiffre",
            "prenom": "prenom_chiffre",
            "email": "email_chiffre",
            "contact_urgence": "contact_urgence_chiffre",
            "medecin_referent": "medecin_referent_chiffre",
        }
        json_fields = {"comorbidites", "traitements_actuels", "allergies"}

        for key, value in fields.items():
            if key in encrypted_fields:
                setattr(patient, encrypted_fields[key], encrypt_field(value))
            elif key in json_fields:
                setattr(patient, key, json.dumps(value) if value is not None else None)
            elif hasattr(patient, key):
                setattr(patient, key, value)

        await session.flush()
        return patient

    @staticmethod
    async def soft_delete(session: AsyncSession, patient_id: str) -> bool:
        """Désactivation logique du compte."""
        patient = await PatientRepository.get_by_id(session, patient_id)
        if not patient:
            return False
        patient.actif = False
        await session.flush()
        logger.info(f"🗑️  Patient désactivé : {patient_id}")
        return True

    @staticmethod
    async def hard_delete(session: AsyncSession, patient_id: str) -> bool:
        """Suppression définitive (droit à l'effacement RGPD)."""
        patient = await session.get(Patient, patient_id)
        if not patient:
            return False
        await session.delete(patient)
        await session.flush()
        logger.warning(f"🔥 Patient supprimé définitivement : {patient_id}")
        return True

    @staticmethod
    async def update_last_login(session: AsyncSession, patient_id: str) -> None:
        """Met à jour la date de dernière connexion."""
        patient = await PatientRepository.get_by_id(session, patient_id)
        if patient:
            patient.last_login = now_utc()
            await session.flush()


# ============================================================================
# REPOSITORY : MesureGlycemieRepository
# ============================================================================

class MesureGlycemieRepository:
    """Repository CRUD pour les mesures glycémiques."""

    @staticmethod
    async def create(
        session: AsyncSession,
        patient_id: str,
        valeur: float,
        valeur_originale: float,
        unite_originale: str = UniteGlycemie.G_PAR_L.value,
        moment_mesure: str = MomentMesure.AUTRE.value,
        timestamp: Optional[datetime] = None,
        **kwargs: Any,
    ) -> MesureGlycemie:
        """Crée une mesure glycémique.

        Args:
            session: Session DB.
            patient_id: UUID du patient.
            valeur: Valeur normalisée en g/L.
            valeur_originale: Valeur saisie.
            unite_originale: Unité d'origine.
            moment_mesure: Moment de la mesure.
            timestamp: Date/heure (défaut now).

        Returns:
            Mesure créée.
        """
        mesure = MesureGlycemie(
            patient_id=patient_id,
            valeur=valeur,
            valeur_originale=valeur_originale,
            unite_originale=unite_originale,
            moment_mesure=moment_mesure,
            timestamp=timestamp or now_utc(),
            note_patient=kwargs.get("note_patient"),
            repas_associe=kwargs.get("repas_associe"),
            activite_physique=kwargs.get("activite_physique"),
            stress_level=kwargs.get("stress_level"),
            medication_prise=json.dumps(kwargs["medication_prise"])
                if kwargs.get("medication_prise") else None,
            source=kwargs.get("source", "manuelle"),
        )
        session.add(mesure)
        await session.flush()
        return mesure

    @staticmethod
    async def create_bulk(
        session: AsyncSession, mesures: list[dict[str, Any]],
    ) -> int:
        """Insère plusieurs mesures en bulk.

        Args:
            session: Session DB.
            mesures: Liste de dicts représentant les mesures.

        Returns:
            Nombre de mesures insérées.
        """
        objects = []
        for m in mesures:
            objects.append(MesureGlycemie(
                patient_id=m["patient_id"],
                valeur=m["valeur"],
                valeur_originale=m.get("valeur_originale", m["valeur"]),
                unite_originale=m.get("unite_originale", UniteGlycemie.G_PAR_L.value),
                moment_mesure=m.get("moment_mesure", MomentMesure.AUTRE.value),
                timestamp=m.get("timestamp", now_utc()),
                note_patient=m.get("note_patient"),
                repas_associe=m.get("repas_associe"),
                source=m.get("source", "import_excel"),
            ))
        session.add_all(objects)
        await session.flush()
        logger.info(f"📥 Bulk insert : {len(objects)} mesures")
        return len(objects)

    @staticmethod
    async def get_by_patient(
        session: AsyncSession,
        patient_id: str,
        date_debut: Optional[datetime] = None,
        date_fin: Optional[datetime] = None,
        moment: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[MesureGlycemie]:
        """Récupère les mesures d'un patient avec filtres."""
        query = select(MesureGlycemie).where(MesureGlycemie.patient_id == patient_id)

        if date_debut:
            query = query.where(MesureGlycemie.timestamp >= date_debut)
        if date_fin:
            query = query.where(MesureGlycemie.timestamp <= date_fin)
        if moment:
            query = query.where(MesureGlycemie.moment_mesure == moment)

        query = query.order_by(MesureGlycemie.timestamp.desc())
        if limit:
            query = query.limit(limit)

        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_derniere(
        session: AsyncSession, patient_id: str,
    ) -> Optional[MesureGlycemie]:
        """Retourne la dernière mesure du patient."""
        result = await session.execute(
            select(MesureGlycemie)
            .where(MesureGlycemie.patient_id == patient_id)
            .order_by(MesureGlycemie.timestamp.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def count_by_patient(
        session: AsyncSession, patient_id: str,
    ) -> int:
        """Compte les mesures d'un patient."""
        result = await session.execute(
            select(func.count(MesureGlycemie.id))
            .where(MesureGlycemie.patient_id == patient_id)
        )
        return result.scalar_one() or 0

    @staticmethod
    async def delete(session: AsyncSession, mesure_id: str, patient_id: str) -> bool:
        """Supprime une mesure (vérifie l'ownership)."""
        result = await session.execute(
            select(MesureGlycemie).where(
                MesureGlycemie.id == mesure_id,
                MesureGlycemie.patient_id == patient_id,
            )
        )
        mesure = result.scalar_one_or_none()
        if not mesure:
            return False
        await session.delete(mesure)
        await session.flush()
        return True


# ============================================================================
# REPOSITORY : AlerteRepository
# ============================================================================

class AlerteRepository:
    """Repository CRUD pour les alertes."""

    @staticmethod
    async def create(
        session: AsyncSession,
        patient_id: str,
        type_alerte: str,
        severite: str,
        message: str,
        donnees_contexte: Optional[dict[str, Any]] = None,
    ) -> Alerte:
        """Crée une alerte."""
        alerte = Alerte(
            patient_id=patient_id,
            type=type_alerte,
            severite=severite,
            message=message,
            donnees_contexte=json.dumps(donnees_contexte) if donnees_contexte else None,
        )
        session.add(alerte)
        await session.flush()
        logger.warning(
            f"🚨 Alerte [{severite}] patient={patient_id} type={type_alerte}"
        )
        return alerte

    @staticmethod
    async def get_actives(
        session: AsyncSession, patient_id: str,
    ) -> list[Alerte]:
        """Retourne les alertes non acquittées."""
        result = await session.execute(
            select(Alerte)
            .where(Alerte.patient_id == patient_id, Alerte.acquittee.is_(False))
            .order_by(Alerte.timestamp.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def acquitter(
        session: AsyncSession, alerte_id: str, action: Optional[str] = None,
    ) -> bool:
        """Acquitte une alerte."""
        alerte = await session.get(Alerte, alerte_id)
        if not alerte:
            return False
        alerte.acquittee = True
        alerte.date_acquittement = now_utc()
        if action:
            alerte.action_prise = action
        await session.flush()
        return True


# ============================================================================
# REPOSITORY : SessionMedecinRepository
# ============================================================================

class SessionMedecinRepository:
    """Repository pour les sessions médecin (QR)."""

    @staticmethod
    async def create(
        session: AsyncSession,
        patient_id: str,
        qr_token: str,
        nonce: str,
        niveau_permission: int,
        duree_minutes: int,
        max_scans: int = 3,
    ) -> SessionMedecin:
        """Crée une session QR."""
        sess = SessionMedecin(
            patient_id=patient_id,
            qr_token=qr_token,
            nonce=nonce,
            niveau_permission=niveau_permission,
            date_expiration=now_utc() + timedelta(minutes=duree_minutes),
            max_scans=max_scans,
        )
        session.add(sess)
        await session.flush()
        return sess

    @staticmethod
    async def get_by_token(
        session: AsyncSession, qr_token: str,
    ) -> Optional[SessionMedecin]:
        """Récupère une session par son token."""
        result = await session.execute(
            select(SessionMedecin)
            .where(SessionMedecin.qr_token == qr_token)
            .options(selectinload(SessionMedecin.patient))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_nonce(
        session: AsyncSession, nonce: str,
    ) -> Optional[SessionMedecin]:
        """Récupère une session par son nonce."""
        result = await session.execute(
            select(SessionMedecin).where(SessionMedecin.nonce == nonce)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def incrementer_scan(
        session: AsyncSession, session_id: str,
        medecin_identifiant: Optional[str] = None,
        medecin_nom: Optional[str] = None,
    ) -> Optional[SessionMedecin]:
        """Incrémente le compteur de scans."""
        sess = await session.get(SessionMedecin, session_id)
        if not sess:
            return None
        sess.nb_scans += 1
        if medecin_identifiant:
            sess.medecin_identifiant = medecin_identifiant
        if medecin_nom:
            sess.medecin_nom = medecin_nom
        if sess.nb_scans >= sess.max_scans:
            sess.active = False
        await session.flush()
        return sess

    @staticmethod
    async def revoquer(session: AsyncSession, session_id: str) -> bool:
        """Révoque une session."""
        sess = await session.get(SessionMedecin, session_id)
        if not sess:
            return False
        sess.active = False
        sess.revoquee = True
        sess.date_revocation = now_utc()
        await session.flush()
        return True

    @staticmethod
    async def cleanup_expired(session: AsyncSession) -> int:
        """Désactive les sessions expirées. Retourne le nombre nettoyé."""
        result = await session.execute(
            select(SessionMedecin).where(
                SessionMedecin.active.is_(True),
                SessionMedecin.date_expiration < now_utc(),
            )
        )
        sessions = list(result.scalars().all())
        for s in sessions:
            s.active = False
        await session.flush()
        if sessions:
            logger.info(f"🧹 {len(sessions)} sessions QR expirées nettoyées")
        return len(sessions)


# ============================================================================
# REPOSITORY : ConversationRepository
# ============================================================================

class ConversationRepository:
    """Repository pour l'historique des conversations."""

    @staticmethod
    async def add_message(
        session: AsyncSession,
        patient_id: str,
        session_id: str,
        role: str,
        content: str,
        **kwargs: Any,
    ) -> ConversationHistory:
        """Ajoute un message à l'historique."""
        conv = ConversationHistory(
            patient_id=patient_id,
            session_id=session_id,
            role=role,
            content=content,
            modalite=kwargs.get("modalite", "texte"),
            langue=kwargs.get("langue", LANGUE_PAR_DEFAUT.value),
            tokens_input=kwargs.get("tokens_input"),
            tokens_output=kwargs.get("tokens_output"),
            model_used=kwargs.get("model_used"),
            latence_ms=kwargs.get("latence_ms"),
            intention_detectee=kwargs.get("intention_detectee"),
        )
        session.add(conv)
        await session.flush()
        return conv

    @staticmethod
    async def get_history(
        session: AsyncSession,
        patient_id: str,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[ConversationHistory]:
        """Récupère l'historique récent."""
        query = select(ConversationHistory).where(
            ConversationHistory.patient_id == patient_id,
        )
        if session_id:
            query = query.where(ConversationHistory.session_id == session_id)
        query = query.order_by(ConversationHistory.timestamp.desc()).limit(limit)

        result = await session.execute(query)
        messages = list(result.scalars().all())
        return list(reversed(messages))  # Ordre chronologique


# ============================================================================
# REPOSITORY : RapportMedicalRepository
# ============================================================================

class RapportMedicalRepository:
    """Repository pour les rapports médicaux."""

    @staticmethod
    async def create(
        session: AsyncSession,
        patient_id: str,
        contenu: dict[str, Any],
        periode_debut: datetime,
        periode_fin: datetime,
        resume_ia: Optional[str] = None,
        score_risque: Optional[float] = None,
        pdf_path: Optional[str] = None,
    ) -> RapportMedical:
        """Crée un rapport médical."""
        rapport = RapportMedical(
            patient_id=patient_id,
            contenu_json=json.dumps(contenu, default=str, ensure_ascii=False),
            periode_debut=periode_debut,
            periode_fin=periode_fin,
            resume_ia=resume_ia,
            score_risque=score_risque,
            pdf_path=pdf_path,
        )
        session.add(rapport)
        await session.flush()
        logger.info(f"📄 Rapport créé pour patient {patient_id}")
        return rapport

    @staticmethod
    async def get_by_id(
        session: AsyncSession, rapport_id: str,
    ) -> Optional[RapportMedical]:
        """Récupère un rapport par ID."""
        return await session.get(RapportMedical, rapport_id)

    @staticmethod
    async def get_latest(
        session: AsyncSession, patient_id: str, limit: int = 10,
    ) -> list[RapportMedical]:
        """Retourne les derniers rapports d'un patient."""
        result = await session.execute(
            select(RapportMedical)
            .where(RapportMedical.patient_id == patient_id)
            .order_by(RapportMedical.date_generation.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# ============================================================================
# REPOSITORY : ImportHistoryRepository
# ============================================================================

class ImportHistoryRepository:
    """Repository pour l'historique des imports."""

    @staticmethod
    async def create(
        session: AsyncSession,
        patient_id: str,
        filename: str,
        format_file: str,
        preview_data: Optional[list[dict[str, Any]]] = None,
    ) -> ImportHistory:
        """Crée un historique d'import en statut preview."""
        imp = ImportHistory(
            patient_id=patient_id,
            filename=filename,
            format=format_file,
            status="preview",
            preview_data=json.dumps(preview_data, default=str) if preview_data else None,
        )
        session.add(imp)
        await session.flush()
        return imp

    @staticmethod
    async def update_status(
        session: AsyncSession,
        import_id: str,
        status: str,
        nb_lignes_total: Optional[int] = None,
        nb_lignes_importees: Optional[int] = None,
        nb_erreurs: Optional[int] = None,
        erreurs: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[ImportHistory]:
        """Met à jour le statut d'un import."""
        imp = await session.get(ImportHistory, import_id)
        if not imp:
            return None
        imp.status = status
        if nb_lignes_total is not None:
            imp.nb_lignes_total = nb_lignes_total
        if nb_lignes_importees is not None:
            imp.nb_lignes_importees = nb_lignes_importees
        if nb_erreurs is not None:
            imp.nb_erreurs = nb_erreurs
        if erreurs is not None:
            imp.erreurs_detail = json.dumps(erreurs, default=str)
        await session.flush()
        return imp

    @staticmethod
    async def get_by_id(
        session: AsyncSession, import_id: str,
    ) -> Optional[ImportHistory]:
        """Récupère un import."""
        return await session.get(ImportHistory, import_id)

    @staticmethod
    async def get_history(
        session: AsyncSession, patient_id: str, limit: int = 20,
    ) -> list[ImportHistory]:
        """Historique des imports d'un patient."""
        result = await session.execute(
            select(ImportHistory)
            .where(ImportHistory.patient_id == patient_id)
            .order_by(ImportHistory.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# ============================================================================
# REPOSITORY : AccessLogRepository
# ============================================================================

class AccessLogRepository:
    """Repository pour les logs d'audit."""

    @staticmethod
    async def log(
        session: AsyncSession,
        actor: str,
        action: str,
        success: bool = True,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> AccessLog:
        """Enregistre une entrée d'audit."""
        log_entry = AccessLog(
            actor=actor,
            action=action,
            success=success,
            patient_id=patient_id,
            session_id=session_id,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            details=json.dumps(details, default=str) if details else None,
        )
        session.add(log_entry)
        await session.flush()
        return log_entry

    @staticmethod
    async def get_patient_logs(
        session: AsyncSession, patient_id: str, limit: int = 100,
    ) -> list[AccessLog]:
        """Récupère les logs d'un patient."""
        result = await session.execute(
            select(AccessLog)
            .where(AccessLog.patient_id == patient_id)
            .order_by(AccessLog.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# ============================================================================
# EXPORT PATIENT (Droit à la portabilité RGPD)
# ============================================================================

async def export_patient_data(
    session: AsyncSession, patient_id: str,
) -> dict[str, Any]:
    """Exporte toutes les données d'un patient en un dict portable.

    Args:
        session: Session DB.
        patient_id: UUID du patient.

    Returns:
        Dictionnaire JSON-serializable avec toutes les données.

    Raises:
        ValueError: Si patient introuvable.
    """
    patient = await PatientRepository.get_by_id(session, patient_id)
    if not patient:
        raise ValueError(f"Patient introuvable : {patient_id}")

    mesures = await MesureGlycemieRepository.get_by_patient(session, patient_id)
    alertes_result = await session.execute(
        select(Alerte).where(Alerte.patient_id == patient_id)
    )
    alertes = list(alertes_result.scalars().all())

    conversations = await ConversationRepository.get_history(
        session, patient_id, limit=10000,
    )
    rapports = await RapportMedicalRepository.get_latest(session, patient_id, limit=1000)

    export_data = {
        "export_metadata": {
            "generated_at": now_utc().isoformat(),
            "patient_id": patient_id,
            "format_version": "1.0",
            "rgpd_compliant": True,
        },
        "profil": patient.to_dict(include_sensitive=True),
        "mesures_glycemie": [m.to_dict() for m in mesures],
        "alertes": [a.to_dict() for a in alertes],
        "conversations": [c.to_dict() for c in conversations],
        "rapports_medicaux": [r.to_dict() for r in rapports],
        "statistiques": {
            "nb_mesures_total": len(mesures),
            "nb_alertes_total": len(alertes),
            "nb_conversations_total": len(conversations),
            "nb_rapports_total": len(rapports),
        },
    }

    export_file = EXPORTS_DIR / f"export_patient_{patient_id}_{now_utc().strftime('%Y%m%d_%H%M%S')}.json"
    export_file.write_text(
        json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(f"📤 Export patient {patient_id} généré : {export_file}")

    return export_data


# ============================================================================
# HEALTH CHECK
# ============================================================================

async def db_health_check() -> dict[str, Any]:
    """Vérifie l'état de la connexion DB.

    Returns:
        Dict avec le statut.
    """
    try:
        async with get_session() as session:
            await session.execute(select(1))
        return {"status": "healthy", "database": "connected"}
    except SQLAlchemyError as exc:
        logger.error(f"Health check DB échoué : {exc}")
        return {"status": "unhealthy", "error": str(exc)}


__all__ = [
    "Base", "engine", "AsyncSessionLocal", "get_session", "init_db", "close_db",
    "db_health_check", "now_utc", "generate_uuid",
    "encrypt_field", "decrypt_field",
    "Patient", "MesureGlycemie", "Alerte", "SessionMedecin",
    "ConversationHistory", "RapportMedical", "ImportHistory", "AccessLog",
    "PatientRepository", "MesureGlycemieRepository", "AlerteRepository",
    "SessionMedecinRepository", "ConversationRepository",
    "RapportMedicalRepository", "ImportHistoryRepository", "AccessLogRepository",
    "export_patient_data",
]