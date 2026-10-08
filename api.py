"""
================================================================================
FICHIER : api.py
RESPONSABILITÉ : API REST FastAPI de DiaBot-RDC
================================================================================
Ce module expose TOUS les endpoints REST de l'application :

  ▸ Authentification (register, login, refresh, logout, forgot-password)
  ▸ Patient (profile CRUD + droit à l'effacement RGPD)
  ▸ Glycémie (mesure, historique, statistiques)
  ▸ Import (upload Excel/CSV, preview, validate)
  ▸ IA (chat texte/audio/image, rapport, alertes)
  ▸ QR Code (generer, scanner, revoquer, log)
  ▸ Visualisations (courbes, dashboard, AGP)
  ▸ Médecin (accès via QR scanné)

Caractéristiques :
- Framework : FastAPI (async)
- Validation : Pydantic v2
- Documentation auto : /docs (Swagger), /redoc
- Middleware : CORS, rate limiting, compression, logging
- Guards : JWT, permissions médecin, OWASP
- Health check endpoint

Auteur : Équipe DiaBot-RDC
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Optional

from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Header, Query,
    Request, Response, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response as FastResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator

from config import (
    API, APP_NAME, APP_VERSION, ENVIRONMENT, SECURITY,
    UPLOADS_DIR, LangueSupportee, LANGUE_PAR_DEFAUT,
    MomentMesure, UniteGlycemie, TypeDiabete,
    NiveauPermissionQR, ModeIA,
)
from database import (
    init_db, close_db, get_session, db_health_check,
    PatientRepository, MesureGlycemieRepository, AlerteRepository,
    SessionMedecinRepository, ConversationRepository,
    RapportMedicalRepository, ImportHistoryRepository, AccessLogRepository,
    export_patient_data, generate_uuid,
)
from security import security_manager, SecurityManager
from multilingual import multilingual_engine
from data_processor import data_processor, convertir_valeur
from visualizations import viz_engine
from qr_manager import qr_manager, PERMISSIONS_PAR_NIVEAU
from ai_engine import diabot, ModaliteEntree

logger = logging.getLogger(__name__)

# ============================================================================
# MODÈLES PYDANTIC (Schemas)
# ============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# AUTHENTIFICATION
# ──────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """Schéma d'inscription."""
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8, max_length=72)
    nom: str = Field(..., min_length=1, max_length=100)
    prenom: str = Field(..., min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, ge=1, le=120)
    sexe: Optional[str] = Field(None, pattern="^[MFA]$")
    type_diabete: str = Field(default=TypeDiabete.TYPE_2.value)
    langue_preferee: str = Field(default=LANGUE_PAR_DEFAUT.value)
    consentement_rgpd: bool = Field(default=False)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Valide la force du mot de passe."""
        if len(v) < 8:
            raise ValueError("Le mot de passe doit faire au moins 8 caractères")
        if not any(c.isdigit() for c in v):
            raise ValueError("Le mot de passe doit contenir au moins un chiffre")
        if not any(c.isalpha() for c in v):
            raise ValueError("Le mot de passe doit contenir au moins une lettre")
        return v

    @field_validator("consentement_rgpd")
    @classmethod
    def validate_consent(cls, v: bool) -> bool:
        """Vérifie le consentement RGPD."""
        if not v:
            raise ValueError("Le consentement RGPD est obligatoire")
        return v


class LoginRequest(BaseModel):
    """Schéma de connexion."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Réponse avec tokens JWT."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Schéma de refresh token."""
    refresh_token: str


# ──────────────────────────────────────────────────────────────────────────────
# PATIENT
# ──────────────────────────────────────────────────────────────────────────────

class PatientProfile(BaseModel):
    """Profil patient."""
    id: str
    username: str
    nom: Optional[str] = None
    prenom: Optional[str] = None
    age: Optional[int] = None
    sexe: Optional[str] = None
    type_diabete: str
    langue_preferee: str
    created_at: Optional[str] = None


class PatientUpdate(BaseModel):
    """Mise à jour profil."""
    nom: Optional[str] = None
    prenom: Optional[str] = None
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, ge=1, le=120)
    sexe: Optional[str] = Field(None, pattern="^[MFA]$")
    type_diabete: Optional[str] = None
    langue_preferee: Optional[str] = None
    contact_urgence: Optional[str] = None
    medecin_referent: Optional[str] = None
    comorbidites: Optional[list[str]] = None
    traitements_actuels: Optional[list[str]] = None
    allergies: Optional[list[str]] = None


# ──────────────────────────────────────────────────────────────────────────────
# GLYCÉMIE
# ──────────────────────────────────────────────────────────────────────────────

class MesureCreate(BaseModel):
    """Création d'une mesure glycémique."""
    valeur: float = Field(..., gt=0, le=10)
    unite: str = Field(default=UniteGlycemie.G_PAR_L.value)
    moment_mesure: str = Field(default=MomentMesure.AUTRE.value)
    timestamp: Optional[datetime] = None
    note_patient: Optional[str] = Field(None, max_length=500)
    repas_associe: Optional[str] = Field(None, max_length=200)
    activite_physique: Optional[str] = Field(None, max_length=200)
    stress_level: Optional[int] = Field(None, ge=0, le=10)


class MesureResponse(BaseModel):
    """Réponse mesure."""
    id: str
    patient_id: str
    valeur: float
    unite_originale: str
    moment_mesure: str
    timestamp: str
    note_patient: Optional[str] = None
    source: str


# ──────────────────────────────────────────────────────────────────────────────
# IA / CHAT
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Requête chat IA."""
    message: str = Field(..., min_length=1, max_length=5000)
    langue: Optional[str] = None
    mode: str = Field(default=ModeIA.PATIENT.value)
    audio_response: bool = False


class ChatResponse(BaseModel):
    """Réponse chat IA."""
    texte: str
    langue: str
    intention: Optional[str] = None
    alerte_urgence: bool = False
    message_urgence: Optional[str] = None
    donnees_glycemiques: Optional[dict[str, Any]] = None
    analyse_image: Optional[dict[str, Any]] = None
    audio_base64: Optional[str] = None
    latence_ms: int = 0
    model_used: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# QR CODE
# ──────────────────────────────────────────────────────────────────────────────

class QRGenerateRequest(BaseModel):
    """Génération de QR code."""
    niveau_permission: int = Field(default=1, ge=1, le=4)
    duree_minutes: int = Field(default=15)
    max_scans: int = Field(default=3, ge=1, le=10)


class QRScanRequest(BaseModel):
    """Scan de QR code (côté médecin)."""
    qr_token: str
    medecin_identifiant: Optional[str] = None
    medecin_nom: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# RÉPONSES GÉNÉRIQUES
# ──────────────────────────────────────────────────────────────────────────────

class SuccessResponse(BaseModel):
    """Réponse de succès générique."""
    success: bool = True
    message: str
    data: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Réponse d'erreur."""
    success: bool = False
    error: str
    detail: Optional[str] = None


# ============================================================================
# LIFESPAN (startup/shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gestion du cycle de vie de l'application.

    Startup : initialisation DB + chargement modèles IA.
    Shutdown : fermeture propre des ressources.
    """
    logger.info(f"🚀 Démarrage {APP_NAME} v{APP_VERSION} [{ENVIRONMENT.value}]")

    # 1. Base de données
    try:
        await init_db()
    except Exception as exc:
        logger.critical(f"❌ Impossible d'initialiser la DB : {exc}")
        raise

    # 2. Modèles IA
    try:
        status = await diabot.initialize()
        diabot.set_multilingual_engine(multilingual_engine)
        logger.info(f"🧠 IA statut : {status}")
    except Exception as exc:
        logger.warning(f"⚠️  IA en mode dégradé : {exc}")

    logger.info("✅ Application prête")
    yield

    # Shutdown
    logger.info("🛑 Arrêt de l'application...")
    try:
        await diabot.cleanup()
        await close_db()
    except Exception as exc:
        logger.error(f"Erreur arrêt : {exc}")
    logger.info("✅ Application arrêtée proprement")


# ============================================================================
# APPLICATION FASTAPI
# ============================================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "API REST de DiaBot-RDC — IA médicale pour le suivi du diabète "
        "en République Démocratique du Congo. Support multilingue (FR, Lingala, "
        "Swahili, Tshiluba, Kikongo). Conforme RGPD."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── MIDDLEWARE ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=API.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def logging_middleware(request: Request, call_next: Any) -> Response:
    """Middleware de logging des requêtes."""
    start = time.time()
    ip = request.client.host if request.client else "unknown"
    try:
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} "
            f"[{duration_ms}ms] IP={ip}"
        )
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response
    except Exception as exc:
        logger.error(f"❌ {request.method} {request.url.path} → {exc}")
        raise


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Any) -> Response:
    """Middleware de rate limiting global."""
    # Exempter certains endpoints
    if request.url.path in ("/health", "/docs", "/redoc", "/openapi.json"):
        return await call_next(request)

    ip = request.client.host if request.client else "unknown"
    key = f"{ip}:{request.url.path}"

    allowed, headers = security_manager.rate_limiter.is_allowed(
        key, max_requests=SECURITY.RATE_LIMIT_PATIENT_PER_MIN,
    )

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": "Trop de requêtes"},
            headers=headers,
        )

    response = await call_next(request)
    for k, v in headers.items():
        response.headers[k] = v
    return response


# ============================================================================
# DÉPENDANCES (Guards)
# ============================================================================

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_patient_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """Guard : extrait et valide le patient_id depuis le JWT.

    Args:
        credentials: Credentials HTTP Bearer.

    Returns:
        patient_id (UUID) du patient authentifié.

    Raises:
        HTTPException 401: Si token invalide.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token d'authentification requis",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = security_manager.jwt.verify_access_token(credentials.credentials)
        return payload.sub
    except Exception as exc:
        logger.warning(f"🔒 Auth échouée : {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_patient_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """Guard optionnel : patient_id si authentifié, sinon None."""
    if not credentials:
        return None
    try:
        payload = security_manager.jwt.verify_access_token(credentials.credentials)
        return payload.sub
    except Exception:
        return None


def get_client_ip(request: Request) -> str:
    """Extrait l'IP du client (gère les proxies)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health", tags=["Système"])
async def health_check() -> dict[str, Any]:
    """Vérifie l'état général du système.

    Returns:
        Statut global incluant DB et modèles IA.
    """
    db_status = await db_health_check()
    ai_status = diabot.get_status()
    return {
        "status": "healthy" if db_status.get("status") == "healthy" else "degraded",
        "app": APP_NAME,
        "version": APP_VERSION,
        "environment": ENVIRONMENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "ai": ai_status,
        "qr_stats": qr_manager.get_stats(),
    }


@app.get("/", tags=["Système"])
async def root() -> dict[str, Any]:
    """Page d'accueil de l'API."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "description": "IA médicale diabétologie pour la RDC",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================================
# ENDPOINTS : AUTHENTIFICATION
# ============================================================================

@app.post(
    f"{API.PREFIX}/auth/register",
    response_model=SuccessResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentification"],
)
async def register(request: Request, payload: RegisterRequest) -> SuccessResponse:
    """Inscription d'un nouveau patient.

    Args:
        payload: Données d'inscription.

    Returns:
        Confirmation d'inscription.

    Raises:
        HTTPException 409: Si username déjà pris.
    """
    ip = get_client_ip(request)

    # Validation sanitization
    sanitized, errors = security_manager.validate_and_sanitize_input(
        payload.model_dump(),
    )
    if errors:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=errors)

    async with get_session() as session:
        existing = await PatientRepository.get_by_username(session, payload.username)
        if existing:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail="Ce nom d'utilisateur est déjà pris",
            )

        hashed = security_manager.password.hash(payload.password)
        patient = await PatientRepository.create(
            session,
            username=payload.username,
            nom=payload.nom,
            prenom=payload.prenom,
            hash_password=hashed,
            email=payload.email,
            age=payload.age,
            sexe=payload.sexe,
            type_diabete=payload.type_diabete,
            langue_preferee=payload.langue_preferee,
            consentement_rgpd=payload.consentement_rgpd,
        )

        # Audit
        await AccessLogRepository.log(
            session, actor=payload.username, action="register",
            patient_id=patient.id, ip_address=ip, success=True,
        )

        # Enregistrer le consentement RGPD
        security_manager.gdpr.record_consent(
            patient.id, "data_processing", True, version="1.0",
        )

    logger.info(f"✅ Nouveau patient inscrit : {payload.username}")
    return SuccessResponse(
        message="Inscription réussie",
        data={"patient_id": patient.id, "username": patient.username},
    )


@app.post(
    f"{API.PREFIX}/auth/login",
    response_model=TokenResponse,
    tags=["Authentification"],
)
async def login(request: Request, payload: LoginRequest) -> TokenResponse:
    """Connexion d'un patient.

    Returns:
        Paire de tokens JWT (access + refresh).
    """
    ip = get_client_ip(request)

    async with get_session() as session:
        patient = await PatientRepository.get_by_username(session, payload.username)
        if not patient or not patient.actif:
            security_manager.auditor.record_login_attempt(
                payload.username, ip, success=False,
            )
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail="Identifiants invalides",
            )

        if not security_manager.password.verify(payload.password, patient.hash_password):
            security_manager.auditor.record_login_attempt(
                payload.username, ip, success=False,
            )
            await AccessLogRepository.log(
                session, actor=payload.username, action="login",
                patient_id=patient.id, ip_address=ip, success=False,
            )
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail="Identifiants invalides",
            )

        # Succès
        await PatientRepository.update_last_login(session, patient.id)
        await AccessLogRepository.log(
            session, actor=payload.username, action="login",
            patient_id=patient.id, ip_address=ip, success=True,
        )
        security_manager.auditor.record_login_attempt(
            payload.username, ip, success=True,
        )

    tokens = security_manager.jwt.create_token_pair(
        patient.id, roles=["patient"],
    )
    logger.info(f"✅ Login : {payload.username}")
    return TokenResponse(**tokens)


@app.post(
    f"{API.PREFIX}/auth/refresh",
    response_model=TokenResponse,
    tags=["Authentification"],
)
async def refresh_token(payload: RefreshRequest) -> TokenResponse:
    """Rafraîchit le token d'accès.

    Returns:
        Nouvelle paire de tokens.
    """
    try:
        verified = security_manager.jwt.verify_refresh_token(payload.refresh_token)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail=f"Refresh token invalide : {exc}",
        )

    tokens = security_manager.jwt.create_token_pair(verified.sub, roles=["patient"])
    return TokenResponse(**tokens)


@app.post(f"{API.PREFIX}/auth/logout", tags=["Authentification"])
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> SuccessResponse:
    """Déconnexion (révocation du token)."""
    if credentials:
        try:
            payload = security_manager.jwt.verify_access_token(credentials.credentials)
            security_manager.jwt.revoke_token(payload.jti)
        except Exception:
            pass
    return SuccessResponse(message="Déconnexion réussie")


# ============================================================================
# ENDPOINTS : PATIENT
# ============================================================================

@app.get(
    f"{API.PREFIX}/patient/profile",
    response_model=PatientProfile,
    tags=["Patient"],
)
async def get_profile(
    patient_id: str = Depends(get_current_patient_id),
) -> PatientProfile:
    """Récupère le profil du patient authentifié."""
    async with get_session() as session:
        patient = await PatientRepository.get_by_id(session, patient_id)
        if not patient:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient introuvable")
        data = patient.to_dict(include_sensitive=True)
        return PatientProfile(**{
            k: v for k, v in data.items()
            if k in PatientProfile.model_fields
        })


@app.put(
    f"{API.PREFIX}/patient/profile",
    response_model=SuccessResponse,
    tags=["Patient"],
)
async def update_profile(
    update: PatientUpdate,
    patient_id: str = Depends(get_current_patient_id),
) -> SuccessResponse:
    """Met à jour le profil du patient."""
    fields = {k: v for k, v in update.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Aucun champ à modifier")

    async with get_session() as session:
        patient = await PatientRepository.update(session, patient_id, **fields)
        if not patient:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient introuvable")

    return SuccessResponse(message="Profil mis à jour")


@app.delete(
    f"{API.PREFIX}/patient/profile",
    response_model=SuccessResponse,
    tags=["Patient"],
)
async def delete_profile(
    hard: bool = Query(False, description="Suppression définitive (RGPD)"),
    patient_id: str = Depends(get_current_patient_id),
) -> SuccessResponse:
    """Désactive ou supprime définitivement le compte (droit à l'effacement)."""
    async with get_session() as session:
        if hard:
            # Exporter avant suppression
            try:
                await export_patient_data(session, patient_id)
            except Exception:
                pass
            success = await PatientRepository.hard_delete(session, patient_id)
            message = "Compte supprimé définitivement (RGPD)"
        else:
            success = await PatientRepository.soft_delete(session, patient_id)
            message = "Compte désactivé"

        if not success:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient introuvable")

    return SuccessResponse(message=message)


@app.get(f"{API.PREFIX}/patient/export", tags=["Patient"])
async def export_data(
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Exporte toutes les données du patient (droit à la portabilité RGPD)."""
    async with get_session() as session:
        try:
            return await export_patient_data(session, patient_id)
        except ValueError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))


# ============================================================================
# ENDPOINTS : GLYCÉMIE
# ============================================================================

@app.post(
    f"{API.PREFIX}/glycemie/mesure",
    response_model=MesureResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Glycémie"],
)
async def creer_mesure(
    payload: MesureCreate,
    patient_id: str = Depends(get_current_patient_id),
) -> MesureResponse:
    """Enregistre une nouvelle mesure glycémique."""
    # Convertir en g/L
    try:
        valeur_gl = convertir_valeur(payload.valeur, payload.unite)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    # Validation plausibilité
    if not security_manager.sanitizer.validate_glycemie_value(
        payload.valeur, payload.unite,
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Valeur glycémique implausible : {payload.valeur} {payload.unite}",
        )

    async with get_session() as session:
        mesure = await MesureGlycemieRepository.create(
            session,
            patient_id=patient_id,
            valeur=valeur_gl,
            valeur_originale=payload.valeur,
            unite_originale=payload.unite,
            moment_mesure=payload.moment_mesure,
            timestamp=payload.timestamp,
            note_patient=payload.note_patient,
            repas_associe=payload.repas_associe,
            activite_physique=payload.activite_physique,
            stress_level=payload.stress_level,
        )

        # Vérifier urgence
        is_urg, severite, msg = diabot.analyzer.detecter_urgence(valeur_gl)
        if is_urg or severite.value in ("urgent", "critique"):
            await AlerteRepository.create(
                session,
                patient_id=patient_id,
                type_alerte="glycemie_critique",
                severite=severite.value,
                message=msg,
                donnees_contexte={"mesure_id": mesure.id, "valeur": valeur_gl},
            )

        return MesureResponse(**mesure.to_dict())


@app.get(f"{API.PREFIX}/glycemie/historique", tags=["Glycémie"])
async def historique_glycemie(
    debut: Optional[datetime] = None,
    fin: Optional[datetime] = None,
    moment: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Récupère l'historique glycémique avec filtres."""
    async with get_session() as session:
        mesures = await MesureGlycemieRepository.get_by_patient(
            session, patient_id,
            date_debut=debut, date_fin=fin,
            moment=moment, limit=limit,
        )
        return {
            "patient_id": patient_id,
            "count": len(mesures),
            "mesures": [m.to_dict() for m in mesures],
        }


@app.get(f"{API.PREFIX}/glycemie/statistiques", tags=["Glycémie"])
async def statistiques_glycemie(
    jours: int = Query(30, ge=1, le=365),
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Calcule les statistiques glycémiques complètes."""
    async with get_session() as session:
        mesures = await MesureGlycemieRepository.get_by_patient(session, patient_id)
        mesures_dict = [m.to_dict() for m in mesures]

    stats = data_processor.calculer_statistiques(mesures_dict, jours=jours)
    return stats


@app.delete(f"{API.PREFIX}/glycemie/mesure/{{mesure_id}}", tags=["Glycémie"])
async def supprimer_mesure(
    mesure_id: str,
    patient_id: str = Depends(get_current_patient_id),
) -> SuccessResponse:
    """Supprime une mesure glycémique."""
    async with get_session() as session:
        success = await MesureGlycemieRepository.delete(session, mesure_id, patient_id)
        if not success:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Mesure introuvable")
    return SuccessResponse(message="Mesure supprimée")


# ============================================================================
# ENDPOINTS : IMPORT
# ============================================================================

@app.post(f"{API.PREFIX}/import/upload", tags=["Import"])
async def upload_file(
    file: UploadFile = File(...),
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Upload et prévisualisation d'un fichier Excel/CSV."""
    # Validation du fichier
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nom de fichier manquant")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".xlsx", ".xls", ".csv"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Format non supporté. Utilisez .xlsx ou .csv",
        )

    # Taille max
    content = await file.read()
    if len(content) > API.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Fichier trop volumineux (max {API.MAX_UPLOAD_SIZE_MB}MB)",
        )

    # Sauvegarder
    safe_name = security_manager.sanitizer.sanitize_filename(file.filename)
    filepath = UPLOADS_DIR / f"{patient_id[:8]}_{int(time.time())}_{safe_name}"
    filepath.write_bytes(content)

    # Traiter
    rapport = data_processor.importer_fichier(filepath, patient_id)

    # Enregistrer en DB
    async with get_session() as session:
        imp = await ImportHistoryRepository.create(
            session,
            patient_id=patient_id,
            filename=safe_name,
            format_file=ext.lstrip("."),
            preview_data=rapport.preview_data,
        )
        await ImportHistoryRepository.update_status(
            session, imp.id,
            status=rapport.statut.value,
            nb_lignes_total=rapport.nb_lignes_total,
            nb_erreurs=len(rapport.erreurs),
            erreurs=[e.to_dict() for e in rapport.erreurs],
        )
        rapport_dict = rapport.to_dict()
        rapport_dict["db_import_id"] = imp.id

    return rapport_dict


@app.post(f"{API.PREFIX}/import/validate/{{import_id}}", tags=["Import"])
async def validate_import(
    import_id: str,
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Valide un import et insère les mesures en DB."""
    try:
        mesures, rapport = data_processor.valider_import(import_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))

    async with get_session() as session:
        inserted = await MesureGlycemieRepository.create_bulk(session, mesures)
        await ImportHistoryRepository.update_status(
            session, import_id,
            status="validated",
            nb_lignes_importees=inserted,
        )

    return {
        "success": True,
        "message": f"{inserted} mesures importées avec succès",
        "import_id": import_id,
    }


@app.get(f"{API.PREFIX}/import/history", tags=["Import"])
async def import_history(
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Historique des imports du patient."""
    async with get_session() as session:
        imports = await ImportHistoryRepository.get_history(session, patient_id)
        return {"count": len(imports), "imports": [i.to_dict() for i in imports]}


# ============================================================================
# ENDPOINTS : IA / CHAT
# ============================================================================

@app.post(
    f"{API.PREFIX}/ia/chat",
    response_model=ChatResponse,
    tags=["IA"],
)
async def chat_ia(
    payload: ChatRequest,
    patient_id: str = Depends(get_current_patient_id),
) -> ChatResponse:
    """Envoie un message texte à l'IA."""
    try:
        mode_ia = ModeIA(payload.mode)
    except ValueError:
        mode_ia = ModeIA.PATIENT

    # Récupérer le contexte glycémique
    async with get_session() as session:
        patient = await PatientRepository.get_by_id(session, patient_id)
        langue = payload.langue or (patient.langue_preferee if patient else "fr")

        mesures = await MesureGlycemieRepository.get_by_patient(
            session, patient_id, limit=100,
        )
        mesures_dict = [m.to_dict() for m in mesures]

    stats = data_processor.calculer_statistiques(mesures_dict, jours=14) \
        if mesures_dict else None

    reponse = await diabot.process_input(
        patient_id=patient_id,
        input_data=payload.message,
        modalite=ModaliteEntree.TEXTE,
        langue=langue,
        mode=mode_ia,
        context_glycemique=stats if isinstance(stats, dict) else None,
        audio_response=payload.audio_response,
    )

    # Logger la conversation
    async with get_session() as session:
        session_id = generate_uuid()
        await ConversationRepository.add_message(
            session, patient_id, session_id,
            role="user", content=payload.message,
            modalite="texte", langue=langue,
        )
        await ConversationRepository.add_message(
            session, patient_id, session_id,
            role="assistant", content=reponse.texte,
            modalite="texte", langue=langue,
            latence_ms=reponse.latence_ms,
            model_used=reponse.model_used,
            intention_detectee=reponse.intention.value if reponse.intention else None,
        )

        # Créer alerte si urgence
        if reponse.alerte_urgence:
            await AlerteRepository.create(
                session, patient_id,
                type_alerte="urgence_ia",
                severite="critique",
                message=reponse.message_urgence or reponse.texte,
            )

    # Encoder l'audio en base64
    audio_b64 = None
    if reponse.audio_response:
        import base64
        audio_b64 = base64.b64encode(reponse.audio_response).decode("utf-8")

    return ChatResponse(
        texte=reponse.texte,
        langue=reponse.langue,
        intention=reponse.intention.value if reponse.intention else None,
        alerte_urgence=reponse.alerte_urgence,
        message_urgence=reponse.message_urgence,
        donnees_glycemiques=reponse.donnees_glycemiques,
        analyse_image=reponse.analyse_image,
        audio_base64=audio_b64,
        latence_ms=reponse.latence_ms,
        model_used=reponse.model_used,
    )


@app.post(f"{API.PREFIX}/ia/chat/audio", tags=["IA"])
async def chat_audio(
    audio: UploadFile = File(...),
    langue: Optional[str] = Form(None),
    audio_response: bool = Form(False),
    patient_id: str = Depends(get_current_patient_id),
) -> ChatResponse:
    """Envoie un fichier audio à l'IA (transcription + réponse)."""
    content = await audio.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "Fichier audio trop volumineux")

    reponse = await diabot.process_input(
        patient_id=patient_id,
        input_data=content,
        modalite=ModaliteEntree.AUDIO,
        langue=langue,
        audio_response=audio_response,
    )

    audio_b64 = None
    if reponse.audio_response:
        import base64
        audio_b64 = base64.b64encode(reponse.audio_response).decode("utf-8")

    return ChatResponse(
        texte=reponse.texte,
        langue=reponse.langue,
        intention=reponse.intention.value if reponse.intention else None,
        alerte_urgence=reponse.alerte_urgence,
        message_urgence=reponse.message_urgence,
        audio_base64=audio_b64,
        latence_ms=reponse.latence_ms,
        model_used=reponse.model_used,
    )


@app.post(f"{API.PREFIX}/ia/chat/image", tags=["IA"])
async def chat_image(
    image: UploadFile = File(...),
    image_type: str = Form("auto"),
    message: Optional[str] = Form(None),
    patient_id: str = Depends(get_current_patient_id),
) -> ChatResponse:
    """Envoie une image à l'IA pour analyse."""
    content = await image.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image trop volumineuse")

    try:
        modalite = ModaliteEntree(f"image_{image_type}") if image_type != "auto" \
            else ModaliteEntree.IMAGE
    except ValueError:
        modalite = ModaliteEntree.IMAGE

    reponse = await diabot.process_input(
        patient_id=patient_id,
        input_data=content,
        modalite=modalite,
    )

    return ChatResponse(
        texte=reponse.texte,
        langue=reponse.langue,
        intention=reponse.intention.value if reponse.intention else None,
        alerte_urgence=reponse.alerte_urgence,
        message_urgence=reponse.message_urgence,
        analyse_image=reponse.analyse_image,
        latence_ms=reponse.latence_ms,
        model_used=reponse.model_used,
    )


@app.get(f"{API.PREFIX}/ia/conversation/history", tags=["IA"])
async def conversation_history(
    limit: int = Query(50, ge=1, le=500),
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Historique des conversations IA."""
    async with get_session() as session:
        messages = await ConversationRepository.get_history(
            session, patient_id, limit=limit,
        )
        return {"count": len(messages), "messages": [m.to_dict() for m in messages]}


@app.post(f"{API.PREFIX}/ia/rapport/generer", tags=["IA"])
async def generer_rapport_ia(
    jours: int = Query(30, ge=7, le=365),
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Génère un rapport médical complet (CV médical)."""
    async with get_session() as session:
        patient = await PatientRepository.get_by_id(session, patient_id)
        if not patient:
            raise HTTPException(404, "Patient introuvable")

        mesures = await MesureGlycemieRepository.get_by_patient(session, patient_id)
        mesures_dict = [m.to_dict() for m in mesures]
        alertes = await AlerteRepository.get_actives(session, patient_id)
        alertes_dict = [a.to_dict() for a in alertes]
        profil = patient.to_dict(include_sensitive=True)

        rapport = await diabot.generer_rapport(
            profil, mesures_dict, alertes_dict, periode_jours=jours,
        )

        # Sauvegarder le rapport
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        db_rapport = await RapportMedicalRepository.create(
            session, patient_id=patient_id,
            contenu=rapport,
            periode_debut=now - timedelta(days=jours),
            periode_fin=now,
            resume_ia=rapport.get("evaluation_globale"),
            score_risque=rapport.get("score_risque", {}).get("score"),
        )
        rapport["id"] = db_rapport.id

    return rapport


@app.get(f"{API.PREFIX}/ia/rapport/{{rapport_id}}", tags=["IA"])
async def get_rapport(
    rapport_id: str,
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Récupère un rapport existant."""
    async with get_session() as session:
        rapport = await RapportMedicalRepository.get_by_id(session, rapport_id)
        if not rapport or rapport.patient_id != patient_id:
            raise HTTPException(404, "Rapport introuvable")
        return rapport.to_dict()


@app.get(f"{API.PREFIX}/ia/alertes", tags=["IA"])
async def get_alertes(
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Récupère les alertes actives."""
    async with get_session() as session:
        alertes = await AlerteRepository.get_actives(session, patient_id)
        return {"count": len(alertes), "alertes": [a.to_dict() for a in alertes]}


# ============================================================================
# ENDPOINTS : QR CODE
# ============================================================================

@app.post(f"{API.PREFIX}/qr/generer", tags=["QR Code"])
async def generer_qr(
    payload: QRGenerateRequest,
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Génère un QR code dynamique pour le patient."""
    result = qr_manager.generer_qr(
        patient_id=patient_id,
        niveau_permission=payload.niveau_permission,
        duree_minutes=payload.duree_minutes,
        max_scans=payload.max_scans,
    )

    if not result.success:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, result.message)

    # Enregistrer en DB
    async with get_session() as session:
        await SessionMedecinRepository.create(
            session,
            patient_id=patient_id,
            qr_token=result.qr_token or "",
            nonce=result.payload.nonce if result.payload else generate_uuid(),
            niveau_permission=payload.niveau_permission,
            duree_minutes=payload.duree_minutes,
            max_scans=payload.max_scans,
        )

    import base64
    return {
        "success": True,
        "qr_token": result.qr_token,
        "qr_image_base64": base64.b64encode(result.qr_image_bytes).decode()
            if result.qr_image_bytes else None,
        "expire_at": result.expire_at.isoformat() if result.expire_at else None,
        "niveau_permission": payload.niveau_permission,
        "scans_restants": result.scans_restants,
        "permissions_detail": PERMISSIONS_PAR_NIVEAU.get(payload.niveau_permission),
    }


@app.get(f"{API.PREFIX}/qr/current", tags=["QR Code"])
async def get_current_qr(
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Récupère le QR actif du patient."""
    qr_info = qr_manager.get_qr_actif(patient_id)
    if not qr_info:
        raise HTTPException(404, "Aucun QR actif. Générez-en un nouveau.")
    return qr_info


@app.post(f"{API.PREFIX}/qr/scanner", tags=["QR Code"])
async def scanner_qr(
    request: Request,
    payload: QRScanRequest,
) -> dict[str, Any]:
    """Scan d'un QR code par un médecin (endpoint public mais traçé)."""
    ip = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "unknown")

    result = qr_manager.verifier_qr(
        qr_token=payload.qr_token,
        medecin_identifiant=payload.medecin_identifiant,
        medecin_nom=payload.medecin_nom,
        ip_address=ip,
        user_agent=user_agent,
    )

    if not result.success:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN
                if result.status in ("expire", "consomme", "revoque")
                else status.HTTP_400_BAD_REQUEST,
            result.message,
        )

    # Audit log
    async with get_session() as session:
        await AccessLogRepository.log(
            session,
            actor=payload.medecin_identifiant or "medecin_anonyme",
            action="qr_scan",
            patient_id=result.patient_id,
            ip_address=ip,
            user_agent=user_agent,
            success=True,
            details={"niveau": result.niveau_permission},
        )

    return {
        "success": True,
        "patient_id": result.patient_id,
        "niveau_permission": result.niveau_permission,
        "scans_restants": result.scans_restants,
        "permissions_detail": PERMISSIONS_PAR_NIVEAU.get(result.niveau_permission or 1),
        "message": result.message,
    }


@app.post(f"{API.PREFIX}/qr/revoquer", tags=["QR Code"])
async def revoquer_qr(
    patient_id: str = Depends(get_current_patient_id),
) -> SuccessResponse:
    """Révoque tous les QR actifs du patient."""
    result = qr_manager.revoquer_qr(patient_id=patient_id)
    return SuccessResponse(message=result.message)


@app.get(f"{API.PREFIX}/qr/access-log", tags=["QR Code"])
async def qr_access_log(
    patient_id: str = Depends(get_current_patient_id),
) -> dict[str, Any]:
    """Journal des accès via QR."""
    logs = qr_manager.get_access_log(patient_id=patient_id)
    return {"count": len(logs), "logs": logs}


# ============================================================================
# ENDPOINTS : VISUALISATIONS
# ============================================================================

@app.get(f"{API.PREFIX}/viz/courbe", tags=["Visualisations"])
async def viz_courbe(
    type_graphique: str = Query("tendance"),
    periode: int = Query(30, ge=1, le=365),
    mobile: bool = Query(False),
    fmt: str = Query("png"),
    patient_id: str = Depends(get_current_patient_id),
) -> FastResponse:
    """Génère un graphique glycémique."""
    async with get_session() as session:
        mesures = await MesureGlycemieRepository.get_by_patient(session, patient_id)
        mesures_dict = [m.to_dict() for m in mesures]

    try:
        img_bytes = viz_engine.generer(
            type_graphique, mesures_dict,
            jours=periode, mobile=mobile, fmt=fmt,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    media_type = "image/svg+xml" if fmt == "svg" else "image/png"
    return FastResponse(content=img_bytes, media_type=media_type)


@app.get(f"{API.PREFIX}/viz/dashboard", tags=["Visualisations"])
async def viz_dashboard(
    periode: int = Query(30, ge=7, le=365),
    patient_id: str = Depends(get_current_patient_id),
) -> FastResponse:
    """Génère le tableau de bord synthétique."""
    async with get_session() as session:
        mesures = await MesureGlycemieRepository.get_by_patient(session, patient_id)
        mesures_dict = [m.to_dict() for m in mesures]

    stats = data_processor.calculer_statistiques(mesures_dict, jours=periode)
    img_bytes = viz_engine.generer_dashboard_complet(
        mesures_dict, statistiques=stats, jours=periode,
    )
    return FastResponse(content=img_bytes, media_type="image/png")


@app.get(f"{API.PREFIX}/viz/agp", tags=["Visualisations"])
async def viz_agp(
    periode: int = Query(14, ge=7, le=90),
    patient_id: str = Depends(get_current_patient_id),
) -> FastResponse:
    """Génère le profil AGP (Ambulatory Glucose Profile)."""
    async with get_session() as session:
        mesures = await MesureGlycemieRepository.get_by_patient(session, patient_id)
        mesures_dict = [m.to_dict() for m in mesures]

    img_bytes = viz_engine.generer("agp", mesures_dict, jours=periode)
    return FastResponse(content=img_bytes, media_type="image/png")


# ============================================================================
# ENDPOINTS : MÉDECIN (accès via QR scanné)
# ============================================================================

async def verify_qr_token(
    qr_token: str = Query(..., description="Token QR du patient"),
) -> dict[str, Any]:
    """Dépendance : vérifie un QR token et retourne les permissions."""
    result = qr_manager.verifier_qr(qr_token=qr_token)
    if not result.success:
        raise HTTPException(403, result.message)
    return {
        "patient_id": result.patient_id,
        "niveau": result.niveau_permission,
        "permissions": PERMISSIONS_PAR_NIVEAU.get(result.niveau_permission or 1),
    }


@app.get(f"{API.PREFIX}/medecin/patient", tags=["Médecin"])
async def medecin_get_patient(
    access: dict[str, Any] = Depends(verify_qr_token),
) -> dict[str, Any]:
    """Accès aux données patient via QR scanné."""
    patient_id = access["patient_id"]
    niveau = access["niveau"]
    perms = access["permissions"]

    async with get_session() as session:
        patient = await PatientRepository.get_by_id(session, patient_id)
        if not patient:
            raise HTTPException(404, "Patient introuvable")

        result: dict[str, Any] = {
            "niveau_acces": niveau,
            "permissions": perms,
        }

        # Niveau 1+ : résumé
        derniere = await MesureGlycemieRepository.get_derniere(session, patient_id)
        if derniere:
            result["derniere_glycemie"] = derniere.to_dict()
        result["type_diabete"] = patient.type_diabete

        # Niveau 2+ : historique 30j
        if niveau >= 2:
            from datetime import timedelta
            debut = datetime.now(timezone.utc) - timedelta(days=30)
            mesures = await MesureGlycemieRepository.get_by_patient(
                session, patient_id, date_debut=debut,
            )
            result["historique_30j"] = [m.to_dict() for m in mesures]
            result["statistiques_30j"] = data_processor.calculer_statistiques(
                [m.to_dict() for m in mesures], jours=30,
            )

        # Niveau 3+ : complet + rapport
        if niveau >= 3:
            result["profil_complet"] = patient.to_dict(include_sensitive=True)
            rapports = await RapportMedicalRepository.get_latest(
                session, patient_id, limit=1,
            )
            if rapports:
                result["dernier_rapport"] = rapports[0].to_dict()

        # Niveau 4 : accès total
        if niveau >= 4:
            conversations = await ConversationRepository.get_history(
                session, patient_id, limit=50,
            )
            result["conversations_recentes"] = [c.to_dict() for c in conversations]

    return result


@app.get(f"{API.PREFIX}/medecin/patient/rapport", tags=["Médecin"])
async def medecin_get_rapport(
    access: dict[str, Any] = Depends(verify_qr_token),
) -> dict[str, Any]:
    """Génère un rapport médical via QR scanné (niveau 3+)."""
    if access["niveau"] < 3:
        raise HTTPException(403, "Niveau de permission insuffisant")

    patient_id = access["patient_id"]
    async with get_session() as session:
        patient = await PatientRepository.get_by_id(session, patient_id)
        mesures = await MesureGlycemieRepository.get_by_patient(session, patient_id)
        alertes = await AlerteRepository.get_actives(session, patient_id)
        profil = patient.to_dict(include_sensitive=True) if patient else {}

    rapport = await diabot.generer_rapport(
        profil,
        [m.to_dict() for m in mesures],
        [a.to_dict() for a in alertes],
    )
    return rapport


# ============================================================================
# ENDPOINTS : UTILITAIRES
# ============================================================================

@app.get(f"{API.PREFIX}/langues", tags=["Système"])
async def get_langues() -> dict[str, Any]:
    """Liste des langues supportées."""
    return {"langues": multilingual_engine.get_supported_languages()}


@app.get(f"{API.PREFIX}/nutrition/{{aliment}}", tags=["Nutrition"])
async def get_nutrition(
    aliment: str,
    langue: str = Query(LANGUE_PAR_DEFAUT.value),
) -> dict[str, Any]:
    """Conseil nutritionnel pour un aliment congolais."""
    conseil = multilingual_engine.get_nutrition_conseil(aliment, langue)
    if not conseil:
        raise HTTPException(404, f"Aliment '{aliment}' non trouvé")
    return conseil


# ============================================================================
# GESTION D'ERREURS GLOBALE
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler personnalisé pour HTTPException."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail, "status": exc.status_code},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler pour toutes les exceptions non gérées."""
    logger.error(f"❌ Exception non gérée : {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Erreur interne du serveur",
            "detail": str(exc) if ENVIRONMENT.value == "development" else None,
        },
    )


__all__ = ["app"]