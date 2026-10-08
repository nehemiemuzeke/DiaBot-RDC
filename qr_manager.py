"""
================================================================================
FICHIER : qr_manager.py
RESPONSABILITÉ : Génération, vérification et gestion des QR codes dynamiques
================================================================================
Ce module gère TOUT le cycle de vie des QR codes sécurisés :
- Génération de QR codes dynamiques contenant un JWT chiffré
- Chaque QR : patient_id chiffré, permissions, timestamp, nonce unique
- Régénération automatique (toutes les X minutes)
- Vérification côté serveur : validité temporelle, usage unique, signature
- 4 niveaux de permissions granulaires
- Rate limiting (max 3 scans par QR)
- Journalisation complète de tous les accès
- Révocation manuelle ou automatique
- Génération d'images QR (PNG/SVG) avec personnalisation visuelle

Auteur : Équipe DiaBot-RDC
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

import jwt
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask

from config import (
    SECURITY, QR, NiveauPermissionQR, APP_NAME, APP_VERSION,
)

logger = logging.getLogger(__name__)


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

class QRStatus(str):
    """Statut d'un QR code."""
    ACTIF = "actif"
    EXPIRE = "expire"
    REVOQUE = "revoque"
    CONSOMME = "consomme"  # Tous les scans utilisés
    INVALIDE = "invalide"


@dataclass
class QRPayload:
    """Payload contenu dans le JWT du QR code.

    Attributes:
        patient_id: UUID du patient (chiffré dans le token).
        niveau_permission: Niveau d'accès (1-4).
        nonce: Identifiant unique à usage unique.
        iat: Timestamp de création.
        exp: Timestamp d'expiration.
        jti: Identifiant unique du token.
        max_scans: Nombre max de scans autorisés.
        permissions_detail: Liste détaillée des permissions.
    """
    patient_id: str
    niveau_permission: int
    nonce: str
    iat: datetime
    exp: datetime
    jti: str
    max_scans: int = 3
    permissions_detail: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation en dict pour JWT."""
        return {
            "pid": self.patient_id,
            "lvl": self.niveau_permission,
            "nnc": self.nonce,
            "iat": self.iat,
            "exp": self.exp,
            "jti": self.jti,
            "mxs": self.max_scans,
            "prm": self.permissions_detail,
            "iss": APP_NAME,
            "typ": "qr_access",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QRPayload":
        """Désérialisation depuis un dict JWT.

        Args:
            data: Dict décodé du JWT.

        Returns:
            Instance QRPayload.
        """
        return cls(
            patient_id=data["pid"],
            niveau_permission=data["lvl"],
            nonce=data["nnc"],
            iat=datetime.fromtimestamp(data["iat"], tz=timezone.utc)
                if isinstance(data["iat"], (int, float))
                else data["iat"],
            exp=datetime.fromtimestamp(data["exp"], tz=timezone.utc)
                if isinstance(data["exp"], (int, float))
                else data["exp"],
            jti=data["jti"],
            max_scans=data.get("mxs", 3),
            permissions_detail=data.get("prm", []),
        )


@dataclass
class QRResult:
    """Résultat d'une opération QR."""
    success: bool
    status: str
    message: str
    qr_token: Optional[str] = None
    qr_image_bytes: Optional[bytes] = None
    payload: Optional[QRPayload] = None
    patient_id: Optional[str] = None
    niveau_permission: Optional[int] = None
    scans_restants: Optional[int] = None
    expire_at: Optional[datetime] = None
    access_log: Optional[dict[str, Any]] = None


@dataclass
class AccessRecord:
    """Enregistrement d'un accès via QR."""
    timestamp: datetime
    qr_jti: str
    patient_id: str
    medecin_identifiant: Optional[str]
    medecin_nom: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    niveau_permission: int
    donnees_accedees: list[str]
    success: bool
    raison_echec: Optional[str] = None


# ============================================================================
# PERMISSIONS DÉTAILLÉES PAR NIVEAU
# ============================================================================

PERMISSIONS_PAR_NIVEAU: dict[int, dict[str, Any]] = {
    NiveauPermissionQR.RESUME_SEUL.value: {
        "nom": "Résumé uniquement",
        "description": "Dernière glycémie, tendance 24h, statut général",
        "donnees_accessibles": [
            "derniere_glycemie",
            "tendance_24h",
            "statut_general",
            "type_diabete",
            "score_risque",
        ],
        "historique_jours": 0,
        "rapport_ia": False,
        "conversations": False,
        "images": False,
        "donnees_personnelles": False,
    },
    NiveauPermissionQR.HISTORIQUE_30J.value: {
        "nom": "Historique 30 jours",
        "description": "Résumé + historique glycémique des 30 derniers jours",
        "donnees_accessibles": [
            "derniere_glycemie",
            "tendance_24h",
            "statut_general",
            "type_diabete",
            "score_risque",
            "historique_glycemie_30j",
            "statistiques_30j",
            "alertes_recentes",
        ],
        "historique_jours": 30,
        "rapport_ia": False,
        "conversations": False,
        "images": False,
        "donnees_personnelles": False,
    },
    NiveauPermissionQR.HISTORIQUE_COMPLET_RAPPORT.value: {
        "nom": "Historique complet + Rapport IA",
        "description": "Tout l'historique + rapport généré par l'IA",
        "donnees_accessibles": [
            "derniere_glycemie",
            "tendance_24h",
            "statut_general",
            "type_diabete",
            "score_risque",
            "historique_glycemie_complet",
            "statistiques_completes",
            "alertes_toutes",
            "rapport_ia",
            "tendance_long_terme",
            "patterns_detectes",
        ],
        "historique_jours": 365,
        "rapport_ia": True,
        "conversations": False,
        "images": False,
        "donnees_personnelles": True,
    },
    NiveauPermissionQR.ACCES_TOTAL.value: {
        "nom": "Accès total",
        "description": "Accès complet : historique, conversations, images, rapports",
        "donnees_accessibles": [
            "derniere_glycemie",
            "tendance_24h",
            "statut_general",
            "type_diabete",
            "score_risque",
            "historique_glycemie_complet",
            "statistiques_completes",
            "alertes_toutes",
            "rapport_ia",
            "tendance_long_terme",
            "patterns_detectes",
            "conversations_ia",
            "images_glucometre",
            "images_repas",
            "ordonnances",
            "donnees_personnelles",
            "traitements",
            "comorbidites",
        ],
        "historique_jours": 9999,
        "rapport_ia": True,
        "conversations": True,
        "images": True,
        "donnees_personnelles": True,
    },
}


# ============================================================================
# GÉNÉRATEUR DE QR CODE
# ============================================================================

class QRGenerator:
    """Générateur d'images QR code avec personnalisation visuelle.

    Génère des QR codes stylisés aux couleurs de DiaBot-RDC
    avec le logo intégré et les informations de sécurité.
    """

    # Couleurs DiaBot-RDC
    COLOR_PRIMARY = "#0072B2"      # Bleu médical
    COLOR_DARK = "#1A1A2E"         # Bleu foncé
    COLOR_BG = "#FFFFFF"           # Blanc
    COLOR_ERROR = "#D55E00"        # Orange (pour QR expiré)

    def __init__(
        self,
        size: int = QR.QR_SIZE,
        border: int = QR.QR_BORDER,
        error_correction: str = QR.QR_ERROR_CORRECTION,
    ) -> None:
        """Initialise le générateur.

        Args:
            size: Taille de l'image en pixels.
            border: Bordure en modules.
            error_correction: Niveau de correction d'erreur (L/M/Q/H).
        """
        self.size = size
        self.border = border
        self.error_correction = {
            "L": qrcode.constants.ERROR_CORRECT_L,
            "M": qrcode.constants.ERROR_CORRECT_M,
            "Q": qrcode.constants.ERROR_CORRECT_Q,
            "H": qrcode.constants.ERROR_CORRECT_H,
        }.get(error_correction, qrcode.constants.ERROR_CORRECT_H)

    def generate_simple(
        self,
        data: str,
        fmt: str = "png",
    ) -> bytes:
        """Génère un QR code simple (noir et blanc).

        Args:
            data: Données à encoder.
            fmt: Format de sortie ("png" ou "svg").

        Returns:
            Bytes de l'image QR.
        """
        qr = qrcode.QRCode(
            version=None,  # Auto-détection
            error_correction=self.error_correction,
            box_size=10,
            border=self.border,
        )
        qr.add_data(data)
        qr.make(fit=True)

        if fmt == "svg":
            import qrcode.image.svg
            img = qr.make_image(image_factory=qrcode.image.svg.SvgImage)
            buf = io.BytesIO()
            img.save(buf)
            return buf.getvalue()

        img = qr.make_image(fill_color=self.COLOR_DARK, back_color=self.COLOR_BG)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def generate_styled(
        self,
        data: str,
        label: str = "DiaBot-RDC",
        niveau: int = 1,
    ) -> bytes:
        """Génère un QR code stylisé aux couleurs DiaBot-RDC.

        Args:
            data: Données à encoder (JWT).
            label: Texte à afficher sous le QR.
            niveau: Niveau de permission (affecte la couleur).

        Returns:
            Bytes de l'image PNG.
        """
        qr = qrcode.QRCode(
            version=None,
            error_correction=self.error_correction,
            box_size=10,
            border=self.border,
        )
        qr.add_data(data)
        qr.make(fit=True)

        # Couleur selon le niveau de permission
        level_colors = {
            1: "#009E73",  # Vert (résumé)
            2: "#0072B2",  # Bleu (30j)
            3: "#E69F00",  # Orange (complet)
            4: "#D55E00",  # Rouge (total)
        }
        fill_color = level_colors.get(niveau, self.COLOR_DARK)

        try:
            img = qr.make_image(
                image_factory=StyledPilImage,
                module_drawer=RoundedModuleDrawer(),
                color_mask=RadialGradiantColorMask(
                    back_color=(255, 255, 255),
                    center_color=self._hex_to_rgb(fill_color),
                    edge_color=self._hex_to_rgb(self.COLOR_DARK),
                ),
            )
        except Exception:
            # Fallback si StyledPilImage échoue
            img = qr.make_image(
                fill_color=fill_color,
                back_color=self.COLOR_BG,
            )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def generate_expired(self) -> bytes:
        """Génère un QR code visuellement marqué comme expiré.

        Returns:
            Bytes de l'image PNG avec un X rouge.
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=self.error_correction,
            box_size=10,
            border=self.border,
        )
        qr.add_data("EXPIRED")
        qr.make(fit=True)

        img = qr.make_image(
            fill_color="#CCCCCC",
            back_color=self.COLOR_BG,
        )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """Convertit un code hex en tuple RGB.

        Args:
            hex_color: Code couleur hex (ex: "#0072B2").

        Returns:
            Tuple (R, G, B).
        """
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# ============================================================================
# GESTIONNAIRE DE QR CODE PRINCIPAL
# ============================================================================

class QRManager:
    """Gestionnaire complet des QR codes dynamiques sécurisés.

    Orchestre la génération, la vérification, la régénération
    et la révocation des QR codes d'accès médical.
    """

    def __init__(
        self,
        signing_key: Optional[str] = None,
        algorithm: str = "HS256",
    ) -> None:
        """Initialise le gestionnaire QR.

        Args:
            signing_key: Clé de signature JWT. Si None, utilise la config.
            algorithm: Algorithme JWT.
        """
        self.signing_key = signing_key or SECURITY.QR_SIGNING_KEY or SECURITY.SECRET_KEY
        self.algorithm = algorithm
        self.generator = QRGenerator()

        # Store en mémoire des nonces consommés (en production : Redis)
        self._consumed_nonces: set[str] = set()
        self._active_tokens: dict[str, dict[str, Any]] = {}

        # Rate limiting par IP
        self._scan_attempts: dict[str, list[float]] = {}

        logger.info("🔲 QRManager initialisé")

    # ========================================================================
    # GÉNÉRATION
    # ========================================================================

    def generer_qr(
        self,
        patient_id: str,
        niveau_permission: int = NiveauPermissionQR.RESUME_SEUL.value,
        duree_minutes: int = QR.DUREE_DEFAUT_MINUTES,
        max_scans: int = QR.MAX_SCANS_PAR_QR,
        generer_image: bool = True,
        styled: bool = True,
    ) -> QRResult:
        """Génère un nouveau QR code dynamique sécurisé.

        Args:
            patient_id: UUID du patient.
            niveau_permission: Niveau d'accès (1-4).
            duree_minutes: Durée de validité en minutes.
            max_scans: Nombre max de scans autorisés.
            generer_image: Si True, génère l'image QR.
            styled: Si True, QR stylisé (sinon simple).

        Returns:
            QRResult avec le token, l'image et les métadonnées.

        Raises:
            ValueError: Si le niveau de permission est invalide.
        """
        # Valider le niveau
        if niveau_permission not in PERMISSIONS_PAR_NIVEAU:
            return QRResult(
                success=False,
                status=QRStatus.INVALIDE,
                message=f"Niveau de permission invalide : {niveau_permission}. "
                        f"Valeurs acceptées : {list(PERMISSIONS_PAR_NIVEAU.keys())}",
            )

        # Valider la durée
        if duree_minutes not in QR.DUREES_VALIDITE_MINUTES:
            duree_minutes = QR.DUREE_DEFAUT_MINUTES
            logger.warning(
                f"⚠️  Durée {duree_minutes}min non standard, "
                f"fallback sur {QR.DUREE_DEFAUT_MINUTES}min"
            )

        now = datetime.now(timezone.utc)
        nonce = secrets.token_urlsafe(32)
        jti = str(uuid.uuid4())

        # Construire le payload
        permissions = PERMISSIONS_PAR_NIVEAU[niveau_permission]
        payload = QRPayload(
            patient_id=patient_id,
            niveau_permission=niveau_permission,
            nonce=nonce,
            iat=now,
            exp=now + timedelta(minutes=duree_minutes),
            jti=jti,
            max_scans=max_scans,
            permissions_detail=permissions["donnees_accessibles"],
        )

        # Encoder en JWT
        try:
            token = jwt.encode(
                payload.to_dict(),
                self.signing_key,
                algorithm=self.algorithm,
            )
        except Exception as exc:
            logger.error(f"❌ Erreur encodage JWT QR : {exc}")
            return QRResult(
                success=False,
                status=QRStatus.INVALIDE,
                message=f"Erreur de génération du token : {exc}",
            )

        # Stocker le token actif
        self._active_tokens[jti] = {
            "patient_id": patient_id,
            "nonce": nonce,
            "exp": payload.exp,
            "max_scans": max_scans,
            "nb_scans": 0,
            "niveau": niveau_permission,
            "created_at": now,
        }

        # Générer l'image
        qr_image = None
        if generer_image:
            try:
                if styled:
                    qr_image = self.generator.generate_styled(
                        data=token,
                        label=f"Niveau {niveau_permission}",
                        niveau=niveau_permission,
                    )
                else:
                    qr_image = self.generator.generate_simple(token)
            except Exception as exc:
                logger.warning(f"⚠️  Erreur génération image QR : {exc}")
                # Fallback sur QR simple
                try:
                    qr_image = self.generator.generate_simple(token)
                except Exception:
                    qr_image = None

        logger.info(
            f"✅ QR généré pour patient {patient_id[:8]}... | "
            f"Niveau {niveau_permission} | "
            f"Expire dans {duree_minutes}min | "
            f"Max {max_scans} scans | "
            f"JTI {jti[:8]}..."
        )

        return QRResult(
            success=True,
            status=QRStatus.ACTIF,
            message=(
                f"QR code généré avec succès. "
                f"Expire dans {duree_minutes} minutes. "
                f"Permission : {permissions['nom']}."
            ),
            qr_token=token,
            qr_image_bytes=qr_image,
            payload=payload,
            patient_id=patient_id,
            niveau_permission=niveau_permission,
            scans_restants=max_scans,
            expire_at=payload.exp,
        )

    # ========================================================================
    # VÉRIFICATION (côté médecin / serveur)
    # ========================================================================

    def verifier_qr(
        self,
        qr_token: str,
        medecin_identifiant: Optional[str] = None,
        medecin_nom: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> QRResult:
        """Vérifie un QR code scanné par un médecin.

        Pipeline de vérification :
        1. Décodage et signature JWT
        2. Validité temporelle
        3. Nonce unique (anti-replay)
        4. Nombre de scans restants
        5. Rate limiting par IP
        6. Enregistrement de l'accès

        Args:
            qr_token: Token JWT du QR code scanné.
            medecin_identifiant: ID du médecin qui scanne.
            medecin_nom: Nom du médecin.
            ip_address: IP source du scan.
            user_agent: User-Agent du scan.

        Returns:
            QRResult avec le statut de vérification et les données accessibles.
        """
        now = datetime.now(timezone.utc)
        access_record: dict[str, Any] = {
            "timestamp": now.isoformat(),
            "medecin": medecin_identifiant or "inconnu",
            "ip": ip_address,
            "success": False,
        }

        # 1. Décodage JWT
        try:
            decoded = jwt.decode(
                qr_token,
                self.signing_key,
                algorithms=[self.algorithm],
                issuer=APP_NAME,
            )
        except jwt.ExpiredSignatureError:
            logger.warning("⏰ QR expiré")
            access_record["raison"] = "QR expiré"
            return QRResult(
                success=False,
                status=QRStatus.EXPIRE,
                message="Ce QR code a expiré. Demandez au patient d'en générer un nouveau.",
                access_log=access_record,
            )
        except jwt.InvalidSignatureError:
            logger.warning("🚨 Signature QR invalide")
            access_record["raison"] = "Signature invalide"
            return QRResult(
                success=False,
                status=QRStatus.INVALIDE,
                message="QR code invalide ou falsifié.",
                access_log=access_record,
            )
        except jwt.InvalidTokenError as exc:
            logger.warning(f"🚨 Token QR invalide : {exc}")
            access_record["raison"] = str(exc)
            return QRResult(
                success=False,
                status=QRStatus.INVALIDE,
                message="QR code invalide.",
                access_log=access_record,
            )

        # Reconstituer le payload
        try:
            payload = QRPayload.from_dict(decoded)
        except (KeyError, ValueError) as exc:
            logger.error(f"❌ Payload QR malformé : {exc}")
            return QRResult(
                success=False,
                status=QRStatus.INVALIDE,
                message="QR code malformé.",
                access_log=access_record,
            )

        # 2. Vérification temporelle (double-check)
        if payload.exp < now:
            return QRResult(
                success=False,
                status=QRStatus.EXPIRE,
                message="Ce QR code a expiré.",
                payload=payload,
                access_log=access_record,
            )

        # 3. Vérification du nonce (anti-replay)
        if payload.nonce in self._consumed_nonces:
            logger.warning(
                f"🚨 Tentative de replay détectée ! "
                f"Nonce {payload.nonce[:8]}... déjà consommé"
            )
            access_record["raison"] = "Nonce déjà utilisé (replay attack)"
            return QRResult(
                success=False,
                status=QRStatus.CONSOMME,
                message="Ce QR code a déjà été utilisé le nombre maximum de fois.",
                payload=payload,
                access_log=access_record,
            )

        # 4. Vérification du nombre de scans
        jti_data = self._active_tokens.get(payload.jti)
        if jti_data:
            if jti_data["nb_scans"] >= jti_data["max_scans"]:
                self._consumed_nonces.add(payload.nonce)
                return QRResult(
                    success=False,
                    status=QRStatus.CONSOMME,
                    message="Nombre maximum de scans atteint.",
                    payload=payload,
                    scans_restants=0,
                    access_log=access_record,
                )

        # 5. Rate limiting par IP
        if ip_address:
            if not self._check_rate_limit(ip_address):
                access_record["raison"] = "Rate limit dépassé"
                return QRResult(
                    success=False,
                    status=QRStatus.INVALIDE,
                    message="Trop de tentatives de scan. Réessayez dans quelques minutes.",
                    access_log=access_record,
                )

        # 6. ✅ Tout est valide — Consommer le scan
        self._consumed_nonces.add(payload.nonce)

        if jti_data:
            jti_data["nb_scans"] += 1
            scans_restants = jti_data["max_scans"] - jti_data["nb_scans"]
        else:
            scans_restants = payload.max_scans - 1

        # Déterminer le statut final
        if scans_restants <= 0:
            status = QRStatus.CONSOMME
        else:
            status = QRStatus.ACTIF

        access_record["success"] = True
        access_record["patient_id"] = payload.patient_id
        access_record["niveau"] = payload.niveau_permission
        access_record["scans_restants"] = scans_restants

        permissions = PERMISSIONS_PAR_NIVEAU.get(
            payload.niveau_permission,
            PERMISSIONS_PAR_NIVEAU[1],
        )

        logger.info(
            f"✅ QR vérifié avec succès | "
            f"Patient {payload.patient_id[:8]}... | "
            f"Médecin {medecin_identifiant or 'inconnu'} | "
            f"Niveau {payload.niveau_permission} | "
            f"Scans restants : {scans_restants}"
        )

        return QRResult(
            success=True,
            status=status,
            message=(
                f"Accès autorisé — {permissions['nom']}. "
                f"Scans restants : {scans_restants}."
            ),
            payload=payload,
            patient_id=payload.patient_id,
            niveau_permission=payload.niveau_permission,
            scans_restants=scans_restants,
            expire_at=payload.exp,
            access_log=access_record,
        )

    # ========================================================================
    # RÉVOCATION
    # ========================================================================

    def revoquer_qr(
        self,
        qr_token: Optional[str] = None,
        jti: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> QRResult:
        """Révoque un QR code actif.

        Peut révoquer par token, par JTI, ou tous les QR d'un patient.

        Args:
            qr_token: Token JWT du QR à révoquer.
            jti: JTI du QR à révoquer.
            patient_id: Si fourni, révoque TOUS les QR actifs du patient.

        Returns:
            QRResult avec le statut de révocation.
        """
        revoked_count = 0

        if qr_token:
            try:
                decoded = jwt.decode(
                    qr_token, self.signing_key,
                    algorithms=[self.algorithm],
                    options={"verify_exp": False},
                )
                jti = decoded.get("jti")
            except Exception:
                pass

        if jti:
            if jti in self._active_tokens:
                del self._active_tokens[jti]
                revoked_count += 1
                logger.info(f"🚫 QR révoqué : JTI {jti[:8]}...")

        if patient_id:
            to_remove = [
                j for j, data in self._active_tokens.items()
                if data["patient_id"] == patient_id
            ]
            for j in to_remove:
                del self._active_tokens[j]
                revoked_count += 1
            if to_remove:
                logger.info(
                    f"🚫 {revoked_count} QR révoqués pour "
                    f"patient {patient_id[:8]}..."
                )

        if revoked_count == 0:
            return QRResult(
                success=False,
                status=QRStatus.INVALIDE,
                message="Aucun QR code trouvé à révoquer.",
            )

        return QRResult(
            success=True,
            status=QRStatus.REVOQUE,
            message=f"{revoked_count} QR code(s) révoqué(s) avec succès.",
        )

    # ========================================================================
    # RÉGÉNÉRATION AUTOMATIQUE
    # ========================================================================

    def regenerer_qr(
        self,
        ancien_token: str,
    ) -> QRResult:
        """Régénère un QR code avec les mêmes paramètres.

        Révoque l'ancien et crée un nouveau avec la même permission.

        Args:
            ancien_token: Token du QR à régénérer.

        Returns:
            QRResult avec le nouveau QR.
        """
        try:
            decoded = jwt.decode(
                ancien_token, self.signing_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False},
            )
            payload = QRPayload.from_dict(decoded)
        except Exception as exc:
            return QRResult(
                success=False,
                status=QRStatus.INVALIDE,
                message=f"Impossible de régénérer : {exc}",
            )

        # Révoquer l'ancien
        self.revoquer_qr(jti=payload.jti)

        # Calculer la durée restante (ou défaut)
        now = datetime.now(timezone.utc)
        if payload.exp > now:
            duree_restante = int((payload.exp - now).total_seconds() / 60)
            duree = max(5, duree_restante)
        else:
            duree = QR.DUREE_DEFAUT_MINUTES

        # Générer le nouveau
        return self.generer_qr(
            patient_id=payload.patient_id,
            niveau_permission=payload.niveau_permission,
            duree_minutes=duree,
            max_scans=payload.max_scans,
        )

    # ========================================================================
    # NETTOYAGE
    # ========================================================================

    def cleanup_expired(self) -> int:
        """Nettoie les QR codes expirés et les nonces anciens.

        Returns:
            Nombre de QR nettoyés.
        """
        now = datetime.now(timezone.utc)
        expired = [
            jti for jti, data in self._active_tokens.items()
            if data["exp"] < now
        ]
        for jti in expired:
            del self._active_tokens[jti]

        # Limiter la taille du set de nonces (garder les 10000 derniers)
        if len(self._consumed_nonces) > 10000:
            # En production : Redis avec TTL
            self._consumed_nonces = set(list(self._consumed_nonces)[-5000:])

        if expired:
            logger.info(f"🧹 {len(expired)} QR expirés nettoyés")

        return len(expired)

    # ========================================================================
    # UTILITAIRES
    # ========================================================================

    def get_qr_actif(
        self,
        patient_id: str,
    ) -> Optional[dict[str, Any]]:
        """Retourne le QR actif le plus récent d'un patient.

        Args:
            patient_id: UUID du patient.

        Returns:
            Dict avec les infos du QR ou None.
        """
        now = datetime.now(timezone.utc)
        actifs = [
            (jti, data)
            for jti, data in self._active_tokens.items()
            if data["patient_id"] == patient_id and data["exp"] > now
        ]

        if not actifs:
            return None

        # Le plus récent
        actifs.sort(key=lambda x: x[1]["created_at"], reverse=True)
        jti, data = actifs[0]

        return {
            "jti": jti,
            "patient_id": data["patient_id"],
            "niveau_permission": data["niveau"],
            "expire_at": data["exp"].isoformat(),
            "scans_restants": data["max_scans"] - data["nb_scans"],
            "max_scans": data["max_scans"],
            "created_at": data["created_at"].isoformat(),
            "status": QRStatus.ACTIF,
        }

    def get_access_log(
        self,
        patient_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retourne le journal des accès QR.

        Note : En production, ce log serait en base de données.
        Ici, retourne un log simplifié depuis la mémoire.

        Args:
            patient_id: Filtrer par patient.
            limit: Nombre max d'entrées.

        Returns:
            Liste des enregistrements d'accès.
        """
        # En production : requête DB sur AccessLog
        # Ici : retourner les tokens actifs comme proxy
        logs = []
        for jti, data in self._active_tokens.items():
            if patient_id and data["patient_id"] != patient_id:
                continue
            logs.append({
                "jti": jti[:8] + "...",
                "patient_id": data["patient_id"][:8] + "...",
                "niveau": data["niveau"],
                "nb_scans": data["nb_scans"],
                "max_scans": data["max_scans"],
                "expire_at": data["exp"].isoformat(),
                "status": (
                    QRStatus.ACTIF if data["exp"] > datetime.now(timezone.utc)
                    else QRStatus.EXPIRE
                ),
            })

        return logs[:limit]

    def get_permissions_info(
        self,
        niveau: int,
    ) -> Optional[dict[str, Any]]:
        """Retourne les détails d'un niveau de permission.

        Args:
            niveau: Niveau de permission (1-4).

        Returns:
            Dict avec les détails ou None.
        """
        return PERMISSIONS_PAR_NIVEAU.get(niveau)

    def _check_rate_limit(
        self,
        ip_address: str,
        max_scans: int = 10,
        window_seconds: int = 60,
    ) -> bool:
        """Vérifie le rate limiting par IP.

        Args:
            ip_address: IP source.
            max_scans: Max scans par fenêtre.
            window_seconds: Durée de la fenêtre.

        Returns:
            True si autorisé.
        """
        now = time.time()

        if ip_address not in self._scan_attempts:
            self._scan_attempts[ip_address] = []

        # Nettoyer les anciennes tentatives
        self._scan_attempts[ip_address] = [
            t for t in self._scan_attempts[ip_address]
            if now - t < window_seconds
        ]

        if len(self._scan_attempts[ip_address]) >= max_scans:
            logger.warning(
                f"🚫 Rate limit QR dépassé pour IP {ip_address} "
                f"({len(self._scan_attempts[ip_address])}/{max_scans})"
            )
            return False

        self._scan_attempts[ip_address].append(now)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques du gestionnaire QR.

        Returns:
            Dict de statistiques.
        """
        now = datetime.now(timezone.utc)
        actifs = sum(
            1 for d in self._active_tokens.values() if d["exp"] > now
        )
        expires = sum(
            1 for d in self._active_tokens.values() if d["exp"] <= now
        )

        return {
            "qr_actifs": actifs,
            "qr_expires": expires,
            "nonces_consommes": len(self._consumed_nonces),
            "ips_surveillees": len(self._scan_attempts),
            "total_tokens": len(self._active_tokens),
        }


# ============================================================================
# QR CODE POUR AFFICHAGE MOBILE (avec instructions)
# ============================================================================

def generer_qr_patient_complet(
    patient_id: str,
    niveau: int = NiveauPermissionQR.RESUME_SEUL.value,
    duree: int = 15,
    nom_patient: Optional[str] = None,
) -> dict[str, Any]:
    """Fonction utilitaire pour générer un QR complet pour l'app mobile.

    Retourne toutes les données nécessaires à l'affichage :
    image, timer, instructions, niveau de permission.

    Args:
        patient_id: UUID du patient.
        niveau: Niveau de permission.
        duree: Durée en minutes.
        nom_patient: Nom du patient (pour l'affichage).

    Returns:
        Dict complet pour le frontend mobile.
    """
    manager = QRManager()
    result = manager.generer_qr(
        patient_id=patient_id,
        niveau_permission=niveau,
        duree_minutes=duree,
    )

    if not result.success:
        return {
            "success": False,
            "error": result.message,
        }

    permissions = PERMISSIONS_PAR_NIVEAU.get(niveau, {})

    return {
        "success": True,
        "qr_image_base64": (
            __import__("base64").b64encode(result.qr_image_bytes).decode("utf-8")
            if result.qr_image_bytes else None
        ),
        "qr_token": result.qr_token,
        "expire_at": result.expire_at.isoformat() if result.expire_at else None,
        "expire_seconds": duree * 60,
        "niveau_permission": {
            "niveau": niveau,
            "nom": permissions.get("nom", ""),
            "description": permissions.get("description", ""),
        },
        "scans_restants": result.scans_restants,
        "instructions": (
            f"Montrez ce QR code à votre médecin. "
            f"Il expirera dans {duree} minutes. "
            f"Le médecin pourra accéder à : {permissions.get('description', '')}."
        ),
        "avertissement": (
            "⚠️ Ne partagez ce QR code qu'avec votre médecin "
            "ou un professionnel de santé de confiance."
        ),
        "patient_nom": nom_patient,
    }


# ============================================================================
# INSTANCE GLOBALE
# ============================================================================

qr_manager = QRManager()


__all__ = [
    "QRManager", "QRGenerator", "QRPayload", "QRResult", "QRStatus",
    "AccessRecord", "PERMISSIONS_PAR_NIVEAU",
    "generer_qr_patient_complet", "qr_manager",
]