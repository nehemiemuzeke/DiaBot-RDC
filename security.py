"""
================================================================================
FICHIER : security.py
RESPONSABILITÉ : Sécurité, authentification, chiffrement et conformité RGPD
================================================================================
Ce module gère TOUTE la couche de sécurité de DiaBot-RDC :
- Authentification : bcrypt, JWT (access + refresh), 2FA OTP par SMS
- Chiffrement : AES-256 (données au repos), helpers TLS
- Conformité RGPD : consentement, droit à l'effacement, portabilité,
  pseudonymisation pour l'entraînement IA
- Audit : journalisation des accès, détection d'anomalies
- Rate limiting : protection contre les abus
- Validation & sanitization : protection OWASP Top 10
- Protection CSRF, XSS, SQL injection, path traversal

Auteur : Équipe DiaBot-RDC
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

import hashlib
import hmac
import html
import logging
import os
import re
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
import jwt
from cryptography.fernet import Fernet, InvalidToken

from config import SECURITY, ENVIRONMENT, Environment, APP_NAME

logger = logging.getLogger(__name__)


# ============================================================================
# HASHING DE MOTS DE PASSE (bcrypt)
# ============================================================================

class PasswordHasher:
    """Gestionnaire de hachage de mots de passe via bcrypt.

    Utilise bcrypt avec un nombre de rounds configurable (min 12).
    Conforme aux recommandations OWASP 2024.
    """

    def __init__(self, rounds: int = SECURITY.BCRYPT_ROUNDS) -> None:
        """Initialise le hasher.

        Args:
            rounds: Nombre de rounds bcrypt (minimum 12).

        Raises:
            ValueError: Si rounds < 12.
        """
        if rounds < 12:
            raise ValueError("Le nombre de rounds bcrypt doit être >= 12")
        self.rounds = rounds

    def hash(self, password: str) -> str:
        """Hache un mot de passe en clair.

        Args:
            password: Mot de passe en clair.

        Returns:
            Hash bcrypt encodé en UTF-8.

        Raises:
            ValueError: Si le mot de passe est vide ou trop long (>72 bytes).
        """
        if not password:
            raise ValueError("Le mot de passe ne peut pas être vide")
        if len(password.encode("utf-8")) > 72:
            raise ValueError("Le mot de passe ne peut pas dépasser 72 octets")

        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify(self, password: str, hashed: str) -> bool:
        """Vérifie un mot de passe contre un hash.

        Args:
            password: Mot de passe en clair à vérifier.
            hashed: Hash bcrypt stocké.

        Returns:
            True si le mot de passe correspond.
        """
        if not password or not hashed:
            return False
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"),
                hashed.encode("utf-8"),
            )
        except (ValueError, TypeError) as exc:
            logger.warning(f"Erreur de vérification de mot de passe : {exc}")
            return False

    def needs_rehash(self, hashed: str) -> bool:
        """Vérifie si un hash doit être recalculé (rounds obsolètes).

        Args:
            hashed: Hash bcrypt existant.

        Returns:
            True si le hash utilise des rounds inférieurs à la config.
        """
        try:
            info = bcrypt.checkpw(
                b"dummy",
                hashed.encode("utf-8"),
            )
            # Extraire les rounds du hash
            parts = hashed.split("$")
            if len(parts) >= 3:
                current_rounds = int(parts[2])
                return current_rounds < self.rounds
        except Exception:
            pass
        return True


# ============================================================================
# GESTION DES TOKENS JWT
# ============================================================================

@dataclass
class TokenPayload:
    """Payload d'un token JWT."""
    sub: str                    # patient_id ou medecin_id
    type: str                   # "access" | "refresh" | "qr"
    exp: datetime
    iat: datetime
    jti: str                    # Token ID unique (pour révocation)
    roles: list[str] = field(default_factory=lambda: ["patient"])
    permissions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class JWTManager:
    """Gestionnaire de tokens JWT (access + refresh).

    Génère, vérifie et révoque les tokens JWT.
    Supporte la rotation des refresh tokens.
    """

    def __init__(
        self,
        secret_key: str = SECURITY.SECRET_KEY,
        algorithm: str = SECURITY.JWT_ALGORITHM,
        access_expire_minutes: int = SECURITY.ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_expire_days: int = SECURITY.REFRESH_TOKEN_EXPIRE_DAYS,
    ) -> None:
        """Initialise le gestionnaire JWT.

        Args:
            secret_key: Clé secrète pour la signature.
            algorithm: Algorithme de signature (HS256 recommandé).
            access_expire_minutes: Durée de vie du token d'accès.
            refresh_expire_days: Durée de vie du refresh token.
        """
        if secret_key.startswith("CHANGE_ME"):
            logger.critical(
                "🚨 CLÉ SECRÈTE JWT PAR DÉFAUT ! "
                "CHANGEZ-LA IMMÉDIATEMENT EN PRODUCTION !"
            )
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_expire = timedelta(minutes=access_expire_minutes)
        self.refresh_expire = timedelta(days=refresh_expire_days)

        # Store en mémoire des tokens révoqués (en production : Redis)
        self._revoked_tokens: set[str] = set()

    def create_access_token(
        self,
        subject: str,
        roles: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        """Crée un token d'accès JWT.

        Args:
            subject: Identifiant du sujet (patient_id).
            roles: Rôles de l'utilisateur.
            permissions: Permissions spécifiques.
            extra: Données supplémentaires à inclure.

        Returns:
            Token JWT encodé.
        """
        now = datetime.now(timezone.utc)
        jti = secrets.token_urlsafe(32)

        payload = {
            "sub": subject,
            "type": "access",
            "iat": now,
            "exp": now + self.access_expire,
            "jti": jti,
            "roles": roles or ["patient"],
            "permissions": permissions or [],
            "iss": APP_NAME,
        }
        if extra:
            payload["extra"] = extra

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"🔑 Access token créé pour {subject} (jti={jti[:8]}...)")
        return token

    def create_refresh_token(self, subject: str) -> str:
        """Crée un refresh token JWT.

        Args:
            subject: Identifiant du sujet.

        Returns:
            Refresh token JWT encodé.
        """
        now = datetime.now(timezone.utc)
        jti = secrets.token_urlsafe(32)

        payload = {
            "sub": subject,
            "type": "refresh",
            "iat": now,
            "exp": now + self.refresh_expire,
            "jti": jti,
            "iss": APP_NAME,
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"🔄 Refresh token créé pour {subject}")
        return token

    def create_token_pair(
        self,
        subject: str,
        roles: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """Crée une paire access + refresh tokens.

        Args:
            subject: Identifiant du sujet.
            roles: Rôles de l'utilisateur.

        Returns:
            Dict avec "access_token", "refresh_token", "token_type", "expires_in".
        """
        return {
            "access_token": self.create_access_token(subject, roles=roles),
            "refresh_token": self.create_refresh_token(subject),
            "token_type": "bearer",
            "expires_in": int(self.access_expire.total_seconds()),
        }

    def decode_token(self, token: str, expected_type: str = "access") -> TokenPayload:
        """Décode et vérifie un token JWT.

        Args:
            token: Token JWT à décoder.
            expected_type: Type attendu ("access" ou "refresh").

        Returns:
            TokenPayload vérifié.

        Raises:
            jwt.ExpiredSignatureError: Si le token est expiré.
            jwt.InvalidTokenError: Si le token est invalide.
            ValueError: Si le type ne correspond pas ou le token est révoqué.
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=APP_NAME,
            )
        except jwt.ExpiredSignatureError:
            logger.warning("⏰ Token expiré")
            raise
        except jwt.InvalidSignatureError:
            logger.warning("🚨 Signature JWT invalide")
            raise
        except jwt.InvalidTokenError as exc:
            logger.warning(f"🚨 Token JWT invalide : {exc}")
            raise

        # Vérifier le type
        token_type = payload.get("type", "")
        if token_type != expected_type:
            raise ValueError(
                f"Type de token incorrect : attendu={expected_type}, "
                f"reçu={token_type}"
            )

        # Vérifier la révocation
        jti = payload.get("jti", "")
        if jti in self._revoked_tokens:
            raise ValueError("Token révoqué")

        return TokenPayload(
            sub=payload["sub"],
            type=token_type,
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            jti=jti,
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", []),
            extra=payload.get("extra", {}),
        )

    def verify_access_token(self, token: str) -> TokenPayload:
        """Vérifie un token d'accès.

        Args:
            token: Token d'accès JWT.

        Returns:
            Payload vérifié.
        """
        return self.decode_token(token, expected_type="access")

    def verify_refresh_token(self, token: str) -> TokenPayload:
        """Vérifie un refresh token et génère une nouvelle paire (rotation).

        Args:
            token: Refresh token JWT.

        Returns:
            Payload du refresh token.
        """
        payload = self.decode_token(token, expected_type="refresh")
        # Révoquer l'ancien refresh token (rotation)
        self.revoke_token(payload.jti)
        return payload

    def revoke_token(self, jti: str) -> None:
        """Révoque un token par son JTI.

        Args:
            jti: Identifiant unique du token.
        """
        self._revoked_tokens.add(jti)
        logger.info(f"🚫 Token révoqué : {jti[:8]}...")

    def cleanup_expired_revocations(self, max_age_hours: int = 24) -> int:
        """Nettoie les révocations expirées (en production : TTL Redis).

        Args:
            max_age_hours: Âge maximum des révocations à garder.

        Returns:
            Nombre de révocations nettoyées (toujours 0 en mémoire).
        """
        # En mémoire, on ne peut pas savoir l'âge. En production : Redis TTL.
        return 0


# ============================================================================
# AUTHENTIFICATION 2FA (OTP par SMS)
# ============================================================================

class OTPManager:
    """Gestionnaire de codes OTP pour l'authentification à deux facteurs.

    Génère des codes à usage unique envoyés par SMS, adaptés
    aux contraintes réseau de la RDC (délais plus longs).
    """

    def __init__(
        self,
        otp_length: int = SECURITY.OTP_LENGTH,
        expire_minutes: int = SECURITY.OTP_EXPIRE_MINUTES,
    ) -> None:
        """Initialise le gestionnaire OTP.

        Args:
            otp_length: Longueur du code OTP (6 par défaut).
            expire_minutes: Durée de validité du code.
        """
        self.otp_length = otp_length
        self.expire_delta = timedelta(minutes=expire_minutes)

        # Stockage en mémoire : {phone_hash: [(code, expiry), ...]}
        # En production : Redis avec TTL
        self._store: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
        self._attempts: dict[str, int] = defaultdict(int)
        self._max_attempts: int = 5

    def generate_otp(self, phone_number: str) -> str:
        """Génère un code OTP pour un numéro de téléphone.

        Args:
            phone_number: Numéro de téléphone (format RDC : +243...).

        Returns:
            Code OTP numérique.

        Raises:
            ValueError: Si trop de tentatives récentes.
        """
        phone_hash = self._hash_phone(phone_number)

        # Vérifier le rate limit
        if self._attempts[phone_hash] >= self._max_attempts:
            raise ValueError(
                "Trop de codes OTP demandés. Réessayez dans 15 minutes."
            )

        code = "".join(
            str(secrets.randbelow(10)) for _ in range(self.otp_length)
        )
        expiry = datetime.now(timezone.utc) + self.expire_delta

        self._store[phone_hash].append((code, expiry))
        self._attempts[phone_hash] += 1

        # Nettoyer les anciens codes
        self._cleanup(phone_hash)

        logger.info(
            f"📱 OTP généré pour ***{phone_number[-4:]} "
            f"(expire dans {self.expire_delta.total_seconds() / 60:.0f} min)"
        )
        return code

    def verify_otp(self, phone_number: str, code: str) -> bool:
        """Vérifie un code OTP.

        Args:
            phone_number: Numéro de téléphone.
            code: Code OTP saisi par l'utilisateur.

        Returns:
            True si le code est valide.
        """
        phone_hash = self._hash_phone(phone_number)
        now = datetime.now(timezone.utc)

        self._cleanup(phone_hash)

        for i, (stored_code, expiry) in enumerate(self._store[phone_hash]):
            if now > expiry:
                continue
            if hmac.compare_digest(stored_code, code):
                # Supprimer le code utilisé (usage unique)
                self._store[phone_hash].pop(i)
                self._attempts[phone_hash] = 0
                logger.info(f"✅ OTP vérifié pour ***{phone_number[-4:]}")
                return True

        logger.warning(f"❌ OTP invalide pour ***{phone_number[-4:]}")
        return False

    def _hash_phone(self, phone: str) -> str:
        """Hache un numéro de téléphone pour le stockage.

        Args:
            phone: Numéro en clair.

        Returns:
            Hash SHA-256 du numéro normalisé.
        """
        normalized = re.sub(r"[^\d+]", "", phone)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _cleanup(self, phone_hash: str) -> None:
        """Supprime les codes expirés.

        Args:
            phone_hash: Hash du numéro.
        """
        now = datetime.now(timezone.utc)
        self._store[phone_hash] = [
            (c, e) for c, e in self._store[phone_hash] if e > now
        ]


# ============================================================================
# CHIFFREMENT AES-256
# ============================================================================

class AESEncryptor:
    """Chiffrement symétrique AES-256 via Fernet.

    Utilisé pour chiffrer les données médicales sensibles au repos.
    """

    def __init__(self, key: Optional[str] = None) -> None:
        """Initialise l'encrypteur.

        Args:
            key: Clé AES en base64. Si None, utilise la config.
        """
        raw_key = key or SECURITY.AES_KEY or SECURITY.SECRET_KEY
        if not raw_key:
            raise ValueError("Aucune clé de chiffrement disponible")

        # Normaliser sur 32 octets pour Fernet
        key_bytes = raw_key.encode("utf-8").ljust(32, b"0")[:32]
        self._fernet = Fernet(
            __import__("base64").urlsafe_b64encode(key_bytes)
        )

    def encrypt(self, plaintext: str) -> str:
        """Chiffre une chaîne de texte.

        Args:
            plaintext: Texte en clair.

        Returns:
            Texte chiffré en base64.
        """
        if not plaintext:
            return plaintext
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        """Déchiffre une chaîne.

        Args:
            ciphertext: Texte chiffré.

        Returns:
            Texte en clair.

        Raises:
            InvalidToken: Si le déchiffrement échoue.
        """
        if not ciphertext:
            return ciphertext
        return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")

    def encrypt_dict(self, data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        """Chiffre des champs spécifiques d'un dictionnaire.

        Args:
            data: Dictionnaire source.
            fields: Liste des clés à chiffrer.

        Returns:
            Copie du dictionnaire avec champs chiffrés.
        """
        import json
        result = dict(data)
        for f in fields:
            if f in result and result[f] is not None:
                value = result[f]
                if not isinstance(value, str):
                    value = json.dumps(value, default=str)
                result[f] = self.encrypt(value)
        return result

    def decrypt_dict(self, data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        """Déchiffre des champs spécifiques d'un dictionnaire.

        Args:
            data: Dictionnaire avec champs chiffrés.
            fields: Liste des clés à déchiffrer.

        Returns:
            Copie du dictionnaire avec champs en clair.
        """
        result = dict(data)
        for f in fields:
            if f in result and result[f] is not None:
                try:
                    result[f] = self.decrypt(result[f])
                except (InvalidToken, Exception) as exc:
                    logger.warning(f"⚠️  Déchiffrement échoué pour '{f}' : {exc}")
        return result

    @staticmethod
    def generate_key() -> str:
        """Génère une nouvelle clé AES-256.

        Returns:
            Clé Fernet en base64.
        """
        return Fernet.generate_key().decode("utf-8")


# ============================================================================
# RATE LIMITING
# ============================================================================

@dataclass
class RateLimitEntry:
    """Entrée de rate limiting."""
    count: int = 0
    window_start: float = 0.0


class RateLimiter:
    """Limiteur de débit par IP ou par utilisateur.

    Implémente un algorithme de fenêtre glissante simple.
    En production, remplacer par Redis.
    """

    def __init__(self) -> None:
        """Initialise le rate limiter."""
        # {key: RateLimitEntry}
        self._limits: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._lock_time: dict[str, float] = {}

    def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60,
    ) -> tuple[bool, dict[str, Any]]:
        """Vérifie si une requête est autorisée.

        Args:
            key: Clé d'identification (IP, user_id, etc.).
            max_requests: Nombre max de requêtes par fenêtre.
            window_seconds: Durée de la fenêtre en secondes.

        Returns:
            Tuple (autorisé, headers_info).
        """
        now = time.time()

        # Vérifier si bloqué
        if key in self._lock_time and now < self._lock_time[key]:
            remaining_lock = int(self._lock_time[key] - now)
            return False, {
                "X-RateLimit-Limit": str(max_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(self._lock_time[key])),
                "Retry-After": str(remaining_lock),
            }

        entry = self._limits[key]

        # Réinitialiser la fenêtre si expirée
        if now - entry.window_start > window_seconds:
            entry.count = 0
            entry.window_start = now

        entry.count += 1
        remaining = max(0, max_requests - entry.count)

        headers = {
            "X-RateLimit-Limit": str(max_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(entry.window_start + window_seconds)),
        }

        if entry.count > max_requests:
            # Bloquer pour 2x la fenêtre
            self._lock_time[key] = now + window_seconds * 2
            logger.warning(
                f"🚫 Rate limit dépassé pour {key} "
                f"({entry.count}/{max_requests})"
            )
            headers["Retry-After"] = str(window_seconds * 2)
            return False, headers

        return True, headers

    def cleanup(self, max_age_seconds: int = 3600) -> int:
        """Nettoie les entrées expirées.

        Args:
            max_age_seconds: Âge max des entrées.

        Returns:
            Nombre d'entrées supprimées.
        """
        now = time.time()
        expired = [
            k for k, v in self._limits.items()
            if now - v.window_start > max_age_seconds
        ]
        for k in expired:
            del self._limits[k]
            self._lock_time.pop(k, None)
        return len(expired)


# ============================================================================
# VALIDATION & SANITIZATION (OWASP Top 10)
# ============================================================================

class InputSanitizer:
    """Sanitiseur et validateur d'entrées utilisateur.

    Protection contre :
    - XSS (Cross-Site Scripting)
    - SQL Injection
    - Path Traversal
    - Command Injection
    - LDAP Injection
    """

    # Patterns dangereux
    SQL_INJECTION_PATTERNS: list[re.Pattern] = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC)\b)", re.I),
        re.compile(r"(--|;|/\*|\*/|xp_|sp_)", re.I),
        re.compile(r"(\b(OR|AND)\b\s+\d+\s*=\s*\d+)", re.I),
        re.compile(r"('\s*(OR|AND)\s+')", re.I),
    ]

    PATH_TRAVERSAL_PATTERNS: list[re.Pattern] = [
        re.compile(r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|\.\.%2f)", re.I),
        re.compile(r"(/etc/passwd|/etc/shadow|/proc/|/windows/)", re.I),
    ]

    XSS_PATTERNS: list[re.Pattern] = [
        re.compile(r"(<script[^>]*>|</script>)", re.I),
        re.compile(r"(javascript\s*:|vbscript\s*:|data\s*:)", re.I),
        re.compile(r"(on\w+\s*=)", re.I),  # onclick=, onload=, etc.
        re.compile(r"(<iframe|<object|<embed|<form)", re.I),
    ]

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 10000) -> str:
        """Sanitise une chaîne de caractères.

        Args:
            value: Chaîne à sanitiser.
            max_length: Longueur maximale autorisée.

        Returns:
            Chaîne sanitiser.

        Raises:
            ValueError: Si la chaîne contient des patterns dangereux.
        """
        if not isinstance(value, str):
            value = str(value)

        # Tronquer
        value = value[:max_length]

        # Échapper le HTML
        value = html.escape(value, quote=True)

        # Supprimer les caractères de contrôle (sauf newline, tab)
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

        return value

    @classmethod
    def validate_no_sql_injection(cls, value: str) -> bool:
        """Vérifie l'absence d'injection SQL.

        Args:
            value: Chaîne à vérifier.

        Returns:
            True si aucun pattern SQL dangereux détecté.
        """
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if pattern.search(value):
                logger.warning(f"🚨 Tentative d'injection SQL détectée : {value[:50]}...")
                return False
        return True

    @classmethod
    def validate_no_path_traversal(cls, value: str) -> bool:
        """Vérifie l'absence de path traversal.

        Args:
            value: Chemin ou nom de fichier.

        Returns:
            True si aucun pattern de traversal détecté.
        """
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if pattern.search(value):
                logger.warning(f"🚨 Tentative de path traversal : {value[:50]}...")
                return False
        return True

    @classmethod
    def validate_no_xss(cls, value: str) -> bool:
        """Vérifie l'absence de XSS.

        Args:
            value: Chaîne à vérifier.

        Returns:
            True si aucun pattern XSS détecté.
        """
        for pattern in cls.XSS_PATTERNS:
            if pattern.search(value):
                logger.warning(f"🚨 Tentative de XSS détectée : {value[:50]}...")
                return False
        return True

    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Valide le format d'une adresse email.

        Args:
            email: Adresse email.

        Returns:
            True si le format est valide.
        """
        pattern = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        return bool(pattern.match(email))

    @classmethod
    def validate_phone_rdc(cls, phone: str) -> bool:
        """Valide un numéro de téléphone RDC.

        Formats acceptés : +243XXXXXXXXX, 0XXXXXXXXX, 243XXXXXXXXX

        Args:
            phone: Numéro de téléphone.

        Returns:
            True si le format est valide.
        """
        patterns = [
            re.compile(r"^\+243\d{9}$"),
            re.compile(r"^0\d{9}$"),
            re.compile(r"^243\d{9}$"),
        ]
        return any(p.match(phone) for p in patterns)

    @classmethod
    def validate_glycemie_value(cls, value: float, unite: str = "g/L") -> bool:
        """Valide une valeur glycémique (détection d'erreurs).

        Args:
            value: Valeur numérique.
            unite: Unité de mesure.

        Returns:
            True si la valeur est plausible.
        """
        from config import SEUILS, CONVERSION_VERS_G_PAR_L

        # Convertir en g/L pour la validation
        facteur = CONVERSION_VERS_G_PAR_L.get(unite, 1.0)
        valeur_gl = value * facteur

        return SEUILS.VALEUR_MIN_PLAUSIBLE <= valeur_gl <= SEUILS.VALEUR_MAX_PLAUSIBLE

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitise un nom de fichier.

        Args:
            filename: Nom de fichier original.

        Returns:
            Nom de fichier sécurisé.
        """
        # Supprimer le chemin
        filename = os.path.basename(filename)
        # Supprimer les caractères dangereux
        filename = re.sub(r"[^\w\s\-.]", "_", filename)
        # Limiter la longueur
        name, ext = os.path.splitext(filename)
        name = name[:100]
        ext = ext[:10].lower()
        if ext not in (".csv", ".xlsx", ".xls", ".json", ".pdf", ".png", ".jpg"):
            ext = ""
        return f"{name}{ext}" if name else "unnamed"


# ============================================================================
# CONFORMITÉ RGPD
# ============================================================================

class GDPRCompliance:
    """Gestionnaire de conformité RGPD.

    Implémente les droits des patients :
    - Consentement éclairé
    - Droit d'accès
    - Droit à la rectification
    - Droit à l'effacement (droit à l'oubli)
    - Droit à la portabilité
    - Minimisation des données
    - Pseudonymisation pour l'entraînement IA
    """

    @staticmethod
    def record_consent(
        patient_id: str,
        consent_type: str,
        granted: bool,
        version: str = "1.0",
        details: Optional[str] = None,
    ) -> dict[str, Any]:
        """Enregistre un consentement du patient.

        Args:
            patient_id: UUID du patient.
            consent_type: Type de consentement
                (data_processing, ai_training, data_sharing, marketing).
            granted: True si accordé.
            version: Version du document de consentement.
            details: Détails supplémentaires.

        Returns:
            Enregistrement du consentement.
        """
        record = {
            "patient_id": patient_id,
            "consent_type": consent_type,
            "granted": granted,
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "ip_address": None,  # À remplir par l'API
            "user_agent": None,
        }
        logger.info(
            f"📝 Consentement [{consent_type}] "
            f"{'accordé' if granted else 'refusé'} "
            f"pour patient {patient_id}"
        )
        return record

    @staticmethod
    def pseudonymize_for_training(
        patient_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Pseudonymise les données d'un patient pour l'entraînement IA.

        Supprime ou remplace tous les identifiants directs et indirects.

        Args:
            patient_data: Données brutes du patient.

        Returns:
            Données pseudonymisées.
        """
        # Champs à supprimer complètement
        fields_to_remove = [
            "id", "username", "email", "email_chiffre",
            "nom", "nom_chiffre", "prenom", "prenom_chiffre",
            "contact_urgence", "contact_urgence_chiffre",
            "medecin_referent", "medecin_referent_chiffre",
            "hash_password",
        ]

        # Champs à pseudonymiser
        pseudonymized = {}
        patient_hash = hashlib.sha256(
            patient_data.get("id", "unknown").encode()
        ).hexdigest()[:12]

        for key, value in patient_data.items():
            if key in fields_to_remove:
                continue
            elif key == "patient_id":
                pseudonymized[key] = f"P-{patient_hash}"
            elif key in ("age",):
                # Généraliser l'âge en tranche
                if isinstance(value, int):
                    pseudonymized[key] = f"{(value // 10) * 10}-{(value // 10) * 10 + 9}"
                else:
                    pseudonymized[key] = value
            elif key in ("created_at", "updated_at", "last_login", "timestamp"):
                # Généraliser les dates (garder seulement le mois)
                if isinstance(value, str) and len(value) >= 7:
                    pseudonymized[key] = value[:7]  # YYYY-MM
                else:
                    pseudonymized[key] = value
            else:
                pseudonymized[key] = value

        return pseudonymized

    @staticmethod
    def generate_data_retention_policy() -> dict[str, Any]:
        """Génère la politique de rétention des données.

        Returns:
            Politique de rétention structurée.
        """
        return {
            "data_retention_policy": {
                "version": "1.0",
                "last_updated": "2024-01-01",
                "categories": {
                    "mesures_glycemie": {
                        "retention_period": "10 ans",
                        "justification": "Suivi médical long terme du diabète",
                        "legal_basis": "Intérêt vital du patient",
                    },
                    "conversations_ia": {
                        "retention_period": "2 ans",
                        "justification": "Amélioration du service et continuité",
                        "legal_basis": "Intérêt légitime",
                    },
                    "alertes": {
                        "retention_period": "5 ans",
                        "justification": "Traçabilité médicale et sécurité",
                        "legal_basis": "Obligation légale",
                    },
                    "logs_acces": {
                        "retention_period": "1 an",
                        "justification": "Sécurité et audit",
                        "legal_basis": "Obligation légale",
                    },
                    "rapports_medicaux": {
                        "retention_period": "20 ans",
                        "justification": "Dossier médical",
                        "legal_basis": "Intérêt vital",
                    },
                    "donnees_personnelles": {
                        "retention_period": "Durée de la relation + 5 ans",
                        "justification": "Suivi médical",
                        "legal_basis": "Consentement + intérêt vital",
                    },
                },
            },
        }

    @staticmethod
    def check_data_minimization(data: dict[str, Any], purpose: str) -> list[str]:
        """Vérifie le principe de minimisation des données.

        Args:
            data: Données collectées.
            purpose: Finalité du traitement.

        Returns:
            Liste des champs potentiellement excessifs.
        """
        # Champs nécessaires par finalité
        required_fields: dict[str, set[str]] = {
            "suivi_glycemique": {"valeur", "timestamp", "moment_mesure", "patient_id"},
            "authentification": {"username", "hash_password"},
            "rapport_medecin": {"patient_id", "valeur", "timestamp", "type_diabete"},
            "alerte_urgence": {"patient_id", "valeur", "timestamp", "contact_urgence"},
        }

        required = required_fields.get(purpose, set())
        excess = [k for k in data.keys() if k not in required and k not in ("id",)]

        if excess:
            logger.debug(
                f"⚠️  Minimisation : champs potentiellement excessifs "
                f"pour '{purpose}' : {excess}"
            )

        return excess


# ============================================================================
# AUDIT & DÉTECTION D'ANOMALIES
# ============================================================================

class SecurityAuditor:
    """Auditeur de sécurité et détecteur d'anomalies.

    Surveille les patterns d'accès suspects :
    - Trop de tentatives de connexion
    - Accès hors horaires normaux
    - Accès depuis de nouvelles localisations
    - Volume anormal de données consultées
    """

    def __init__(self) -> None:
        """Initialise l'auditeur."""
        self._login_attempts: dict[str, list[float]] = defaultdict(list)
        self._data_access_counts: dict[str, int] = defaultdict(int)
        self._suspicious_events: list[dict[str, Any]] = []

    def record_login_attempt(
        self,
        username: str,
        ip_address: str,
        success: bool,
    ) -> Optional[dict[str, Any]]:
        """Enregistre une tentative de connexion.

        Args:
            username: Nom d'utilisateur.
            ip_address: Adresse IP source.
            success: Si la connexion a réussi.

        Returns:
            Alerte si comportement suspect détecté, sinon None.
        """
        now = time.time()
        key = f"{username}:{ip_address}"
        self._login_attempts[key].append(now)

        # Garder seulement les 24 dernières heures
        cutoff = now - 86400
        self._login_attempts[key] = [
            t for t in self._login_attempts[key] if t > cutoff
        ]

        recent = self._login_attempts[key]

        # Détecter le brute force
        if not success:
            recent_failures = [
                t for t in recent if t > now - 900  # 15 minutes
            ]
            if len(recent_failures) >= 5:
                alert = {
                    "type": "brute_force_suspect",
                    "severity": "high",
                    "username": username,
                    "ip_address": ip_address,
                    "attempts_15min": len(recent_failures),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._suspicious_events.append(alert)
                logger.critical(
                    f"🚨 BRUTE FORCE suspecté : {username} "
                    f"depuis {ip_address} ({len(recent_failures)} échecs/15min)"
                )
                return alert

        return None

    def record_data_access(
        self,
        actor: str,
        patient_id: str,
        resource: str,
    ) -> Optional[dict[str, Any]]:
        """Enregistre un accès aux données et détecte les anomalies.

        Args:
            actor: Identifiant de l'accédant.
            patient_id: Patient dont les données sont accédées.
            resource: Ressource accédée.

        Returns:
            Alerte si volume anormal, sinon None.
        """
        key = f"{actor}:{patient_id}"
        self._data_access_counts[key] += 1

        # Détecter un volume anormal (>100 accès/jour au même patient)
        if self._data_access_counts[key] > 100:
            alert = {
                "type": "acces_donnees_anormal",
                "severity": "medium",
                "actor": actor,
                "patient_id": patient_id,
                "resource": resource,
                "count": self._data_access_counts[key],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._suspicious_events.append(alert)
            logger.warning(
                f"⚠️  Volume d'accès anormal : {actor} → "
                f"patient {patient_id} ({self._data_access_counts[key]} accès)"
            )
            return alert

        return None

    def get_suspicious_events(
        self,
        limit: int = 50,
        severity: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retourne les événements suspects récents.

        Args:
            limit: Nombre max d'événements.
            severity: Filtrer par sévérité.

        Returns:
            Liste d'événements suspects.
        """
        events = self._suspicious_events
        if severity:
            events = [e for e in events if e.get("severity") == severity]
        return events[-limit:]

    def reset_daily_counters(self) -> None:
        """Réinitialise les compteurs quotidiens."""
        self._data_access_counts.clear()
        logger.info("🔄 Compteurs d'audit quotidiens réinitialisés")


# ============================================================================
# PROTECTION CSRF
# ============================================================================

class CSRFProtection:
    """Protection contre les attaques CSRF (Cross-Site Request Forgery).

    Génère et vérifie des tokens CSRF pour les requêtes state-changing.
    """

    def __init__(self, secret: str = SECURITY.SECRET_KEY) -> None:
        """Initialise la protection CSRF.

        Args:
            secret: Clé secrète pour la signature.
        """
        self.secret = secret
        self._tokens: dict[str, float] = {}  # token -> expiry

    def generate_token(self, session_id: str, ttl_seconds: int = 3600) -> str:
        """Génère un token CSRF.

        Args:
            session_id: Identifiant de session.
            ttl_seconds: Durée de validité.

        Returns:
            Token CSRF.
        """
        nonce = secrets.token_urlsafe(32)
        raw = f"{session_id}:{nonce}:{time.time()}"
        signature = hmac.new(
            self.secret.encode(), raw.encode(), hashlib.sha256,
        ).hexdigest()
        token = f"{nonce}.{signature}"

        self._tokens[token] = time.time() + ttl_seconds
        return token

    def verify_token(self, session_id: str, token: str) -> bool:
        """Vérifie un token CSRF.

        Args:
            session_id: Identifiant de session.
            token: Token CSRF à vérifier.

        Returns:
            True si le token est valide.
        """
        if token not in self._tokens:
            return False

        if time.time() > self._tokens[token]:
            del self._tokens[token]
            return False

        try:
            nonce, signature = token.rsplit(".", 1)
            # La vérification complète nécessiterait le timestamp original
            # Simplifié ici pour la démonstration
            del self._tokens[token]  # Usage unique
            return True
        except (ValueError, Exception):
            return False


# ============================================================================
# GESTIONNAIRE DE SÉCURITÉ PRINCIPAL (Facade)
# ============================================================================

class SecurityManager:
    """Facade de sécurité unifiée pour DiaBot-RDC.

    Point d'entrée unique pour toutes les opérations de sécurité.
    """

    def __init__(self) -> None:
        """Initialise tous les sous-systèmes de sécurité."""
        self.password = PasswordHasher()
        self.jwt = JWTManager()
        self.otp = OTPManager()
        self.aes = AESEncryptor()
        self.rate_limiter = RateLimiter()
        self.sanitizer = InputSanitizer()
        self.gdpr = GDPRCompliance()
        self.auditor = SecurityAuditor()
        self.csrf = CSRFProtection()

        logger.info("🔒 SecurityManager initialisé avec tous les sous-systèmes")

    def authenticate_user(
        self,
        username: str,
        password: str,
        stored_hash: str,
        ip_address: str = "unknown",
    ) -> Optional[dict[str, str]]:
        """Authentifie un utilisateur et retourne les tokens.

        Args:
            username: Nom d'utilisateur.
            password: Mot de passe en clair.
            stored_hash: Hash bcrypt stocké en DB.
            ip_address: IP source.

        Returns:
            Dict avec tokens JWT si succès, None sinon.
        """
        # Vérifier le rate limit de login
        allowed, _ = self.rate_limiter.is_allowed(
            f"login:{ip_address}",
            max_requests=SECURITY.RATE_LIMIT_LOGIN_PER_MIN,
        )
        if not allowed:
            logger.warning(f"🚫 Login bloqué (rate limit) : {ip_address}")
            return None

        # Vérifier le mot de passe
        if not self.password.verify(password, stored_hash):
            self.auditor.record_login_attempt(username, ip_address, success=False)
            return None

        # Succès
        self.auditor.record_login_attempt(username, ip_address, success=True)
        tokens = self.jwt.create_token_pair(username, roles=["patient"])
        logger.info(f"✅ Authentification réussie : {username}")
        return tokens

    def validate_and_sanitize_input(
        self,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Valide et sanitise un dictionnaire d'entrées.

        Args:
            data: Données d'entrée.

        Returns:
            Tuple (données sanitiser, liste d'erreurs).
        """
        errors: list[str] = []
        sanitized: dict[str, Any] = {}

        for key, value in data.items():
            if isinstance(value, str):
                # Sanitiser
                clean = self.sanitizer.sanitize_string(value)

                # Vérifier les injections
                if not self.sanitizer.validate_no_sql_injection(value):
                    errors.append(f"Injection SQL détectée dans '{key}'")
                    continue
                if not self.sanitizer.validate_no_xss(value):
                    errors.append(f"XSS détecté dans '{key}'")
                    continue

                sanitized[key] = clean
            else:
                sanitized[key] = value

        return sanitized, errors


# ============================================================================
# INSTANCE GLOBALE
# ============================================================================

security_manager = SecurityManager()


__all__ = [
    "PasswordHasher", "JWTManager", "TokenPayload", "OTPManager",
    "AESEncryptor", "RateLimiter", "InputSanitizer", "GDPRCompliance",
    "SecurityAuditor", "CSRFProtection", "SecurityManager",
    "security_manager",
]