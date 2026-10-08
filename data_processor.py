"""
================================================================================
FICHIER : data_processor.py
RESPONSABILITÉ : Import, validation, nettoyage et export de données glycémiques
================================================================================
Ce module gère TOUT le pipeline de traitement des données :
- Import de fichiers Excel (.xlsx) via openpyxl et CSV via pandas
- Validation multi-niveaux : format, types, plages, cohérence
- Détection de valeurs aberrantes (glycémie < 0.20 ou > 6.00 g/L)
- Conversion automatique d'unités (mg/dL ↔ g/L ↔ mmol/L)
- Détection et gestion des doublons temporels
- Vérification de la cohérence chronologique
- Rapport d'importation détaillé avec prévisualisation
- Nettoyage, normalisation et imputation
- Agrégation temporelle (heure, jour, semaine, mois)
- Export des données nettoyées (CSV, JSON)

Auteur : Équipe DiaBot-RDC
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

import io
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from config import (
    SEUILS, CONVERSION_VERS_G_PAR_L, UniteGlycemie, MomentMesure,
    UPLOADS_DIR, EXPORTS_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

class ImportStatus(str, Enum):
    """Statut d'un import de données."""
    EN_ATTENTE = "en_attente"
    PREVIEW = "preview"
    VALIDATED = "validated"
    PARTIAL = "partial"
    ERROR = "error"
    ROLLED_BACK = "rolled_back"


class ErreurType(str, Enum):
    """Types d'erreurs de validation."""
    FORMAT_COLONNE = "format_colonne"
    VALEUR_MANQUANTE = "valeur_manquante"
    VALEUR_ABERRENTE = "valeur_aberrante"
    UNITE_INCONNUE = "unite_inconnue"
    DATE_INVALIDE = "date_invalide"
    DOUBLON_TEMPOREL = "doublon_temporel"
    INCOHERENCE_CHRONOLOGIQUE = "incoherence_chronologique"
    TYPE_INCORRECT = "type_incorrect"
    PLAGE_INVALIDE = "plage_invalide"
    LIGNE_VIDE = "ligne_vide"


@dataclass
class ErreurValidation:
    """Représente une erreur de validation sur une ligne."""
    ligne: int
    colonne: str
    type_erreur: ErreurType
    message: str
    valeur_originale: Any = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation en dict."""
        return {
            "ligne": self.ligne,
            "colonne": self.colonne,
            "type": self.type_erreur.value,
            "message": self.message,
            "valeur_originale": str(self.valeur_originale),
            "suggestion": self.suggestion,
        }


@dataclass
class RapportImport:
    """Rapport complet d'une opération d'import."""
    import_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    format_fichier: str = ""
    statut: ImportStatus = ImportStatus.EN_ATTENTE
    nb_lignes_total: int = 0
    nb_lignes_valides: int = 0
    nb_lignes_rejetees: int = 0
    nb_lignes_corrigees: int = 0
    erreurs: list[ErreurValidation] = field(default_factory=list)
    avertissements: list[str] = field(default_factory=list)
    colonnes_detectees: list[str] = field(default_factory=list)
    mapping_colonnes: dict[str, str] = field(default_factory=dict)
    unite_detectee: str = UniteGlycemie.G_PAR_L.value
    preview_data: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duree_traitement_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation complète."""
        return {
            "import_id": self.import_id,
            "filename": self.filename,
            "format": self.format_fichier,
            "statut": self.statut.value,
            "nb_lignes_total": self.nb_lignes_total,
            "nb_lignes_valides": self.nb_lignes_valides,
            "nb_lignes_rejetees": self.nb_lignes_rejetees,
            "nb_lignes_corrigees": self.nb_lignes_corrigees,
            "taux_reussite": round(
                self.nb_lignes_valides / max(self.nb_lignes_total, 1) * 100, 1
            ),
            "erreurs": [e.to_dict() for e in self.erreurs],
            "avertissements": self.avertissements,
            "colonnes_detectees": self.colonnes_detectees,
            "mapping_colonnes": self.mapping_colonnes,
            "unite_detectee": self.unite_detectee,
            "preview": self.preview_data[:20],
            "timestamp": self.timestamp.isoformat(),
            "duree_ms": self.duree_traitement_ms,
        }

    @property
    def taux_reussite(self) -> float:
        """Taux de réussite en pourcentage."""
        if self.nb_lignes_total == 0:
            return 0.0
        return self.nb_lignes_valides / self.nb_lignes_total * 100


# ============================================================================
# MAPPING DES COLONNES (détection automatique)
# ============================================================================

# Alias possibles pour chaque colonne standard (FR, EN, variantes)
COLONNE_ALIASES: dict[str, list[str]] = {
    "date": [
        "date", "date_heure", "datetime", "timestamp", "horodatage",
        "date and time", "date/time", "date_time", "time",
        "jour", "date de mesure", "date mesure",
    ],
    "valeur": [
        "valeur", "glycemie", "glycémie", "glucose", "blood glucose",
        "bg", "sugar", "valeur glycémie", "valeur glycemie",
        "glucose level", "reading", "mesure", "resultat", "résultat",
        "valeur (g/l)", "valeur (mg/dl)", "valeur (mmol/l)",
    ],
    "unite": [
        "unite", "unité", "unit", "unite de mesure", "unité de mesure",
    ],
    "moment": [
        "moment", "moment de mesure", "moment mesure", "repas",
        "meal", "meal time", "type de mesure", "type mesure",
        "contexte", "context", "periode", "période",
    ],
    "note": [
        "note", "notes", "commentaire", "commentaires", "remarque",
        "remarques", "comment", "comments", "observation",
    ],
    "repas": [
        "repas", "repas associe", "repas associé", "meal",
        "food", "nourriture", "alimentation",
    ],
    "activite": [
        "activite", "activité", "activite physique", "activité physique",
        "exercise", "sport", "physical activity",
    ],
    "stress": [
        "stress", "stress level", "niveau de stress", "niveau stress",
    ],
    "medicament": [
        "medicament", "médicament", "medication", "traitement",
        "insuline", "insulin", "dose",
    ],
}


def _detecter_colonne(nom_colonne: str) -> Optional[str]:
    """Détecte le rôle d'une colonne à partir de son nom.

    Args:
        nom_colonne: Nom de la colonne (nettoyé).

    Returns:
        Nom standard de la colonne ou None.
    """
    nom_clean = nom_colonne.strip().lower()
    # Supprimer les accents pour la comparaison
    nom_clean = re.sub(r"[éèêë]", "e", nom_clean)
    nom_clean = re.sub(r"[àâä]", "a", nom_clean)
    nom_clean = re.sub(r"[îï]", "i", nom_clean)
    nom_clean = re.sub(r"[ôö]", "o", nom_clean)
    nom_clean = re.sub(r"[ùûü]", "u", nom_clean)
    nom_clean = re.sub(r"[ç]", "c", nom_clean)

    for standard, aliases in COLONNE_ALIASES.items():
        for alias in aliases:
            alias_clean = re.sub(r"[éèêë]", "e", alias)
            alias_clean = re.sub(r"[àâä]", "a", alias_clean)
            alias_clean = re.sub(r"[îï]", "i", alias_clean)
            alias_clean = re.sub(r"[ôö]", "o", alias_clean)
            alias_clean = re.sub(r"[ùûü]", "u", alias_clean)
            alias_clean = re.sub(r"[ç]", "c", alias_clean)
            if nom_clean == alias_clean or nom_clean in alias_clean:
                return standard

    return None


# ============================================================================
# DÉTECTION AUTOMATIQUE D'UNITÉ
# ============================================================================

def _detecter_unite(valeurs: pd.Series) -> str:
    """Détecte automatiquement l'unité des valeurs glycémiques.

    Heuristique basée sur la médiane des valeurs :
    - Médiane < 5 → probablement g/L
    - Médiane 5-30 → probablement mmol/L
    - Médiane > 30 → probablement mg/dL

    Args:
        valeurs: Série de valeurs numériques.

    Returns:
        Code unité détectée.
    """
    valeurs_numeriques = pd.to_numeric(valeurs, errors="coerce").dropna()
    if valeurs_numeriques.empty:
        return UniteGlycemie.G_PAR_L.value

    mediane = valeurs_numeriques.median()

    if mediane < 5.0:
        return UniteGlycemie.G_PAR_L.value
    elif mediane < 30.0:
        return UniteGlycemie.MMOL_PAR_L.value
    else:
        return UniteGlycemie.MG_PAR_DL.value


# ============================================================================
# CONVERSION D'UNITÉS
# ============================================================================

def convertir_valeur(
    valeur: float,
    unite_source: str,
    unite_cible: str = UniteGlycemie.G_PAR_L.value,
) -> float:
    """Convertit une valeur glycémique d'une unité à une autre.

    Args:
        valeur: Valeur numérique à convertir.
        unite_source: Unité d'origine.
        unite_cible: Unité cible (défaut : g/L).

    Returns:
        Valeur convertie.

    Raises:
        ValueError: Si l'unité est inconnue.
    """
    if unite_source == unite_cible:
        return valeur

    # Normaliser les noms d'unités
    unite_source = _normaliser_unite(unite_source)
    unite_cible = _normaliser_unite(unite_cible)

    if unite_source == unite_cible:
        return valeur

    # Convertir d'abord en g/L (unité de référence)
    facteur_source = CONVERSION_VERS_G_PAR_L.get(unite_source)
    if facteur_source is None:
        raise ValueError(f"Unité source inconnue : {unite_source}")

    valeur_gl = valeur * facteur_source

    # Puis convertir de g/L vers l'unité cible
    facteur_cible = CONVERSION_VERS_G_PAR_L.get(unite_cible)
    if facteur_cible is None:
        raise ValueError(f"Unité cible inconnue : {unite_cible}")

    return round(valeur_gl / facteur_cible, 4)


def _normaliser_unite(unite: str) -> str:
    """Normalise le nom d'une unité.

    Args:
        unite: Nom d'unité (peu importe le format).

    Returns:
        Nom d'unité normalisé.
    """
    unite_lower = unite.strip().lower()
    mapping = {
        "g/l": "g/L",
        "gl": "g/L",
        "g.dl": "g/L",
        "mg/dl": "mg/dL",
        "mgdl": "mg/dL",
        "mg.dl": "mg/dL",
        "mmol/l": "mmol/L",
        "mmoll": "mmol/L",
        "mmol.l": "mmol/L",
    }
    return mapping.get(unite_lower, unite)


# ============================================================================
# PARSERS DE DATE
# ============================================================================

FORMATS_DATE_SUPPORTES: list[str] = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y",
]


def parser_date(valeur: Any) -> Optional[datetime]:
    """Tente de parser une date depuis différents formats.

    Args:
        valeur: Valeur à parser (str, datetime, int, float).

    Returns:
        Datetime UTC ou None si échec.
    """
    if isinstance(valeur, datetime):
        if valeur.tzinfo is None:
            return valeur.replace(tzinfo=timezone.utc)
        return valeur.astimezone(timezone.utc)

    if isinstance(valeur, (int, float)):
        # Timestamp Unix
        try:
            if valeur > 1e12:  # Millisecondes
                valeur = valeur / 1000
            dt = datetime.fromtimestamp(valeur, tz=timezone.utc)
            if 2000 <= dt.year <= 2100:
                return dt
        except (OSError, ValueError, OverflowError):
            pass

    if isinstance(valeur, str):
        valeur = valeur.strip()
        if not valeur:
            return None

        # Essayer pandas d'abord (très flexible)
        try:
            dt = pd.to_datetime(valeur, dayfirst=True)
            if pd.notna(dt):
                if dt.tzinfo is None:
                    dt = dt.tz_localize(timezone.utc)
                return dt.to_pydatetime()
        except (ValueError, TypeError):
            pass

        # Essayer chaque format
        for fmt in FORMATS_DATE_SUPPORTES:
            try:
                dt = datetime.strptime(valeur, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

    return None


# ============================================================================
# NORMALISATION DES MOMENTS DE MESURE
# ============================================================================

MOMENT_ALIASES: dict[str, list[str]] = {
    MomentMesure.A_JEUN.value: [
        "a jeun", "à jeun", "jeun", "fasting", "fast", "a_jeun",
        "matin a jeun", "réveil", "reveil", "waking",
    ],
    MomentMesure.AVANT_REPAS.value: [
        "avant repas", "avant le repas", "pre repas", "pré repas",
        "before meal", "pre meal", "avant manger",
    ],
    MomentMesure.POSTPRANDIAL_1H.value: [
        "postprandial 1h", "1h apres repas", "1h après repas",
        "1h post", "1h after meal", "post 1h",
    ],
    MomentMesure.POSTPRANDIAL_2H.value: [
        "postprandial", "postprandial 2h", "2h apres repas",
        "2h après repas", "2h post", "after meal", "post meal",
        "2h after", "post 2h", "apres repas", "après repas",
    ],
    MomentMesure.AVANT_COUCHER.value: [
        "coucher", "avant coucher", "bedtime", "before bed",
        "soir", "soiree", "soirée",
    ],
    MomentMesure.NUIT.value: [
        "nuit", "night", "nocturne", "nocturnal", "3h", "4h",
    ],
    MomentMesure.APRES_ACTIVITE.value: [
        "apres activite", "après activité", "after exercise",
        "post exercise", "sport",
    ],
}


def normaliser_moment(valeur: Any) -> str:
    """Normalise le moment de mesure.

    Args:
        valeur: Valeur brute du moment.

    Returns:
        Code moment normalisé.
    """
    if not valeur or (isinstance(valeur, float) and np.isnan(valeur)):
        return MomentMesure.AUTRE.value

    val_str = str(valeur).strip().lower()
    val_clean = re.sub(r"[éèêë]", "e", val_str)
    val_clean = re.sub(r"[àâä]", "a", val_clean)
    val_clean = re.sub(r"[îï]", "i", val_clean)
    val_clean = re.sub(r"[ôö]", "o", val_clean)

    for standard, aliases in MOMENT_ALIASES.items():
        for alias in aliases:
            alias_clean = re.sub(r"[éèêë]", "e", alias)
            alias_clean = re.sub(r"[àâä]", "a", alias_clean)
            if val_clean == alias_clean or alias_clean in val_clean:
                return standard

    return MomentMesure.AUTRE.value


# ============================================================================
# LECTURE DE FICHIERS
# ============================================================================

def lire_fichier_excel(
    filepath: Path,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """Lit un fichier Excel et retourne un DataFrame.

    Args:
        filepath: Chemin du fichier .xlsx.
        sheet_name: Nom de l'onglet (défaut : premier).

    Returns:
        DataFrame pandas.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier est corrompu.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    try:
        # Vérifier que c'est bien un Excel valide
        wb = load_workbook(str(filepath), read_only=True, data_only=True)
        sheets = wb.sheetnames
        wb.close()

        target_sheet = sheet_name or sheets[0]
        logger.info(
            f"📊 Lecture Excel : {filepath.name} "
            f"(onglet '{target_sheet}', {len(sheets)} onglets)"
        )

        df = pd.read_excel(
            str(filepath),
            sheet_name=target_sheet,
            engine="openpyxl",
            dtype=str,  # Tout lire en string pour validation manuelle
        )

        # Nettoyer les noms de colonnes
        df.columns = [
            str(c).strip().replace("\n", " ").replace("\r", "")
            for c in df.columns
        ]

        return df

    except Exception as exc:
        logger.error(f"❌ Erreur lecture Excel : {exc}")
        raise ValueError(f"Fichier Excel invalide ou corrompu : {exc}") from exc


def lire_fichier_csv(
    filepath: Path,
    encoding: Optional[str] = None,
    separateur: Optional[str] = None,
) -> pd.DataFrame:
    """Lit un fichier CSV et retourne un DataFrame.

    Détecte automatiquement l'encodage et le séparateur.

    Args:
        filepath: Chemin du fichier .csv.
        encoding: Encodage (auto-détection si None).
        separateur: Séparateur (auto-détection si None).

    Returns:
        DataFrame pandas.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier est illisible.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    # Auto-détection de l'encodage
    encodings_to_try = [encoding] if encoding else [
        "utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1",
    ]

    # Auto-détection du séparateur
    separators_to_try = [separateur] if separateur else [
        ",", ";", "\t", "|",
    ]

    for enc in encodings_to_try:
        for sep in separators_to_try:
            try:
                df = pd.read_csv(
                    str(filepath),
                    encoding=enc,
                    sep=sep,
                    dtype=str,
                    on_bad_lines="skip",
                )
                # Vérifier que le résultat est raisonnable
                if len(df.columns) >= 2 and len(df) > 0:
                    logger.info(
                        f"📊 Lecture CSV : {filepath.name} "
                        f"(enc={enc}, sep='{sep}', "
                        f"{len(df)} lignes, {len(df.columns)} colonnes)"
                    )
                    df.columns = [
                        str(c).strip().replace("\n", " ")
                        for c in df.columns
                    ]
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            except Exception:
                continue

    raise ValueError(
        f"Impossible de lire le CSV : {filepath.name}. "
        f"Vérifiez l'encodage et le format."
    )


# ============================================================================
# PIPELINE DE VALIDATION PRINCIPAL
# ============================================================================

class DataValidator:
    """Validateur multi-niveaux pour les données glycémiques.

    Pipeline de validation :
    1. Structure (colonnes requises)
    2. Types (dates, nombres)
    3. Plages (valeurs plausibles)
    4. Cohérence (chronologie, doublons)
    5. Unités (détection et conversion)
    """

    def __init__(self) -> None:
        """Initialise le validateur."""
        self.erreurs: list[ErreurValidation] = []
        self.avertissements: list[str] = []

    def reset(self) -> None:
        """Réinitialise les erreurs et avertissements."""
        self.erreurs.clear()
        self.avertissements.clear()

    def _add_erreur(
        self,
        ligne: int,
        colonne: str,
        type_erreur: ErreurType,
        message: str,
        valeur: Any = None,
        suggestion: Optional[str] = None,
    ) -> None:
        """Ajoute une erreur de validation."""
        self.erreurs.append(ErreurValidation(
            ligne=ligne,
            colonne=colonne,
            type_erreur=type_erreur,
            message=message,
            valeur_originale=valeur,
            suggestion=suggestion,
        ))

    def _add_avertissement(self, message: str) -> None:
        """Ajoute un avertissement."""
        self.avertissements.append(message)

    def valider_structure(
        self,
        df: pd.DataFrame,
        mapping: dict[str, str],
    ) -> bool:
        """Valide la structure du DataFrame (colonnes requises).

        Args:
            df: DataFrame à valider.
            mapping: Mapping colonnes détectées → colonnes standard.

        Returns:
            True si la structure est valide.
        """
        # Colonnes obligatoires
        required = {"date", "valeur"}
        mapped_columns = set(mapping.values())

        missing = required - mapped_columns
        if missing:
            for col in missing:
                self._add_erreur(
                    ligne=0,
                    colonne=col,
                    type_erreur=ErreurType.FORMAT_COLONNE,
                    message=f"Colonne obligatoire '{col}' non trouvée",
                    suggestion=(
                        "Assurez-vous que votre fichier contient une colonne "
                        "de date et une colonne de valeur glycémique"
                    ),
                )
            return False

        return True

    def valider_types(
        self,
        df: pd.DataFrame,
        mapping: dict[str, str],
    ) -> pd.DataFrame:
        """Valide et convertit les types de données.

        Args:
            df: DataFrame source.
            mapping: Mapping des colonnes.

        Returns:
            DataFrame avec types convertis.
        """
        result = df.copy()

        # Inverser le mapping : nom_original → nom_standard
        inv_mapping = {v: k for k, v in mapping.items()}

        for idx, row in result.iterrows():
            ligne_num = idx + 2  # +2 pour l'en-tête et l'index 0-based

            # Valider la date
            col_date = inv_mapping.get("date", "")
            if col_date and col_date in result.columns:
                date_val = row.get(col_date)
                parsed = parser_date(date_val)
                if parsed is None and pd.notna(date_val) and str(date_val).strip():
                    self._add_erreur(
                        ligne=ligne_num,
                        colonne=col_date,
                        type_erreur=ErreurType.DATE_INVALIDE,
                        message=f"Date non reconnue : '{date_val}'",
                        valeur=date_val,
                        suggestion="Formats acceptés : JJ/MM/AAAA HH:MM ou AAAA-MM-JJ HH:MM",
                    )

            # Valider la valeur
            col_valeur = inv_mapping.get("valeur", "")
            if col_valeur and col_valeur in result.columns:
                val = row.get(col_valeur)
                if pd.isna(val) or str(val).strip() == "":
                    self._add_erreur(
                        ligne=ligne_num,
                        colonne=col_valeur,
                        type_erreur=ErreurType.VALEUR_MANQUANTE,
                        message="Valeur glycémique manquante",
                        suggestion="Renseignez la valeur de glycémie",
                    )
                else:
                    try:
                        # Nettoyer la valeur (virgule → point, espaces)
                        val_str = str(val).strip().replace(",", ".").replace(" ", "")
                        val_float = float(val_str)
                    except (ValueError, TypeError):
                        self._add_erreur(
                            ligne=ligne_num,
                            colonne=col_valeur,
                            type_erreur=ErreurType.TYPE_INCORRECT,
                            message=f"Valeur non numérique : '{val}'",
                            valeur=val,
                            suggestion="La valeur doit être un nombre (ex: 1.25 ou 125)",
                        )

        return result

    def valider_plages(
        self,
        df: pd.DataFrame,
        mapping: dict[str, str],
        unite: str,
    ) -> None:
        """Valide les plages de valeurs glycémiques.

        Args:
            df: DataFrame à valider.
            mapping: Mapping des colonnes.
            unite: Unité des valeurs.
        """
        inv_mapping = {v: k for k, v in mapping.items()}
        col_valeur = inv_mapping.get("valeur", "")
        if not col_valeur or col_valeur not in df.columns:
            return

        for idx, row in df.iterrows():
            ligne_num = idx + 2
            val = row.get(col_valeur)
            if pd.isna(val):
                continue

            try:
                val_str = str(val).strip().replace(",", ".")
                val_float = float(val_str)
            except (ValueError, TypeError):
                continue

            # Convertir en g/L pour la validation
            try:
                val_gl = convertir_valeur(val_float, unite)
            except ValueError:
                val_gl = val_float

            # Vérifier les valeurs aberrantes
            if val_gl < SEUILS.VALEUR_MIN_PLAUSIBLE:
                self._add_erreur(
                    ligne=ligne_num,
                    colonne=col_valeur,
                    type_erreur=ErreurType.VALEUR_ABERRENTE,
                    message=(
                        f"Valeur trop basse : {val_float} {unite} "
                        f"({val_gl:.2f} g/L). Probablement une erreur de saisie."
                    ),
                    valeur=val_float,
                    suggestion=(
                        f"Vérifiez la valeur. Le minimum plausible est "
                        f"{SEUILS.VALEUR_MIN_PLAUSIBLE} g/L"
                    ),
                )
            elif val_gl > SEUILS.VALEUR_MAX_PLAUSIBLE:
                # Peut-être une erreur d'unité (mg/dL saisi comme g/L)
                if unite == UniteGlycemie.G_PAR_L.value and val_float > 30:
                    self._add_erreur(
                        ligne=ligne_num,
                        colonne=col_valeur,
                        type_erreur=ErreurType.VALEUR_ABERRENTE,
                        message=(
                            f"Valeur suspecte : {val_float} g/L. "
                            f"Est-ce en mg/dL ? ({val_float} mg/dL = "
                            f"{convertir_valeur(val_float, 'mg/dL'):.2f} g/L)"
                        ),
                        valeur=val_float,
                        suggestion="Vérifiez l'unité de mesure",
                    )
                else:
                    self._add_erreur(
                        ligne=ligne_num,
                        colonne=col_valeur,
                        type_erreur=ErreurType.VALEUR_ABERRENTE,
                        message=(
                            f"Valeur trop haute : {val_float} {unite} "
                            f"({val_gl:.2f} g/L)"
                        ),
                        valeur=val_float,
                        suggestion=(
                            f"Le maximum plausible est "
                            f"{SEUILS.VALEUR_MAX_PLAUSIBLE} g/L"
                        ),
                    )

    def valider_chronologie(
        self,
        df: pd.DataFrame,
        mapping: dict[str, str],
    ) -> None:
        """Vérifie la cohérence chronologique et les doublons.

        Args:
            df: DataFrame à valider.
            mapping: Mapping des colonnes.
        """
        inv_mapping = {v: k for k, v in mapping.items()}
        col_date = inv_mapping.get("date", "")
        if not col_date or col_date not in df.columns:
            return

        # Parser toutes les dates
        dates_parsed: list[tuple[int, Optional[datetime]]] = []
        for idx, row in df.iterrows():
            dt = parser_date(row.get(col_date))
            dates_parsed.append((idx + 2, dt))

        # Vérifier les doublons temporels (même date à la minute près)
        seen_dates: dict[str, int] = {}
        for ligne, dt in dates_parsed:
            if dt is None:
                continue
            dt_key = dt.strftime("%Y-%m-%d %H:%M")
            if dt_key in seen_dates:
                self._add_erreur(
                    ligne=ligne,
                    colonne=col_date,
                    type_erreur=ErreurType.DOUBLON_TEMPOREL,
                    message=(
                        f"Doublon temporel détecté : {dt_key} "
                        f"(déjà présent ligne {seen_dates[dt_key]})"
                    ),
                    suggestion="Supprimez le doublon ou vérifiez l'heure exacte",
                )
            else:
                seen_dates[dt_key] = ligne

        # Vérifier la cohérence chronologique
        valid_dates = [(l, d) for l, d in dates_parsed if d is not None]
        if len(valid_dates) > 1:
            out_of_order = 0
            for i in range(1, len(valid_dates)):
                if valid_dates[i][1] < valid_dates[i - 1][1]:
                    out_of_order += 1

            if out_of_order > 0:
                self._add_avertissement(
                    f"{out_of_order} lignes ne sont pas dans l'ordre "
                    f"chronologique. Elles seront réordonnées automatiquement."
                )

    def valider_complet(
        self,
        df: pd.DataFrame,
        mapping: dict[str, str],
        unite: str,
    ) -> bool:
        """Exécute le pipeline de validation complet.

        Args:
            df: DataFrame à valider.
            mapping: Mapping des colonnes.
            unite: Unité des valeurs.

        Returns:
            True si aucune erreur bloquante.
        """
        self.reset()

        # 1. Structure
        if not self.valider_structure(df, mapping):
            return False

        # 2. Types
        self.valider_types(df, mapping)

        # 3. Plages
        self.valider_plages(df, mapping, unite)

        # 4. Chronologie
        self.valider_chronologie(df, mapping)

        # Résumé
        nb_erreurs = len(self.erreurs)
        nb_warnings = len(self.avertissements)
        if nb_erreurs > 0:
            logger.warning(
                f"⚠️  Validation : {nb_erreurs} erreurs, "
                f"{nb_warnings} avertissements"
            )
        else:
            logger.info(
                f"✅ Validation réussie ({nb_warnings} avertissements)"
            )

        # Bloquant si > 50% d'erreurs
        taux_erreur = nb_erreurs / max(len(df), 1)
        return taux_erreur < 0.5


# ============================================================================
# PROCESSEUR DE DONNÉES PRINCIPAL
# ============================================================================

class DataProcessor:
    """Processeur de données glycémiques complet.

    Orchestre l'import, la validation, le nettoyage et l'export.
    """

    def __init__(self) -> None:
        """Initialise le processeur."""
        self.validator = DataValidator()
        self._active_imports: dict[str, dict[str, Any]] = {}

    def importer_fichier(
        self,
        filepath: Path,
        patient_id: str,
        sheet_name: Optional[str] = None,
    ) -> RapportImport:
        """Importe un fichier de données glycémiques.

        Pipeline complet :
        1. Lecture du fichier
        2. Détection des colonnes
        3. Détection de l'unité
        4. Validation
        5. Prévisualisation

        Args:
            filepath: Chemin du fichier.
            patient_id: UUID du patient.
            sheet_name: Onglet Excel (optionnel).

        Returns:
            Rapport d'importation complet.
        """
        import time
        start_time = time.time()

        rapport = RapportImport(
            filename=filepath.name,
            format_fichier=filepath.suffix.lower().lstrip("."),
        )

        try:
            # 1. Lecture
            if filepath.suffix.lower() in (".xlsx", ".xls"):
                df = lire_fichier_excel(filepath, sheet_name)
            elif filepath.suffix.lower() == ".csv":
                df = lire_fichier_csv(filepath)
            else:
                raise ValueError(
                    f"Format non supporté : {filepath.suffix}. "
                    f"Formats acceptés : .xlsx, .csv"
                )

            rapport.nb_lignes_total = len(df)
            rapport.colonnes_detectees = list(df.columns)

            if df.empty:
                rapport.statut = ImportStatus.ERROR
                rapport.avertissements.append("Le fichier est vide")
                return rapport

            # 2. Détection des colonnes
            mapping: dict[str, str] = {}  # standard → original
            for col in df.columns:
                detected = _detecter_colonne(col)
                if detected and detected not in mapping:
                    mapping[detected] = col

            rapport.mapping_colonnes = mapping
            logger.info(f"🔍 Mapping détecté : {mapping}")

            # 3. Détection de l'unité
            col_valeur = mapping.get("valeur", "")
            if col_valeur:
                rapport.unite_detectee = _detecter_unite(df[col_valeur])
                logger.info(f"📏 Unité détectée : {rapport.unite_detectee}")

            # Vérifier si l'unité est dans le nom de la colonne
            for col in df.columns:
                col_lower = col.lower()
                if "mg/dl" in col_lower or "mgdl" in col_lower:
                    rapport.unite_detectee = UniteGlycemie.MG_PAR_DL.value
                elif "mmol" in col_lower:
                    rapport.unite_detectee = UniteGlycemie.MMOL_PAR_L.value
                elif "g/l" in col_lower or "gl" in col_lower:
                    rapport.unite_detectee = UniteGlycemie.G_PAR_L.value

            # 4. Validation
            is_valid = self.validator.valider_complet(
                df, mapping, rapport.unite_detectee,
            )
            rapport.erreurs = self.validator.erreurs
            rapport.avertissements = self.validator.avertissements

            # 5. Prévisualisation (10 premières lignes traitées)
            preview_df = self._nettoyer_dataframe(
                df.head(10), mapping, rapport.unite_detectee,
            )
            rapport.preview_data = preview_df.to_dict("records")

            # Statut
            if is_valid:
                rapport.statut = ImportStatus.PREVIEW
                rapport.nb_lignes_valides = (
                    rapport.nb_lignes_total - len(rapport.erreurs)
                )
                rapport.nb_lignes_rejetees = len(
                    set(e.ligne for e in rapport.erreurs)
                )
            else:
                rapport.statut = ImportStatus.ERROR

            # Stocker pour validation ultérieure
            self._active_imports[rapport.import_id] = {
                "df": df,
                "mapping": mapping,
                "unite": rapport.unite_detectee,
                "patient_id": patient_id,
            }

        except Exception as exc:
            rapport.statut = ImportStatus.ERROR
            rapport.avertissements.append(f"Erreur fatale : {str(exc)}")
            logger.error(f"❌ Import échoué : {exc}", exc_info=True)

        rapport.duree_traitement_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"📋 Rapport import {rapport.import_id[:8]} : "
            f"{rapport.nb_lignes_total} lignes, "
            f"{rapport.taux_reussite:.1f}% valides, "
            f"{rapport.duree_traitement_ms}ms"
        )

        return rapport

    def valider_import(
        self,
        import_id: str,
    ) -> tuple[list[dict[str, Any]], RapportImport]:
        """Valide et retourne les données nettoyées d'un import.

        Args:
            import_id: ID de l'import en attente.

        Returns:
            Tuple (liste de mesures nettoyées, rapport mis à jour).

        Raises:
            ValueError: Si l'import n'existe pas.
        """
        if import_id not in self._active_imports:
            raise ValueError(f"Import introuvable : {import_id}")

        data = self._active_imports[import_id]
        df = data["df"]
        mapping = data["mapping"]
        unite = data["unite"]
        patient_id = data["patient_id"]

        rapport = RapportImport(
            import_id=import_id,
            filename=data.get("filename", ""),
            format_fichier=data.get("format", ""),
        )

        # Nettoyer tout le DataFrame
        cleaned_df = self._nettoyer_dataframe(df, mapping, unite)

        # Construire les mesures
        mesures: list[dict[str, Any]] = []
        lignes_rejetees = 0

        inv_mapping = {v: k for k, v in mapping.items()}
        col_date = inv_mapping.get("date", "")
        col_valeur = inv_mapping.get("valeur", "")
        col_moment = inv_mapping.get("moment", "")
        col_note = inv_mapping.get("note", "")
        col_repas = inv_mapping.get("repas", "")
        col_activite = inv_mapping.get("activite", "")
        col_stress = inv_mapping.get("stress", "")
        col_medoc = inv_mapping.get("medicament", "")

        for idx, row in cleaned_df.iterrows():
            # Parser la date
            dt = parser_date(row.get(col_date)) if col_date else None
            if dt is None:
                lignes_rejetees += 1
                continue

            # Parser la valeur
            val_raw = row.get(col_valeur)
            if pd.isna(val_raw):
                lignes_rejetees += 1
                continue

            try:
                val_str = str(val_raw).strip().replace(",", ".")
                val_float = float(val_str)
                val_gl = convertir_valeur(val_float, unite)
            except (ValueError, TypeError):
                lignes_rejetees += 1
                continue

            # Vérifier la plausibilité finale
            if val_gl < SEUILS.VALEUR_MIN_PLAUSIBLE or val_gl > SEUILS.VALEUR_MAX_PLAUSIBLE:
                lignes_rejetees += 1
                continue

            mesure = {
                "patient_id": patient_id,
                "valeur": round(val_gl, 3),
                "valeur_originale": val_float,
                "unite_originale": unite,
                "moment_mesure": normaliser_moment(row.get(col_moment))
                    if col_moment else MomentMesure.AUTRE.value,
                "timestamp": dt,
                "note_patient": str(row.get(col_note, "")).strip()
                    if col_note and pd.notna(row.get(col_note)) else None,
                "repas_associe": str(row.get(col_repas, "")).strip()
                    if col_repas and pd.notna(row.get(col_repas)) else None,
                "activite_physique": str(row.get(col_activite, "")).strip()
                    if col_activite and pd.notna(row.get(col_activite)) else None,
                "stress_level": self._parse_int(row.get(col_stress))
                    if col_stress else None,
                "source": "import_excel" if unite else "import_csv",
            }
            mesures.append(mesure)

        # Trier par timestamp
        mesures.sort(key=lambda m: m["timestamp"])

        rapport.nb_lignes_total = len(df)
        rapport.nb_lignes_valides = len(mesures)
        rapport.nb_lignes_rejetees = lignes_rejetees
        rapport.statut = ImportStatus.VALIDATED

        # Nettoyer la mémoire
        del self._active_imports[import_id]

        logger.info(
            f"✅ Import validé : {len(mesures)} mesures importées "
            f"pour patient {patient_id}"
        )

        return mesures, rapport

    def _nettoyer_dataframe(
        self,
        df: pd.DataFrame,
        mapping: dict[str, str],
        unite: str,
    ) -> pd.DataFrame:
        """Nettoie et normalise un DataFrame.

        Args:
            df: DataFrame brut.
            mapping: Mapping des colonnes.
            unite: Unité des valeurs.

        Returns:
            DataFrame nettoyé.
        """
        result = df.copy()
        inv_mapping = {v: k for k, v in mapping.items()}

        # Supprimer les lignes entièrement vides
        result = result.dropna(how="all")

        # Nettoyer les valeurs glycémiques
        col_valeur = inv_mapping.get("valeur", "")
        if col_valeur and col_valeur in result.columns:
            result[col_valeur] = result[col_valeur].apply(
                lambda x: self._clean_numeric(x)
            )

        # Trier par date si possible
        col_date = inv_mapping.get("date", "")
        if col_date and col_date in result.columns:
            result["_parsed_date"] = result[col_date].apply(parser_date)
            result = result.sort_values("_parsed_date", na_position="last")
            result = result.drop(columns=["_parsed_date"])

        return result.reset_index(drop=True)

    @staticmethod
    def _clean_numeric(value: Any) -> Optional[float]:
        """Nettoie une valeur numérique.

        Args:
            value: Valeur brute.

        Returns:
            Float nettoyé ou None.
        """
        if pd.isna(value):
            return None
        try:
            s = str(value).strip().replace(",", ".").replace(" ", "")
            s = re.sub(r"[^\d.\-]", "", s)
            return float(s) if s else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_int(value: Any) -> Optional[int]:
        """Parse un entier depuis une valeur quelconque.

        Args:
            value: Valeur à parser.

        Returns:
            Entier ou None.
        """
        if pd.isna(value):
            return None
        try:
            return int(float(str(value).strip()))
        except (ValueError, TypeError):
            return None

    # ========================================================================
    # AGRÉGATION TEMPORELLE
    # ========================================================================

    @staticmethod
    def agreger_par_periode(
        mesures: list[dict[str, Any]],
        periode: str = "jour",
    ) -> list[dict[str, Any]]:
        """Agrège les mesures par période temporelle.

        Args:
            mesures: Liste de mesures normalisées.
            periode: Période d'agrégation
                ("heure", "jour", "semaine", "mois").

        Returns:
            Liste de dicts agrégés avec stats par période.
        """
        if not mesures:
            return []

        df = pd.DataFrame(mesures)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        freq_map = {
            "heure": "h",
            "jour": "D",
            "semaine": "W",
            "mois": "ME",
        }
        freq = freq_map.get(periode, "D")

        grouped = df.set_index("timestamp").groupby(pd.Grouper(freq=freq))

        result = []
        for period, group in grouped:
            if group.empty:
                continue

            valeurs = group["valeur"].dropna()
            if valeurs.empty:
                continue

            stats = {
                "periode": period.isoformat(),
                "nb_mesures": len(valeurs),
                "moyenne": round(float(valeurs.mean()), 3),
                "min": round(float(valeurs.min()), 3),
                "max": round(float(valeurs.max()), 3),
                "ecart_type": round(float(valeurs.std()), 3)
                    if len(valeurs) > 1 else 0.0,
                "mediane": round(float(valeurs.median()), 3),
            }

            # TIR (Time In Range)
            total = len(valeurs)
            in_range = len(
                valeurs[
                    (valeurs >= SEUILS.NORMAL_JEUN_MIN)
                    & (valeurs <= SEUILS.NORMAL_POSTPRANDIAL_MAX)
                ]
            )
            below = len(valeurs[valeurs < SEUILS.HYPOGLYCEMIE_MAX])
            above = len(valeurs[valeurs > SEUILS.NORMAL_POSTPRANDIAL_MAX])

            stats["tir_percent"] = round(in_range / total * 100, 1)
            stats["tbr_percent"] = round(below / total * 100, 1)
            stats["tar_percent"] = round(above / total * 100, 1)

            result.append(stats)

        return result

    # ========================================================================
    # STATISTIQUES GLOBALES
    # ========================================================================

    @staticmethod
    def calculer_statistiques(
        mesures: list[dict[str, Any]],
        jours: int = 30,
    ) -> dict[str, Any]:
        """Calcule les statistiques glycémiques complètes.

        Implémente les métriques standards ADA/IDF :
        - Moyenne, écart-type, CV%
        - TIR, TBR, TAR
        - GMI (Glucose Management Indicator)
        - LBGI, HBGI (Low/High Blood Glucose Index)

        Args:
            mesures: Liste de mesures normalisées (en g/L).
            jours: Période d'analyse en jours.

        Returns:
            Dict complet de statistiques.
        """
        if not mesures:
            return {"erreur": "Aucune mesure disponible"}

        df = pd.DataFrame(mesures)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filtrer par période
        cutoff = datetime.now(timezone.utc) - timedelta(days=jours)
        df = df[df["timestamp"] >= cutoff]

        if df.empty:
            return {"erreur": f"Aucune mesure dans les {jours} derniers jours"}

        valeurs = df["valeur"].dropna()
        n = len(valeurs)

        if n == 0:
            return {"erreur": "Aucune valeur valide"}

        # Statistiques de base
        moyenne = float(valeurs.mean())
        ecart_type = float(valeurs.std()) if n > 1 else 0.0
        cv_percent = (ecart_type / moyenne * 100) if moyenne > 0 else 0.0

        # TIR / TBR / TAR
        in_range = len(
            valeurs[
                (valeurs >= SEUILS.NORMAL_JEUN_MIN)
                & (valeurs <= SEUILS.NORMAL_POSTPRANDIAL_MAX)
            ]
        )
        below_range = len(valeurs[valeurs < SEUILS.HYPOGLYCEMIE_MAX])
        above_range = len(valeurs[valeurs > SEUILS.NORMAL_POSTPRANDIAL_MAX])

        tir = in_range / n * 100
        tbr = below_range / n * 100
        tar = above_range / n * 100

        # GMI (Glucose Management Indicator)
        # Formule ADA : GMI (%) = 3.31 + 0.02392 × mean glucose (mg/dL)
        moyenne_mgdl = moyenne * 100  # g/L → mg/dL
        gmi = 3.31 + 0.02392 * moyenne_mgdl

        # LBGI (Low Blood Glucose Index)
        # Formule : LBGI = 10 × mean(f(Xl)) où f(Xl) = 22.77 × (ln(Xl)^1.504 - 5.381)
        # Xl = min(valeur_mgdl, 112.5)
        def _lbgi_transform(v_mgdl: float) -> float:
            xl = min(v_mgdl, 112.5)
            if xl <= 0:
                return 0.0
            ln_val = np.log(xl)
            f_xl = 22.77 * (ln_val ** 1.504 - 5.381)
            return max(f_xl, 0.0)

        lbgi = float(10 * valeurs.apply(lambda v: _lbgi_transform(v * 100)).mean())

        # HBGI (High Blood Glucose Index)
        def _hbgi_transform(v_mgdl: float) -> float:
            xh = max(v_mgdl, 112.5)
            if xh <= 0:
                return 0.0
            ln_val = np.log(xh)
            f_xh = 22.77 * (ln_val ** 1.504 - 5.381)
            return max(f_xh, 0.0)

        hbgi = float(10 * valeurs.apply(lambda v: _hbgi_transform(v * 100)).mean())

        # Distribution par moment de mesure
        distribution_moment = {}
        if "moment_mesure" in df.columns:
            for moment, group in df.groupby("moment_mesure"):
                vals = group["valeur"].dropna()
                if not vals.empty:
                    distribution_moment[moment] = {
                        "nb": len(vals),
                        "moyenne": round(float(vals.mean()), 3),
                        "min": round(float(vals.min()), 3),
                        "max": round(float(vals.max()), 3),
                    }

        # Score de risque global (0-100)
        risque_score = 0.0
        if tbr > 4:
            risque_score += min((tbr - 4) * 10, 30)
        if tar > 25:
            risque_score += min((tar - 25) * 1.5, 30)
        if cv_percent > 36:
            risque_score += min((cv_percent - 36) * 1, 20)
        if lbgi > 2.5:
            risque_score += min(lbgi * 4, 20)
        risque_score = min(round(risque_score, 1), 100.0)

        return {
            "periode_jours": jours,
            "nb_mesures": n,
            "moyenne_gl": round(moyenne, 3),
            "moyenne_mgdl": round(moyenne_mgdl, 1),
            "ecart_type": round(ecart_type, 3),
            "cv_percent": round(cv_percent, 1),
            "min": round(float(valeurs.min()), 3),
            "max": round(float(valeurs.max()), 3),
            "mediane": round(float(valeurs.median()), 3),
            "tir_percent": round(tir, 1),
            "tbr_percent": round(tbr, 1),
            "tar_percent": round(tar, 1),
            "gmi_percent": round(gmi, 1),
            "lbgi": round(lbgi, 2),
            "hbgi": round(hbgi, 2),
            "risque_score": risque_score,
            "distribution_moment": distribution_moment,
            "cibles_ada": {
                "tir_cible": "≥ 70%",
                "tbr_cible": "< 4%",
                "tar_cible": "< 25%",
                "cv_cible": "< 36%",
                "gmi_cible": "< 7%",
            },
            "evaluation": {
                "tir_ok": tir >= SEUILS.TIR_CIBLE_MIN_PERCENT,
                "tbr_ok": tbr <= SEUILS.TBR_LIMITE_PERCENT,
                "tar_ok": tar <= SEUILS.TAR_LIMITE_PERCENT,
                "cv_ok": cv_percent < 36,
            },
        }

    # ========================================================================
    # EXPORT
    # ========================================================================

    @staticmethod
    def exporter_csv(
        mesures: list[dict[str, Any]],
        patient_id: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Exporte les mesures en fichier CSV.

        Args:
            mesures: Liste de mesures.
            patient_id: UUID du patient.
            filename: Nom de fichier (auto-généré si None).

        Returns:
            Chemin du fichier exporté.
        """
        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"glycemie_{patient_id[:8]}_{ts}.csv"

        filepath = EXPORTS_DIR / filename
        df = pd.DataFrame(mesures)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        df.to_csv(str(filepath), index=False, encoding="utf-8-sig")
        logger.info(f"📤 Export CSV : {filepath} ({len(df)} lignes)")
        return filepath

    @staticmethod
    def exporter_json(
        mesures: list[dict[str, Any]],
        patient_id: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Exporte les mesures en fichier JSON.

        Args:
            mesures: Liste de mesures.
            patient_id: UUID du patient.
            filename: Nom de fichier.

        Returns:
            Chemin du fichier exporté.
        """
        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"glycemie_{patient_id[:8]}_{ts}.json"

        filepath = EXPORTS_DIR / filename

        # Sérialiser les datetimes
        serializable = []
        for m in mesures:
            entry = dict(m)
            if "timestamp" in entry and isinstance(entry["timestamp"], datetime):
                entry["timestamp"] = entry["timestamp"].isoformat()
            serializable.append(entry)

        filepath.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"📤 Export JSON : {filepath} ({len(serializable)} mesures)")
        return filepath

    # ========================================================================
    # GÉNÉRATION DE DONNÉES SYNTHÉTIQUES (pour tests)
    # ========================================================================

    @staticmethod
    def generer_donnees_synthetiques(
        patient_id: str,
        jours: int = 30,
        mesures_par_jour: int = 4,
        type_diabete: str = "type_2",
    ) -> list[dict[str, Any]]:
        """Génère des données glycémiques synthétiques réalistes.

        Utile pour les tests et les démonstrations.

        Args:
            patient_id: UUID du patient.
            jours: Nombre de jours à générer.
            mesures_par_jour: Mesures par jour.
            type_diabete: Type de diabète (affecte la variabilité).

        Returns:
            Liste de mesures synthétiques.
        """
        np.random.seed(42)
        mesures: list[dict[str, Any]] = []

        # Paramètres selon le type de diabète
        if type_diabete == "type_1":
            base_mean = 1.30
            base_std = 0.45
        else:
            base_mean = 1.20
            base_std = 0.30

        moments = [
            MomentMesure.A_JEUN,
            MomentMesure.POSTPRANDIAL_2H,
            MomentMesure.AVANT_COUCHER,
            MomentMesure.AVANT_REPAS,
        ]

        now = datetime.now(timezone.utc)

        for day_offset in range(jours):
            date = now - timedelta(days=jours - day_offset)

            # Effet de l'aube (glycémie plus haute le matin)
            dawn_effect = np.random.normal(0.10, 0.05)

            for m_idx in range(mesures_par_jour):
                moment = moments[m_idx % len(moments)]
                hour = {
                    MomentMesure.A_JEUN: 6,
                    MomentMesure.AVANT_REPAS: 12,
                    MomentMesure.POSTPRANDIAL_2H: 14,
                    MomentMesure.AVANT_COUCHER: 22,
                }.get(moment, 12)

                # Variation selon le moment
                moment_offset = {
                    MomentMesure.A_JEUN: dawn_effect,
                    MomentMesure.POSTPRANDIAL_2H: np.random.normal(0.25, 0.10),
                    MomentMesure.AVANT_COUCHER: np.random.normal(0.05, 0.08),
                    MomentMesure.AVANT_REPAS: np.random.normal(0.0, 0.08),
                }.get(moment, 0.0)

                valeur = np.random.normal(
                    base_mean + moment_offset, base_std,
                )
                valeur = max(0.40, min(4.00, valeur))

                timestamp = date.replace(
                    hour=hour,
                    minute=np.random.randint(0, 60),
                    second=0,
                    microsecond=0,
                )

                mesures.append({
                    "patient_id": patient_id,
                    "valeur": round(float(valeur), 3),
                    "valeur_originale": round(float(valeur), 3),
                    "unite_originale": UniteGlycemie.G_PAR_L.value,
                    "moment_mesure": moment.value,
                    "timestamp": timestamp,
                    "note_patient": None,
                    "repas_associe": "fufu + pondu"
                        if moment == MomentMesure.POSTPRANDIAL_2H else None,
                    "activite_physique": "marche 30min"
                        if np.random.random() < 0.3 else None,
                    "stress_level": np.random.randint(1, 8)
                        if np.random.random() < 0.2 else None,
                    "source": "synthetique",
                })

        mesures.sort(key=lambda m: m["timestamp"])
        logger.info(
            f"🧪 {len(mesures)} mesures synthétiques générées "
            f"({jours}j, {type_diabete})"
        )
        return mesures


# ============================================================================
# INSTANCE GLOBALE
# ============================================================================

data_processor = DataProcessor()


__all__ = [
    "DataProcessor", "DataValidator", "RapportImport", "ImportStatus",
    "ErreurValidation", "ErreurType",
    "convertir_valeur", "parser_date", "normaliser_moment",
    "lire_fichier_excel", "lire_fichier_csv",
    "data_processor",
]