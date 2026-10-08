"""
================================================================================
FICHIER : visualizations.py
RESPONSABILITÉ : Génération de graphiques et visualisations glycémiques
================================================================================
Ce module génère TOUTES les visualisations pour DiaBot-RDC :
- Courbe glycémique journalière (type AGP - Ambulatory Glucose Profile)
- Graphiques de tendance (7/14/30/90 jours)
- Heatmap jour × heure des glycémies
- Distribution des glycémies (histogramme)
- TIR pie chart (Time In Range)
- Comparaison période vs période
- Corrélation repas/glycémie
- Courbe de prédiction avec intervalle de confiance
- Mini-sparklines pour résumé rapide
- Tableau de bord synthétique combiné

Caractéristiques :
- Zones colorées (hypo=rouge, normal=vert, hyper=orange)
- Annotations intelligentes (événements notables)
- Palette daltonisme-safe (colorblind-friendly)
- Export PNG/SVG haute résolution
- Optimisé mobile (tailles adaptatives)

Auteur : Équipe DiaBot-RDC
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Backend non-interactif (serveur)

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from config import SEUILS, EXPORTS_DIR

logger = logging.getLogger(__name__)


# ============================================================================
# PALETTE DE COULEURS (Daltonisme-safe)
# ============================================================================

# Palette Okabe-Ito modifiée (accessible aux daltoniens)
class Couleurs:
    """Palette de couleurs accessible (daltonisme-safe)."""

    # Zones glycémiques
    HYPO_CRITIQUE = "#D55E00"     # Orange-rouge (vermillon)
    HYPO_MODEREE = "#E69F00"      # Orange
    NORMAL = "#009E73"            # Vert-bleu (teal)
    HYPER_MODEREE = "#F0E442"     # Jaune
    HYPER_SEVERE = "#CC79A7"      # Rose
    URGENCE = "#D55E00"           # Vermillon

    # Courbes
    COURBE_PRINCIPALE = "#0072B2"  # Bleu
    COURBE_PREDICTION = "#56B4E9"  # Bleu ciel
    INTERVALLE_CONFIANCE = "#56B4E9"
    MOYENNE = "#000000"           # Noir
    MEDIANE = "#E69F00"           # Orange

    # UI
    FOND = "#FFFFFF"
    GRILLE = "#E0E0E0"
    TEXTE = "#333333"
    TEXTE_SECONDAIRE = "#666666"
    BORDURE = "#CCCCCC"

    # TIR Pie
    TIR_IN_RANGE = "#009E73"
    TIR_BELOW = "#D55E00"
    TIR_ABOVE = "#E69F00"
    TIR_VERY_HIGH = "#CC79A7"

    # Heatmap
    HEATMAP_COLD = "#0072B2"
    HEATMAP_WARM = "#D55E00"

    @classmethod
    def zones_bg(cls) -> list[tuple[float, float, str, float]]:
        """Retourne les zones de fond colorées pour les graphiques glycémiques.

        Returns:
            Liste de tuples (y_min, y_max, couleur, alpha).
        """
        return [
            (0.0, SEUILS.URGENCE_HYPO, cls.HYPO_CRITIQUE, 0.15),
            (SEUILS.URGENCE_HYPO, SEUILS.HYPOGLYCEMIE_MAX, cls.HYPO_MODEREE, 0.10),
            (SEUILS.HYPOGLYCEMIE_MAX, SEUILS.NORMAL_POSTPRANDIAL_MAX, cls.NORMAL, 0.08),
            (SEUILS.NORMAL_POSTPRANDIAL_MAX, SEUILS.HYPERGLYCEMIE_MODEREE_MAX, cls.HYPER_MODEREE, 0.08),
            (SEUILS.HYPERGLYCEMIE_MODEREE_MAX, SEUILS.HYPERGLYCEMIE_SEVERE_MAX, cls.HYPER_SEVERE, 0.10),
            (SEUILS.HYPERGLYCEMIE_SEVERE_MAX, 5.0, cls.URGENCE, 0.15),
        ]


# ============================================================================
# CONFIGURATION MATPLOTLIB GLOBALE
# ============================================================================

def _setup_style() -> None:
    """Configure le style matplotlib global."""
    plt.rcParams.update({
        "figure.facecolor": Couleurs.FOND,
        "axes.facecolor": Couleurs.FOND,
        "axes.edgecolor": Couleurs.BORDURE,
        "axes.labelcolor": Couleurs.TEXTE,
        "text.color": Couleurs.TEXTE,
        "xtick.color": Couleurs.TEXTE_SECONDAIRE,
        "ytick.color": Couleurs.TEXTE_SECONDAIRE,
        "grid.color": Couleurs.GRILLE,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


_setup_style()


# ============================================================================
# UTILITAIRES
# ============================================================================

def _fig_to_bytes(fig: Figure, fmt: str = "png") -> bytes:
    """Convertit une figure matplotlib en bytes.

    Args:
        fig: Figure matplotlib.
        fmt: Format de sortie ("png" ou "svg").

    Returns:
        Bytes de l'image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    data = buf.getvalue()
    plt.close(fig)
    return data


def _add_zones_glycemiques(ax: plt.Axes, y_min: float = 0.0, y_max: float = 4.0) -> None:
    """Ajoute les zones colorées de fond sur un axe.

    Args:
        ax: Axe matplotlib.
        y_min: Valeur Y minimale.
        y_max: Valeur Y maximale.
    """
    for zone_ymin, zone_ymax, couleur, alpha in Couleurs.zones_bg():
        ymin = max(zone_ymin, y_min)
        ymax = min(zone_ymax, y_max)
        if ymin < ymax:
            ax.axhspan(ymin, ymax, color=couleur, alpha=alpha, zorder=0)

    # Lignes de seuil
    seuils_importants = [
        (SEUILS.URGENCE_HYPO, "Hypo sévère", Couleurs.HYPO_CRITIQUE),
        (SEUILS.HYPOGLYCEMIE_MAX, "Hypo", Couleurs.HYPO_MODEREE),
        (SEUILS.NORMAL_JEUN_MIN, "Normal min", Couleurs.NORMAL),
        (SEUILS.NORMAL_POSTPRANDIAL_MAX, "Postprandial max", Couleurs.NORMAL),
        (SEUILS.HYPERGLYCEMIE_MODEREE_MAX, "Hyper modérée", Couleurs.HYPER_SEVERE),
        (SEUILS.URGENCE_HYPER, "Urgence", Couleurs.URGENCE),
    ]
    for val, label, couleur in seuils_importants:
        if y_min <= val <= y_max:
            ax.axhline(
                y=val, color=couleur, linestyle="--",
                linewidth=0.8, alpha=0.6, zorder=1,
            )


def _prepare_dataframe(mesures: list[dict[str, Any]]) -> pd.DataFrame:
    """Prépare un DataFrame à partir des mesures.

    Args:
        mesures: Liste de dicts de mesures.

    Returns:
        DataFrame trié par timestamp.
    """
    if not mesures:
        return pd.DataFrame()

    df = pd.DataFrame(mesures)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _get_figure(
    width: float = 8.0,
    height: float = 4.5,
    mobile: bool = False,
) -> tuple[Figure, plt.Axes]:
    """Crée une figure avec les bonnes dimensions.

    Args:
        width: Largeur en pouces.
        height: Hauteur en pouces.
        mobile: Si True, dimensions adaptées mobile.

    Returns:
        Tuple (Figure, Axes).
    """
    if mobile:
        width, height = 5.0, 3.5
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(Couleurs.FOND)
    return fig, ax


# ============================================================================
# 1. COURBE GLYCÉMIQUE JOURNALIÈRE (AGP simplifié)
# ============================================================================

def courbe_journaliere(
    mesures: list[dict[str, Any]],
    date_cible: Optional[datetime] = None,
    mobile: bool = False,
    fmt: str = "png",
) -> bytes:
    """Génère la courbe glycémique d'une journée (type AGP).

    Affiche les mesures de la journée avec les zones de couleur,
    les seuils et les annotations d'événements.

    Args:
        mesures: Liste de mesures.
        date_cible: Date de la journée (défaut : aujourd'hui).
        mobile: Mode mobile.
        fmt: Format de sortie.

    Returns:
        Bytes de l'image.
    """
    df = _prepare_dataframe(mesures)
    if df.empty:
        return _empty_chart("Aucune donnée pour cette journée", mobile, fmt)

    if date_cible is None:
        date_cible = datetime.now(timezone.utc)

    # Filtrer la journée
    jour_debut = date_cible.replace(hour=0, minute=0, second=0, microsecond=0)
    jour_fin = jour_debut + timedelta(days=1)
    df_jour = df[(df["timestamp"] >= jour_debut) & (df["timestamp"] < jour_fin)]

    if df_jour.empty:
        return _empty_chart(
            f"Aucune mesure le {jour_debut.strftime('%d/%m/%Y')}",
            mobile, fmt,
        )

    fig, ax = _get_figure(mobile=mobile)

    # Zones de fond
    _add_zones_glycemiques(ax, y_min=0.0, y_max=max(3.5, df_jour["valeur"].max() * 1.1))

    # Courbe principale
    ax.plot(
        df_jour["timestamp"], df_jour["valeur"],
        color=Couleurs.COURBE_PRINCIPALE, linewidth=2.5,
        marker="o", markersize=8, markerfacecolor="white",
        markeredgecolor=Couleurs.COURBE_PRINCIPALE, markeredgewidth=2,
        zorder=5, label="Glycémie",
    )

    # Colorer les points selon la zone
    for _, row in df_jour.iterrows():
        val = row["valeur"]
        if val < SEUILS.HYPOGLYCEMIE_MAX:
            color = Couleurs.HYPO_CRITIQUE
        elif val <= SEUILS.NORMAL_POSTPRANDIAL_MAX:
            color = Couleurs.NORMAL
        elif val <= SEUILS.HYPERGLYCEMIE_MODEREE_MAX:
            color = Couleurs.HYPER_MODEREE
        else:
            color = Couleurs.HYPER_SEVERE
        ax.plot(
            row["timestamp"], val,
            "o", color=color, markersize=10, zorder=6,
        )

    # Annotations des repas
    if "repas_associe" in df_jour.columns:
        for _, row in df_jour.iterrows():
            if pd.notna(row.get("repas_associe")) and row["repas_associe"]:
                ax.annotate(
                    "🍽️",
                    xy=(row["timestamp"], row["valeur"]),
                    xytext=(0, 15), textcoords="offset points",
                    fontsize=12, ha="center", zorder=7,
                )

    # Formatage
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(jour_debut, jour_fin)
    ax.set_ylim(0, max(3.5, df_jour["valeur"].max() * 1.15))
    ax.set_xlabel("Heure")
    ax.set_ylabel("Glycémie (g/L)")
    ax.set_title(
        f"Glycémie du {jour_debut.strftime('%d/%m/%Y')}",
        fontweight="bold", pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Statistiques du jour en encart
    stats_text = (
        f"Moy: {df_jour['valeur'].mean():.2f} | "
        f"Min: {df_jour['valeur'].min():.2f} | "
        f"Max: {df_jour['valeur'].max():.2f} | "
        f"n={len(df_jour)}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=Couleurs.BORDURE),
    )

    fig.autofmt_xdate(rotation=0)
    return _fig_to_bytes(fig, fmt)


# ============================================================================
# 2. GRAPHIQUE DE TENDANCE (7/14/30/90 jours)
# ============================================================================

def courbe_tendance(
    mesures: list[dict[str, Any]],
    jours: int = 30,
    mobile: bool = False,
    fmt: str = "png",
) -> bytes:
    """Génère le graphique de tendance sur N jours.

    Affiche la courbe lissée avec la moyenne mobile et les zones.

    Args:
        mesures: Liste de mesures.
        jours: Nombre de jours (7, 14, 30, 90).
        mobile: Mode mobile.
        fmt: Format de sortie.

    Returns:
        Bytes de l'image.
    """
    df = _prepare_dataframe(mesures)
    if df.empty:
        return _empty_chart("Aucune donnée disponible", mobile, fmt)

    cutoff = datetime.now(timezone.utc) - timedelta(days=jours)
    df = df[df["timestamp"] >= cutoff]

    if df.empty:
        return _empty_chart(f"Aucune donnée sur les {jours} derniers jours", mobile, fmt)

    fig, ax = _get_figure(width=9.0, height=4.5, mobile=mobile)

    y_max = max(3.5, df["valeur"].quantile(0.98) * 1.1)
    _add_zones_glycemiques(ax, y_min=0.0, y_max=y_max)

    # Points individuels (semi-transparents)
    ax.scatter(
        df["timestamp"], df["valeur"],
        color=Couleurs.COURBE_PRINCIPALE, alpha=0.3,
        s=15, zorder=3, label="Mesures",
    )

    # Moyenne mobile (lissage)
    df_daily = df.set_index("timestamp").resample("D")["valeur"].mean().dropna()
    if len(df_daily) >= 2:
        window = max(1, min(3, len(df_daily) // 3))
        df_smooth = df_daily.rolling(window=window, center=True).mean()
        ax.plot(
            df_smooth.index, df_smooth.values,
            color=Couleurs.COURBE_PRINCIPALE, linewidth=2.5,
            zorder=5, label=f"Moyenne mobile ({window}j)",
        )

    # Moyenne globale
    moyenne = df["valeur"].mean()
    ax.axhline(
        y=moyenne, color=Couleurs.MOYENNE,
        linestyle="-.", linewidth=1.2, alpha=0.7, zorder=4,
        label=f"Moyenne ({moyenne:.2f} g/L)",
    )

    # Formatage
    if jours <= 14:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    elif jours <= 30:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    ax.set_ylim(0, y_max)
    ax.set_xlabel("Date")
    ax.set_ylabel("Glycémie (g/L)")
    ax.set_title(
        f"Tendance glycémique — {jours} derniers jours",
        fontweight="bold", pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.autofmt_xdate(rotation=30)
    return _fig_to_bytes(fig, fmt)


# ============================================================================
# 3. HEATMAP JOUR × HEURE
# ============================================================================

def heatmap_glycemie(
    mesures: list[dict[str, Any]],
    jours: int = 14,
    mobile: bool = False,
    fmt: str = "png",
) -> bytes:
    """Génère une heatmap jour × heure des glycémies.

    Permet de repérer visuellement les patterns récurrents
    (hypoglycémies nocturnes, hyperglycémies postprandiales, etc.)

    Args:
        mesures: Liste de mesures.
        jours: Nombre de jours à afficher.
        mobile: Mode mobile.
        fmt: Format de sortie.

    Returns:
        Bytes de l'image.
    """
    df = _prepare_dataframe(mesures)
    if df.empty:
        return _empty_chart("Aucune donnée", mobile, fmt)

    cutoff = datetime.now(timezone.utc) - timedelta(days=jours)
    df = df[df["timestamp"] >= cutoff].copy()

    if df.empty:
        return _empty_chart(f"Aucune donnée sur {jours}j", mobile, fmt)

    # Extraire jour et heure
    df["date"] = df["timestamp"].dt.date
    df["heure"] = df["timestamp"].dt.hour

    # Créer la matrice
    pivot = df.pivot_table(
        values="valeur", index="date", columns="heure",
        aggfunc="mean", fill_value=np.nan,
    )

    if pivot.empty:
        return _empty_chart("Données insuffisantes pour la heatmap", mobile, fmt)

    # Remplir les heures manquantes
    heures_completes = list(range(0, 24, 2))  # Toutes les 2h
    for h in heures_completes:
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[sorted(pivot.columns)]

    fig_width = 9.0 if not mobile else 6.0
    fig_height = max(3.0, len(pivot) * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Colormap personnalisée (vert = normal, rouge = danger)
    from matplotlib.colors import LinearSegmentedColormap
    colors_heatmap = [
        Couleurs.HYPO_CRITIQUE,   # Très bas
        Couleurs.HYPO_MODEREE,    # Bas
        Couleurs.NORMAL,          # Normal
        Couleurs.HYPER_MODEREE,   # Haut
        Couleurs.HYPER_SEVERE,    # Très haut
    ]
    cmap = LinearSegmentedColormap.from_list(
        "glycemie", colors_heatmap, N=256,
    )

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=cmap,
        vmin=0.4,
        vmax=3.0,
        interpolation="nearest",
    )

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{h:02d}h" for h in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    date_labels = [
        d.strftime("%d/%m") if isinstance(d, datetime) else str(d)[-5:]
        for d in pivot.index
    ]
    ax.set_yticklabels(date_labels, fontsize=8)
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Date")
    ax.set_title(
        f"Heatmap glycémique — {jours} derniers jours",
        fontweight="bold", pad=10,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Glycémie (g/L)", fontsize=9)
    cbar.set_ticks([0.5, 0.7, 1.1, 1.4, 2.5, 3.0])
    cbar.set_ticklabels(["0.5", "0.7", "1.1", "1.4", "2.5", "3.0"])

    return _fig_to_bytes(fig, fmt)


# ============================================================================
# 4. DISTRIBUTION DES GLYCÉMIES (Histogramme)
# ============================================================================

def histogramme_distribution(
    mesures: list[dict[str, Any]],
    jours: int = 30,
    mobile: bool = False,
    fmt: str = "png",
) -> bytes:
    """Génère l'histogramme de distribution des glycémies.

    Args:
        mesures: Liste de mesures.
        jours: Période d'analyse.
        mobile: Mode mobile.
        fmt: Format de sortie.

    Returns:
        Bytes de l'image.
    """
    df = _prepare_dataframe(mesures)
    if df.empty:
        return _empty_chart("Aucune donnée", mobile, fmt)

    cutoff = datetime.now(timezone.utc) - timedelta(days=jours)
    df = df[df["timestamp"] >= cutoff]

    if df.empty:
        return _empty_chart("Aucune donnée récente", mobile, fmt)

    fig, ax = _get_figure(mobile=mobile)

    valeurs = df["valeur"].values
    bins = np.arange(0.2, 4.2, 0.1)

    # Histogramme coloré par zone
    counts, bin_edges, patches = ax.hist(
        valeurs, bins=bins, edgecolor="white", linewidth=0.5, zorder=3,
    )

    for patch, left_edge in zip(patches, bin_edges[:-1]):
        mid = left_edge + 0.05
        if mid < SEUILS.HYPOGLYCEMIE_MAX:
            patch.set_facecolor(Couleurs.HYPO_CRITIQUE)
        elif mid <= SEUILS.NORMAL_POSTPRANDIAL_MAX:
            patch.set_facecolor(Couleurs.NORMAL)
        elif mid <= SEUILS.HYPERGLYCEMIE_MODEREE_MAX:
            patch.set_facecolor(Couleurs.HYPER_MODEREE)
        else:
            patch.set_facecolor(Couleurs.HYPER_SEVERE)
        patch.set_alpha(0.75)

    # Lignes de seuil
    for val, label, couleur in [
        (SEUILS.HYPOGLYCEMIE_MAX, "Hypo", Couleurs.HYPO_CRITIQUE),
        (SEUILS.NORMAL_JEUN_MIN, "Normal min", Couleurs.NORMAL),
        (SEUILS.NORMAL_POSTPRANDIAL_MAX, "Normal max", Couleurs.NORMAL),
        (SEUILS.HYPERGLYCEMIE_MODEREE_MAX, "Hyper", Couleurs.HYPER_SEVERE),
    ]:
        ax.axvline(x=val, color=couleur, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            val, ax.get_ylim()[1] * 0.95, label,
            rotation=90, fontsize=7, ha="right", va="top",
            color=couleur, fontweight="bold",
        )

    # Moyenne et médiane
    ax.axvline(
        x=valeurs.mean(), color=Couleurs.MOYENNE,
        linestyle="-", linewidth=2, label=f"Moyenne ({valeurs.mean():.2f})",
    )
    ax.axvline(
        x=np.median(valeurs), color=Couleurs.MEDIANE,
        linestyle=":", linewidth=2, label=f"Médiane ({np.median(valeurs):.2f})",
    )

    ax.set_xlim(0.2, min(4.0, max(valeurs) * 1.1))
    ax.set_xlabel("Glycémie (g/L)")
    ax.set_ylabel("Nombre de mesures")
    ax.set_title(
        f"Distribution glycémique — {jours}j (n={len(valeurs)})",
        fontweight="bold", pad=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    return _fig_to_bytes(fig, fmt)


# ============================================================================
# 5. TIR PIE CHART (Time In Range)
# ============================================================================

def tir_pie_chart(
    mesures: list[dict[str, Any]],
    jours: int = 30,
    mobile: bool = False,
    fmt: str = "png",
) -> bytes:
    """Génère le graphique circulaire du Time In Range.

    Segments :
    - Very Low (< 0.54) : rouge
    - Low (0.54 - 0.70) : orange
    - In Range (0.70 - 1.40) : vert
    - High (1.40 - 2.50) : jaune
    - Very High (> 2.50) : rose

    Args:
        mesures: Liste de mesures.
        jours: Période d'analyse.
        mobile: Mode mobile.
        fmt: Format de sortie.

    Returns:
        Bytes de l'image.
    """
    df = _prepare_dataframe(mesures)
    if df.empty:
        return _empty_chart("Aucune donnée", mobile, fmt)

    cutoff = datetime.now(timezone.utc) - timedelta(days=jours)
    df = df[df["timestamp"] >= cutoff]

    if df.empty:
        return _empty_chart("Aucune donnée récente", mobile, fmt)

    valeurs = df["valeur"]
    total = len(valeurs)

    # Calculer les segments
    very_low = len(valeurs[valeurs < SEUILS.URGENCE_HYPO])
    low = len(valeurs[(valeurs >= SEUILS.URGENCE_HYPO) & (valeurs < SEUILS.HYPOGLYCEMIE_MAX)])
    in_range = len(valeurs[(valeurs >= SEUILS.HYPOGLYCEMIE_MAX) & (valeurs <= SEUILS.NORMAL_POSTPRANDIAL_MAX)])
    high = len(valeurs[(valeurs > SEUILS.NORMAL_POSTPRANDIAL_MAX) & (valeurs <= SEUILS.HYPERGLYCEMIE_MODEREE_MAX)])
    very_high = len(valeurs[valeurs > SEUILS.HYPERGLYCEMIE_MODEREE_MAX])

    labels = [
        f"Très bas\n< {SEUILS.URGENCE_HYPO}\n({very_low/total*100:.1f}%)",
        f"Bas\n{SEUILS.URGENCE_HYPO}-{SEUILS.HYPOGLYCEMIE_MAX}\n({low/total*100:.1f}%)",
        f"Dans la cible\n{SEUILS.HYPOGLYCEMIE_MAX}-{SEUILS.NORMAL_POSTPRANDIAL_MAX}\n({in_range/total*100:.1f}%)",
        f"Haut\n{SEUILS.NORMAL_POSTPRANDIAL_MAX}-{SEUILS.HYPERGLYCEMIE_MODEREE_MAX}\n({high/total*100:.1f}%)",
        f"Très haut\n> {SEUILS.HYPERGLYCEMIE_MODEREE_MAX}\n({very_high/total*100:.1f}%)",
    ]
    sizes = [very_low, low, in_range, high, very_high]
    colors = [
        Couleurs.HYPO_CRITIQUE,
        Couleurs.HYPO_MODEREE,
        Couleurs.TIR_IN_RANGE,
        Couleurs.TIR_ABOVE,
        Couleurs.TIR_VERY_HIGH,
    ]

    # Filtrer les segments vides
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if not non_zero:
        return _empty_chart("Aucune donnée", mobile, fmt)

    labels_f, sizes_f, colors_f = zip(*non_zero)

    fig, ax = _get_figure(width=7.0, height=5.5, mobile=mobile)

    wedges, texts, autotexts = ax.pie(
        sizes_f,
        labels=None,
        autopct="%1.1f%%",
        colors=colors_f,
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
    )

    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
        t.set_color("white")

    # Légende
    ax.legend(
        wedges, labels_f,
        title="Zones glycémiques",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title_fontsize=9,
    )

    # Texte central
    tir_pct = in_range / total * 100
    ax.text(
        0, 0, f"TIR\n{tir_pct:.0f}%",
        ha="center", va="center",
        fontsize=16, fontweight="bold",
        color=Couleurs.NORMAL if tir_pct >= 70 else Couleurs.HYPO_CRITIQUE,
    )

    ax.set_title(
        f"Time In Range — {jours} derniers jours",
        fontweight="bold", pad=15,
    )

    return _fig_to_bytes(fig, fmt)


# ============================================================================
# 6. COMPARAISON PÉRIODE VS PÉRIODE
# ============================================================================

def comparaison_periodes(
    mesures: list[dict[str, Any]],
    jours_recent: int = 14,
    jours_precedent: int = 14,
    mobile: bool = False,
    fmt: str = "png",
) -> bytes:
    """Compare deux périodes consécutives.

    Args:
        mesures: Liste de mesures.
        jours_recent: Jours de la période récente.
        jours_precedent: Jours de la période précédente.
        mobile: Mode mobile.
        fmt: Format de sortie.

    Returns:
        Bytes de l'image.
    """
    df = _prepare_dataframe(mesures)
    if df.empty:
        return _empty_chart("Aucune donnée", mobile, fmt)

    now = datetime.now(timezone.utc)
    debut_recent = now - timedelta(days=jours_recent)
    debut_precedent = debut_recent - timedelta(days=jours_precedent)

    df_recent = df[df["timestamp"] >= debut_recent]
    df_precedent = df[
        (df["timestamp"] >= debut_precedent) & (df["timestamp"] < debut_recent)
    ]

    if df_recent.empty and df_precedent.empty:
        return _empty_chart("Données insuffisantes pour la comparaison", mobile, fmt)

    fig, ax = _get_figure(mobile=mobile)

    y_max = 3.5
    if not df_recent.empty:
        y_max = max(y_max, df_recent["valeur"].quantile(0.98) * 1.1)
    if not df_precedent.empty:
        y_max = max(y_max, df_precedent["valeur"].quantile(0.98) * 1.1)

    _add_zones_glycemiques(ax, y_max=y_max)

    # Période précédente (fond, plus transparent)
    if not df_precedent.empty:
        ax.scatter(
            df_precedent["timestamp"], df_precedent["valeur"],
            color=Couleurs.HYPER_MODEREE, alpha=0.3, s=20,
            label=f"Période précédente ({jours_precedent}j)",
            zorder=3,
        )

    # Période récente (premier plan)
    if not df_recent.empty:
        ax.scatter(
            df_recent["timestamp"], df_recent["valeur"],
            color=Couleurs.COURBE_PRINCIPALE, alpha=0.6, s=25,
            label=f"Période récente ({jours_recent}j)",
            zorder=4,
        )

    # Moyennes
    if not df_precedent.empty:
        ax.axhline(
            y=df_precedent["valeur"].mean(),
            color=Couleurs.HYPER_MODEREE, linestyle="--", linewidth=1.5,
        )
    if not df_recent.empty:
        ax.axhline(
            y=df_recent["valeur"].mean(),
            color=Couleurs.COURBE_PRINCIPALE, linestyle="-", linewidth=1.5,
        )

    # Séparateur vertical
    ax.axvline(
        x=debut_recent, color=Couleurs.TEXTE,
        linestyle=":", linewidth=1, alpha=0.5,
    )

    ax.set_ylim(0, y_max)
    ax.set_xlabel("Date")
    ax.set_ylabel("Glycémie (g/L)")
    ax.set_title(
        f"Comparaison : {jours_precedent}j précédents vs {jours_recent}j récents",
        fontweight="bold", pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.autofmt_xdate(rotation=30)
    return _fig_to_bytes(fig, fmt)


# ============================================================================
# 7. CORRÉLATION REPAS / GLYCÉMIE
# ============================================================================

def correlation_repas(
    mesures: list[dict[str, Any]],
    mobile: bool = False,
    fmt: str = "png",
) -> bytes:
    """Génère un boxplot de la glycémie par type de repas.

    Args:
        mesures: Liste de mesures avec champ "repas_associe".
        mobile: Mode mobile.
        fmt: Format de sortie.

    Returns:
        Bytes de l'image.
    """
    df = _prepare_dataframe(mesures)
    if df.empty or "repas_associe" not in df.columns:
        return _empty_chart("Données repas indisponibles", mobile, fmt)

    df_meals = df[df["repas_associe"].notna() & (df["repas_associe"] != "")].copy()
    if df_meals.empty:
        return _empty_chart("Aucune mesure associée à un repas", mobile, fmt)

    # Grouper par repas
    grouped = df_meals.groupby("repas_associe")["valeur"]
    repas_labels = []
    repas_data = []
    for name, group in grouped:
        if len(group) >= 2:  # Au moins 2 mesures pour être significatif
            repas_labels.append(str(name)[:20])
            repas_data.append(group.values)

    if not repas_data:
        return _empty_chart("Pas assez de données par repas", mobile, fmt)

    fig, ax = _get_figure(width=8.0, height=5.0, mobile=mobile)

    bp = ax.boxplot(
        repas_data,
        labels=repas_labels,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color=Couleurs.MOYENNE, linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # Colorer les boîtes
    box_colors = [Couleurs.COURBE_PRINCIPALE, Couleurs.NORMAL,
                  Couleurs.HYPER_MODEREE, Couleurs.HYPER_SEVERE,
                  Couleurs.MEDIANE]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Zones de fond
    _add_zones_glycemiques(ax, y_max=max(3.5, df_meals["valeur"].max() * 1.1))

    ax.set_ylim(0, max(3.5, df_meals["valeur"].max() * 1.1))
    ax.set_xlabel("Repas")
    ax.set_ylabel("Glycémie (g/L)")
    ax.set_title(
        "Glycémie par type de repas",
        fontweight="bold", pad=10,
    )
    ax.grid(True, axis="y", alpha=0.3)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    return _fig_to_bytes(fig, fmt)


# ============================================================================
# 8. COURBE DE PRÉDICTION AVEC INTERVALLE DE CONFIANCE
# ============================================================================

def courbe_prediction(
    mesures: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    mobile: bool = False,
    fmt: str = "png",
) -> bytes:
    """Génère la courbe de prédiction glycémique.

    Affiche l'historique récent + la prédiction future
    avec l'intervalle de confiance à 95%.

    Args:
        mesures: Mesures historiques.
        predictions: Prédictions futures avec "timestamp", "valeur",
                     "borne_inf", "borne_sup".
        mobile: Mode mobile.
        fmt: Format de sortie.

    Returns:
        Bytes de l'image.
    """
    df_hist = _prepare_dataframe(mesures)
    df_pred = _prepare_dataframe(predictions) if predictions else pd.DataFrame()

    if df_hist.empty:
        return _empty_chart("Aucune donnée pour la prédiction", mobile, fmt)

    fig, ax = _get_figure(width=9.0, height=4.5, mobile=mobile)

    y_max = 3.5

    # Historique (dernières 24-48h)
    cutoff_hist = datetime.now(timezone.utc) - timedelta(hours=48)
    df_recent = df_hist[df_hist["timestamp"] >= cutoff_hist]

    if not df_recent.empty:
        ax.plot(
            df_recent["timestamp"], df_recent["valeur"],
            color=Couleurs.COURBE_PRINCIPALE, linewidth=2,
            marker="o", markersize=4, zorder=5, label="Historique",
        )
        y_max = max(y_max, df_recent["valeur"].max() * 1.1)

    # Prédiction
    if not df_pred.empty and "valeur" in df_pred.columns:
        ax.plot(
            df_pred["timestamp"], df_pred["valeur"],
            color=Couleurs.COURBE_PREDICTION, linewidth=2.5,
            linestyle="--", zorder=5, label="Prédiction",
        )

        # Intervalle de confiance
        if "borne_inf" in df_pred.columns and "borne_sup" in df_pred.columns:
            ax.fill_between(
                df_pred["timestamp"],
                df_pred["borne_inf"].clip(lower=0),
                df_pred["borne_sup"],
                color=Couleurs.INTERVALLE_CONFIANCE,
                alpha=0.2, zorder=3,
                label="IC 95%",
            )
            y_max = max(y_max, df_pred["borne_sup"].max() * 1.05)

    # Séparateur présent/futur
    now = datetime.now(timezone.utc)
    ax.axvline(
        x=now, color=Couleurs.TEXTE,
        linestyle="|", linewidth=2, alpha=0.5, zorder=6,
    )
    ax.text(
        now, y_max * 0.95, " Maintenant ",
        fontsize=8, ha="center", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    _add_zones_glycemiques(ax, y_max=y_max)

    ax.set_ylim(0, y_max)
    ax.set_xlabel("Date/Heure")
    ax.set_ylabel("Glycémie (g/L)")
    ax.set_title(
        "Prédiction glycémique (24-48h)",
        fontweight="bold", pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.autofmt_xdate(rotation=30)
    return _fig_to_bytes(fig, fmt)


# ============================================================================
# 9. MINI-SPARKLINES
# ============================================================================

def sparkline(
    mesures: list[dict[str, Any]],
    jours: int = 7,
    width: float = 2.5,
    height: float = 0.8,
    fmt: str = "png",
) -> bytes:
    """Génère une mini-sparkline pour le résumé rapide.

    Args:
        mesures: Liste de mesures.
        jours: Période.
        width: Largeur.
        height: Hauteur.
        fmt: Format.

    Returns:
        Bytes de l'image.
    """
    df = _prepare_dataframe(mesures)
    if df.empty:
        fig, ax = plt.subplots(figsize=(width, height))
        ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return _fig_to_bytes(fig, fmt)

    cutoff = datetime.now(timezone.utc) - timedelta(days=jours)
    df = df[df["timestamp"] >= cutoff]

    fig, ax = plt.subplots(figsize=(width, height))

    # Ligne simple
    color = Couleurs.NORMAL
    if not df.empty:
        derniere_val = df["valeur"].iloc[-1]
        if derniere_val < SEUILS.HYPOGLYCEMIE_MAX:
            color = Couleurs.HYPO_CRITIQUE
        elif derniere_val > SEUILS.HYPERGLYCEMIE_MODEREE_MAX:
            color = Couleurs.HYPER_SEVERE

    ax.plot(
        df["timestamp"], df["valeur"],
        color=color, linewidth=1.5, zorder=3,
    )
    ax.fill_between(
        df["timestamp"], df["valeur"],
        alpha=0.15, color=color, zorder=2,
    )

    ax.axis("off")
    return _fig_to_bytes(fig, fmt)


# ============================================================================
# 10. TABLEAU DE BORD SYNTHÉTIQUE
# ============================================================================

def tableau_de_bord(
    mesures: list[dict[str, Any]],
    statistiques: Optional[dict[str, Any]] = None,
    jours: int = 30,
    fmt: str = "png",
) -> bytes:
    """Génère le tableau de bord synthétique complet.

    Combine 4 graphiques en une seule image :
    1. Tendance (30j) en haut à gauche
    2. TIR pie en haut à droite
    3. Distribution en bas à gauche
    4. Heatmap en bas à droite

    Args:
        mesures: Liste de mesures.
        statistiques: Stats pré-calculées (optionnel).
        jours: Période d'analyse.
        fmt: Format de sortie.

    Returns:
        Bytes de l'image combinée.
    """
    df = _prepare_dataframe(mesures)
    if df.empty:
        return _empty_chart("Aucune donnée pour le tableau de bord", mobile=False, fmt=fmt)

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(Couleurs.FOND)
    fig.suptitle(
        f"📊 Tableau de bord DiaBot-RDC — {jours} derniers jours",
        fontsize=15, fontweight="bold", y=0.98,
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    cutoff = datetime.now(timezone.utc) - timedelta(days=jours)
    df_period = df[df["timestamp"] >= cutoff]

    # ── 1. Tendance (haut gauche) ──
    ax1 = fig.add_subplot(gs[0, 0])
    if not df_period.empty:
        ax1.scatter(
            df_period["timestamp"], df_period["valeur"],
            color=Couleurs.COURBE_PRINCIPALE, alpha=0.3, s=10,
        )
        df_daily = df_period.set_index("timestamp").resample("D")["valeur"].mean().dropna()
        if len(df_daily) >= 2:
            ax1.plot(
                df_daily.index, df_daily.values,
                color=Couleurs.COURBE_PRINCIPALE, linewidth=2,
            )
        _add_zones_glycemiques(ax1, y_max=3.5)
        ax1.set_ylim(0, 3.5)
        ax1.set_title("Tendance", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.2)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        fig.autofmt_xdate(rotation=30)
    else:
        ax1.text(0.5, 0.5, "Pas de données", ha="center", va="center")

    # ── 2. TIR Pie (haut droite) ──
    ax2 = fig.add_subplot(gs[0, 1])
    if not df_period.empty:
        valeurs = df_period["valeur"]
        total = len(valeurs)
        in_r = len(valeurs[(valeurs >= SEUILS.HYPOGLYCEMIE_MAX) & (valeurs <= SEUILS.NORMAL_POSTPRANDIAL_MAX)])
        below = len(valeurs[valeurs < SEUILS.HYPOGLYCEMIE_MAX])
        above = len(valeurs[valeurs > SEUILS.NORMAL_POSTPRANDIAL_MAX])

        sizes = [below, in_r, above]
        colors_pie = [Couleurs.TIR_BELOW, Couleurs.TIR_IN_RANGE, Couleurs.TIR_ABOVE]
        labels_pie = [
            f"Bas\n{below/total*100:.0f}%",
            f"Cible\n{in_r/total*100:.0f}%",
            f"Haut\n{above/total*100:.0f}%",
        ]

        non_zero = [(s, c, l) for s, c, l in zip(sizes, colors_pie, labels_pie) if s > 0]
        if non_zero:
            s_f, c_f, l_f = zip(*non_zero)
            ax2.pie(
                s_f, labels=l_f, colors=c_f,
                autopct="", startangle=90,
                wedgeprops=dict(width=0.4, edgecolor="white"),
            )
            tir_pct = in_r / total * 100
            ax2.text(
                0, 0, f"{tir_pct:.0f}%",
                ha="center", va="center", fontsize=18, fontweight="bold",
                color=Couleurs.NORMAL if tir_pct >= 70 else Couleurs.HYPO_CRITIQUE,
            )
        ax2.set_title("Time In Range", fontsize=11, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "Pas de données", ha="center", va="center")

    # ── 3. Distribution (bas gauche) ──
    ax3 = fig.add_subplot(gs[1, 0])
    if not df_period.empty:
        bins = np.arange(0.2, 3.5, 0.15)
        counts, bin_edges, patches = ax3.hist(
            df_period["valeur"], bins=bins,
            edgecolor="white", linewidth=0.3,
        )
        for patch, left in zip(patches, bin_edges[:-1]):
            mid = left + 0.075
            if mid < SEUILS.HYPOGLYCEMIE_MAX:
                patch.set_facecolor(Couleurs.HYPO_CRITIQUE)
            elif mid <= SEUILS.NORMAL_POSTPRANDIAL_MAX:
                patch.set_facecolor(Couleurs.NORMAL)
            else:
                patch.set_facecolor(Couleurs.HYPER_MODEREE)
            patch.set_alpha(0.7)
        ax3.set_title("Distribution", fontsize=11, fontweight="bold")
        ax3.set_xlabel("g/L", fontsize=9)
        ax3.grid(True, axis="y", alpha=0.2)
    else:
        ax3.text(0.5, 0.5, "Pas de données", ha="center", va="center")

    # ── 4. Stats clés (bas droite) ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    if not df_period.empty and statistiques:
        stats = statistiques
        lines = [
            f"📊 Statistiques ({jours}j)",
            "",
            f"📏 Moyenne : {stats.get('moyenne_gl', 0):.2f} g/L",
            f"📉 Min : {stats.get('min', 0):.2f} g/L",
            f"📈 Max : {stats.get('max', 0):.2f} g/L",
            f"📐 Écart-type : {stats.get('ecart_type', 0):.2f}",
            f"🔄 CV% : {stats.get('cv_percent', 0):.1f}%",
            "",
            f"🟢 TIR : {stats.get('tir_percent', 0):.1f}%",
            f"🔴 TBR : {stats.get('tbr_percent', 0):.1f}%",
            f"🟡 TAR : {stats.get('tar_percent', 0):.1f}%",
            f"📋 GMI : {stats.get('gmi_percent', 0):.1f}%",
            "",
            f"⚠️ Score risque : {stats.get('risque_score', 0):.0f}/100",
            f"📝 Nb mesures : {stats.get('nb_mesures', 0)}",
        ]
        text = "\n".join(lines)
        ax4.text(
            0.05, 0.95, text,
            transform=ax4.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#F5F5F5",
                edgecolor=Couleurs.BORDURE,
            ),
        )
    elif not df_period.empty:
        moy = df_period["valeur"].mean()
        ax4.text(
            0.5, 0.5,
            f"Moyenne\n{moy:.2f} g/L\n\nn={len(df_period)}",
            ha="center", va="center", fontsize=14, fontweight="bold",
        )

    return _fig_to_bytes(fig, fmt)


# ============================================================================
# UTILITAIRE : Graphique vide
# ============================================================================

def _empty_chart(
    message: str,
    mobile: bool = False,
    fmt: str = "png",
) -> bytes:
    """Génère un graphique vide avec un message.

    Args:
        message: Message à afficher.
        mobile: Mode mobile.
        fmt: Format.

    Returns:
        Bytes de l'image.
    """
    fig, ax = _get_figure(mobile=mobile)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        fontsize=12, color=Couleurs.TEXTE_SECONDAIRE,
        transform=ax.transAxes,
    )
    ax.axis("off")
    return _fig_to_bytes(fig, fmt)


# ============================================================================
# AGP COMPLET (Ambulatory Glucose Profile)
# ============================================================================

def agp_profile(
    mesures: list[dict[str, Any]],
    jours: int = 14,
    fmt: str = "png",
) -> bytes:
    """Génère un profil AGP (Ambulatory Glucose Profile) complet.

    Le profil AGP standard montre :
    - Médiane (ligne centrale)
    - 25e-75e percentiles (zone sombre)
    - 5e-95e percentiles (zone claire)
    - Zones glycémiques en fond

    Args:
        mesures: Liste de mesures.
        jours: Période d'analyse (14j standard).
        fmt: Format de sortie.

    Returns:
        Bytes de l'image AGP.
    """
    df = _prepare_dataframe(mesures)
    if df.empty:
        return _empty_chart("Données insuffisantes pour l'AGP", fmt=fmt)

    cutoff = datetime.now(timezone.utc) - timedelta(days=jours)
    df = df[df["timestamp"] >= cutoff].copy()

    if len(df) < 20:
        return _empty_chart(f"Seulement {len(df)} mesures (min 20 pour l'AGP)", fmt=fmt)

    # Extraire l'heure fractionnaire
    df["heure_frac"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

    # Bins horaires (toutes les 30 min)
    bins = np.arange(0, 24.5, 0.5)
    df["heure_bin"] = pd.cut(df["heure_frac"], bins=bins, labels=False)

    # Calculer les percentiles par bin
    agg = df.groupby("heure_bin")["valeur"].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    agg = agg.unstack()

    if agg.empty:
        return _empty_chart("Données insuffisantes", fmt=fmt)

    heures = [(b + 0.5) * 0.5 for b in agg.index]

    fig, ax = _get_figure(width=10.0, height=5.0)

    y_max = max(3.5, agg[0.95].max() * 1.05) if 0.95 in agg.columns else 3.5
    _add_zones_glycemiques(ax, y_max=y_max)

    # 5e-95e percentile (zone claire)
    if 0.05 in agg.columns and 0.95 in agg.columns:
        ax.fill_between(
            heures,
            agg[0.05].clip(lower=0),
            agg[0.95],
            color=Couleurs.COURBE_PRINCIPALE, alpha=0.1,
            label="5e-95e percentile",
        )

    # 25e-75e percentile (zone sombre)
    if 0.25 in agg.columns and 0.75 in agg.columns:
        ax.fill_between(
            heures,
            agg[0.25],
            agg[0.75],
            color=Couleurs.COURBE_PRINCIPALE, alpha=0.25,
            label="25e-75e percentile",
        )

    # Médiane
    if 0.50 in agg.columns:
        ax.plot(
            heures, agg[0.50],
            color=Couleurs.COURBE_PRINCIPALE, linewidth=3,
            label="Médiane", zorder=5,
        )

    ax.set_xlim(0, 24)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Glycémie (g/L)")
    ax.set_title(
        f"Profil AGP — {jours} derniers jours",
        fontweight="bold", pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    return _fig_to_bytes(fig, fmt)


# ============================================================================
# MOTEUR DE VISUALISATION PRINCIPAL
# ============================================================================

class VisualizationEngine:
    """Moteur de visualisation unifié pour DiaBot-RDC.

    Point d'entrée unique pour la génération de tous les graphiques.
    """

    def __init__(self) -> None:
        """Initialise le moteur."""
        logger.info("📊 VisualizationEngine initialisé")

    def generer(
        self,
        type_graphique: str,
        mesures: list[dict[str, Any]],
        **kwargs: Any,
    ) -> bytes:
        """Génère un graphique par type.

        Args:
            type_graphique: Type de graphique.
                "journalier", "tendance", "heatmap", "distribution",
                "tir", "comparaison", "repas", "prediction",
                "sparkline", "dashboard", "agp".
            mesures: Données de mesures.
            **kwargs: Arguments spécifiques au type.

        Returns:
            Bytes de l'image générée.

        Raises:
            ValueError: Si le type est inconnu.
        """
        generators = {
            "journalier": courbe_journaliere,
            "tendance": courbe_tendance,
            "heatmap": heatmap_glycemie,
            "distribution": histogramme_distribution,
            "tir": tir_pie_chart,
            "comparaison": comparaison_periodes,
            "repas": correlation_repas,
            "prediction": courbe_prediction,
            "sparkline": sparkline,
            "dashboard": tableau_de_bord,
            "agp": agp_profile,
        }

        generator = generators.get(type_graphique)
        if generator is None:
            raise ValueError(
                f"Type de graphique inconnu : '{type_graphique}'. "
                f"Types disponibles : {list(generators.keys())}"
            )

        try:
            return generator(mesures, **kwargs)
        except Exception as exc:
            logger.error(
                f"❌ Erreur génération graphique '{type_graphique}' : {exc}",
                exc_info=True,
            )
            return _empty_chart(
                f"Erreur lors de la génération du graphique",
                fmt=kwargs.get("fmt", "png"),
            )

    def generer_dashboard_complet(
        self,
        mesures: list[dict[str, Any]],
        statistiques: Optional[dict[str, Any]] = None,
        jours: int = 30,
    ) -> bytes:
        """Génère le tableau de bord complet.

        Args:
            mesures: Toutes les mesures du patient.
            statistiques: Stats pré-calculées.
            jours: Période.

        Returns:
            Bytes de l'image dashboard.
        """
        return tableau_de_bord(
            mesures, statistiques=statistiques, jours=jours,
        )


# ============================================================================
# INSTANCE GLOBALE
# ============================================================================

viz_engine = VisualizationEngine()


__all__ = [
    "VisualizationEngine", "viz_engine",
    "courbe_journaliere", "courbe_tendance", "heatmap_glycemie",
    "histogramme_distribution", "tir_pie_chart", "comparaison_periodes",
    "correlation_repas", "courbe_prediction", "sparkline",
    "tableau_de_bord", "agp_profile",
    "Couleurs",
]