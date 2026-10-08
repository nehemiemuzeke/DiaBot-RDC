"""
================================================================================
FICHIER : main.py
RESPONSABILITÉ : Point d'entrée de DiaBot-RDC
================================================================================
Ce module est le lanceur principal de l'application. Il :
1. Charge la configuration (config.py)
2. Configure le logging global
3. Initialise la base de données (database.py)
4. Charge les modèles IA avec graceful degradation (ai_engine.py)
5. Initialise le système multilingue (multilingual.py)
6. Démarre le serveur FastAPI (api.py)
7. Lance le scheduler de tâches périodiques
8. Gère l'arrêt gracieux (cleanup des ressources)

Modes d'exécution :
- development : uvicorn avec reload, logs DEBUG
- production : uvicorn multi-workers, logs INFO
- testing : mode allégé sans GPU

Auteur : Équipe DiaBot-RDC
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import uvicorn

from config import (
    APP_NAME, APP_VERSION, API, ENVIRONMENT, Environment,
    configure_logging, get_all_config, validate_config,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SCHEDULER DE TÂCHES PÉRIODIQUES
# ============================================================================

class TaskScheduler:
    """Scheduler simple pour les tâches périodiques.

    En production, remplacer par Celery ou APScheduler.
    """

    def __init__(self) -> None:
        """Initialise le scheduler."""
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Démarre toutes les tâches périodiques."""
        self._running = True
        logger.info("⏰ Scheduler démarré")

        self._tasks.append(
            asyncio.create_task(self._daily_analysis_loop())
        )
        self._tasks.append(
            asyncio.create_task(self._weekly_report_loop())
        )
        self._tasks.append(
            asyncio.create_task(self._cleanup_loop())
        )
        self._tasks.append(
            asyncio.create_task(self._qr_cleanup_loop())
        )

    async def stop(self) -> None:
        """Arrête toutes les tâches."""
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        logger.info("⏰ Scheduler arrêté")

    async def _daily_analysis_loop(self) -> None:
        """Analyse quotidienne des glycémies de chaque patient.

        S'exécute tous les jours à 6h00 UTC.
        """
        while self._running:
            try:
                await self._wait_until(hour=6, minute=0)
                if not self._running:
                    break
                logger.info("📊 Début analyse quotidienne...")
                await self._run_daily_analysis()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"❌ Erreur analyse quotidienne : {exc}")
                await asyncio.sleep(3600)  # Retry dans 1h

    async def _weekly_report_loop(self) -> None:
        """Génération hebdomadaire de rapports.

        S'exécute chaque lundi à 7h00 UTC.
        """
        while self._running:
            try:
                await self._wait_until(hour=7, minute=0, weekday=0)
                if not self._running:
                    break
                logger.info("📄 Début génération rapports hebdo...")
                await self._run_weekly_reports()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"❌ Erreur rapports hebdo : {exc}")
                await asyncio.sleep(3600)

    async def _cleanup_loop(self) -> None:
        """Nettoyage quotidien des données temporaires.

        S'exécute tous les jours à 3h00 UTC.
        """
        while self._running:
            try:
                await self._wait_until(hour=3, minute=0)
                if not self._running:
                    break
                logger.info("🧹 Début nettoyage quotidien...")
                await self._run_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"❌ Erreur nettoyage : {exc}")
                await asyncio.sleep(3600)

    async def _qr_cleanup_loop(self) -> None:
        """Nettoyage des QR codes expirés.

        S'exécute toutes les 30 minutes.
        """
        while self._running:
            try:
                await asyncio.sleep(1800)  # 30 min
                if not self._running:
                    break
                from qr_manager import qr_manager
                cleaned = qr_manager.cleanup_expired()
                if cleaned > 0:
                    logger.debug(f"🧹 {cleaned} QR expirés nettoyés")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"❌ Erreur cleanup QR : {exc}")

    async def _run_daily_analysis(self) -> None:
        """Exécute l'analyse glycémique quotidienne."""
        try:
            from database import (
                get_session, PatientRepository,
                MesureGlycemieRepository, AlerteRepository,
            )
            from ai_engine import diabot
            from config import SEUILS

            async with get_session() as session:
                # Récupérer tous les patients actifs
                from database import Patient
                from sqlalchemy import select

                result = await session.execute(
                    select(Patient).where(Patient.actif.is_(True))
                )
                patients = result.scalars().all()

                for patient in patients:
                    try:
                        mesures = await MesureGlycemieRepository.get_by_patient(
                            session, patient.id, limit=200,
                        )
                        if not mesures:
                            continue

                        mesures_dict = [m.to_dict() for m in mesures]
                        analyse = await diabot.analyser_glycemie(
                            mesures_dict, patient.id, jours=7,
                        )

                        # Créer une alerte si score de risque élevé
                        if analyse.get("score_risque", 0) > 60:
                            await AlerteRepository.create(
                                session,
                                patient_id=patient.id,
                                type_alerte="risque_eleve_quotidien",
                                severite="attention",
                                message=(
                                    f"Analyse quotidienne : score de risque "
                                    f"{analyse['score_risque']}/100. "
                                    f"TIR={analyse['tir_percent']}%."
                                ),
                                donnees_contexte=analyse,
                            )

                    except Exception as exc:
                        logger.error(
                            f"Erreur analyse patient {patient.id[:8]} : {exc}"
                        )

                logger.info(
                    f"✅ Analyse quotidienne terminée : "
                    f"{len(patients)} patients traités"
                )

        except Exception as exc:
            logger.error(f"❌ Erreur analyse quotidienne : {exc}")

    async def _run_weekly_reports(self) -> None:
        """Génère les rapports hebdomadaires."""
        try:
            from database import (
                get_session, Patient,
                PatientRepository, MesureGlycemieRepository,
                AlerteRepository, RapportMedicalRepository,
            )
            from sqlalchemy import select
            from ai_engine import diabot

            async with get_session() as session:
                result = await session.execute(
                    select(Patient).where(Patient.actif.is_(True))
                )
                patients = result.scalars().all()
                count = 0

                for patient in patients:
                    try:
                        mesures = await MesureGlycemieRepository.get_by_patient(
                            session, patient.id, limit=500,
                        )
                        if len(mesures) < 10:
                            continue

                        alertes = await AlerteRepository.get_actives(
                            session, patient.id,
                        )
                        profil = patient.to_dict(include_sensitive=True)

                        rapport = await diabot.generer_rapport(
                            profil,
                            [m.to_dict() for m in mesures],
                            [a.to_dict() for a in alertes],
                            periode_jours=7,
                        )

                        now = datetime.now(timezone.utc)
                        await RapportMedicalRepository.create(
                            session,
                            patient_id=patient.id,
                            contenu=rapport,
                            periode_debut=now - timedelta(days=7),
                            periode_fin=now,
                            resume_ia=rapport.get("evaluation_globale"),
                            score_risque=rapport.get("score_risque", {}).get("score"),
                        )
                        count += 1

                    except Exception as exc:
                        logger.error(
                            f"Erreur rapport patient {patient.id[:8]} : {exc}"
                        )

                logger.info(f"✅ {count} rapports hebdo générés")

        except Exception as exc:
            logger.error(f"❌ Erreur rapports hebdo : {exc}")

    async def _run_cleanup(self) -> None:
        """Nettoie les données temporaires."""
        try:
            from database import get_session, SessionMedecinRepository
            from security import security_manager

            async with get_session() as session:
                cleaned = await SessionMedecinRepository.cleanup_expired(session)

            security_manager.rate_limiter.cleanup()
            security_manager.jwt.cleanup_expired_revocations()

            logger.info(f"✅ Nettoyage terminé (sessions: {cleaned})")

        except Exception as exc:
            logger.error(f"❌ Erreur nettoyage : {exc}")

    @staticmethod
    async def _wait_until(
        hour: int,
        minute: int = 0,
        weekday: int | None = None,
    ) -> None:
        """Attend jusqu'à l'heure spécifiée.

        Args:
            hour: Heure cible (0-23).
            minute: Minute cible (0-59).
            weekday: Jour de la semaine (0=lundi, None=tous les jours).
        """
        while True:
            now = datetime.now(timezone.utc)
            target = now.replace(
                hour=hour, minute=minute, second=0, microsecond=0,
            )

            if target <= now:
                target += timedelta(days=1)

            if weekday is not None:
                days_ahead = weekday - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target = now + timedelta(days=days_ahead)
                target = target.replace(
                    hour=hour, minute=minute, second=0, microsecond=0,
                )

            wait_seconds = (target - now).total_seconds()
            if wait_seconds > 0:
                await asyncio.sleep(min(wait_seconds, 3600))
                if weekday is None:
                    break
            else:
                break


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

async def startup_checks() -> dict[str, Any]:
    """Exécute les vérifications de démarrage.

    Returns:
        Dict des résultats de health check.
    """
    from database import db_health_check
    from ai_engine import diabot

    results: dict[str, Any] = {}

    # 1. Configuration
    config_warnings = validate_config()
    results["config"] = {
        "status": "ok" if not config_warnings else "warnings",
        "warnings": config_warnings,
    }

    # 2. Base de données
    try:
        db_status = await db_health_check()
        results["database"] = db_status
    except Exception as exc:
        results["database"] = {"status": "error", "error": str(exc)}

    # 3. Modèles IA
    try:
        ai_status = diabot.get_status()
        results["ai"] = ai_status
    except Exception as exc:
        results["ai"] = {"status": "error", "error": str(exc)}

    # 4. Dossiers
    from config import DATA_DIR, MODELS_DIR, LOGS_DIR, UPLOADS_DIR, EXPORTS_DIR
    dirs_ok = all(
        d.exists() for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, UPLOADS_DIR, EXPORTS_DIR]
    )
    results["directories"] = {"status": "ok" if dirs_ok else "missing"}

    return results


def print_banner() -> None:
    """Affiche la bannière de démarrage."""
    config = get_all_config()
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🏥  {APP_NAME} v{APP_VERSION}                              ║
║   IA Médicale Diabétologie — RDC                             ║
║                                                              ║
║   🌍 Langues : FR, Lingala, Swahili, Tshiluba, Kikongo      ║
║   🧠 LLM : {config.get('llm_model', 'N/A'):<44} ║
║   👁️  Vision : {config.get('vision_model', 'N/A'):<41} ║
║   🎙️  STT : {config.get('stt_model', 'N/A'):<44} ║
║   🔊 TTS : {config.get('tts_model', 'N/A'):<44} ║
║   📊 DB : {config.get('database', {}).get('url_scheme', 'N/A'):<45} ║
║   🌐 API : http://{config.get('api', {}).get('host', '0.0.0.0')}:{config.get('api', {}).get('port', 8000)}{config.get('api', {}).get('prefix', '')}{' ' * 20}║
║   📚 Docs : http://localhost:{config.get('api', {}).get('port', 8000)}/docs{' ' * 28}║
║   🏗️  Env : {ENVIRONMENT.value:<44} ║
║                                                              ║
║   ⚠️  Outil d'aide au suivi — PAS un dispositif médical     ║
║   🔒 Conforme RGPD — Données chiffrées AES-256              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main() -> None:
    """Point d'entrée principal de l'application."""
    # 1. Logging
    configure_logging()
    print_banner()

    # 2. Vérifications pré-démarrage
    config_warnings = validate_config()
    for w in config_warnings:
        logger.warning(w)

    if ENVIRONMENT == Environment.PRODUCTION:
        critical_warnings = [
            w for w in config_warnings if "SECRET_KEY" in w or "AES" in w
        ]
        if critical_warnings:
            logger.critical(
                "🚨 CONFIGURATION SÉCURITÉ INCOMPLETE EN PRODUCTION ! "
                "Vérifiez les variables d'environnement."
            )

    # 3. Configuration Uvicorn
    is_dev = ENVIRONMENT in (Environment.DEVELOPMENT, Environment.TESTING)

    uvicorn_config = {
        "app": "api:app",
        "host": API.HOST,
        "port": API.PORT,
        "workers": 1 if is_dev else API.WORKERS,
        "reload": is_dev,
        "log_level": LOG_LEVEL.lower(),
        "access_log": True,
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
        "timeout_keep_alive": 65,
        "limit_concurrency": 1000,
        "limit_max_requests": 10000 if is_dev else 0,
    }

    # 4. Gestion des signaux
    def handle_shutdown(signum: int, frame: Any) -> None:
        """Gère les signaux d'arrêt."""
        sig_name = signal.Signals(signum).name
        logger.info(f"🛑 Signal reçu : {sig_name}. Arrêt en cours...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # 5. Lancement
    logger.info(
        f"🚀 Démarrage du serveur : "
        f"http://{API.HOST}:{API.PORT}{API.PREFIX}"
    )
    logger.info(f"📚 Documentation : http://{API.HOST}:{API.PORT}/docs")

    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt par l'utilisateur (Ctrl+C)")
    except Exception as exc:
        logger.critical(f"💥 Erreur fatale : {exc}", exc_info=True)
        sys.exit(1)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    main()