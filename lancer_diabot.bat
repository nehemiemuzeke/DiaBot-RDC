@echo off
title DiaBot-RDC Server
color 0B

echo ========================================================
echo   Lancement de DiaBot-RDC
echo ========================================================
echo.

call venv\Scripts\activate.bat

if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "uploads" mkdir uploads
if not exist "exports" mkdir exports
if not exist "translations" mkdir translations
if not exist "rag_knowledge_base" mkdir rag_knowledge_base

echo Demarrage de l'API...
start http://localhost:8000/docs
python main.py
pause