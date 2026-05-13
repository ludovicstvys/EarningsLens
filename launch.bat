@echo off
setlocal

pushd "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Missing .venv. Create it first with:
    echo     py -3.11 -m venv .venv
    pause
    popd
    endlocal
    exit /b 1
)

call ".venv\Scripts\activate.bat"

python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Missing dependencies. Install them with:
    echo     pip install -r requirements.txt
    pause
    popd
    endlocal
    exit /b 1
)

if not exist ".models\local-llm" goto missing_models
if not exist ".models\finbert" goto missing_models
if not exist ".models\sentence-embedder" goto missing_models
goto run

:missing_models
echo Local model cache is missing. Prefetch once with:
echo     python scripts\prefetch_models.py
echo The app can still start, but it may download models on first use.

:run
streamlit run app.py
if errorlevel 1 pause

popd
endlocal
