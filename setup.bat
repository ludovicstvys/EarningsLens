@echo off
setlocal enabledelayedexpansion

pushd "%~dp0"

echo.
echo ==================================================
echo   EarningsLens -- one-click setup (Windows)
echo ==================================================
echo.

REM 1. Find a Python interpreter
set "PY="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PY=py -3.11"
        goto :have_python
    )
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto :have_python
)

echo Error: Python is not installed or not on PATH.
echo.
echo Install Python 3.11 from https://www.python.org/downloads/windows/
echo During the installer, make sure to TICK "Add Python to PATH".
echo Then double-click this file again.
echo.
pause
popd
endlocal
exit /b 1

:have_python
%PY% -c "import sys; print('Using Python', sys.version.split()[0])"
echo.

REM 2. Create venv
if not exist ".venv\Scripts\activate.bat" (
    echo Step 1/5  Creating virtual environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        popd
        endlocal
        exit /b 1
    )
) else (
    echo Step 1/5  Reusing existing .venv
)

call ".venv\Scripts\activate.bat"

REM 3. Upgrade pip
echo Step 2/5  Upgrading pip ...
python -m pip install --upgrade pip --quiet
if errorlevel 1 goto :pip_failed

REM 4. Install pinned dependencies
echo Step 3/5  Installing Python dependencies (this can take a few minutes) ...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 goto :pip_failed

REM 5. Copy .env if missing
if not exist ".env" (
    echo Step 4/5  Creating .env from .env.example ...
    copy ".env.example" ".env" >nul
) else (
    echo Step 4/5  Reusing existing .env
)

REM 6. Prefetch Hugging Face models
echo Step 5/5  Downloading local models (~1.5 GB total) ...
echo           FinBERT ~440 MB, MiniLM ~80 MB, SmolLM2 ~1 GB
echo           This is one-time; future launches are instant.
python scripts\prefetch_models.py
if errorlevel 1 goto :prefetch_failed

echo.
echo ==================================================
echo   Setup complete!
echo ==================================================
echo.
echo   To start the app, double-click   launch.bat
echo   Or from a terminal:               streamlit run app.py
echo.
pause
popd
endlocal
exit /b 0

:pip_failed
echo.
echo Dependency installation failed. Check your internet connection and try again.
pause
popd
endlocal
exit /b 1

:prefetch_failed
echo.
echo Model download failed. The app can still start, but models will be downloaded
echo on first use. Run "python scripts\prefetch_models.py" later to retry.
pause
popd
endlocal
exit /b 0
