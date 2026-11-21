@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo   CCTV TRAFFIC AI SERVICE - NSSM WINDOWS INSTALLER
echo =======================================================

:: 1. Check for Administrator privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ban phai chay file batch nay voi quyen Administrator (Run as Administrator)!
    pause
    exit /b 1
)

:: 2. Resolve Paths
cd /d "%~dp0\.."
set "PROJECT_DIR=%CD%"
set "PYTHON_EXE=%PROJECT_DIR%\venv\Scripts\python.exe"
set "SERVICE_NAME=TrafficAIService"
set "NSSM_EXE=%PROJECT_DIR%\scripts\nssm.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Khong tim thay Python venv tai: %PYTHON_EXE%
    echo Vui long kiem tra lai moi truong ao venv!
    pause
    exit /b 1
)

:: Check if nssm is in PATH or in scripts/
where nssm >nul 2>&1
if %errorlevel% equ 0 (
    set "NSSM_CMD=nssm"
) else (
    if exist "%NSSM_EXE%" (
        set "NSSM_CMD=%NSSM_EXE%"
    ) else (
        echo [CANH BAO] Khong tim thay nssm.exe trong PATH hoac trong thu muc scripts\
        echo Dang tien hanh tim kiem nssm tren may tinh...
        set "NSSM_CMD=nssm"
    )
)

echo.
echo [1/5] Kiem tra va dung service cu neu da ton tai...
%NSSM_CMD% stop %SERVICE_NAME% >nul 2>&1
%NSSM_CMD% remove %SERVICE_NAME% confirm >nul 2>&1

echo [2/5] Cai dat Windows Service: %SERVICE_NAME%...
%NSSM_CMD% install %SERVICE_NAME% "%PYTHON_EXE%" "-m uvicorn src.main:app --host 0.0.0.0 --port 8000"
if %errorlevel% neq 0 (
    echo [ERROR] Cai dat service bang NSSM that bai!
    pause
    exit /b 1
)

echo [3/5] Cau hinh thong so van hanh va Graceful Shutdown...
%NSSM_CMD% set %SERVICE_NAME% AppDirectory "%PROJECT_DIR%"
%NSSM_CMD% set %SERVICE_NAME% Start SERVICE_AUTO_START
%NSSM_CMD% set %SERVICE_NAME% AppRestartDelay 5000

:: Graceful Shutdown: Gui tin hieu Console Control Event va cho toi da 10000ms (10 giay) de thoat sach
%NSSM_CMD% set %SERVICE_NAME% AppStopMethodConsole 10000
%NSSM_CMD% set %SERVICE_NAME% AppStopMethodThreads 5000

:: Chuyen huong stdout va stderr vao thu muc logs/
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"
%NSSM_CMD% set %SERVICE_NAME% AppStdout "%PROJECT_DIR%\logs\service_stdout.log"
%NSSM_CMD% set %SERVICE_NAME% AppStderr "%PROJECT_DIR%\logs\service_stderr.log"

echo [4/5] Khoi dong Service...
%NSSM_CMD% start %SERVICE_NAME%

echo [5/5] Kiem tra trang thai Service...
%NSSM_CMD% status %SERVICE_NAME%

echo.
echo =======================================================
echo   CAI DAT DICH VU THANH CONG!
echo   Service Name: %SERVICE_NAME%
echo   Web Dashboard: http://localhost:8000/
echo   API Docs:      http://localhost:8000/docs
echo =======================================================
pause
