@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo   CCTV TRAFFIC AI SERVICE - NSSM WINDOWS UNINSTALLER
echo =======================================================

:: 1. Check for Administrator privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ban phai chay file batch nay voi quyen Administrator (Run as Administrator)!
    pause
    exit /b 1
)

set "SERVICE_NAME=TrafficAIService"
set "NSSM_EXE=%~dp0nssm.exe"

where nssm >nul 2>&1
if %errorlevel% equ 0 (
    set "NSSM_CMD=nssm"
) else (
    if exist "%NSSM_EXE%" (
        set "NSSM_CMD=%NSSM_EXE%"
    ) else (
        set "NSSM_CMD=nssm"
    )
)

echo [1/2] Dang dung Windows Service: %SERVICE_NAME%...
%NSSM_CMD% stop %SERVICE_NAME%

echo [2/2] Dang go bo Windows Service: %SERVICE_NAME%...
%NSSM_CMD% remove %SERVICE_NAME% confirm

echo.
echo =======================================================
echo   DA GO BO DICH VU %SERVICE_NAME% THANH CONG!
echo =======================================================
pause
