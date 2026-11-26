@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   PanoramaStudio - He thong Ghep anh Toan canh (Xu ly anh so)
echo ============================================================
if not exist ".venv\Scripts\python.exe" (
    echo [LOI] Khong tim thay moi truong ao .venv!
    echo Hay chay: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
echo Dang khoi dong may chu ung dung tai http://127.0.0.1:8000 ...
".venv\Scripts\python.exe" app.py
pause
