@echo off
echo Starting Training Data Generator Backend...
echo.
call "%~dp0venv\Scripts\activate.bat"
cd /d "%~dp0backend"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
echo.
python app.py
pause
