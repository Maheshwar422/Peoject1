@echo off
echo ===================================================
echo QT-2.3 Quantum Vision Defect Detection System
echo ===================================================
echo.
echo [1/2] Installing requirements...
python -m pip install -r requirements.txt
echo.
echo [2/2] Launching Streamlit Web App...
echo.
python -m streamlit run app.py --browser.gatherUsageStats=false
pause
