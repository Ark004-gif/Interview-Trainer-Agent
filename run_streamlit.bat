@echo off
echo ============================================================
echo  IBM AICTE -- Interview Trainer Agent
echo  Starting Streamlit App...
echo ============================================================
call venv\Scripts\activate.bat
streamlit run streamlit_app.py --server.port 8501 --server.headless false
pause
