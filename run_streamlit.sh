#!/bin/bash
echo "Starting IBM AICTE Interview Trainer Agent (Streamlit)..."
source venv/bin/activate
streamlit run streamlit_app.py --server.port 8501
