"""
Main entry point — IBM AICTE Interview Trainer Agent
Run: python main.py
     OR: uvicorn main:app --reload
"""
import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from app.api.routes import app  # noqa: F401 — exposes 'app' for uvicorn

if __name__ == "__main__":
    import uvicorn
    from app.config import settings

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Create log directory
    Path("./logs").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  IBM AICTE — Interview Trainer Agent")
    print("  Powered by IBM Granite (watsonx.ai) + RAG")
    print("=" * 60)
    print(f"  > API:     http://{settings.APP_HOST}:{settings.APP_PORT}")
    print(f"  > UI:      http://localhost:{settings.APP_PORT}")
    print(f"  > Docs:    http://localhost:{settings.APP_PORT}/api/docs")
    print(f"  > Model:   {settings.IBM_GRANITE_MODEL_ID}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
