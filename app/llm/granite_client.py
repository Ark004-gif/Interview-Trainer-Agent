"""
IBM Granite LLM Client
Wraps ibm-watsonx-ai to call IBM Granite models for text generation.
"""
import logging
from typing import Optional

from app.config import settings

# ibm-watsonx-ai is an optional heavy dependency.
# The app runs in demo mode when it is not installed.
try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    _IBM_SDK_AVAILABLE = True
except ImportError:  # noqa: BLE001
    _IBM_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class GraniteLLMClient:
    """
    Thread-safe singleton wrapper for IBM watsonx.ai Granite inference.
    Falls back to a mock response when IBM credentials are not yet configured,
    so the application remains runnable during development.
    """

    def __init__(self):
        self._model: Optional[ModelInference] = None
        self._initialized = False

    def _setup(self) -> None:
        if self._initialized:
            return

        if not _IBM_SDK_AVAILABLE:
            logger.warning(
                "ibm-watsonx-ai SDK not installed. "
                "Run: pip install ibm-watsonx-ai  to enable IBM Granite. "
                "Running in DEMO mode."
            )
            self._initialized = True
            return

        if not settings.IBM_WATSONX_API_KEY or settings.IBM_WATSONX_API_KEY.startswith(
            "your_"
        ):
            logger.warning(
                "IBM watsonx.ai credentials not configured. "
                "Running in DEMO mode — responses will be rule-based."
            )
            self._initialized = True
            return

        try:
            credentials = Credentials(
                url=settings.IBM_WATSONX_URL,
                api_key=settings.IBM_WATSONX_API_KEY,
            )
            self._model = ModelInference(
                model_id=settings.IBM_GRANITE_MODEL_ID,
                credentials=credentials,
                project_id=settings.IBM_WATSONX_PROJECT_ID,
                params={
                    GenParams.MAX_NEW_TOKENS: settings.LLM_MAX_NEW_TOKENS,
                    GenParams.TEMPERATURE: settings.LLM_TEMPERATURE,
                    GenParams.TOP_P: settings.LLM_TOP_P,
                    GenParams.TOP_K: settings.LLM_TOP_K,
                    GenParams.REPETITION_PENALTY: settings.LLM_REPETITION_PENALTY,
                },
            )
            logger.info(
                "IBM Granite model '%s' initialized successfully.",
                settings.IBM_GRANITE_MODEL_ID,
            )
        except Exception as exc:  # noqa: BLE001
            err_str = str(exc)
            if "invalid_instance_status" in err_str or "Inactive" in err_str:
                logger.warning(
                    "IBM WML instance is Inactive. Reactivate it at "
                    "https://cloud.ibm.com > Resource List > Watson Machine Learning > Resume. "
                    "Running in DEMO mode."
                )
            elif "no_associated_service_instance" in err_str:
                logger.warning(
                    "No WML service instance associated with this project. "
                    "Running in DEMO mode."
                )
            else:
                logger.error("Failed to initialize IBM Granite: %s", exc)
        finally:
            self._initialized = True

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt. Returns a mock answer in demo mode."""
        self._setup()

        if self._model is None:
            return self._demo_response(prompt)

        try:
            response = self._model.generate_text(prompt=prompt)
            return response.strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Granite generation error: %s", exc)
            return self._demo_response(prompt)

    # ── Demo / fallback ───────────────────────────────────────────────────────

    @staticmethod
    def _demo_response(prompt: str) -> str:
        """Silent fallback when IBM Granite is unavailable."""
        return ""


# Module-level singleton
granite_client = GraniteLLMClient()
