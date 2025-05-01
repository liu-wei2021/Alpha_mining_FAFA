import os
from langchain.chat_models import ChatOpenAI

from src.config import settings


def get_deepseek_llm():
    """
    Initialize and return a ChatOpenAI client configured for Deepseek.
    Reads credentials and settings from your config.yaml via settings.
    """
    # Set the API key in environment if provided
    if settings.deepseek_api_key:
        os.environ["OPENAI_API_KEY"] = settings.deepseek_api_key

    # Instantiate the LLM
    llm = ChatOpenAI(
        model_name=settings.deepseek_model,
        openai_api_base=settings.deepseek_api_base,
        temperature=settings.deepseek_temperature
    )
    return llm
