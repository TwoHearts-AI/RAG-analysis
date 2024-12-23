from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class Settings(BaseSettings):
    # Service settings
    DEBUG: bool = Field(False, description="Debug mode for FastAPI")

    # Qdrant Configuration
    QDRANT_URL: str = Field("qdrant:6333", description="Qdrant server")

    MISTRAL_API_KEY: str = Field(..., description="Mistral API key")
    MISTRAL_MODEL: str = Field(
        "mistral-large-latest",
        description="Mistral model name to use"
    )

    # Setting up langchain tracing
    LANGCHAIN_API_KEY: str = Field(..., description="Langchain API key")
    LANGCHAIN_TRACING_V2: bool = Field(True)
    LANGCHAIN_ENDPOINT: str = Field(..., description="Langchain API endpoint")
    LANGCHAIN_PROJECT: str = Field(..., description="Project name")

    TELEGRAM_TOKEN: str = Field(..., description="Token for tg bot")

    class Config:
        env_file = ".env"


# Load environment variables from the specified .env file
load_dotenv()

# Create settings instance
CONFIG = Settings()
