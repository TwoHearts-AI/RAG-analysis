from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class Settings(BaseSettings):
    # Service settings
    DEBUG: bool = Field(False, description="Debug mode for FastAPI")

    # Qdrant Configuration
    QDRANT_HOST: str = Field("qdrant", description="Qdrant server host")
    QDRANT_PORT: int = Field(6333, description="Qdrant server port")

    # Model Configuration
    EMBEDDING_MODEL: str = Field(
        "cointegrated/rubert-tiny",
        description="Model name for embeddings generation"
    )
    MISTRAL_API_KEY: str = Field(..., description="Mistral API key")
    MISTRAL_MODEL: str = Field(
        "mistral-large-latest",
        description="Mistral model name to use"
    )

    class Config:
        env_file = ".env"


# Load environment variables from the specified .env file
load_dotenv()

# Create settings instance
CONFIG = Settings()
