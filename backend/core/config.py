from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    tavily_api_key: str = ""
    openrouter_api_key: str = ""
    jina_api_key: str = ""
    openrouter_model: str = "openai/gpt-oss-120b:free"
    cors_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
