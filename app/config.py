from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    deepseek_api_key: str = Field(..., env="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        env="DEEPSEEK_BASE_URL"
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        env="DEEPSEEK_MODEL"
    )

    tesseract_cmd: str = Field(
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        env="TESSERACT_CMD"
    )


    max_image_size_mb: int = Field(
        default=10,
        env="MAX_IMAGE_SIZE_MB"
    )
    max_files_per_request: int = Field(
        default=10,
        env="MAX_FILES_PER_REQUEST"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()