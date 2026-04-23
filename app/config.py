from pydantic import Field
from pydantic_settings import BaseSettings


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

    max_file_size_mb: int = Field(
        default=10,
        env="MAX_FILE_SIZE_MB"
    )

    app_name: str = Field(default="OCR Question Generator API", env="APP_NAME")
    app_env: str = Field(default="development", env="APP_ENV")
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=True, env="DEBUG")

    smtp_host: str = Field(..., env="SMTP_HOST")
    smtp_port: int = Field(..., env="SMTP_PORT")
    smtp_username: str = Field(..., env="SMTP_USERNAME")
    smtp_password: str = Field(..., env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")
    mail_from: str = Field(..., env="MAIL_FROM")
    support_email: str = Field(..., env="SUPPORT_EMAIL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()