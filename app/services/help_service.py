from app.models.request_models import HelpContactRequest
from app.services.email_service import EmailService


class HelpService:
    def __init__(self) -> None:
        self.email_service = EmailService()

    def submit_help_request(self, payload: HelpContactRequest) -> None:
        self.email_service.send_help_email_to_author(
            user_email=payload.email,
            subject=payload.subject,
            message_text=payload.message,
        )

        # Optional auto-reply to user
        self.email_service.send_auto_reply_to_user(
            user_email=payload.email,
            original_subject=payload.subject,
        )