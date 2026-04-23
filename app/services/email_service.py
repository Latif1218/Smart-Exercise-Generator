import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import settings


class EmailService:
    def __init__(self) -> None:
        self.smtp_host = settings.smtp_host
        self.smtp_port = settings.smtp_port
        self.smtp_username = settings.smtp_username
        self.smtp_password = settings.smtp_password
        self.smtp_use_tls = settings.smtp_use_tls
        self.mail_from = settings.mail_from
        self.support_email = settings.support_email

    def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        reply_to: str | None = None,
    ) -> None:
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.mail_from
        message["To"] = to_email

        if reply_to:
            message["Reply-To"] = reply_to

        html_part = MIMEText(html_body, "html", "utf-8")
        message.attach(html_part)

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.smtp_use_tls:
                server.starttls()

            server.login(self.smtp_username, self.smtp_password)
            server.sendmail(
                self.mail_from,
                to_email,
                message.as_string(),
            )

    def send_help_email_to_author(
        self,
        user_email: str,
        subject: str,
        message_text: str,
    ) -> None:
        html_body = f"""
        <html>
            <body>
                <h2>New Help Center Message</h2>
                <p><strong>From:</strong> {user_email}</p>
                <p><strong>Subject:</strong> {subject}</p>
                <hr>
                <p><strong>Message:</strong></p>
                <p>{message_text}</p>
            </body>
        </html>
        """

        self.send_email(
            to_email=self.support_email,
            subject=f"Help Center: {subject}",
            html_body=html_body,
            reply_to=user_email,
        )

    def send_auto_reply_to_user(
        self,
        user_email: str,
        original_subject: str,
    ) -> None:
        html_body = f"""
        <html>
            <body>
                <p>Hello,</p>
                <p>We have received your message regarding:</p>
                <p><strong>{original_subject}</strong></p>
                <p>Our team will review your request and get back to you soon.</p>
                <br>
                <p>Best regards,<br>Support Team</p>
            </body>
        </html>
        """

        self.send_email(
            to_email=user_email,
            subject="We received your help request",
            html_body=html_body,
        )