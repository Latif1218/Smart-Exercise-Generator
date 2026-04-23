from fastapi import APIRouter, HTTPException, status

from app.models.request_models import HelpContactRequest
from app.models.response_models import HelpContactResponse
from app.services.help_service import HelpService

router = APIRouter(
    prefix="/help",
    tags=["Help"]
)

help_service = HelpService()


@router.post(
    "/help/contact",
    response_model=HelpContactResponse,
    status_code=status.HTTP_200_OK,
)
def submit_help_contact(payload: HelpContactRequest) -> HelpContactResponse:
    try:
        help_service.submit_help_request(payload)

        return HelpContactResponse(
            success=True,
            message="Your message has been sent successfully.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send help request: {str(exc)}",
        )