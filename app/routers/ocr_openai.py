from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List

from app.models.response_models import OCRResponse, ErrorResponse
from app.services.openai_ocr_service import openai_ocr_service

router = APIRouter(
    prefix="/ocr/openai",
    tags=["OCR — OpenAI Vision"],
    responses={404: {"model": ErrorResponse}}
)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}


def _validate_image_content_type(file: UploadFile):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, WebP, BMP, TIFF"
        )


@router.post(
    "/single",
    response_model=OCRResponse,
    summary="Single Image OCR using OpenAI GPT-4o"
)
async def ocr_single_image_openai(
    image: UploadFile = File(..., description="Image containing text (textbook page, notes, etc.)")
) -> OCRResponse:
    """Extract clean text from a single image using OpenAI GPT-4o Vision"""

    _validate_image_content_type(image)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        page = openai_ocr_service.extract_text_from_single_image(image_bytes, page_number=1)

        return OCRResponse(
            success=True,
            pages=[page],
            merged_text=page.extracted_text,
            total_pages=1
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenAI OCR processing failed: {str(e)}"
        )


@router.post(
    "/multiple",
    response_model=OCRResponse,
    summary="Multiple Images OCR using OpenAI GPT-4o"
)
async def ocr_multiple_images_openai(
    images: List[UploadFile] = File(..., description="Multiple images (max 10)")
) -> OCRResponse:
    """Extract text from multiple images using OpenAI GPT-4o Vision"""

    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed.")

    for img in images:
        _validate_image_content_type(img)

    extracted_pages = []
    failed_pages = []

    for idx, image in enumerate(images, 1):
        try:
            image_bytes = await image.read()
            if not image_bytes:
                failed_pages.append(f"Page {idx}: Empty file")
                continue

            page = openai_ocr_service.extract_text_from_single_image(image_bytes, page_number=idx)
            extracted_pages.append(page)

        except Exception as e:
            failed_pages.append(f"Page {idx}: {str(e)}")

    if not extracted_pages:
        raise HTTPException(
            status_code=422,
            detail=f"No text extracted. Details: {'; '.join(failed_pages)}"
        )

    merged_text = "\n\n".join([page.extracted_text for page in extracted_pages])

    return OCRResponse(
        success=True,
        pages=extracted_pages,
        merged_text=merged_text,
        total_pages=len(extracted_pages)
    )