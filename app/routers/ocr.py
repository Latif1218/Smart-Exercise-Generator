from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List

from app.models.response_models import OCRResponse, ErrorResponse
from app.services.ocr_service import (
    extract_text_from_single_image_endpoint,
    extract_text_from_multiple_images
)

router = APIRouter(
    prefix="/ocr",
    tags=["OCR — Image to Text"]
)


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff"
}


def _validate_image_content_type(file: UploadFile) -> None:
    """Validate that the uploaded file is a supported image type"""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, WebP, BMP, TIFF"
        )
    


@router.post(
    "/single",
    response_model=OCRResponse,
    summary="Single Image OCR",
    description="""
    Upload a single image and extract text from it.
    
    Returns:
    - Extracted text from the image
    - Cleaned and formatted text ready for exercise generation
    """
)
async def ocr_single_image(
    image: UploadFile = File(..., description="Image containing text, textbook page, notes")
) -> OCRResponse:
    """Extract text from a single image using OCR"""

    _validate_image_content_type(image)

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Empty file uploaded."
        )

    try:
        result = extract_text_from_single_image_endpoint(image_bytes)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )
    



@router.post(
    "/multiple",
    response_model=OCRResponse,
    summary="Multiple Images OCR",
    description="""
    Upload multiple images and extract text from all of them, then merge the results.
    
    Features:
    - Up to 20 images can be processed at once
    - Each page is tracked individually
    - Extracted text from all pages is merged and returned
    """
)
async def ocr_multiple_images(
    images: List[UploadFile] = File(..., description="Multiple textbook pages, maximum 20 files")
) -> OCRResponse:
    """Extract and merge text from multiple images using OCR"""

    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="At least one image must be provided."
        )

    if len(images) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A maximum of 20 images can be processed at once. You provided {len(images)}."
        )

    # Validate all files
    for img in images:
        _validate_image_content_type(img)

    # Read all image bytes
    images_bytes = []
    for img in images:
        img_bytes = await img.read()
        if img_bytes:
            images_bytes.append(img_bytes)

    if not images_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All uploaded images are empty."
        )

    try:
        result = extract_text_from_multiple_images(images_bytes)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )