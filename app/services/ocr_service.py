import pytesseract
from PIL import Image
import io
from typing import List
from app.config import settings
from app.utils.image_utils import preprocess_image_for_ocr, validate_image, numpy_to_pil
from app.utils.text_utils import clean_ocr_text, merge_pages_text
from app.models.response_models import ExtractedPage, OCRResponse


if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

def extract_text_from_single_image(image_bytes: bytes, page_number: int = 1) -> ExtractedPage:
    """
    Extract text from a single image using OCR.
    This function is used in the 'Single image flow' in Figma.

    Args:
        image_bytes: Raw image bytes data
        page_number: Page serial number (useful for multiple image flow)

    Returns:
        ExtractedPage object containing page number and extracted text
    """
    validate_image(image_bytes, settings.max_image_size_mb)

    preprocessed_np = preprocess_image_for_ocr(image_bytes)

    pil_image = numpy_to_pil(preprocessed_np)

    raw_text = pytesseract.image_to_string(
        pil_image,
        lang='eng',
        config='--oem 3 --psm 6'
    )

    cleaned_text = clean_ocr_text(raw_text)

    if not cleaned_text:
        raise ValueError(
            f"No text could be extracted from page {page_number}. Please ensure the image is clear."
        )

    return ExtractedPage(
        page_number=page_number,
        extracted_text=cleaned_text
    )



def extract_text_from_multiple_images(images_bytes: List[bytes]) -> OCRResponse:
    """
    Perform OCR on multiple images and merge the extracted text.
    This function is used in the 'Multiple image flow' in Figma.

    Args:
        images_bytes: List of image bytes

    Returns:
        OCRResponse containing extracted pages and merged text
    """
    if not images_bytes:
        raise ValueError("At least one image must be provided.")

    if len(images_bytes) > 20:
        raise ValueError("A maximum of 20 images can be processed at once.")

    extracted_pages = []
    failed_pages = []

    for idx, image_bytes in enumerate(images_bytes, 1):
        try:
            page = extract_text_from_single_image(image_bytes, page_number=idx)
            extracted_pages.append(page)
        except Exception as e:
            failed_pages.append(f"Page {idx}: {str(e)}")

    if not extracted_pages:
        error_details = "; ".join(failed_pages)
        raise ValueError(
            f"No text could be extracted from any image. Details: {error_details}"
        )

    all_texts = [page.extracted_text for page in extracted_pages]
    merged_text = merge_pages_text(all_texts)

    return OCRResponse(
        success=True,
        pages=extracted_pages,
        merged_text=merged_text,
        total_pages=len(extracted_pages)
    )


def extract_text_from_single_image_endpoint(image_bytes: bytes) -> OCRResponse:
    """
    Wrapper function for the single image OCR endpoint.
    Used in the 'Single' mode of the Flutter app.
    """
    page = extract_text_from_single_image(image_bytes, page_number=1)

    return OCRResponse(
        success=True,
        pages=[page],
        merged_text=page.extracted_text,
        total_pages=1
    )