import cv2 
import numpy as np                   
from PIL import Image
import io
from typing import Union

def preprocess_image_for_ocr(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess the image to improve OCR accuracy.
    Textbook photos often contain shadows or slight tilt,
    so this function reduces noise and enhances text visibility.
    """

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image. Please provide a valid image file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  
        2   
    )

    deskewed = _deskew_image(thresh)

    return deskewed

def _deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct image skew (tilt).
    Useful for images captured using a phone where alignment is not perfect.
    """
    try:
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 15:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    except Exception:
        return image
    

def validate_image(image_bytes: bytes, max_size_mb: int = 10) -> bool:
    """
    Validate the image file.
    Checks file size and verifies that the image is not corrupted.
    """
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Image size is {size_mb:.1f}MB. Maximum allowed is {max_size_mb}MB.")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  
        return True
    except Exception:
        raise ValueError("Invalid image file. Supported formats: JPEG, PNG, WebP")


def numpy_to_pil(np_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV numpy array to PIL Image.
    """
    if len(np_image.shape) == 2:
        return Image.fromarray(np_image)
    else:
        rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)