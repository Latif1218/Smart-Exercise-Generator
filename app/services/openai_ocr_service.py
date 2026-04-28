import base64
from openai import OpenAI
from typing import List

from app.config import settings
from app.models.response_models import ExtractedPage, OCRResponse


class OpenAIOCRService:
    def __init__(self):
        self.api_key = getattr(settings, 'openai_api_key', None)
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env file!")

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"         

    def _encode_image(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64"""
        return base64.b64encode(image_bytes).decode("utf-8")

    def extract_text_from_single_image(self, image_bytes: bytes, page_number: int = 1) -> ExtractedPage:
        """Extract clean readable text from image using GPT-4o Vision"""
        
        base64_image = self._encode_image(image_bytes)

        system_prompt = """You are an expert OCR and text formatter for textbooks and academic materials.

Extract ALL text from the given image as accurately and naturally as possible.
- Automatically fix broken words and sentences
- Join text that continues across lines
- Add proper punctuation and create well-structured paragraphs
- Handle English and Bengali text correctly
- Preserve headings, numbers, formulas, and bullet points
- Make the final output clean, readable, and well-formatted

Return ONLY the extracted and formatted text. No explanations or extra comments."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please extract and format all readable text from this textbook page:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=8192
            )

            extracted_text = response.choices[0].message.content.strip()

            if not extracted_text or extracted_text.lower() in ["no text", "empty"]:
                extracted_text = "No readable text could be extracted from the image."

            return ExtractedPage(
                page_number=page_number,
                extracted_text=extracted_text
            )

        except Exception as e:
            print(f"OpenAI Vision Error: {e}")
            raise Exception(f"OpenAI Vision OCR failed: {str(e)}")


# Global instance
openai_ocr_service = OpenAIOCRService()