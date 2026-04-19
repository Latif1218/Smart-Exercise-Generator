from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class QuestionType(str, Enum):
    """Type of questions — the three options shown in Figma"""

    MCQ = "mcq"
    FILL_IN_THE_BLANK = "fill_in_the_blank"
    SHORT_ANSWER = "short_answer"


class ContentType(str, Enum):
    """Options available on the 'Content Type Selection' screen in Figma"""
    READING_PASSAGE = "reading_passage"    
    WORKSHEET_EXAM_PAPER = "worksheet_exam_paper"  


class GenerateExerciseRequest(BaseModel):
    """
    Request body for generating exercises from OCR-extracted text.
    Data will be sent in this format from the Flutter app.
    """
    extracted_text: str = Field(
        ...,
        description="Text extracted from the image using OCR",
        min_length=10
    )
    content_type: ContentType = Field(
        default=ContentType.READING_PASSAGE,
        description="Type of content — reading passage or exam paper"
    )
    question_types: List[QuestionType] = Field(
        ...,
        description="List of question types to generate",
        min_length=1
    )
    number_of_questions: int = Field(
        default=10,
        description="Total number of questions to generate",
        ge=1,
        le=50
    )