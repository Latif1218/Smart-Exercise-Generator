from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.models.request_models import QuestionType


class MCQOption(BaseModel):
    """A single option for an MCQ question"""
    label: str = Field(..., description="Option label — A, B, C, D")
    text: str = Field(..., description="Content of the option")


class Question(BaseModel):
    """
    A generated question.
    Structure follows how questions are displayed in the 'Generated Exercises' screen in Figma.
    """
    question_number: int = Field(..., description="Serial number of the question")
    question_type: QuestionType = Field(..., description="Type of the question")
    question_text: str = Field(..., description="Text of the question")
    options: Optional[List[MCQOption]] = Field(
        default=None,
        description="List of options for MCQ; None for other question types"
    )
    answer: str = Field(..., description="Correct answer")


class ExtractedPage(BaseModel):
    """
    OCR result for a single image page.
    In a multiple image flow, each page is tracked separately.
    """
    page_number: int = Field(..., description="Page serial number")
    extracted_text: str = Field(..., description="Text extracted using OCR")


class OCRResponse(BaseModel):
    """
    Response for the OCR endpoint.
    Contains one page for a single image, or multiple pages for multiple images.
    """
    success: bool = Field(..., description="Indicates whether the request was successful")
    pages: List[ExtractedPage] = Field(..., description="List of extracted pages")
    merged_text: str = Field(
        ...,
        description="Combined text from all pages — displayed in the 'Merged Extracted Text' page in Figma"
    )
    total_pages: int = Field(..., description="Total number of processed pages")


class GenerateExerciseResponse(BaseModel):
    """
    Response for the exercise generation endpoint.
    Matches the structure shown in the 'Generated Exercises' screen in Figma.
    """
    success: bool = Field(..., description="Indicates whether the request was successful")
    questions: List[Question] = Field(..., description="List of generated questions")
    total_questions: int = Field(..., description="Total number of questions generated")
    question_type_breakdown: Dict[str, int] = Field(
        ...,
        description="Breakdown of question count by type"
    )


class ErrorResponse(BaseModel):
    """Standard error response format"""
    success: bool = False
    error: str = Field(..., description="Error message")
    detail: Optional[Any] = Field(default=None, description="Detailed error information")



class HelpContactResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None