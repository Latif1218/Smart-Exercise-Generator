from fastapi import APIRouter, HTTPException, status

from app.models.request_models import GenerateExerciseRequest
from app.models.response_models import GenerateExerciseResponse
from app.services.exercise_service import generate_exercises

router = APIRouter(
    prefix="/exercise",
    tags=["Exercise — AI Question Generation"]
)


@router.post(
    "/generate",
    response_model=GenerateExerciseResponse,
    summary="Generate Exercises from Text",
    description="""
    Generate practice exercises from OCR-extracted text using AI.
    
    Figma flow:
    1. Get extracted text from the OCR endpoint
    2. User selects question types on the 'Select Exercise Type' screen
    3. Call this endpoint — DeepSeek LLM generates questions
    4. Display results on the 'Generated Exercises' screen
    
    Supported question types:
    - **MCQ**: 4-option multiple choice questions
    - **Fill in the Blank**: Key words replaced with blanks
    - **Short Answer**: Questions requiring brief explanations
    
    Content types:
    - **reading_passage**: Generate comprehension questions from general text
    - **worksheet_exam_paper**: Generate new exercises based on an existing exam paper
    """
)
async def generate_exercise(
    request: GenerateExerciseRequest
) -> GenerateExerciseResponse:
    """
    Generate exercises from OCR-extracted text.

    The request body must include extracted_text, question_types, and number_of_questions.
    """
    if not request.question_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one question type must be selected."
        )

    if len(request.extracted_text.strip()) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is too short. Minimum 50 characters are required."
        )

    try:
        result = await generate_exercises(request)
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=str(e)
        )

    except Exception as e:
        error_msg = str(e)

        if "API key" in error_msg or "401" in error_msg or "403" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DeepSeek API key issue. A valid API key must be configured in Phase 2."
            )

        if "timeout" in error_msg.lower() or "connect" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="DeepSeek API timeout. Please check your internet connection or try again later."
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occurred during exercise generation: {error_msg}"
        )