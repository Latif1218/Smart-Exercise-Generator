from typing import List, Dict
from collections import Counter

from app.models.request_models import GenerateExerciseRequest, QuestionType, ContentType
from app.models.response_models import GenerateExerciseResponse, Question, MCQOption
from app.services.llm_service import generate_questions_with_deepseek
from app.utils.text_utils import split_text_into_chunks


def _parse_questions_from_llm_response(llm_response: dict) -> List[Question]:
    """
    Convert DeepSeek JSON response into Question objects.
    Validate the response format and create structured objects.
    """
    raw_questions = llm_response.get("questions", [])

    if not raw_questions:
        raise ValueError("LLM did not generate any questions.")

    parsed_questions = []

    for i, raw_q in enumerate(raw_questions):
        try:
            q_type_str = raw_q.get("question_type", "").lower()
            try:
                question_type = QuestionType(q_type_str)
            except ValueError:
                continue

            options = None
            if question_type == QuestionType.MCQ and raw_q.get("options"):
                options = [
                    MCQOption(
                        label=opt.get("label", ""),
                        text=opt.get("text", "")
                    )
                    for opt in raw_q["options"]
                    if opt.get("label") and opt.get("text")
                ]

                if len(options) != 4:
                    continue

            question = Question(
                question_number=raw_q.get("question_number", i + 1),
                question_type=question_type,
                question_text=raw_q.get("question_text", ""),
                options=options,
                answer=str(raw_q.get("answer", ""))
            )
            parsed_questions.append(question)

        except Exception:
            continue

    if not parsed_questions:
        raise ValueError("No questions could be parsed successfully.")

    return parsed_questions


def _calculate_question_breakdown(questions: List[Question]) -> Dict[str, int]:
    """
    Count how many questions were generated for each question type.
    This breakdown can be displayed in the 'Generated Exercises' screen in Figma.
    """
    type_counts = Counter(q.question_type.value for q in questions)
    return dict(type_counts)


async def generate_exercises(request: GenerateExerciseRequest) -> GenerateExerciseResponse:
    """
    Generate exercises based on the user's request.
    This function sends OCR text to DeepSeek and returns structured questions.

    Figma flow:
    1. 'Select Exercise Type' screen sends question types and count
    2. This function calls the LLM to generate questions
    3. Results are displayed in the 'Generated Exercises' screen

    Args:
        request: GenerateExerciseRequest — contains text, question types, and count

    Returns:
        GenerateExerciseResponse — contains generated questions
    """
    text_chunks = split_text_into_chunks(request.extracted_text, max_chars=3000)

    all_questions = []

    if len(text_chunks) == 1:
        llm_response = await generate_questions_with_deepseek(
            text=request.extracted_text,
            question_types=request.question_types,
            number_of_questions=request.number_of_questions,
            content_type=request.content_type
        )
        all_questions = _parse_questions_from_llm_response(llm_response)
    else:
        llm_response = await generate_questions_with_deepseek(
            text=text_chunks[0],
            question_types=request.question_types,
            number_of_questions=request.number_of_questions,
            content_type=request.content_type
        )
        all_questions = _parse_questions_from_llm_response(llm_response)

    for idx, question in enumerate(all_questions, 1):
        question.question_number = idx

    breakdown = _calculate_question_breakdown(all_questions)

    return GenerateExerciseResponse(
        success=True,
        questions=all_questions,
        total_questions=len(all_questions),
        question_type_breakdown=breakdown
    )