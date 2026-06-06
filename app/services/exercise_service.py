# Smart-Exercise-Generator\app\services\exercise_service.py

import re
import logging
from typing import List, Dict
from collections import Counter

from app.models.request_models import GenerateExerciseRequest, QuestionType, ContentType
from app.models.response_models import GenerateExerciseResponse, Question, MCQOption
from app.services.llm_service import generate_questions_with_deepseek
from app.utils.text_utils import split_text_into_chunks

logger = logging.getLogger(__name__)


# ================================================================
# HELPER 1 — AUTO CONTENT TYPE DETECTION
# ================================================================

def _detect_content_type(text: str) -> ContentType:
    """
    Automatically detect whether the input text is a worksheet/exam paper
    or a reading passage, based on structural patterns in the text.

    This overrides the user-provided content_type if worksheet signals are found.
    This is necessary because:
      - Users may select the wrong content type
      - The app may default to reading_passage even for worksheets
      - Real-world photos often contain full worksheets with existing questions

    Returns:
        ContentType.WORKSHEET_EXAM_PAPER  — if worksheet patterns are detected
        ContentType.READING_PASSAGE       — otherwise
    """
    worksheet_signals = [
        # Numbered sentences with blanks: "1. She ________ to school."
        r'\d+[\.\)]\s+.{3,}_{4,}',
        # 5 or more underscores used as a blank
        r'_{5,}',
        # Exercise or section headers
        r'\b(fill\s+in\s+the\s+blank|exercise\s*\d+|section\s+[a-z])\b',
        # Verb hints in brackets: "(go)", "(write)", "(be)"
        r'\(\s*[a-z]{2,15}\s*\)',
        # MCQ option lines: "A) walk  B) walked"
        r'\b[A-D]\)\s+\w+',
    ]

    match_count = sum(
        1 for pattern in worksheet_signals
        if re.search(pattern, text, re.IGNORECASE)
    )

    # 2 or more signals → treat as worksheet
    if match_count >= 2:
        logger.info(f"[ContentDetection] Worksheet signals found: {match_count}. Treating as WORKSHEET_EXAM_PAPER.")
        return ContentType.WORKSHEET_EXAM_PAPER

    logger.info(f"[ContentDetection] Worksheet signals found: {match_count}. Treating as READING_PASSAGE.")
    return ContentType.READING_PASSAGE


# ================================================================
# HELPER 2 — PARSE LLM RESPONSE INTO QUESTION OBJECTS
# ================================================================

def _parse_questions_from_llm_response(llm_response: dict) -> List[Question]:
    """
    Convert DeepSeek JSON response into validated Question objects.

    Skips malformed questions silently and raises only if
    no valid questions could be parsed at all.
    """
    raw_questions = llm_response.get("questions", [])

    if not raw_questions:
        raise ValueError("LLM did not generate any questions.")

    parsed_questions = []

    for i, raw_q in enumerate(raw_questions):
        try:
            # --- Validate question type ---
            q_type_str = raw_q.get("question_type", "").lower().strip()
            try:
                question_type = QuestionType(q_type_str)
            except ValueError:
                logger.warning(f"[Parse] Skipping question {i+1}: unknown type '{q_type_str}'")
                continue

            # --- Validate answer ---
            answer_raw = raw_q.get("answer")
            if not answer_raw:
                logger.warning(f"[Parse] Skipping question {i+1}: answer is null or empty")
                continue
            answer = str(answer_raw).strip()

            # --- Parse MCQ options ---
            options = None
            if question_type == QuestionType.MCQ:
                raw_options = raw_q.get("options")
                if not raw_options:
                    logger.warning(f"[Parse] Skipping MCQ question {i+1}: options missing")
                    continue

                options = [
                    MCQOption(
                        label=opt.get("label", "").strip(),
                        text=opt.get("text", "").strip()
                    )
                    for opt in raw_options
                    if opt.get("label") and opt.get("text")
                ]

                if len(options) != 4:
                    logger.warning(f"[Parse] Skipping MCQ question {i+1}: expected 4 options, got {len(options)}")
                    continue

            # --- Validate question text ---
            question_text = raw_q.get("question_text", "").strip()
            if not question_text:
                logger.warning(f"[Parse] Skipping question {i+1}: question_text is empty")
                continue

            # --- Build Question object ---
            question = Question(
                question_number=raw_q.get("question_number", i + 1),
                question_type=question_type,
                question_text=question_text,
                options=options,
                answer=answer
            )
            parsed_questions.append(question)

        except Exception as e:
            logger.warning(f"[Parse] Skipping question {i+1} due to unexpected error: {e}")
            continue

    if not parsed_questions:
        raise ValueError("No valid questions could be parsed from the LLM response.")

    return parsed_questions


# ================================================================
# HELPER 3 — QUESTION TYPE BREAKDOWN
# ================================================================

def _calculate_question_breakdown(questions: List[Question]) -> Dict[str, int]:
    """
    Count how many questions were generated for each question type.
    Used in the 'Generated Exercises' screen to show the breakdown summary.
    """
    type_counts = Counter(q.question_type.value for q in questions)
    return dict(type_counts)


# ================================================================
# MAIN SERVICE FUNCTION
# ================================================================

async def generate_exercises(request: GenerateExerciseRequest) -> GenerateExerciseResponse:
    """
    Generate exercises based on the user's request.

    Flow:
    1. Auto-detect content type from text (overrides user-provided type if needed)
    2. Split text into chunks if too long
    3. Send to DeepSeek LLM for question generation
    4. Parse and validate the LLM response
    5. Return structured questions

    Figma flow:
    - 'Select Exercise Type' screen → sends question types and count
    - This function calls the LLM → generates questions
    - Results displayed in 'Generated Exercises' screen

    Args:
        request: GenerateExerciseRequest — contains text, question types, count, content_type

    Returns:
        GenerateExerciseResponse — contains validated, structured questions
    """

    # ---- Step 1: Auto-detect and override content type if needed ----
    detected_type = _detect_content_type(request.extracted_text)

    if detected_type != request.content_type:
        logger.info(
            f"[ContentType] Overriding user-provided '{request.content_type.value}' "
            f"→ auto-detected '{detected_type.value}'"
        )
        request.content_type = detected_type

    # ---- Step 2: Split text into chunks if needed ----
    text_chunks = split_text_into_chunks(request.extracted_text, max_chars=3000)

    logger.info(
        f"[Generate] content_type={request.content_type.value} | "
        f"question_types={[qt.value for qt in request.question_types]} | "
        f"number_of_questions={request.number_of_questions} | "
        f"chunks={len(text_chunks)}"
    )

    # ---- Step 3: Call LLM ----
    # Note: If text has multiple chunks, we use only the first chunk.
    # This ensures the LLM focuses on a single coherent section.
    # Future improvement: merge chunk results intelligently.
    text_to_use = request.extracted_text if len(text_chunks) == 1 else text_chunks[0]

    if len(text_chunks) > 1:
        logger.info(f"[Chunking] Text split into {len(text_chunks)} chunks. Using chunk 1 only.")

    llm_response = await generate_questions_with_deepseek(
        text=text_to_use,
        question_types=request.question_types,
        number_of_questions=request.number_of_questions,
        content_type=request.content_type
    )

    # ---- Step 4: Parse LLM response ----
    all_questions = _parse_questions_from_llm_response(llm_response)

    # ---- Step 5: Re-number questions sequentially ----
    for idx, question in enumerate(all_questions, 1):
        question.question_number = idx

    # ---- Step 6: Build breakdown summary ----
    breakdown = _calculate_question_breakdown(all_questions)

    logger.info(f"[Generate] Successfully generated {len(all_questions)} questions. Breakdown: {breakdown}")

    return GenerateExerciseResponse(
        success=True,
        questions=all_questions,
        total_questions=len(all_questions),
        question_type_breakdown=breakdown
    )