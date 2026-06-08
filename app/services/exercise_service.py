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
    """
    worksheet_signals = [
        r'\d+[\.\)]\s+.{3,}_{4,}',
        r'_{5,}',
        r'\b(fill\s+in\s+the\s+blank|exercise\s*\d+|section\s+[a-z])\b',
        r'\(\s*[a-z]{2,15}\s*\)',
        r'\b[A-D]\)\s+\w+',
        r'\(\d+\)\s+[a-z]+',
        r'I\s+\(\d+\)',
        r'\(\d+\)\s+[a-zA-Z]+ed\b',
    ]

    match_count = sum(
        1 for pattern in worksheet_signals
        if re.search(pattern, text, re.IGNORECASE)
    )

    if match_count >= 2:
        logger.info(f"[ContentDetection] Worksheet signals: {match_count} → WORKSHEET_EXAM_PAPER")
        return ContentType.WORKSHEET_EXAM_PAPER

    logger.info(f"[ContentDetection] Worksheet signals: {match_count} → READING_PASSAGE")
    return ContentType.READING_PASSAGE


# ================================================================
# HELPER 2 — SPLIT TEXT INTO EXERCISE SECTIONS
# ================================================================

# def _split_text_by_sections(text: str) -> List[Dict]:
#     """
#     Split merged OCR text into individual exercise sections.
#     Each section has its own grammar topic.

#     Handles formats like:
#     - "Prepositions\\n\\nFill in the blanks..."
#     - "II. Fill in the blanks with the appropriate part of speech.\\n\\n"
#     - "Exercise 1\\n\\nFill in the blanks with correct articles..."

#     Returns list of dicts:
#     [
#         {"title": "Prepositions", "text": "..."},
#         {"title": "II. Fill in the blanks...", "text": "..."},
#         {"title": "Exercise 1", "text": "..."},
#     ]
#     """

#     # Header patterns — a line is a section header if it matches one of these
#     header_patterns = [
#         # Priority 1: Topic name alone on a line (MOST IMPORTANT)
#         r'^(?:Prepositions?|Articles?|Tenses?|Vocabulary|Grammar|'
#         r'Parts?\s+of\s+Speech|Punctuation|Comprehension)\s*$',

#         # Priority 2: "Exercise 1", "Exercise 2" alone on a line
#         r'^Exercise\s*\d+\b.*$',

#         # Priority 3: Roman numeral headers "II.", "III."
#         r'^[IVX]+\.\s+.+$',

#         # Priority 4: "Fill in the blanks with X" — ONLY if short (< 60 chars)
#         # and no topic name found above
#         r'^Fill\s+in\s+the\s+blank[s]?\s+with\s+.{3,50}[.:]?\s*$',
#     ]

#     combined_pattern = '|'.join(f'(?:{p})' for p in header_patterns)

#     # Split text into lines and find header line indices
#     lines = text.split('\n')
#     header_indices = []

#     for i, line in enumerate(lines):
#         stripped = line.strip()
#         if stripped and len(stripped) < 100:
#             if re.match(combined_pattern, stripped, re.IGNORECASE):
#                 header_indices.append((i, stripped))
#                 print(f"  HEADER FOUND at line {i}: '{stripped}'")  # debug

#     logger.info(
#         f"[SectionSplit] Found {len(header_indices)} potential headers: "
#         f"{[h[1][:40] for h in header_indices]}"
#     )

#     # After finding all headers, remove duplicates that are too close
#     # (within 3 lines of each other — keep the first one)
#     filtered_headers = []


#     for idx, title in header_indices:
#         if not filtered_headers or idx - filtered_headers[-1][0] > 3:
#             filtered_headers.append((idx, title))
#     header_indices = filtered_headers

#     # Need at least 2 headers to split into multiple sections
#     if len(header_indices) < 2:
#         return [{"title": "Main", "text": text}]

#     # Add sentinel at end
#     header_indices.append((len(lines), "End"))

#     # Extract section texts using header positions
#     sections = []
#     for i in range(len(header_indices) - 1):
#         start_line = header_indices[i][0]
#         end_line = header_indices[i + 1][0]
#         title = header_indices[i][1]

#         section_lines = lines[start_line:end_line]
#         section_text = '\n'.join(section_lines).strip()

#         # Only keep sections with meaningful content
#         if len(section_text) > 50:
#             sections.append({
#                 "title": title,
#                 "text": section_text
#             })

#     if len(sections) <= 1:
#         return [{"title": "Main", "text": text}]

#     logger.info(
#         f"[SectionSplit] Split into {len(sections)} sections: "
#         f"{[s['title'][:40] for s in sections]}"
#     )
#     return sections



def _split_text_by_sections(text: str) -> List[Dict]:
    """
    Split merged OCR text into individual exercise sections.
    Uses a two-pass approach:
    Pass 1: Find exact topic name headers (highest priority)
    Pass 2: Find other structural headers
    """

    lines = text.split('\n')
    print(f"Total lines: {len(lines)}")
    for i, line in enumerate(lines):
        if 'exercise' in line.lower() or 'Exercise' in line:
            print(f"Line {i}: repr={repr(line)}")
    header_indices = []

    # ================================================================
    # PASS 1 — Exact topic name on its own line (HIGHEST PRIORITY)
    # These are single words/phrases that appear alone on a line
    # e.g. "Prepositions", "Articles", "Exercise 1"
    # ================================================================
    exact_topics = [
        'prepositions', 'preposition',
        'articles', 'article',
        'tenses', 'tense',
        'vocabulary',
        'grammar',
        'parts of speech',
        'punctuation',
        'comprehension',
    ]

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Check exact topic names
        if stripped.lower() in exact_topics:
            header_indices.append((i, stripped))
            print(f"  HEADER FOUND (exact topic) at line {i}: '{stripped}'")
            continue

        # Check "Exercise N" pattern — alone on a line or with short subtitle
        if re.match(r'^Exercise\s*\d+\s*$', stripped, re.IGNORECASE):
            header_indices.append((i, stripped))
            print(f"  HEADER FOUND (exercise) at line {i}: '{stripped}'")
            continue

        # Check Roman numeral headers: "II.", "III.", "IV." etc.
        if re.match(r'^[IVX]+\.\s+.{5,80}$', stripped, re.IGNORECASE):
            header_indices.append((i, stripped))
            print(f"  HEADER FOUND (roman) at line {i}: '{stripped}'")
            continue

    # ================================================================
    # PASS 2 — If fewer than 2 headers found, try broader patterns
    # ================================================================
    if len(header_indices) < 2:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 100:
                continue

            # Avoid duplicating already found headers
            if any(hi[0] == i for hi in header_indices):
                continue

            # "Fill in the blanks with X" — short subtitle lines
            if re.match(
                r'^Fill\s+in\s+the\s+blank[s]?\s+with\s+.{3,60}[.:]?\s*$',
                stripped, re.IGNORECASE
            ):
                header_indices.append((i, stripped))
                print(f"  HEADER FOUND (fill-in subtitle) at line {i}: '{stripped}'")

    # Sort by line position
    header_indices.sort(key=lambda x: x[0])

    # ================================================================
    # DEDUPLICATION — Remove headers that are too close to each other
    # Keep the SHORTER, more specific one (topic name beats subtitle)
    # ================================================================
    filtered_headers = []
    for idx, title in header_indices:
        if not filtered_headers:
            filtered_headers.append((idx, title))
            continue

        prev_idx, prev_title = filtered_headers[-1]
        if idx - prev_idx <= 3:
            # Too close — keep the shorter, more specific title
            if len(title) < len(prev_title):
                filtered_headers[-1] = (idx, title)
            # else keep the previous one
        else:
            filtered_headers.append((idx, title))

    header_indices = filtered_headers

    logger.info(
        f"[SectionSplit] Final headers ({len(header_indices)}): "
        f"{[h[1][:40] for h in header_indices]}"
    )

    # Need at least 2 headers to split
    if len(header_indices) < 2:
        return [{"title": "Main", "text": text}]

    # Add sentinel
    header_indices.append((len(lines), "End"))

    # Extract sections
    sections = []
    for i in range(len(header_indices) - 1):
        start_line = header_indices[i][0]
        end_line = header_indices[i + 1][0]
        title = header_indices[i][1]

        section_text = '\n'.join(lines[start_line:end_line]).strip()

        if len(section_text) > 50:
            sections.append({"title": title, "text": section_text})

    if len(sections) <= 1:
        return [{"title": "Main", "text": text}]

    logger.info(
        f"[SectionSplit] Split into {len(sections)} sections: "
        f"{[s['title'][:40] for s in sections]}"
    )
    return sections



# ================================================================
# HELPER 3 — PARSE LLM RESPONSE
# ================================================================

def _parse_questions_from_llm_response(llm_response: dict) -> List[Question]:
    """Convert DeepSeek JSON response into validated Question objects."""
    raw_questions = llm_response.get("questions", [])

    if not raw_questions:
        raise ValueError("LLM did not generate any questions.")

    parsed_questions = []

    for i, raw_q in enumerate(raw_questions):
        try:
            q_type_str = raw_q.get("question_type", "").lower().strip()
            try:
                question_type = QuestionType(q_type_str)
            except ValueError:
                logger.warning(f"[Parse] Skipping Q{i+1}: unknown type '{q_type_str}'")
                continue

            answer_raw = raw_q.get("answer")
            if not answer_raw:
                logger.warning(f"[Parse] Skipping Q{i+1}: answer is null or empty")
                continue
            answer = str(answer_raw).strip()

            options = None
            if question_type == QuestionType.MCQ:
                raw_options = raw_q.get("options")
                if not raw_options:
                    logger.warning(f"[Parse] Skipping MCQ Q{i+1}: options missing")
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
                    logger.warning(
                        f"[Parse] Skipping MCQ Q{i+1}: "
                        f"expected 4 options, got {len(options)}"
                    )
                    continue

            question_text = raw_q.get("question_text", "").strip()
            if not question_text:
                logger.warning(f"[Parse] Skipping Q{i+1}: question_text is empty")
                continue

            question = Question(
                question_number=raw_q.get("question_number", i + 1),
                question_type=question_type,
                question_text=question_text,
                options=options,
                answer=answer
            )
            parsed_questions.append(question)

        except Exception as e:
            logger.warning(f"[Parse] Skipping Q{i+1} due to error: {e}")
            continue

    if not parsed_questions:
        raise ValueError("No valid questions could be parsed from the LLM response.")

    return parsed_questions


# ================================================================
# HELPER 4 — QUESTION TYPE BREAKDOWN
# ================================================================

def _calculate_question_breakdown(questions: List[Question]) -> Dict[str, int]:
    """Count questions per type for the breakdown summary."""
    type_counts = Counter(q.question_type.value for q in questions)
    return dict(type_counts)


# ================================================================
# HELPER 5 — MCQ ANSWER DISTRIBUTION VALIDATION
# ================================================================

def _validate_mcq_answer_distribution(questions: List[Question]) -> bool:
    """
    Check if MCQ answers are distributed across A, B, C, D.
    Only enforced when there are 4 or more MCQ questions.
    """
    mcq_answers = [
        q.answer.upper() for q in questions
        if q.question_type == QuestionType.MCQ
    ]

    if len(mcq_answers) < 4:
        return True

    for label in ["A", "B", "C", "D"]:
        if label not in mcq_answers:
            logger.warning(
                f"[AnswerDist] Label '{label}' missing. "
                f"Distribution: {Counter(mcq_answers)}"
            )
            return False

    logger.info(f"[AnswerDist] OK: {Counter(mcq_answers)}")
    return True


# ================================================================
# HELPER 6 — PARAGRAPH E COVERAGE VALIDATION
# ================================================================

def _enforce_paragraph_e_coverage(
    questions: List[Question],
    text: str
) -> bool:
    """
    Check if any question covers Paragraph E content (for reading passages).
    Returns True if text has no Paragraph E, or if at least one question covers it.
    """
    para_e_keywords = [
        "isotope", "müller", "muller", "valle isarco",
        "val senales", "mica", "south tyrol", "bolzano",
        "bressanone", "wolfgang"
    ]

    text_lower = text.lower()
    text_has_para_e = any(kw in text_lower for kw in para_e_keywords)

    # If text doesn't have Paragraph E content — no need to check
    if not text_has_para_e:
        return True

    # Check if any question covers Paragraph E
    for q in questions:
        q_text = q.question_text.lower()
        q_answer = q.answer.lower()
        for keyword in para_e_keywords:
            if keyword in q_text or keyword in q_answer:
                logger.info(f"[ParaE] Coverage found: '{keyword}'")
                return True

    return False


# ================================================================
# MAIN SERVICE FUNCTION
# ================================================================

async def generate_exercises(request: GenerateExerciseRequest) -> GenerateExerciseResponse:
    """
    Generate exercises based on the user's request.

    Key improvements:
    1. Calls LLM separately for EACH question type
    2. Splits merged multi-topic text into sections
    3. Calls LLM separately for EACH section when multiple topics detected
    4. Validates MCQ answer distribution and balance
    5. Validates Paragraph E coverage for reading passages
    """

    # ---- Step 1: Auto-detect content type ----
    detected_type = _detect_content_type(request.extracted_text)

    if detected_type != request.content_type:
        logger.info(
            f"[ContentType] Overriding '{request.content_type.value}' "
            f"to '{detected_type.value}'"
        )
        request.content_type = detected_type

    # ---- Step 2: Use full text (no chunking for section detection) ----
    text_chunks = split_text_into_chunks(request.extracted_text, max_chars=3000)

    if len(text_chunks) > 1:
        logger.info(
            f"[Chunking] Text has {len(text_chunks)} chunks. "
            f"Using FULL text for section detection."
        )

    # Always use full text so all sections are detected
    # Each section will be sent to LLM separately anyway
    text_to_use = request.extracted_text
    

    # ---- Step 2b: Split into sections if multiple topics detected ----
    sections = _split_text_by_sections(text_to_use)
    print(f"SECTIONS DETECTED: {len(sections)}")
    for s in sections:
        print(f"  - '{s['title']}': first 50 chars = '{s['text'][:50]}'")

    if len(sections) > 1:
        logger.info(
            f"[Sections] Detected {len(sections)} sections: "
            f"{[s['title'] for s in sections]}"
        )
    else:
        logger.info("[Sections] Single section detected.")

    # ---- Step 3: Calculate per-type targets ----
    requested_types = request.question_types
    num_types = len(requested_types)
    questions_per_type = request.number_of_questions // num_types
    remainder = request.number_of_questions % num_types

    type_targets: Dict[QuestionType, int] = {}
    for i, qt in enumerate(requested_types):
        type_targets[qt] = questions_per_type + (1 if i < remainder else 0)

    logger.info(
        f"[Generate] content_type={request.content_type.value} | "
        f"targets={type_targets} | "
        f"total={request.number_of_questions}"
    )

    # ---- Step 4: Call LLM with retry logic ----
    MAX_RETRIES = 2
    all_questions: List[Question] = []

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"[LLM] Attempt {attempt}/{MAX_RETRIES}")
        attempt_questions: List[Question] = []

        for qt in requested_types:
            target_count = type_targets[qt]
            all_type_questions: List[Question] = []

            if len(sections) > 1:
                # ============================================================
                # MULTI-SECTION MODE:
                # Call LLM separately for each section
                # ============================================================
                per_section = target_count // len(sections)
                sec_remainder = target_count % len(sections)

                for sec_idx, section in enumerate(sections):
                    sec_count = per_section + (1 if sec_idx < sec_remainder else 0)
                    if sec_count == 0:
                        sec_count = 1

                    logger.info(
                        f"[LLM] Section '{section['title'][:30]}': "
                        f"requesting {sec_count * 3} '{qt.value}' "
                        f"(need {sec_count})"
                    )

                    llm_response = await generate_questions_with_deepseek(
                        text=section["text"],
                        question_types=[qt],
                        number_of_questions=sec_count * 3,
                        content_type=request.content_type
                    )

                    parsed = _parse_questions_from_llm_response(llm_response)
                    filtered = [q for q in parsed if q.question_type == qt]

                    logger.info(
                        f"[LLM] Section '{section['title'][:30]}': "
                        f"got {len(filtered)} valid '{qt.value}'"
                    )

                    all_type_questions.extend(filtered[:sec_count])

            else:
                # ============================================================
                # SINGLE SECTION MODE:
                # Call LLM once for the entire text
                # ============================================================
                request_count = target_count * 3

                logger.info(
                    f"[LLM] Requesting {request_count} '{qt.value}' "
                    f"(need {target_count})"
                )

                llm_response = await generate_questions_with_deepseek(
                    text=text_to_use,
                    question_types=[qt],
                    number_of_questions=request_count,
                    content_type=request.content_type
                )

                parsed = _parse_questions_from_llm_response(llm_response)
                filtered = [q for q in parsed if q.question_type == qt]

                logger.info(
                    f"[LLM] Got {len(filtered)} valid '{qt.value}' "
                    f"(parsed {len(parsed)}, requested {request_count})"
                )

                all_type_questions.extend(filtered[:target_count])

            attempt_questions.extend(all_type_questions[:target_count])

        # ---- Step 5: Validate MCQ answer distribution ----
        answer_dist_ok = _validate_mcq_answer_distribution(attempt_questions)

        # ---- Step 5b: Validate MCQ answer balance (no label > 3 times) ----
        mcq_answers = [
            q.answer.upper() for q in attempt_questions
            if q.question_type == QuestionType.MCQ
        ]
        answer_balanced = not any(
            mcq_answers.count(label) > 3
            for label in ["A", "B", "C", "D"]
        )
        if not answer_balanced:
            logger.warning(
                f"[AnswerBalance] Unbalanced: {Counter(mcq_answers)}. "
                f"{'Retrying...' if attempt < MAX_RETRIES else 'Max retries reached.'}"
            )

        # ---- Step 5c: Validate Paragraph E coverage ----
        para_e_ok = _enforce_paragraph_e_coverage(attempt_questions, text_to_use)
        if not para_e_ok:
            logger.warning(
                f"[ParaE] No Paragraph E coverage. "
                f"{'Retrying...' if attempt < MAX_RETRIES else 'Max retries reached.'}"
            )

        # ---- Step 6: Check if we have enough of each type ----
        type_counts = Counter(q.question_type for q in attempt_questions)
        enough = all(
            type_counts.get(qt, 0) >= type_targets[qt]
            for qt in requested_types
        )

        if enough and answer_dist_ok and answer_balanced and para_e_ok:
            all_questions = attempt_questions
            logger.info(
                f"[LLM] Attempt {attempt} succeeded: "
                f"{len(all_questions)} questions. "
                f"Breakdown: {Counter(q.question_type.value for q in all_questions)}"
            )
            break
        else:
            if not enough:
                missing = {
                    qt.value: type_targets[qt] - type_counts.get(qt, 0)
                    for qt in requested_types
                    if type_counts.get(qt, 0) < type_targets[qt]
                }
                logger.warning(
                    f"[LLM] Attempt {attempt}: Missing: {missing}. "
                    f"{'Retrying...' if attempt < MAX_RETRIES else 'Done.'}"
                )
            if not answer_dist_ok:
                logger.warning(
                    f"[LLM] Attempt {attempt}: MCQ missing labels. "
                    f"{'Retrying...' if attempt < MAX_RETRIES else 'Done.'}"
                )

            if attempt == MAX_RETRIES:
                all_questions = attempt_questions
                logger.warning(
                    f"[LLM] Max retries reached. "
                    f"Returning {len(all_questions)} questions."
                )

    # ---- Step 7: Validate at least one question exists ----
    if not all_questions:
        raise ValueError(
            f"No questions could be generated. "
            f"Requested: {[t.value for t in requested_types]}. "
            f"Please try again."
        )

    # ---- Step 8: Re-number questions sequentially ----
    for idx, question in enumerate(all_questions, 1):
        question.question_number = idx

    # ---- Step 9: Build breakdown summary ----
    breakdown = _calculate_question_breakdown(all_questions)

    logger.info(
        f"[Generate] Done. {len(all_questions)} questions. "
        f"Breakdown: {breakdown}"
    )

    return GenerateExerciseResponse(
        success=True,
        questions=all_questions,
        total_questions=len(all_questions),
        question_type_breakdown=breakdown
    )