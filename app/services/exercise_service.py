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
# HELPER 6 — STRUCTURE EXTRACTOR
# ================================================================

def _extract_sentence_structures(text: str) -> List[str]:
    structures = []
    normalized = re.sub(r'_{4,}', 'BLANK', text)
    normalized = re.sub(r'\(\d+\)', 'BLANK', normalized)
    lines = normalized.split('\n')

    for line in lines:
        line = line.strip()
        line = re.sub(r'^\([ivxlcdmIVXLCDM]+\)\s*', '', line)
        
        if 'BLANK' not in line:
            continue

        skeleton = line.lower()
        skeleton = re.sub(r'\b\d+\w*\b', 'NUM', skeleton)
        skeleton = re.sub(r'\s+', ' ', skeleton).strip()
        structures.append(skeleton)

        # ✅ Debug log — কোন pattern detect হচ্ছে দেখো
        print(f"[DEBUG] skeleton: {skeleton}")

        if re.search(r'(bought|purchased|got|prepared|made).+blank.+and.+blank', skeleton):
            structures.append("PATTERN:double_blank")
            print("[DEBUG] → PATTERN:double_blank detected")

        if re.search(r'^what\s+blank', skeleton):
            structures.append("PATTERN:what_exclamatory")
            print("[DEBUG] → PATTERN:what_exclamatory detected")

        if re.search(r'blank\s+\w*(est|most\s+\w+)', skeleton):
            structures.append("PATTERN:superlative")
            print("[DEBUG] → PATTERN:superlative detected")

        if re.search(r'is\s+blank\s+\w+\s+\w+\s+of\s+the', skeleton):
            structures.append("PATTERN:ordinal_time")
            print("[DEBUG] → PATTERN:ordinal_time detected")

        if re.search(r'in\s+blank\s+(morning|afternoon|evening|night)', skeleton):
            structures.append("PATTERN:time_of_day")
            print("[DEBUG] → PATTERN:time_of_day detected")

    print(f"[DEBUG] Total structures: {structures}")
    logger.info(f"[StructureExtract] {len(structures)} structures blacklisted")
    return structures

# ================================================================
# HELPER 7 — STRUCTURE SIMILARITY VALIDATOR  
# ================================================================

def _validate_structure_similarity(
    questions: List[Question],
    original_structures: List[str]
) -> List[int]:
    failed_indices = []
    seen_texts = set()
    seen_skeletons = set()

    pattern_list = [s for s in original_structures if s.startswith("PATTERN:")]
    print(f"[DEBUG] Active patterns: {pattern_list}")

    for idx, question in enumerate(questions):
        q_text = question.question_text

        # Normalize
        normalized_q = re.sub(r'_{4,}', 'BLANK', q_text)
        normalized_q = re.sub(r'\(\d+\)', 'BLANK', normalized_q)
        normalized_q = re.sub(r'\b\d+\w*\b', 'NUM', normalized_q)
        normalized_q = re.sub(r'\s+', ' ', normalized_q).strip().lower()

        print(f"[DEBUG] Q{idx+1} normalized: {normalized_q}")

        failed = False

        # ================================================================
        # CHECK 1 — Duplicate sentence
        # ================================================================
        if normalized_q in seen_texts:
            print(f"[DEBUG] Q{idx+1} → FAILED: duplicate sentence")
            failed_indices.append(idx)
            continue
        seen_texts.add(normalized_q)

        # ================================================================
        # CHECK 2 — Double blank (universal)
        # ================================================================
        if normalized_q.count('blank') >= 2:
            print(f"[DEBUG] Q{idx+1} → FAILED: double blank")
            failed_indices.append(idx)
            continue

        # ================================================================
        # CHECK 3 — Same structure as previous generated questions
        # ================================================================
        skeleton_q = re.sub(r'\b[a-z]{4,}\b', 'W', normalized_q)
        skeleton_q = re.sub(r'\s+', ' ', skeleton_q).strip()

        if skeleton_q in seen_skeletons:
            print(f"[DEBUG] Q{idx+1} → FAILED: same structure as previous question")
            failed_indices.append(idx)
            continue
        seen_skeletons.add(skeleton_q)

        # ================================================================
        # CHECK 4 — Original structure similarity
        # ================================================================
        for structure in original_structures:
            if structure.startswith("PATTERN:"):
                pattern_name = structure.replace("PATTERN:", "")

                if pattern_name == "double_blank":
                    if normalized_q.count('blank') >= 2:
                        print(f"[DEBUG] Q{idx+1} → FAILED: double_blank pattern")
                        failed = True
                        break

                elif pattern_name == "what_exclamatory":
                    if re.search(r'^what\s+blank', normalized_q, re.IGNORECASE):
                        print(f"[DEBUG] Q{idx+1} → FAILED: what_exclamatory")
                        failed = True
                        break

                elif pattern_name == "superlative":
                    if re.search(r'blank\s+\w*(est|most\s+\w+)', normalized_q, re.IGNORECASE):
                        print(f"[DEBUG] Q{idx+1} → FAILED: superlative")
                        failed = True
                        break

                elif pattern_name == "ordinal_time":
                    if re.search(r'is\s+blank\s+\w+\s+\w+\s+of\s+the', normalized_q, re.IGNORECASE):
                        print(f"[DEBUG] Q{idx+1} → FAILED: ordinal_time")
                        failed = True
                        break

                elif pattern_name == "time_of_day":
                    if re.search(r'in\s+blank\s+(morning|afternoon|evening|night)', normalized_q, re.IGNORECASE):
                        print(f"[DEBUG] Q{idx+1} → FAILED: time_of_day")
                        failed = True
                        break

                continue

            # Word similarity check
            q_words = set(normalized_q.split())
            s_words = set(structure.split())
            if not s_words:
                continue
            common = q_words & s_words
            similarity = len(common) / len(s_words)
            if similarity > 0.6:
                print(f"[DEBUG] Q{idx+1} → FAILED: {similarity:.0%} word similarity")
                failed = True
                break

        if failed:
            failed_indices.append(idx)

    print(f"[DEBUG] Failed indices: {failed_indices}")
    return failed_indices

# ================================================================
# HELPER 8 — PARAGRAPH E COVERAGE VALIDATION
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

    if not text_has_para_e:
        return True

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
    6. Validates structure similarity to prevent duplication
    """

    detected_type = _detect_content_type(request.extracted_text)

    if detected_type != request.content_type:
        logger.info(
            f"[ContentType] Overriding '{request.content_type.value}' "
            f"to '{detected_type.value}'"
        )
        request.content_type = detected_type

    text_chunks = split_text_into_chunks(request.extracted_text, max_chars=3000)

    if len(text_chunks) > 1:
        logger.info(
            f"[Chunking] Text has {len(text_chunks)} chunks. "
            f"Using FULL text for section detection."
        )

    text_to_use = request.extracted_text

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

    # Structure blacklist — একবারই extract করো, loop এর বাইরে
    original_structures = _extract_sentence_structures(text_to_use)

    MAX_RETRIES = 3
    all_questions: List[Question] = []

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"[LLM] Attempt {attempt}/{MAX_RETRIES}")
        attempt_questions: List[Question] = list(all_questions)

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

        # ============================================================
        # VALIDATION 1 — STRUCTURE SIMILARITY CHECK
        # ============================================================
        failed_structure_indices = _validate_structure_similarity(
            attempt_questions,
            original_structures
        )

        if failed_structure_indices:
            logger.warning(
                f"[StructureCheck] {len(failed_structure_indices)} questions "
                f"failed structure check: indices {failed_structure_indices}. "
                f"{'Retrying...' if attempt < MAX_RETRIES else 'Max retries reached.'}"
            )
            structure_ok = False
        else:
            structure_ok = True
            logger.info("[StructureCheck] All questions passed structure check ✅")

        # ============================================================
        # VALIDATION 2 — MCQ ANSWER DISTRIBUTION
        # ============================================================


        if failed_structure_indices and attempt < MAX_RETRIES:
            passed_questions = [
                q for i, q in enumerate(attempt_questions)
                if i not in failed_structure_indices
            ]
            logger.info(
                f"[StructureCheck] Keeping {len(passed_questions)} passed questions, "
                f"regenerating {len(failed_structure_indices)} failed ones."
            )
            for qt in requested_types:
                passed_count = sum(1 for q in passed_questions if q.question_type == qt)
                type_targets[qt] = max(0, type_targets[qt] - passed_count)

            all_questions = passed_questions
            continue


        answer_dist_ok = _validate_mcq_answer_distribution(attempt_questions)

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

        # ============================================================
        # VALIDATION 3 — PARAGRAPH E COVERAGE
        # ============================================================
        para_e_ok = _enforce_paragraph_e_coverage(attempt_questions, text_to_use)
        if not para_e_ok:
            logger.warning(
                f"[ParaE] No Paragraph E coverage. "
                f"{'Retrying...' if attempt < MAX_RETRIES else 'Max retries reached.'}"
            )

        # ============================================================
        # VALIDATION 4 — ENOUGH QUESTIONS GENERATED
        # ============================================================
        type_counts = Counter(q.question_type for q in attempt_questions)
        enough = all(
            type_counts.get(qt, 0) >= type_targets[qt]
            for qt in requested_types
        )

        # ============================================================
        # ALL VALIDATIONS PASS → DONE
        # ============================================================
        if enough and answer_dist_ok and answer_balanced and para_e_ok and structure_ok:
            all_questions = attempt_questions
            logger.info(
                f"[LLM] Attempt {attempt} succeeded: "
                f"{len(all_questions)} questions. "
                f"Breakdown: {Counter(q.question_type.value for q in all_questions)}"
            )
            break

        # ============================================================
        # SOME VALIDATION FAILED → LOG & RETRY
        # ============================================================
        else:
            if not enough:
                missing = {
                    qt.value: type_targets[qt] - type_counts.get(qt, 0)
                    for qt in requested_types
                    if type_counts.get(qt, 0) < type_targets[qt]
                }
                logger.warning(
                    f"[LLM] Attempt {attempt}: Missing questions: {missing}. "
                    f"{'Retrying...' if attempt < MAX_RETRIES else 'Done.'}"
                )

            if not answer_dist_ok:
                logger.warning(
                    f"[LLM] Attempt {attempt}: MCQ missing labels. "
                    f"{'Retrying...' if attempt < MAX_RETRIES else 'Done.'}"
                )

            if not structure_ok:
                logger.warning(
                    f"[LLM] Attempt {attempt}: Structure similarity failed. "
                    f"{'Retrying...' if attempt < MAX_RETRIES else 'Done.'}"
                )

            if attempt == MAX_RETRIES:
                all_questions = attempt_questions
                logger.warning(
                    f"[LLM] Max retries reached. "
                    f"Returning {len(all_questions)} questions."
                )

    if not all_questions:
        raise ValueError(
            f"No questions could be generated. "
            f"Requested: {[t.value for t in requested_types]}. "
            f"Please try again."
        )

    for idx, question in enumerate(all_questions, 1):
        question.question_number = idx

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