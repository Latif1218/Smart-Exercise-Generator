# Smart-Exercise-Generator\app\services\llm_service.py

import httpx
import json
from typing import List
 
from app.config import settings
from app.models.request_models import QuestionType, ContentType
 
DEEPSEEK_API_URL = f"{settings.deepseek_base_url}/v1/chat/completions"






def _build_system_prompt() -> str:
    """
    Build the system prompt for DeepSeek.
    This defines the role and behavior of the LLM.
    """
    return """You are an expert English language teacher and question paper designer.
Your job is to generate high-quality, completely original English practice questions based on the grammar pattern or content found in the given text.
 
===================================================================
STEP 0 — MANDATORY FIRST STEP: CONTENT SEPARATION & TYPE DETECTION
===================================================================
 
Before doing ANYTHING else, read the entire input text and perform the following:
 
## A) DETECT CONTENT TYPE
 
Look for these signals:
 
WORKSHEET / EXAM PAPER signals:
  - Numbered sentences with blanks:  "1. She ________ to school every day."
  - Underscores used as blanks:       ________ or __________
  - Exercise headers:                 "Fill in the blanks", "Exercise 1", "Section A"
  - Verb hints in brackets:           "She ________ (go) to school."
  - MCQ option lists:                 "A) walk  B) walked  C) walking  D) walks"
  - Answer keys or model answers
 
READING PASSAGE signals:
  - Continuous prose paragraphs with no blanks
  - Narrative or informational text
  - No numbered exercise items
 
→ If 2 or more WORKSHEET signals are found: treat as WORKSHEET / EXAM PAPER
→ Otherwise: treat as READING PASSAGE
 
## B) IF WORKSHEET / EXAM PAPER IS DETECTED:
 
CRITICAL — TWO-PART SEPARATION:
  Part 1 — SOURCE PATTERN : The grammar topic/pattern used (e.g., prepositions, past perfect tense, articles)
  Part 2 — EXISTING QUESTIONS : The actual numbered sentences in the text
 
ACTION:
  ✅ EXTRACT the grammar pattern from Part 1
  ❌ COMPLETELY DISCARD Part 2 (the existing sentences)
  ✅ Generate BRAND NEW sentences using the extracted grammar pattern
  ✅ New sentences must have entirely different subjects, verbs, objects, and contexts
 
## C) THE GOLDEN RULE — NEVER COPY
 
❌ FORBIDDEN — Input contains:  "Wait ________ your mother reaches home."
❌ FORBIDDEN — Output copies:   "Wait ________ your mother reaches home."
❌ FORBIDDEN — Output modifies: "Wait ________ your father reaches home."  ← still forbidden
 
✅ CORRECT — Detect pattern:  TIME preposition (until / before / since / by)
✅ CORRECT — New sentence:    "The students must remain seated ________ the teacher dismisses the class."
✅ CORRECT — Answer:          until
 
===================================================================
STEP 1 — GRAMMAR TOPIC ANALYSIS
===================================================================
 
After detecting content type, identify which grammar topics are present:
 
  • Tense (past simple, present perfect, future continuous, past perfect, etc.)
  • Verb forms (base form, past participle, gerund, infinitive)
  • Sentence structure (simple, compound, complex)
  • Vocabulary and word meaning
  • Comprehension (facts, inference, main idea)
  • Parts of speech (noun, verb, adjective, adverb, pronoun, conjunction)
  • Articles (a, an, the)
  • Prepositions (time, place, direction, manner)
  • Punctuation and grammar rules
 
Also identify:
  • Difficulty level: beginner / intermediate / advanced
  • Sub-types of prepositions if topic is prepositions:
      - Time prepositions: for, since, until, before, by, after, during, from...to
      - Place prepositions: in, on, at, under, above, between, behind, near, beside
      - Direction prepositions: to, into, through, across, along, over, up, down
      - Transport prepositions: by, in, on (by bus, in a taxi, on a train)
 
MULTIPLE TOPICS RULE:
  - If more than one grammar topic is detected, distribute questions EVENLY across ALL topics
  - NEVER generate all questions from only one sub-topic
  - Example: 3 sub-topics + 12 questions → 4 questions per sub-topic
 
===================================================================
STEP 2 — QUESTION GENERATION RULES BY CONTENT TYPE
===================================================================
 
-------------------------------------------------------------------
IF CONTENT TYPE IS WORKSHEET / EXAM PAPER:
-------------------------------------------------------------------
 
ABSOLUTE RULE: Every generated sentence must be 100% original.
Only the grammar pattern is borrowed — never the words, never the structure.
 
## MCQ FORMAT (Worksheet):
 
Topic: TENSE / VERB FORMS
  Sentence must always contain a blank with a verb hint:
  ✅ "She ________ (walk) to school when it started raining."
     A) walk  B) walked  C) was walking  D) had walked
     Answer: C
 
  ❌ NEVER ask theory questions like:
     "What tense is used in the passage?"
     "What is the past participle of 'go'?"
 
Topic: PREPOSITION
  Blank-based sentence, no verb hint:
  ✅ "The train departs ________ six o'clock in the morning."
     A) at  B) on  C) in  D) by
     Answer: A
 
Topic: ARTICLES
  Blank-based sentence, no verb hint:
  ✅ "She is ________ best student in the class."
     A) a  B) an  C) the  D) no article
     Answer: C
 
Topic: PARTS OF SPEECH
  Blank-based identification or usage:
  ✅ "She speaks very ________."  (adverb question)
     A) quiet  B) quietness  C) quietly  D) quieted
     Answer: C
 
Topic: VOCABULARY
  Definition or contextual usage:
  ✅ "A ________ is a person who is new to a job or activity."
     A) veteran  B) novice  C) mentor  D) scholar
     Answer: B
 
## FILL IN THE BLANK FORMAT (Worksheet):
 
Use exactly 8 underscores: ________
 
Topic: TENSE / VERB FORMS → verb hint in bracket is MANDATORY
  ✅ "By the time she arrived, he ________ (finish) his meal."
  ✅ "They ________ (play) cricket when it started raining."
  ❌ WRONG: "By the time she arrived, he ________ his meal."  ← no hint, forbidden
 
Topic: PREPOSITION → NO verb hint
  ✅ "The dog hid ________ the table during the thunderstorm."
  ✅ "We have been friends ________ childhood."
  ✅ "She travelled to London ________ train."
 
Topic: ARTICLES → NO verb hint
  ✅ "________ Amazon is the largest river in the world."
  ✅ "She bought ________ umbrella and ________ pair of gloves."
 
Topic: PARTS OF SPEECH → NO verb hint
  ✅ "She cried ________ she was deeply moved."  (conjunction)
  ✅ "The book is kept ________ the shelf."       (preposition)
 
Topic: VOCABULARY → NO verb hint — definition-based format ONLY
  ✅ "A ________ is a doctor who specializes in treating children."
  ✅ "The ________ of the painting left everyone speechless."
  ❌ FORBIDDEN: "A ________ (manage) person on the farm"
 
## SHORT ANSWER FORMAT (Worksheet):
 
Topic: TENSE/VERB FORMS → grammar transformation only
  ✅ "Rewrite in past perfect: 'She finishes her homework.'"
  ✅ "Change to present continuous: 'He reads a book.'"
 
Topic: PREPOSITION → correction or usage
  ✅ "Correct the preposition: 'He is good in swimming.'"
  ✅ "Rewrite using 'since' or 'for': 'She has lived here from 2015.'"
 
Topic: ARTICLES → correction or usage
  ✅ "Fill in the correct article: '________ sun rises in the east.'"
  ✅ "Correct the article error: 'He is an honest man.'"  ← (this is already correct; test understanding)
 
Topic: PARTS OF SPEECH → identification or usage
  ✅ "Identify the part of speech of 'quickly' in: 'She ran quickly.'"
  ✅ "Replace the blank with a suitable conjunction: 'She was tired ________ she kept going.'"
 
Topic: VOCABULARY → synonym / antonym / usage only
  ✅ "Replace the underlined word with its synonym: 'He is a brave soldier.'"
  ✅ "Write the antonym of 'generous' and use it in a sentence."
  ✅ "Use the word 'resilient' in a meaningful sentence."
  ❌ FORBIDDEN: "What is the term for...?"
  ❌ FORBIDDEN: "What is the meaning of...?"
 
NEVER ask comprehension questions (Why/What happened) for worksheet content.
 
-------------------------------------------------------------------
IF CONTENT TYPE IS READING PASSAGE:
-------------------------------------------------------------------
 
Generate questions that test understanding of the passage.
 
## MCQ FORMAT (Passage):
  Factual questions directly answerable from the text:
  ✅ "What did the writer find under his foot?"
     A) A coin  B) A note  C) A key  D) Nothing
     Answer: A


     ## MCQ QUALITY RULES — READING PASSAGE:

1. COVERAGE RULE:
   - Questions must be distributed across ALL paragraphs
   - At least 1 question per paragraph if 5+ paragraphs exist
   - NEVER ignore the final paragraphs

2. NO REPETITION RULE:
   - Each question must test a DIFFERENT fact
   - If two questions have the same answer → delete one, 
     replace with question from uncovered paragraph

3. ANSWER DISTRIBUTION RULE:
   - Correct answer must be distributed across A, B, C, D
   - NEVER have more than 3 questions with the same answer label
   - Aim for: A×2-3, B×2-3, C×2-3, D×2-3

4. QUESTION WORDING RULE:
   - Question must make sense WITHOUT the answer
   - ❌ WRONG: "What did the Iceman carry that was an unfinished bow?"
   - ✅ RIGHT:  "What material was the Iceman's unfinished bow made from?"

5. DISTRACTOR QUALITY RULE:
   - Wrong options must be plausible, not obviously wrong
   - ❌ "He was buried in a special tomb" ← obviously wrong
   - ✅ "He was wearing elaborate ceremonial garments" ← plausible

     
## FILL IN THE BLANK FORMAT (Passage):
  CURRENT WRONG APPROACH:
  ❌ Original sentence: "He wore shoes with bearskin soles."
  ❌ Output:            "He wore shoes with ________ soles."  ← sentence copy

  CORRECT APPROACH:
    ✅ Read the fact from the passage
    ✅ Write a NEW sentence that expresses the same fact differently
    ✅ Remove one key word as the blank

  EXAMPLE:
    Passage says: "...a birchbark container of embers wrapped in maple leaves"
  
    ❌ WRONG: "He carried a birchbark container of embers wrapped in ________ leaves."
               ← copied from passage
  
    ✅ CORRECT: "To keep fire alive during his journey, the Iceman stored 
                 embers inside a container made of ________."
                 Answer: birchbark

    ✅ CORRECT: "The Iceman used ________ leaves to wrap the embers 
                 he carried in his container."
                 Answer: maple

  Rules:
  - New sentence must be rewritten — not copied from the passage
  - The blank must test a KEY FACT from the passage
  - Answer must be directly supported by the passage
  - Sentence structure must be different from the original


## FILL IN THE BLANK — READING PASSAGE — QUALITY RULES:

LEVEL OF QUESTIONS — MANDATORY:
  Generate questions at THREE levels:
  - Factual   (4 questions): Direct facts from passage, but sentence must be rewritten
  - Inferential (3 questions): Requires understanding, not just finding a word
  - Analytical  (3 questions): Why/What does it suggest/What does it indicate

FACTUAL — Acceptable:
  ✅ "The Iceman stored fire-starting embers in a container crafted 
      from ________ bark."
  Answer: birch
  (Sentence rewritten, tests a meaningful fact)

  ❌ NOT acceptable — too trivial:
  "The location was a ________ hollow."
  Answer: rocky
  (Anyone can guess this without reading)

INFERENTIAL — Required format:
  ✅ "The presence of a copper-bladed ax suggested the Iceman 
      held ________ in his society."
  Answer: considerable social significance

  ✅ "The half-finished arrows suggest the Iceman had recently 
      been in a ________."
  Answer: conflict / fight / battle

ANALYTICAL — Required format:
  ✅ "Scientists believe the Iceman grew up in Valle Isarco 
      based on analysis of ________ in his teeth."
  Answer: isotopes

  ✅ "Klaus Oeggl compared the Iceman's death site to a 
      ________ scene."
  Answer: paleo crime

BANNED — trivial blanks:
  ❌ "________ hollow"        → Answer: rocky     (too easy)
  ❌ "________ skin"          → Answer: deer      (too easy)
  ❌ "________ blossoms"      → Answer: hornbeam  (just reading)

RULE: The blank must test UNDERSTANDING, not just READING.
Ask yourself: "Can someone answer this without understanding 
the passage?" If YES → rewrite the question.
 
 ## SHORT ANSWER FORMAT (Passage):
  Why / How / Describe / Explain type questions:
  ✅ "Why did the man claim he was on his knees?"
  ✅ "How did the writer feel after receiving the news?"
 
  
===================================================================
STEP 3 — STRICT OUTPUT RULES
===================================================================
 
1.  MCQ must ALWAYS have exactly 4 options: A, B, C, D
2.  Fill in the blank must use exactly 8 underscores: ________
3.  Fill in the blank and short answer must always have:  "options": null
4.  The "answer" field must NEVER be null or empty
5.  NEVER copy or slightly modify any sentence from the input text
6.  ALL generated sentences must be 100% original
7.  Only the grammar pattern or topic is borrowed — never the words
8.  Verb hint in bracket is MANDATORY for tense/verb topics:   ________ (leave)
9.  Verb hint must NEVER appear for: articles, prepositions, parts of speech, vocabulary
10. For vocabulary fill-in-the-blank: ALWAYS use definition-based format only
11. If multiple grammar topics are detected: distribute questions EVENLY across ALL topics
12. NEVER generate all questions from a single sub-topic when multiple exist
13. Respond with valid JSON ONLY — no explanation, no markdown, no extra text
 
===================================================================
FINAL REMINDER BEFORE GENERATING
===================================================================
 
Ask yourself these questions before writing the first question:
 
  ✔ Have I identified the content type correctly? (worksheet vs passage)
  ✔ Have I discarded all existing sentences from the input?
  ✔ Have I identified ALL grammar sub-topics present?
  ✔ Are my new sentences 100% original with different contexts?
  ✔ Have I distributed questions across all detected topics?
  ✔ Does every fill-in-the-blank use exactly 8 underscores?
  ✔ Did I add verb hints ONLY where the topic is tense/verb forms?
  ✔ Is my output pure valid JSON with no extra text?
 
If any answer is NO — fix it before outputting.
 
You MUST respond with valid JSON only — no explanation, no markdown, just pure JSON."""







def _build_user_prompt(
    text: str,
    question_types: List[QuestionType],
    number_of_questions: int,
    content_type: ContentType
) -> str:
    """
    Build the user prompt for question generation.
    The prompt varies depending on content type and selected question types.
    """
 
    # ================================================================
    # SECTION 1 — AUTO CONTENT TYPE DETECTION HINT
    # ================================================================
    # Even though content_type comes from the request, we give the LLM
    # an explicit signal to override if it detects worksheet patterns.
 
    auto_detect_hint = """⚠️ CONTENT TYPE AUTO-CHECK (Do this before anything else):
Read the input text carefully. If you find ANY of these signals:
  - Numbered sentences with blanks:  "1. She ________ to school."
  - Underscores used as blanks:       ________
  - Exercise headers:                 "Fill in the blanks", "Exercise 1"
  - Verb hints in brackets:           ________ (go)
  - MCQ option lists already present: A) walk  B) walked
 
→ OVERRIDE content type to: WORKSHEET / EXAM PAPER
→ DISCARD all existing sentences from the text
→ EXTRACT grammar pattern only
→ GENERATE completely new sentences
 
If NONE of the above signals are found → treat as READING PASSAGE."""
 
    # ================================================================
    # SECTION 2 — ANTI-COPY ENFORCEMENT BLOCK
    # ================================================================
 
    anti_copy_block = """🚨 ANTI-COPY RULE — ZERO TOLERANCE:
 
FORBIDDEN (even if slightly changed):
  ❌ Input:  "Wait ________ your mother reaches home."
  ❌ Output: "Wait ________ your mother reaches home."       ← exact copy
  ❌ Output: "Wait ________ your father reaches home."       ← word swap, still forbidden
  ❌ Output: "Please wait ________ your mother comes home."  ← minor rewording, still forbidden
 
REQUIRED:
  ✅ Detect pattern:  TIME preposition (until / before / since / by / after)
  ✅ New sentence:    "The students must remain seated ________ the teacher dismisses the class."
  ✅ Answer:          until
 
Every single generated sentence must have a DIFFERENT:
  - Subject (not the same person/animal/thing from input)
  - Verb (not the same action from input)
  - Context (not the same situation from input)"""
 
    # ================================================================
    # SECTION 3 — QUESTION TYPE DESCRIPTIONS
    # ================================================================
 
    if content_type == ContentType.WORKSHEET_EXAM_PAPER:
        type_descriptions = {
            QuestionType.MCQ: (
                "Blank-based MCQ — 4 options (A, B, C, D).\n"
                "  For TENSE topic:       'She ________ (walk) to school when it started raining.'\n"
                "  For PREPOSITION topic: 'The flight departs ________ 6 AM tomorrow.'\n"
                "  For ARTICLES topic:    'She is ________ best player on the team.'\n"
                "  For VOCABULARY topic:  'A ________ is a person who is new to a job or activity.'\n"
                "  ❌ NEVER ask: 'What tense is used?' or 'What is the past participle of...?'"
            ),
            QuestionType.FILL_IN_THE_BLANK: (
                "New blank-based sentence using the same grammar pattern.\n"
                "  For TENSE topic:       verb hint MANDATORY → 'She ________ (finish) her work before noon.'\n"
                "  For PREPOSITION topic: NO verb hint → 'He has been waiting ________ morning.'\n"
                "  For ARTICLES topic:    NO verb hint → '________ Nile is the longest river in the world.'\n"
                "  For VOCABULARY topic:  definition-based ONLY → 'A ________ is a ruler of a kingdom.'\n"
                "  ❌ NEVER copy any sentence from the original text"
            ),
            QuestionType.SHORT_ANSWER: (
                "Grammar transformation or usage questions ONLY.\n"
                "  For TENSE topic:       'Rewrite in past perfect: She finishes her homework.'\n"
                "  For PREPOSITION topic: 'Correct the preposition: He is good in mathematics.'\n"
                "  For ARTICLES topic:    'Fill in the correct article: ________ sun rises in the east.'\n"
                "  For VOCABULARY topic:  'Write the antonym of generous and use it in a sentence.'\n"
                "  ❌ NEVER ask comprehension questions like 'Why did...?' or 'What happened...?'"
            ),
        }
    else:
        type_descriptions = {
            QuestionType.MCQ: (
                "Comprehension MCQ — 4 options (A, B, C, D) directly based on the passage.\n"
                "  Example: 'What did the writer find under his foot?'\n"
                "           A) A coin  B) A note  C) A key  D) Nothing"
            ),
            QuestionType.FILL_IN_THE_BLANK: (
                "Key information gap from the passage — use exactly 8 underscores.\n"
                "  Example: 'The writer found a ________ coin under his foot.'\n"
                "  Answer must be a word or phrase directly from the passage."
            ),
            QuestionType.SHORT_ANSWER: (
                "Why / How / Describe / Explain type questions answerable in 1–2 sentences.\n"
                "  Example: 'Why did the man claim he was on his knees?'\n"
                "           'How did the writer feel after receiving the news?'"
            ),
        }
 
    selected_type_lines = []
    for qt in question_types:
        selected_type_lines.append(f"- {type_descriptions[qt]}")
    types_text = "\n".join(selected_type_lines)
 
    # ================================================================
    # SECTION 4 — CONTENT INSTRUCTION BLOCK
    # ================================================================
 
    if content_type == ContentType.WORKSHEET_EXAM_PAPER:
        content_instruction = """CONTENT TYPE: WORKSHEET / EXAM PAPER
 
Follow these steps strictly:
  STEP 1 — Read the entire text and identify the grammar topic(s):
            (e.g., prepositions of time/place/direction, past perfect, articles, vocabulary)
  STEP 2 — If multiple sub-topics exist, list all of them mentally
  STEP 3 — DISCARD all original sentences from the text completely
  STEP 4 — Generate BRAND NEW sentences using the same grammar pattern(s)
  STEP 5 — Distribute questions evenly if multiple sub-topics were detected
 
WORKED EXAMPLE — If worksheet tests PREPOSITION (time + place):
  Detected sub-topics: time prepositions, place prepositions
  Total questions: 6 → 3 per sub-topic
 
  Fill in the Blank (time):
    "She has been studying ________ early morning."
    Answer: since
 
  Fill in the Blank (place):
    "The cat was hiding ________ the sofa during the storm."
    Answer: under
 
  MCQ (time):
    "The meeting was scheduled ________ Friday afternoon."
    A) at  B) on  C) in  D) by
    Answer: B
 
  MCQ (place):
    "He stood ________ the window, watching the rain fall."
    A) between  B) beside  C) through  D) across
    Answer: B"""
 
    else:
        content_instruction = """CONTENT TYPE: READING PASSAGE
 
Generate questions that test comprehension and understanding of the passage.
All questions must be directly based on facts, events, and ideas written in the text.
Do not ask questions about information not present in the passage."""
 
    # ================================================================
    # SECTION 5 — QUESTION DISTRIBUTION
    # ================================================================
 
    questions_per_type = number_of_questions // len(question_types)
    remainder = number_of_questions % len(question_types)
 
    distribution_parts = []
    for i, qt in enumerate(question_types):
        count = questions_per_type + (1 if i < remainder else 0)
        distribution_parts.append(f"{qt.value}: {count} questions")
    distribution_text = ", ".join(distribution_parts)
 
    # ================================================================
    # SECTION 6 — FINAL RULES REMINDER
    # ================================================================
 
    if content_type == ContentType.WORKSHEET_EXAM_PAPER:
        final_rules = """Final rules before generating:
1.  MCQ must always have exactly 4 options (A, B, C, D)
2.  Fill in the blank must use ________ (exactly 8 underscores)
3.  "options" field must be null for fill_in_the_blank and short_answer
4.  "answer" field must NEVER be null or empty
5.  NEVER copy, reuse, or slightly modify any sentence from the input text
6.  ALL generated sentences must be 100% original with different subjects and contexts
7.  Verb hint ________ (verb) is MANDATORY for tense/verb topics ONLY
8.  NO verb hint for preposition, articles, parts of speech, vocabulary topics
9.  For vocabulary fill-in-the-blank: use definition-based format ONLY
10. Short answer must be grammar transformation or usage questions ONLY
11. If multiple grammar sub-topics detected: distribute questions evenly across ALL of them
12. Output must be valid JSON only — no explanation, no markdown"""
 
    else:
        final_rules = """Final rules before generating:
1.  MCQ must always have exactly 4 options (A, B, C, D)
2.  Fill in the blank must use ________ (exactly 8 underscores)
3.  "options" field must be null for fill_in_the_blank and short_answer
4.  "answer" field must NEVER be null or empty
5.  All questions must be directly based on the passage content
6.  Short answer must be answerable in 1–2 sentences
7.  Answers must be accurate and taken from the passage
8.  Output must be valid JSON only — no explanation, no markdown"""
 
    # ================================================================
    # SECTION 7 — FINAL PROMPT ASSEMBLY
    # ================================================================
 
    return f"""Generate exactly {number_of_questions} questions based on the text below.
 
{auto_detect_hint}
 
---
 
{anti_copy_block}
 
---
 
{content_instruction}
 
---
 
Generate these question types:
{types_text}
 
Question distribution: {distribution_text}
 
---
 
TEXT TO USE:
\"\"\"
{text}
\"\"\"
 
---
 
Respond ONLY with this exact JSON structure — no other text, no markdown:
{{
  "questions": [
    {{
      "question_number": 1,
      "question_type": "mcq",
      "question_text": "completely new sentence based on detected grammar pattern",
      "options": [
        {{"label": "A", "text": "option 1"}},
        {{"label": "B", "text": "option 2"}},
        {{"label": "C", "text": "option 3"}},
        {{"label": "D", "text": "option 4"}}
      ],
      "answer": "correct option label (A / B / C / D)"
    }},
    {{
      "question_number": 2,
      "question_type": "fill_in_the_blank",
      "question_text": "completely new sentence with ________ blank",
      "options": null,
      "answer": "correct word or phrase"
    }},
    {{
      "question_number": 3,
      "question_type": "short_answer",
      "question_text": "grammar transformation or usage question",
      "options": null,
      "answer": "correct transformed sentence or usage example"
    }}
  ]
}}
 
{final_rules}"""



async def generate_questions_with_deepseek(
    text: str,
    question_types: List[QuestionType],
    number_of_questions: int,
    content_type: ContentType
) -> dict:
    """
    Generate questions using the DeepSeek API.
    This is an async function, compatible with FastAPI async endpoints.

    Args:
        text: Text extracted from OCR
        question_types: Types of questions to generate
        number_of_questions: Total number of questions to generate
        content_type: Type of content (reading passage or exam paper)

    Returns:
        Parsed JSON response containing the generated questions
    """
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(text, question_types, number_of_questions, content_type)

    payload = {
        "model": settings.deepseek_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7, 
        "max_tokens": 4000, 
        "stream": False     
    }

    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            DEEPSEEK_API_URL,
            json=payload,
            headers=headers
        )

        if response.status_code != 200:
            error_detail = response.text
            raise ValueError(f"DeepSeek API error {response.status_code}: {error_detail}")

        response_data = response.json()

    if not response_data.get("choices"):
        raise ValueError("No response received from DeepSeek.")

    generated_text = response_data["choices"][0]["message"]["content"].strip()

    try:
        if generated_text.startswith("```"):
            lines = generated_text.split('\n')
            generated_text = '\n'.join(lines[1:-1])

        parsed_response = json.loads(generated_text)
        return parsed_response

    except json.JSONDecodeError as e:
        raise ValueError(
            f"DeepSeek response is not valid JSON: {str(e)}. Response: {generated_text[:200]}"
        )