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

Your job is to generate high-quality English practice questions from the given text.

---

## BEFORE GENERATING, ANALYZE THE TEXT FOR:

1. **Language Topics Present:**
   - Tense (past simple, present perfect, future continuous, etc.)
   - Verb forms (base form, past participle, gerund, infinitive)
   - Sentence structure (simple, compound, complex)
   - Vocabulary and word meaning
   - Comprehension (facts, inference, main idea)
   - Parts of speech (noun, verb, adjective, adverb, pronoun)
   - Articles (a, an, the)
   - Prepositions
   - Punctuation and grammar rules

2. **Difficulty Level:** beginner / intermediate / advanced

3. **Content Type Detection:**
   - WORKSHEET/EXAM PAPER: Contains blanks, verb hints in brackets, exercise numbers
   - READING PASSAGE: Contains continuous prose text for comprehension

4. **Multiple Topic Detection — CRITICAL:**
   - Carefully read the ENTIRE text before generating
   - Identify ALL grammar topics present, not just the first one
   - Example: A worksheet may contain Tense + Articles + Parts of Speech together
   - List all detected topics mentally before starting question generation
   - Distribute questions proportionally across ALL detected topics

   *** IF MULTIPLE TOPICS ARE PRESENT: ***
   - NEVER generate all questions from only one topic
   - MUST cover every detected topic at least once
   - Distribute evenly: if 12 questions and 3 topics → 4 questions per topic
   - Each question type (MCQ, Fill in Blank, Short Answer) should also cover multiple topics

---

## QUESTION GENERATION RULES BY CONTENT TYPE:

### IF CONTENT TYPE IS WORKSHEET/EXAM PAPER:

**CRITICAL RULE: NEVER copy, reuse, or slightly modify any sentence from the original text.
Identify the grammar pattern only, then create COMPLETELY NEW sentences.**

**TENSE / VERB FORMS:**

- MCQ format MUST always be blank-based:
  CORRECT FORMAT:
  "She ________ (walk) to school when it started raining."
  A) walk  B) walked  C) was walking  D) had walked

  NEVER use these formats:
  - "What tense is used in Exercise 4?"
  - "What is the past participle of 'hear'?"
  - "Which tense is primarily used in the passage?"

- Fill in the Blank format:
  Use exactly 8 underscores: ________

  IF grammar topic is TENSE or VERB FORMS:
    Always include verb hint in bracket — MANDATORY
    CORRECT: "By the time he arrived, she ________ (leave) the room."
    WRONG: "By the time he arrived, she ________ the room."

  IF grammar topic is ARTICLES:
    No verb hint needed
    CORRECT: "________ Eiffel Tower is one of the most visited monuments."
    CORRECT: "She bought ________ umbrella and ________ bag."

  IF grammar topic is PARTS OF SPEECH:
    No verb hint needed
    CORRECT: "She cried ________ she was sad." (conjunction)
    CORRECT: "The plane flew ________ the mountains." (preposition/adverb)

  IF grammar topic is PREPOSITION:
    No verb hint needed
    CORRECT: "She walked ________ the park to find her keys."

  IF grammar topic is VOCABULARY or WORD MEANING:
    *** STRICTLY NO verb hint in bracket — this is a HARD RULE ***
    *** ANY format like "________ (verb)" is COMPLETELY FORBIDDEN ***
    Use definition-based format ONLY:
    CORRECT: "A ________ is a person who rules a country as its supreme leader."
    CORRECT: "The ________ of the forest calmed the troubled mind."
    WRONG (FORBIDDEN): "A person who ________ (manage) a farm"
    WRONG (FORBIDDEN): "The room was ________ (illuminate) by a lamp"

- Short Answer format (WORKSHEET ONLY):
  Questions must match the grammar topic detected:

  IF TENSE/VERB FORMS:
  - "Rewrite in past perfect: 'She finishes her work.'"
  - "Change the sentence to past continuous tense."

  IF ARTICLES:
  - "Fill in the correct article: '________ sun rises in the east.'"
  - "Correct the article in the sentence: 'She is an honest woman.'"

  IF PARTS OF SPEECH:
  - "Identify the part of speech of the underlined word: 'She runs fast.'"
  - "Replace the blank with the correct conjunction: 'She was tired ________ she kept working.'"

  IF PREPOSITION:
  - "Correct the preposition: 'He is good in mathematics.'"
  - "Rewrite using since or for: They lived here from 2010."

  IF VOCABULARY/WORD MEANING:
  - "Replace the underlined word with its synonym: 'He is a brave man.'"
  - "Use the word 'lachrymose' in a meaningful sentence."
  - "Write the antonym of 'benevolent' and use it in a sentence."

  *** STRICTLY FORBIDDEN for Vocabulary Short Answer: ***
  - NEVER ask "What is the term for...?" — this is a comprehension question
  - NEVER ask "What is the meaning of...?" — this is a comprehension question
  - ONLY ask transformation/usage questions

  NEVER ask comprehension questions like "Why did the man...?" for any worksheet content.

---

### IF CONTENT TYPE IS READING PASSAGE:

**COMPREHENSION:**

- MCQ: Factual questions directly from the passage
  "What did the writer find under his foot?"
  A) A coin  B) A note  C) A key  D) Nothing

- Fill in the Blank: Key information gaps from the passage
  "The writer found a ________ coin under his foot."

- Short Answer: "Why", "How", "Describe", "Explain" type questions
  - "Why did the man claim he was on his knees?"
  - "How did the writer feel about his exam performance?"

---

## STRICT OUTPUT RULES:

1. MCQ must ALWAYS have exactly 4 options (A, B, C, D)
2. Fill in the blank must use ________ (8 underscores)
3. Fill in the blank and short answer must have options: null
4. answer field must NEVER be null or empty
5. All questions must match the grammar pattern found in the text
6. NEVER copy or slightly modify sentences from the original text
7. ALL new sentences must be completely original
8. Only the grammar pattern/structure should match the original
9. For worksheet fill-in-the-blank:
   - TENSE/VERB FORMS topic: verb hint in bracket is MANDATORY e.g. ________ (leave)
   - ARTICLES topic: NO verb hint e.g. "________ Nile is the longest river."
   - PARTS OF SPEECH topic: NO verb hint e.g. "She cried ________ she was sad."
   - PREPOSITION topic: NO verb hint e.g. "She walked ________ the park."
   - VOCABULARY topic: NO verb hint EVER, definition-based format ONLY
     e.g. "A ________ is a person who rules a country." Answer: monarch
10. For worksheet short answer:
    - TENSE: grammar transformation only
    - ARTICLES: article correction/usage only
    - PARTS OF SPEECH: identification/usage only
    - PREPOSITION: preposition correction/usage only
    - VOCABULARY: synonym/antonym/usage only — NEVER "What is the term for...?"
11. *** MULTIPLE TOPICS RULE — STRICTLY ENFORCED: ***
    - If the text contains multiple grammar topics, questions MUST cover ALL topics
    - NEVER generate all questions from only one topic
    - Distribute questions evenly across all detected topics
    - Example: Tense + Articles + Parts of Speech → each topic gets equal questions
12. You MUST respond with valid JSON only — no explanation, no markdown, just pure JSON"""






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

    # ---- Type Descriptions ----
    if content_type == ContentType.WORKSHEET_EXAM_PAPER:
        type_descriptions = {
            QuestionType.MCQ: "Blank-based MCQ with 4 tense/verb options (A, B, C, D). Format: 'She ________ (go) to school.' — NEVER ask theory questions like 'What tense is used?'",
            QuestionType.FILL_IN_THE_BLANK: "New sentence with same grammar pattern as the worksheet. Verb hint in bracket is MANDATORY. Format: 'By the time he arrived, she ________ (leave) the room.'",
            QuestionType.SHORT_ANSWER: "Grammar transformation questions ONLY. Format: 'Rewrite in past perfect: She finishes her work.' — NEVER ask comprehension questions."
        }
    else:
        type_descriptions = {
            QuestionType.MCQ: "Comprehension MCQ with 4 options (A, B, C, D) directly based on the passage",
            QuestionType.FILL_IN_THE_BLANK: "Key information gap from the passage. Format: 'The writer found a ________ coin under his foot.'",
            QuestionType.SHORT_ANSWER: "Why / How / Explain / Describe type comprehension questions answerable in 1-2 sentences"
        }

    selected_types = [type_descriptions[qt] for qt in question_types]
    types_text = "\n".join([f"- {t}" for t in selected_types])

    # ---- Content Instruction ----
    if content_type == ContentType.WORKSHEET_EXAM_PAPER:
        content_instruction = """CONTENT TYPE: WORKSHEET / EXAM PAPER

STEP 1: Carefully read the text and identify the grammar pattern
        (e.g., past perfect, past continuous, present perfect)
STEP 2: Do NOT use any sentence from the original text
STEP 3: Create COMPLETELY NEW sentences using the SAME grammar pattern

EXAMPLE — If worksheet tests past perfect tense:
  MCQ:
  "By the time she arrived, he ________ (leave) the room."
  A) leave  B) left  C) was leaving  D) had left
  Answer: D

  Fill in the Blank:
  "She ________ (finish) her work before the bell rang."
  Answer: had finished

  Short Answer:
  "Rewrite in past perfect: 'He eats his dinner.'"
  Answer: He had eaten his dinner."""

    else:
        content_instruction = """CONTENT TYPE: READING PASSAGE

Generate questions that test comprehension and understanding of the passage.
Questions must be directly based on facts, events, and ideas in the text."""

    # ---- Question Distribution ----
    questions_per_type = number_of_questions // len(question_types)
    remainder = number_of_questions % len(question_types)

    distribution_parts = []
    for i, qt in enumerate(question_types):
        count = questions_per_type + (1 if i < remainder else 0)
        distribution_parts.append(f"{qt.value}: {count} questions")
    distribution_text = ", ".join(distribution_parts)

    # ---- Important Rules ----
    if content_type == ContentType.WORKSHEET_EXAM_PAPER:
        important_rules = """Important rules:
1. MCQ must always have exactly 4 options (A, B, C, D)
2. Fill in the blank must use ________ (8 underscores)
3. NEVER copy, reuse, or slightly modify any sentence from the original text
4. ALL sentences must be completely new and original
5. Verb hint in bracket is MANDATORY for every fill in the blank: ________ (verb)
6. Short answer must be grammar transformation questions ONLY
7. NEVER ask comprehension questions like 'Why did...?' or 'What did...?' for worksheet content
8. Answers must be grammatically correct"""

    else:
        important_rules = """Important rules:
1. MCQ must always have exactly 4 options (A, B, C, D)
2. Fill in the blank must use ________ (8 underscores)
3. All questions must be directly based on the passage
4. Fill in the blank and short answer must have options: null
5. Short answer must be answerable in 1-2 sentences
6. Answers must be accurate and factual"""

    return f"""Based on the following text, generate exactly {number_of_questions} questions.

{content_instruction}

Generate these types of questions:
{types_text}

Question distribution: {distribution_text}

TEXT TO USE:
\"\"\"
{text}
\"\"\"

Respond ONLY with this exact JSON structure, no other text:
{{
  "questions": [
    {{
      "question_number": 1,
      "question_type": "mcq",
      "question_text": "question based on detected grammar pattern from text",
      "options": [
        {{"label": "A", "text": "option 1"}},
        {{"label": "B", "text": "option 2"}},
        {{"label": "C", "text": "option 3"}},
        {{"label": "D", "text": "option 4"}}
      ],
      "answer": "correct option label"
    }},
    {{
      "question_number": 2,
      "question_type": "fill_in_the_blank",
      "question_text": "new sentence with blank based on detected grammar pattern",
      "options": null,
      "answer": "correct answer"
    }},
    {{
      "question_number": 3,
      "question_type": "short_answer",
      "question_text": "grammar transformation question based on detected pattern",
      "options": null,
      "answer": "correct answer"
   }}
  ]
}}

{important_rules}"""




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