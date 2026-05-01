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
   - Pronunciation patterns
   - Parts of speech (noun, verb, adjective, adverb, pronoun)
   - Punctuation and grammar rules

2. **Difficulty Level:** beginner / intermediate / advanced

3. **Content Pattern** (for worksheet): detect instruction type and replicate format exactly

---

## QUESTION GENERATION RULES BY TOPIC:

**TENSE:**
- MCQ: Test correct tense identification or usage in context
- Fill in Blank: "She ___ (go) to school yesterday." → answer: went
- Short Answer: "Identify the tense" / "Rewrite in past perfect"

**VERB FORMS:**
- MCQ: "The correct past participle of 'write' is ___"
- Fill in Blank: "He has ___ (finish) his work." → answer: finished
- Short Answer: "Use the verb in three different forms"

**SENTENCE STRUCTURE:**
- MCQ: "Which of the following is a compound sentence?"
- Fill in Blank: "Join the sentences using a suitable ___." → answer: conjunction
- Short Answer: "Convert the simple sentence into a complex sentence"

**COMPREHENSION (Reading Passage):**
- MCQ: Factual questions directly from the passage
- Fill in Blank: Key information gaps from the passage
- Short Answer: "Why", "How", "Describe", "Explain" type questions

**VOCABULARY:**
- MCQ: Word meaning, synonym, antonym
- Fill in Blank: Use correct word in context
- Short Answer: Use the word in a meaningful sentence

---

## STRICT OUTPUT RULES:

1. MCQ must ALWAYS have exactly 4 options (A, B, C, D)
2. Fill in the blank must use ________ (8 underscores)
3. Fill in the blank and short answer must have options: null
4. answer field must NEVER be null or empty
5. All questions must be based strictly on the provided text
6. Questions must match the language topic found in the text
7. You MUST respond with valid JSON only — no explanation, no markdown, just pure JSON"""


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
    type_descriptions = {
        QuestionType.MCQ: "Multiple Choice Questions (MCQ) with 4 options (A, B, C, D) and the correct answer",
        QuestionType.FILL_IN_THE_BLANK: "Fill in the Blank questions where key words are replaced with blanks",
        QuestionType.SHORT_ANSWER: "Short Answer questions that require 1-2 sentence answers"
    }

    selected_types = [type_descriptions[qt] for qt in question_types]
    types_text = "\n".join([f"- {t}" for t in selected_types])

    if content_type == ContentType.WORKSHEET_EXAM_PAPER:
        content_instruction = "The content is an existing exam paper or worksheet. Generate NEW exercises based on the same grammar/topic focus, not repeating the exact questions."
    else:
        content_instruction = "The content is a reading passage. Generate questions that test comprehension and understanding of the text."

    questions_per_type = number_of_questions // len(question_types)
    remainder = number_of_questions % len(question_types)

    distribution_parts = []
    for i, qt in enumerate(question_types):
        count = questions_per_type + (1 if i < remainder else 0)
        distribution_parts.append(f"{qt.value}: {count} questions")
    distribution_text = ", ".join(distribution_parts)

    return f"""Based on the following text, generate exactly {number_of_questions} questions.

Content Type Instruction: {content_instruction}

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
      "question_text": "What is...?",
      "options": [
        {{"label": "A", "text": "First option"}},
        {{"label": "B", "text": "Second option"}},
        {{"label": "C", "text": "Third option"}},
        {{"label": "D", "text": "Fourth option"}}
      ],
      "answer": "A"
    }},
    {{
      "question_number": 2,
      "question_type": "fill_in_the_blank",
      "question_text": "The process of ________ involves...",
      "options": null,
      "answer": "photosynthesis"
    }},
    {{
      "question_number": 3,
      "question_type": "short_answer",
      "question_text": "Explain why...?",
      "options": null,
      "answer": "Brief answer here"
    }}
  ]
}}

Important rules:
1. MCQ questions must always have exactly 4 options (A, B, C, D)
2. Fill in the blank must use ________ (8 underscores)
3. Short answer questions should be answerable in 1-2 sentences
4. All questions must be directly based on the provided text
5. Answers must be accurate and factual"""



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