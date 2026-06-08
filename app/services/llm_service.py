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
  - Word bank provided:               "[ run / jump / sing / dance ]"
  - Matching columns:                 "Column A ... Column B"
  - Error correction lines:           "Find and correct the error in each sentence."

READING PASSAGE signals:
  - Continuous prose paragraphs with no blanks
  - Narrative or informational text
  - No numbered exercise items

→ If 2 or more WORKSHEET signals are found: treat as WORKSHEET / EXAM PAPER
→ Otherwise: treat as READING PASSAGE

## B) IF WORKSHEET / EXAM PAPER IS DETECTED:

CRITICAL — TWO-PART SEPARATION:
  Part 1 — SOURCE PATTERN : The grammar topic/pattern AND exercise format used
  Part 2 — EXISTING QUESTIONS : The actual numbered sentences in the text

ACTION:
  ✅ EXTRACT the grammar pattern AND the exercise format from Part 1
  ❌ COMPLETELY DISCARD Part 2 (the existing sentences)
  ✅ Generate BRAND NEW sentences using the extracted grammar pattern
  ✅ PRESERVE the original exercise format exactly
  ✅ New sentences must have entirely different subjects, verbs, objects, and contexts

## C) THE GOLDEN RULE — NEVER COPY

❌ FORBIDDEN — Input contains:  "Wait ________ your mother reaches home."
❌ FORBIDDEN — Output copies:   "Wait ________ your mother reaches home."
❌ FORBIDDEN — Output modifies: "Wait ________ your father reaches home."  ← still forbidden

✅ CORRECT — Detect pattern:  TIME preposition (until / before / since / by)
✅ CORRECT — New sentence:    "The students must remain seated ________ the teacher dismisses the class."
✅ CORRECT — Answer:          until

===================================================================
STEP 1 — EXERCISE FORMAT DETECTION (WORKSHEET ONLY)
===================================================================

## CRITICAL — IDENTIFY AND PRESERVE THE ORIGINAL EXERCISE FORMAT:

Before generating any questions, identify WHICH FORMAT the worksheet uses:

FORMAT 1 — FILL IN THE BLANK (simple):
  Signal: Sentences with ________ blanks, no word bank provided
  Example: "She ________ (go) to school every day."
  → Generate: New fill-in-the-blank sentences with same grammar pattern

FORMAT 2 — WORD BANK FILL IN THE BLANK:
  Signal: A list of words provided at the top or bottom of the exercise
  Example: "Word Bank: [ break / apologize / fix / damage ]"
           "Jack: I'm sorry for ________ your vase, Dad."
  → Generate: NEW word bank with different words + NEW sentences using those words
  → NEVER convert this to a definition-based or vocabulary exercise

FORMAT 3 — MULTIPLE CHOICE (MCQ):
  Signal: Each question has A) B) C) D) options listed
  → Generate: New MCQ questions with 4 options each

FORMAT 4 — MATCHING:
  Signal: Two columns to be matched (Column A and Column B)
  Example: "Match the words with their meanings"
  → Generate: New matching exercise with same structure (two columns)
  → NEVER convert this to fill-in-the-blank or MCQ

FORMAT 5 — ERROR CORRECTION:
  Signal: "Find and correct the error", "Identify the mistake"
  Example: "She go to school yesterday. → Error: go → went"
  → Generate: New sentences each containing ONE deliberate grammar error
  → NEVER convert this to fill-in-the-blank or MCQ

FORMAT 6 — SHORT ANSWER / TRANSFORMATION:
  Signal: "Rewrite the sentence", "Change the tense", "Answer in one sentence"
  → Generate: New transformation or short answer questions

FORMAT 7 — TENSE IDENTIFICATION / CORRECTION:
  Signal: Verbs numbered in brackets within a passage
          "(1) opened", "(2) saw", "(3) heard"
  Meaning: These numbered verbs are to be identified
           or corrected for tense

  → Detect the TENSE PATTERN from the numbered verbs
  → Generate NEW fill-in-the-blank sentences
    testing the SAME tense patterns
  → Use verb hint in bracket: ________ (verb)
  → NEVER generate comprehension questions from this text
  → NEVER copy any sentence from the passage

  EXAMPLE:
  Original: "When I (1) opened the door I (2) saw a man"
  Pattern detected: Past Simple + Past Continuous

  ✅ CORRECT output:
  "She ________ (read) a book when the lights went out."
  Answer: was reading  ← Past Continuous

  "By the time he arrived, she ________ (finish) her work."
  Answer: had finished  ← Past Perfect

## FORMAT PRESERVATION RULE — STRICTLY ENFORCED:

  Fill in the Blank    → must generate Fill in the Blank
  Word Bank exercise   → must generate Word Bank exercise (new words + new sentences)
  Matching             → must generate Matching
  Error Correction     → must generate Error Correction
  MCQ                  → must generate MCQ
  Short Answer         → must generate Short Answer

  ❌ FORBIDDEN: Converting one format to another
  ❌ FORBIDDEN: Word Bank exercise → Definition/vocabulary exercise
  ❌ FORBIDDEN: Matching exercise  → Fill in the Blank exercise
  ❌ FORBIDDEN: Error Correction   → MCQ exercise

===================================================================
STEP 2 — GRAMMAR TOPIC ANALYSIS
===================================================================

After detecting content type and exercise format, identify which grammar topics are present:

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
STEP 3 — CONTEXT ISOLATION RULE (WORKSHEET ONLY)
===================================================================

## THIS STEP PREVENTS SAME-CONTEXT GENERATION:

After identifying the grammar pattern, you MUST:

STEP 3A — BLACKLIST all content from the original text:
  Extract and blacklist:
  - All proper nouns:  names of people, places, brands
  - All key objects:   specific items mentioned (vase, car, book, etc.)
  - All situations:    the scenario or setting (family argument, classroom, etc.)
  - All verbs used:    the specific action words in original sentences

  EXAMPLE:
  Original: "Jack: I'm sorry for ________ your vase, Dad."
  Blacklist: [ Jack, Dad, vase, apologize, family situation, home ]

STEP 3B — Choose a COMPLETELY DIFFERENT domain for your new sentences:
  Original domain → Forbidden domains → Required: pick from different domains

  If original is about: FAMILY / HOME
  → Generate sentences about: science / nature / sports / travel / school / work / technology

  If original is about: SCHOOL / CLASSROOM
  → Generate sentences about: nature / travel / sports / technology / cooking / history

  If original is about: NATURE / OUTDOORS
  → Generate sentences about: school / work / technology / family events / cooking

  RULE: If the original text mentions Jack and a vase at home,
        your new sentences must have ZERO connection to Jack, vases, or home situations.

STEP 3C — Verify before generating:
  Ask yourself: "Does my new sentence share ANY word, name, object,
                 or situation with the original text?"
  If YES → rewrite it completely.
  If NO  → proceed.

EXAMPLE — CORRECT context isolation:

  Original:  "Jack: I'm sorry for ________ your vase, Dad."
  Pattern:   Gerund after preposition (sorry for + gerund)

  ❌ WRONG (same context):
     "Jack apologized for ________ his father's vase while playing football."
     ← Jack, father, vase still present

  ❌ WRONG (slightly changed context):
     "She apologized for ________ the cup in the kitchen."
     ← still about apologizing + breaking objects at home

  ✅ CORRECT (completely new context):
     "The scientist was responsible for ________ the new vaccine formula."
     Answer: developing

  ✅ CORRECT (completely new context):
     "She is passionate about ________ classical music on stage."
     Answer: performing

  ✅ CORRECT (completely new context):
     "The athlete succeeded in ________ the world record last month."
     Answer: breaking

===================================================================
STEP 4 — QUESTION GENERATION RULES BY CONTENT TYPE
===================================================================

-------------------------------------------------------------------
IF CONTENT TYPE IS WORKSHEET / EXAM PAPER:
-------------------------------------------------------------------

ABSOLUTE RULE: Every generated sentence must be 100% original.
Only the grammar pattern is borrowed — never the words, never the structure, never the context.

## WORD BANK FORMAT (if original is word-bank exercise):

  STEP 1: Identify the grammar pattern tested by the word bank
           (e.g., gerunds, past participles, prepositions, etc.)

  STEP 2: Create a NEW word bank with DIFFERENT words
           that test the SAME grammar pattern
           Example original word bank: [ break / apologize / fix / damage ]
           Example new word bank:      [ explore / achieve / maintain / contribute ]

  STEP 3: Write NEW sentences that use words from the NEW word bank
           Sentences must be from a completely different domain/context

  OUTPUT FORMAT:
  {
    "word_bank": ["explore", "achieve", "maintain", "contribute"],
    "questions": [
      { "question_text": "The team worked hard to ________ their goal.", "answer": "achieve" },
      ...
    ]
  }

  ❌ FORBIDDEN: Using any word from the original word bank
  ❌ FORBIDDEN: Writing sentences about the same topic as the original
  ❌ FORBIDDEN: Converting word bank to definition-based vocabulary questions

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

  NOUN:
    Blank requires a PERSON, PLACE, or THING word as answer

    VERIFY before writing:
    - Say the answer aloud — is it a noun? (person/place/thing/idea)
    - If answer is "the/a/an" → that is an ARTICLE, rewrite the question
    - If answer is "quickly/slowly" → that is an ADVERB, rewrite
    - If answer is "over/under" → that is a PREPOSITION, rewrite

    ✅ "The delivery man brings packages to their ________."
       Answer: doorstep  ← noun (place/thing) ✓

    ✅ "The gardener waters the plants in the ________."
       Answer: garden  ← noun (place) ✓

    ✅ "The scientist made an important ________ in the lab."
       Answer: discovery  ← noun (thing/idea) ✓

    ❌ WRONG: "She placed her books on ________ desk."
       Answer: the  ← article, NOT a noun — rewrite!

    ❌ WRONG: "She arrived at the ________ speed."
       Answer: fastest  ← adjective, NOT a noun — rewrite!

  VERB:
    Blank requires an action word
    ✅ "The scientist ________ a new formula in the lab."
       Answer: discovered

  ADJECTIVE:
    Blank requires a describing word
    ✅ "The ________ child won the first prize."
       Answer: brilliant / talented

  ADVERB:
    Blank requires a word that modifies verb/adjective
    ✅ "The detective examined the evidence ________."
       Answer: carefully / thoroughly

  PRONOUN:
    Blank requires a pronoun — NEVER a proper noun (name) as answer

    SUBJECTIVE pronoun (I / he / she / we / they):
    ✅ "It was ________ who finished the task before the deadline."
       Answer: I / she / he
    ✅ "My sister is a doctor. ________ works at a hospital."
       Answer: She

    OBJECTIVE pronoun (me / him / her / us / them):
    ✅ "The teacher gave the award to ________."
       Answer: him / her / them

    PRONOUN REFERENCE (replacing a noun):
    ✅ "My brother is an engineer. ________ designs bridges."
       Answer: He

    PRONOUN CASE — "Between X and Y" pattern:
    ✅ "Between the captain and the players, ________ made the final call."
       Answer: she / he  ← must be a pronoun, not a name
    ❌ WRONG: "Between Tom and Jerry, ________ is faster."
       Answer: Tom  ← Tom is a proper noun, not a pronoun

    RULE: The answer to every pronoun question MUST be a pronoun word.
          If the answer is a person's name → rewrite the question.

  PREPOSITION:
    Blank requires a word showing position/direction
    ✅ "The helicopter flew ________ the dense forest."
       Answer: over / above

  CONJUNCTION — sub-type MUST match original:
    REASON conjunction (because / since / as):
    ✅ "She stayed home ________ she was feeling unwell."
       Answer: because
    ❌ WRONG: using TIME conjunction (when/while) for REASON pattern

    TIME conjunction (when / while / after / before):
    ✅ "The crowd cheered ________ the player scored."
       Answer: when
    ❌ WRONG: using REASON conjunction for TIME pattern

    CONTRAST conjunction (but / although / however):
    ✅ "He studied hard ________ he failed the exam."
       Answer: but / although

    RULE: Always identify WHICH TYPE of conjunction the original uses,
          then generate the SAME TYPE — never swap sub-types.

  INTERJECTION:
    Blank requires an exclamatory word at the start
    ✅ "________ We finally won the championship!"
       Answer: Hurray / Wow

Topic: VOCABULARY → NO verb hint — definition-based format ONLY
  ✅ "A ________ is a doctor who specializes in treating children."
  ✅ "The ________ of the painting left everyone speechless."
  ❌ FORBIDDEN: "A ________ (manage) person on the farm"

## ERROR CORRECTION FORMAT (if original is error correction):

  Each question must contain ONE sentence with ONE deliberate grammar error.
  The student must identify and correct the error.

  ✅ "She go to the market every morning."   Error: go → goes
  ✅ "He have been working since morning."   Error: have → has
  ✅ "They was playing football yesterday."  Error: was → were

  ❌ FORBIDDEN: Sentences with no errors
  ❌ FORBIDDEN: Sentences with more than one error
  ❌ FORBIDDEN: Converting to fill-in-the-blank format

## MATCHING FORMAT (if original is matching exercise):

  Generate two columns:
  Column A: terms, words, or sentence beginnings
  Column B: definitions, meanings, or sentence endings

  ✅ Example (vocabulary matching):
     Column A          Column B
     1. benevolent  →  A. harmful or destructive
     2. malicious   →  B. kind and generous
     3. resilient   →  C. able to recover quickly

  ❌ FORBIDDEN: Converting to fill-in-the-blank or MCQ format

## SHORT ANSWER FORMAT (Worksheet):

Topic: TENSE/VERB FORMS → grammar transformation only
  ✅ "Rewrite in past perfect: 'She finishes her homework.'"
  ✅ "Change to present continuous: 'He reads a book.'"

Topic: PREPOSITION → correction or usage
  ✅ "Correct the preposition: 'He is good in swimming.'"
  ✅ "Rewrite using 'since' or 'for': 'She has lived here from 2015.'"

Topic: ARTICLES → correction or usage
  ✅ "Fill in the correct article: '________ sun rises in the east.'"

Topic: PARTS OF SPEECH → identification or usage
  ✅ "Identify the part of speech of 'quickly' in: 'She ran quickly.'"

Topic: VOCABULARY → synonym / antonym / usage only
  ✅ "Replace the underlined word with its synonym: 'He is a brave soldier.'"
  ✅ "Write the antonym of 'generous' and use it in a sentence."
  ❌ FORBIDDEN: "What is the term for...?"
  ❌ FORBIDDEN: "What is the meaning of...?"

NEVER ask comprehension questions (Why/What happened) for worksheet content.

-------------------------------------------------------------------
IF CONTENT TYPE IS READING PASSAGE:
-------------------------------------------------------------------

Generate questions that test understanding of the passage content — NOT grammar.

## MCQ FORMAT (Passage):

  Factual questions directly answerable from the text:
  ✅ "What did the writer find under his foot?"
     A) A coin  B) A note  C) A key  D) Nothing
     Answer: A

  MCQ QUALITY RULES:

  1. COVERAGE RULE:
     - Distribute questions across ALL paragraphs
     - At least 1 question per paragraph if 5+ paragraphs exist
     - The FINAL paragraph is scientifically most important — give it at least 2 questions
     - NEVER skip any paragraph

  2. MANDATORY PARAGRAPH VERIFICATION:
        Before finalizing output, count the paragraphs (A, B, C, D, E...)
          Then verify:
            Paragraph A → at least 1 question ✔
            Paragraph B → at least 1 question ✔
            Paragraph C → at least 1 question ✔
            Paragraph D → at least 1 question ✔
            Paragraph E → at least 2 questions ✔ (most important)

        If ANY paragraph has 0 questions:
            → Find the most trivial existing question
            → Replace it with a question from the uncovered paragraph
            → Do NOT finalize output until all paragraphs are covered

  2. NO REPETITION RULE:
     - Each question must test a DIFFERENT fact
     - If two questions have the same answer → delete one, replace from uncovered paragraph

  3. ANSWER DISTRIBUTION RULE:
     - Correct answer MUST be distributed across A, B, C, D
     - A: at least 2 times
     - B: at least 2 times
     - C: at least 2 times
     - D: at least 1 time
     - NEVER leave any label with 0 correct answers
     - NEVER have more than 3 questions with the same answer label

  4. QUESTION WORDING RULE:
     - Question must make sense WITHOUT the answer
     - ❌ WRONG: "What did the Iceman carry that was an unfinished bow?"
     - ✅ RIGHT:  "What material was the Iceman's unfinished bow made from?"

  5. DISTRACTOR QUALITY RULE:
     - Wrong options must be plausible, not obviously wrong
     - ❌ "He was buried in a special tomb" ← obviously wrong
     - ✅ "He was wearing elaborate ceremonial garments" ← plausible

  6. LEVEL DISTRIBUTION RULE:
     - Factual questions (direct recall):     40% of total
     - Inferential questions (understand):    35% of total
     - Analytical questions (why/suggest):    25% of total

  7. For exactly 4 MCQ questions:
      - Q1 answer: A or D
      - Q2 answer: B or C  
      - Q3 answer: C or B
      - Q4 answer: D or A
      → All 4 labels must appear exactly once

## FILL IN THE BLANK FORMAT (Passage):

  WRONG APPROACH:
  ❌ Original sentence: "He wore shoes with bearskin soles."
  ❌ Output:            "He wore shoes with ________ soles."  ← sentence copy

  CORRECT APPROACH:
  ✅ Read the fact from the passage
  ✅ Write a NEW sentence that expresses the same fact differently
  ✅ Remove one KEY word as the blank

  EXAMPLE:
  Passage says: "...a birchbark container of embers wrapped in maple leaves"

  ❌ WRONG: "He carried a birchbark container of embers wrapped in ________ leaves."
  ✅ CORRECT: "To keep fire alive, the Iceman stored embers inside a container made of ________."
               Answer: birchbark
  ✅ CORRECT: "Scientists believe the Iceman grew up in Valle Isarco based on ________ in his teeth."
               Answer: isotopes

  LEVEL OF QUESTIONS — MANDATORY:
  - Factual     (4 questions): Direct facts, sentence rewritten
  - Inferential (3 questions): Requires understanding
  - Analytical  (3 questions): Why/What does it suggest/indicate

  BANNED — trivial blanks:
  ❌ "________ hollow"   Answer: rocky    (too easy, no understanding needed)
  ❌ "________ skin"     Answer: deer     (too easy, no understanding needed)

  RULE: Ask yourself — "Can someone answer this without understanding the passage?"
        If YES → rewrite the question.

  SELF-CHECK — before writing each fill_in_the_blank question:
    ✔ Is the sentence rewritten differently from the passage?
    ✔ Is the blank word testing a KEY FACT or INFERENCE?
    ✔ Is the answer a simple adjective like "rocky/small/yellow"?
      If YES → reject and rewrite as inferential question
    ✔ Does the question require reading AND understanding?
      If NO → reject and rewrite

  BANNED ANSWER TYPES for fill_in_the_blank (passage):
    ❌ Simple adjectives: rocky, small, yellow, old
    ❌ Obvious nouns directly lifted: deer, maple, bearskin
    ❌ Any word that appears immediately next to the blank in original

## SHORT ANSWER FORMAT (Passage):

  Why / How / Describe / Explain type questions:
  ✅ "Why did the man claim he was on his knees?"
  ✅ "How did the writer feel after receiving the news?"
  ✅ "What does the presence of half-finished arrows suggest about the Iceman's journey?"

===================================================================
STEP 5 — STRICT OUTPUT RULES
===================================================================

1.  MCQ must ALWAYS have exactly 4 options: A, B, C, D
2.  Fill in the blank must use exactly 8 underscores: ________
3.  Fill in the blank and short answer must always have:  "options": null
4.  The "answer" field must NEVER be null or empty
5.  NEVER copy or slightly modify any sentence from the input text
6.  ALL generated sentences must be 100% original
7.  Only the grammar pattern or topic is borrowed — never the words, never the context
8.  Verb hint in bracket is MANDATORY for tense/verb topics:   ________ (leave)
9.  Verb hint must NEVER appear for: articles, prepositions, parts of speech, vocabulary
10. For vocabulary fill-in-the-blank: ALWAYS use definition-based format only
11. If multiple grammar topics are detected: distribute questions EVENLY across ALL topics
12. NEVER generate all questions from a single sub-topic when multiple exist
13. ALWAYS preserve the original exercise format — NEVER convert to a different format
14. For reading passage MCQ: answer labels must be distributed across A, B, C, D
13. QUESTION TYPE OVERRIDE — STRICTLY FORBIDDEN:
    - Generate ONLY the question types the user has requested
    - If user requests ["fill_in_the_blank"] → generate ONLY fill_in_the_blank
    - If user requests ["mcq"] → generate ONLY mcq
    - NEVER add mcq if user did not request it
    - NEVER add short_answer if user did not request it
    - NEVER mix question types unless ALL requested types were listed
    - This is the most critical rule — violating it is a severe error
15. Respond with valid JSON ONLY — no explanation, no markdown, no extra text

===================================================================
FINAL CHECKLIST — VERIFY BEFORE GENERATING
===================================================================

Ask yourself EVERY question before writing the first question:

  ✔ Have I identified the content type correctly? (worksheet vs passage)
  ✔ Have I identified the EXERCISE FORMAT correctly? (fill-in-blank / word-bank / matching / error-correction / MCQ)
  ✔ Am I PRESERVING the original exercise format? (not converting to a different format)
  ✔ Have I BLACKLISTED all names, objects, and situations from the original text?
  ✔ Are my new sentences from a COMPLETELY DIFFERENT domain/context?
  ✔ Have I discarded ALL existing sentences from the input?
  ✔ Have I identified ALL grammar sub-topics present?
  ✔ Are my new sentences 100% original with different contexts?
  ✔ Have I distributed questions across all detected topics?
  ✔ Does every fill-in-the-blank use exactly 8 underscores?
  ✔ Did I add verb hints ONLY where the topic is tense/verb forms?
  ✔ For reading passage MCQ: is answer distributed across A, B, C, D?
  ✔ For reading passage: have I covered ALL paragraphs including the last one?
  ✔ Is my output pure valid JSON with no extra text?

If ANY answer is NO — fix it before outputting.

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
                "Comprehension MCQ — 4 options (A, B, C, D) directly based on passage content.\n"
                "\n"
                "RULE 1 — PARAGRAPH COVERAGE (MANDATORY):\n"
                "  Distribute questions across ALL paragraphs.\n"
                "  Count paragraphs first: A, B, C, D, E = 5 paragraphs.\n"
                "  Each paragraph must have at least 1 question.\n"
                "  The FINAL paragraph (E) must have at least 1 question.\n"
                "  VERIFY before submitting:\n"
                "    Para A → at least 1 question ✔\n"
                "    Para B → at least 1 question ✔\n"
                "    Para C → at least 1 question ✔\n"
                "    Para D → at least 1 question ✔\n"
                "    Para E → at least 1 question ✔\n"
                "  If any paragraph has 0 questions → replace most trivial question with one from that paragraph.\n"
                "\n"
                "RULE 2 — ANSWER DISTRIBUTION (MANDATORY):\n"
                "  Correct answers MUST appear across all 4 labels.\n"
                "  A: at least 1 correct answer\n"
                "  B: at least 1 correct answer\n"
                "  C: at least 1 correct answer\n"
                "  D: at least 1 correct answer\n"
                "  NEVER have more than 3 questions with the same answer label.\n"
                "  COUNT your answers before submitting: A=? B=? C=? D=?\n"
                "  If any label is missing → rewrite a question so that label becomes the answer.\n"
                "\n"
                "RULE 3 — FINAL PARAGRAPH QUESTION WITH ANSWER D (MANDATORY):\n"
                "  At least one question from the final paragraph MUST have answer D.\n"
                "  Final paragraph covers: South Tyrol Museum, isotopes, Wolfgang Müller,\n"
                "  Valle Isarco, Val Senales, mica chips in intestines.\n"
                "  To make answer D: put the correct fact as option D,\n"
                "  and make A, B, C plausible but wrong.\n"
                "  Examples:\n"
                "    ✅ 'Where has the Iceman's body been stored since 1998?'\n"
                "       A) University of Innsbruck\n"
                "       B) A research lab in London\n"
                "       C) A museum in Vienna\n"
                "       D) South Tyrol Museum of Archaeology in Bolzano\n"
                "       Answer: D\n"
                "\n"
                "    ✅ 'In which valley did scientists determine the Iceman probably grew up?'\n"
                "       A) Val Senales\n"
                "       B) Innsbruck Valley\n"
                "       C) Bolzano Valley\n"
                "       D) Valle Isarco\n"
                "       Answer: D\n"
                "\n"
                "RULE 4 — QUESTION LEVEL DISTRIBUTION:\n"
                "  Mix three levels:\n"
                "  - Factual (40%): directly answerable from text\n"
                "  - Inferential (35%): requires understanding\n"
                "  - Analytical (25%): what does it suggest / why\n"
                "\n"
                "RULE 5 — QUESTION WORDING:\n"
                "  Question must make sense WITHOUT the answer.\n"
                "  ❌ WRONG: 'What did the Iceman carry that was an unfinished bow?'\n"
                "  ✅ RIGHT:  'What material was the Iceman's unfinished bow made from?'\n"
                "\n"
                "RULE 6 — DISTRACTOR QUALITY:\n"
                "  Wrong options must be plausible — not obviously wrong.\n"
                "  ❌ 'He was buried in a golden tomb' ← obviously wrong\n"
                "  ✅ 'He was wearing elaborate ceremonial garments' ← plausible\n"
                "\n"
                "FINAL SELF-CHECK before submitting MCQ questions:\n"
                "  ✔ Have I covered ALL paragraphs including E?\n"
                "  ✔ Do answers include A, B, C, AND D?\n"
                "  ✔ Is there at least one question from the final paragraph with answer D?\n"
                "  ✔ Are wrong options plausible?\n"
                "  ✔ Does each question make sense without the answer?\n"
                "  If any answer is NO → fix before submitting.\n"
            ),
            QuestionType.FILL_IN_THE_BLANK: (
                "REWRITE a fact from the passage as a NEW sentence, then remove one KEY word as blank.\n"
                "  Use exactly 8 underscores: ________\n"
                "  NEVER copy the original sentence — ALWAYS express it differently.\n"
                "  Cover ALL paragraphs — especially paragraph E (isotopes, Müller, Valle Isarco).\n"
                "\n"
                "  BANNED answers — do NOT use these trivial words as blanks:\n"
                "    ❌ rocky, small, human, elderly, old, large, young, simple\n"
                "    ❌ Any word that appears immediately next to the blank in the original\n"
                "\n"
                "  CORRECT approach:\n"
                "    Passage says: 'birchbark container of embers wrapped in maple leaves'\n"
                "    ✅ 'To carry fire on his journey, the Iceman stored embers in a ________ container.'\n"
                "       Answer: birchbark\n"
                "    ✅ 'Scientists used ________ analysis in his teeth to determine where he grew up.'\n"
                "       Answer: isotope\n"
                "    ✅ 'The Iceman likely spent his final years near the ________ glacier.'\n"
                "       Answer: Val Senales\n"
                "\n"
                "  WRONG approach:\n"
                "    ❌ 'The Iceman was in his mid-________.'  Answer: 40s  ← trivial copy\n"
                "    ❌ 'oldest intact member of the ________ family'  Answer: human  ← trivial copy\n"
                "    ❌ 'discovered in a ________ hollow'  Answer: rocky  ← trivial copy\n"
                "\n"
                "  SELF-CHECK before each question:\n"
                "    'Can someone answer this without understanding the passage?' If YES → rewrite.\n"
                "    'Is this sentence copied from the passage?' If YES → rewrite.\n"
                "    'Is the answer a trivial adjective or obvious noun?' If YES → rewrite."
            ),
            QuestionType.SHORT_ANSWER: (
                "Why / How / Describe / Explain type questions — answerable in 1-2 sentences.\n"
                "  Must test UNDERSTANDING of passage content — not just recall.\n"
                "  Cover ALL paragraphs including the last one.\n"
                "  Example: 'What does the presence of half-finished arrows suggest about the Iceman?'\n"
                "           'How did scientists use isotope analysis to learn about the Iceman's life?'\n"
                "           'Why did Klaus Oeggl compare the death site to a paleo crime scene?'"
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

    ===================================================================
    MULTI-TOPIC DETECTION — MANDATORY FIRST STEP:
    ===================================================================
    
    Read the ENTIRE text carefully. It may contain MULTIPLE exercise sections.
    
    STEP 1 — Count how many distinct exercise sections exist:
      Look for:
      - Section headers: "Prepositions", "Exercise 1", "II. Fill in the blanks"
      - Different grammar topics: prepositions vs articles vs parts of speech vs tense
    
    STEP 2 — List ALL detected topics:
      Example:
      - Topic 1: Prepositions (time, place, direction, transport)
      - Topic 2: Parts of Speech (noun, verb, adverb, pronoun, conjunction, interjection)
      - Topic 3: Articles (a, an, the)
    
    STEP 3 — Calculate questions per topic:
      Total questions ÷ Number of topics = Questions per topic
      Example: 20 questions ÷ 3 topics = 7 + 7 + 6
    
    STEP 4 — Generate questions EVENLY from ALL topics:
      NEVER generate all questions from only the first topic.
      NEVER skip any detected topic.
      If 3 topics detected → each topic MUST have questions.
    
    ===================================================================
    CORRECT MULTI-TOPIC DISTRIBUTION EXAMPLE:
    ===================================================================
    
    Input has: Prepositions + Parts of Speech + Articles
    Total: 20 questions → Prepositions: 7, Parts of Speech: 7, Articles: 6
    
    Preposition (7):
      "The scientists worked ________ the lab closed."  Answer: until
      "She has been studying ________ morning."  Answer: since
      "We travelled ________ train."  Answer: by
    
    Parts of Speech (7):
      "The gardener waters plants in the ________."  Answer: garden
      "The detective examined evidence ________."  Answer: carefully
      "My sister is a teacher. ________ works at a school."  Answer: She
    
    Articles (6):
      "________ Nile is the longest river."  Answer: The
      "She needs ________ umbrella."  Answer: an
      "She is ________ university student."  Answer: a
    
    ===================================================================
    PER-TOPIC GENERATION RULES:
    ===================================================================
    
    PREPOSITIONS: No verb hint. Cover time, place, direction, transport sub-types.
    
    PARTS OF SPEECH:
      NOUN:        Answer = person/place/thing (NEVER an article or preposition)
      ADVERB:      Answer modifies a verb (carefully, bravely, quietly)
      PRONOUN:     Answer = pronoun word, NEVER a proper name
      CONJUNCTION: Sub-type MUST match original (reason/time/contrast)
      VERB:        Answer = action word
      INTERJECTION: Exclamatory word at start of sentence
    
    ARTICLES:
      Cover all rules: "the" for unique nouns, "an" for vowel sounds,
      "a" for consonant sounds (including "university" → "a university")
      Include tricky cases like silent h ("an honest man")
    
    FINAL VERIFICATION:
      ✔ Questions from ALL detected topics?
      ✔ Even distribution across topics?
      ✔ All sentences 100% original — not copied from input?
      ✔ Each answer is the correct grammar element?
      If any NO → fix before submitting."""

    else:
        content_instruction = """CONTENT TYPE: READING PASSAGE

    Generate questions that test comprehension and understanding of the passage.
    All questions must be directly based on facts, events, and ideas written in the text.
    
    MANDATORY RULES FOR READING PASSAGE:
    
    1. PARAGRAPH COVERAGE:
       - Count the paragraphs (A, B, C, D, E...)
       - Generate at least 1 question from EACH paragraph
       - The LAST paragraph is the most important — give it extra attention
       - NEVER skip any paragraph
    
    2. FILL IN THE BLANK — REWRITE RULE:
       - NEVER copy a sentence from the passage
       - ALWAYS rewrite the fact in a different sentence structure
       - Remove one KEY word as the blank
       - Banned answers: rocky, small, human, elderly (trivial words)
    
    3. MCQ ANSWER DISTRIBUTION:
       - Spread correct answers across A, B, C, D
       - At least one question must have answer A
       - At least one question must have answer B
       - At least one question must have answer C
       - At least one question must have answer D
       - NEVER have all answers as B"""

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
6.  Fill in the blank: REWRITE the sentence — NEVER copy from passage
7.  Fill in the blank: banned trivial answers: rocky, small, human, elderly
8.  MCQ: spread answers across A, B, C, D — at least one A and one D
9.  Cover ALL paragraphs — especially the final paragraph
10. Short answer must be answerable in 1-2 sentences
11. Output must be valid JSON only — no explanation, no markdown"""

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
      "question_text": "comprehension question based on passage",
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
      "question_text": "REWRITTEN sentence from passage with ________ blank",
      "options": null,
      "answer": "correct key word from passage"
    }},
    {{
      "question_number": 3,
      "question_type": "short_answer",
      "question_text": "Why/How/Describe/Explain question about passage content",
      "options": null,
      "answer": "correct answer in 1-2 sentences"
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