# Smart-Exercise-Generator
This is a AI and Backend System Which Extract the text using OCR from image. Find the main context about the text using AI Model. Generate the  (MCQ/Fill in the Blank/Short Answer) Question for Practice. Which helping Student/Teacher both.


## Smart Exercise Generator Backend API
    
    This API serves as the backend for the Flutter mobile application.
    
    ### Workflow:
    1. **OCR Endpoints** — Upload images and extract text
    2. **Exercise Endpoints** — Generate questions from extracted text using AI
    
    ### Main Features:
    - Single/Multiple image OCR (Tesseract)
    - AI Exercise Generation (DeepSeek LLM)
    - MCQ, Fill in the Blank, Short Answer questions
    - Clean JSON responses for Flutter app
    
    ### Phases:
    - **Phase 1 (Current):** UI/UX + Frontend + OCR + Basic Exercise Generation
    - **Phase 2 (Next):** Full AI Integration, History tracking, PDF export backend