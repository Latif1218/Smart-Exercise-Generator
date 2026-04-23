from openai import OpenAI, chat
DEEPSEEK_API_KEY="Ssk-fa5901e19ca04a7485740a2aa59ac190"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-chat"
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

response = client.chat.completions.create(
    model=DEEPSEEK_MODEL,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Bangladesh?"
        }
    ]
)

print(response.choices[0].message.content)