from openai import OpenAI, chat
DEEPSEEK_API_KEY="sk-7b05c0ae5d654e97a389e0acb50a499b"
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
            "content": "Tell me about something SDG?"
        }
    ]
)

print(response.choices[0].message.content)