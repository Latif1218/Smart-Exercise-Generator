from openai import OpenAI
from app.config import settings

client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url
)

response = client.chat.completions.create(
    model=settings.deepseek_model,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Bangladesh?"
        }
    ]
)

print(response.choices[0].message.content)