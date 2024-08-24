import os
import openai
from langchain import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = "Which countries boarders Nigeria."

response = openai.Completion.create(
    engine = "text-embedding-ada-002",
    prompt = prompt,
    temperature = 0.4,
    max_tokens = 64
)

print(response.choices[0].text)