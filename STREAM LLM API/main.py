import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

mera_api = os.getenv('GROQ_API_KEY')
if not mera_api:
    raise ValueError("API ka fault hain")

client = Groq(api_key=mera_api)

model = "qwen/qwen3.8-27b"
role = "user"
prompt = "write simple universal binary search code in cpp"

message = {
    "role":role,
    "content":prompt
}
messages=[message]

stream = client.chat.completions.create(
    messages=messages,
    model=model,
    stream=True
)

for chunk in stream:
    content  = chunk.choices[0].delta.content
    if content:
        print(content,end="",flush=True)