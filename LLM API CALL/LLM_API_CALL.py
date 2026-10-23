# LLM API(GROQ_API) CALL WITH CUSTOM 
# 1. TEMPRETURE, 
# 2. SYS MSSG, 
# 3. TOKEN USAGES, 
# 4. FINISH REASON.

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY ERROR")

client = Groq(api_key = my_api_key)

model = "qwen/qwen3.8-27b"
role = "user"
prompt = "can i apraisal?"

message= {
    "role": role,
    "content" : prompt
}
sys_message= {
    "role" : "system",
    "content" : "you are a Strict HR"
}
messages = [sys_message, message]

response = client.chat.completions.create(
    model = model, 
    messages = messages, 
    temperature = 0, 
    max_tokens=100
    )

total_token_used = response.usage.prompt_tokens + response.usage.completion_tokens

finish_reason = ""
if (response.choices[0].finish_reason == "length"):
    finish_reason = "\nMAX TOKEN LIMIT IS REACHED !"

print(
    f"{response.choices[0].message.content}"
    f"\n\n\nTOTAL TOKEN USED: {total_token_used}"
    f"{finish_reason}"
)