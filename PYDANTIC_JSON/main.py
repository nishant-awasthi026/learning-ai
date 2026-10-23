# LLM API(GROQ_API) CALL WITH CUSTOM
# 1. TEMPERATURE,
# 2. SYS MESSAGE,
# 3. TOKEN USAGE,
# 4. FINISH REASON,
# 5. PYDANTIC + JSON EXTRACTION (validate + retry).

import os
import re
import json
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY ERROR")

client = Groq(api_key=my_api_key)

model = "qwen/qwen3.8-27b"
max_retries = 2


# ---------------------------------------------------------------------------
# STEP 1: SCHEMA -> the predefined things to extract (edit this for your use case)
# ---------------------------------------------------------------------------
class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AppraisalRequest(BaseModel):
    employee_name: Optional[str] = Field(None, description="Name of the employee, only if stated.")
    job_role: Optional[str] = Field(None, description="Current job title or role.")
    years_of_experience: Optional[float] = Field(None, ge=0, description="Years in the company or role, as a number.")
    achievements: list[str] = Field(default_factory=list, description="Concrete achievements mentioned. Empty list if none.")
    appraisal_requested: bool = Field(..., description="True if the employee is asking for appraisal/hike/promotion.")
    expected_hike_percent: Optional[float] = Field(None, ge=0, le=100, description="Hike percent expected, number only.")
    reason: Optional[str] = Field(None, description="Main reason the employee gives for the request.")
    urgency: Urgency = Field(..., description="Urgency implied by the text.")


# ---------------------------------------------------------------------------
# STEP 2: SYSTEM MESSAGE -> persona + rules + JSON schema generated from Pydantic
# ---------------------------------------------------------------------------
schema_text = json.dumps(AppraisalRequest.model_json_schema(), indent=2)

sys_message = {
    "role": "system",
    "content": (
        "You are a Strict HR. Read the employee's long, vague message and extract "
        "ONLY the requested fields.\n"
        "Rules:\n"
        "- Use only information stated in the text. Never guess or invent.\n"
        "- If an optional field is not mentioned, use null.\n"
        "- Return ONLY a valid JSON object that matches this JSON schema, nothing else:\n"
        f"{schema_text}"
    ),
}

# long vague paragraph (your input text)
paragraph = """
Hi, so I have been working here for almost 3 years now as a backend developer,
my name is Amit Verma by the way, and honestly I feel like my work has not been
recognised properly. Last year I moved our payment service to a new queue
system which cut the failures a lot, and I also trained two new joiners.
My friend in another team got a good hike recently so I was thinking maybe
15 percent would be fair for me. I am not in a big hurry but the appraisal
cycle closes at the end of this month so it would be nice to hear soon.
"""

message = {
    "role": "user",
    "content": f"<text>\n{paragraph}\n</text>\nReturn the JSON.",  # word "JSON" is required for JSON mode
}

messages = [sys_message, message]


# ---------------------------------------------------------------------------
# STEP 3: CALL + VALIDATE + RETRY
# ---------------------------------------------------------------------------
def clean_json_text(text):
    # reasoning models (like Qwen) can add <think>...</think>, remove it before parsing
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


total_token_used = 0
finish_reason = ""
result = None

for attempt in range(max_retries + 1):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,                            # JSON needs more room than 100
        response_format={"type": "json_object"},    # forces valid JSON output
    )

    total_token_used += response.usage.prompt_tokens + response.usage.completion_tokens
    choice = response.choices[0]

    # FINISH REASON: if cut off, the JSON is incomplete, so retrying will not help
    if choice.finish_reason == "length":
        finish_reason = "\nMAX TOKEN LIMIT IS REACHED !"
        break

    raw = clean_json_text(choice.message.content)

    try:
        # parses the JSON string AND validates types/enums/ranges in one go
        result = AppraisalRequest.model_validate_json(raw)
        break
    except ValidationError as e:
        errors = "\n".join(
            f"- {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        print(f"[attempt {attempt + 1}] VALIDATION FAILED:\n{errors}\n")

        # send the exact errors back so the model can fix its own output
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"Your JSON failed validation:\n{errors}\nReturn the corrected JSON only.",
        })


# ---------------------------------------------------------------------------
# STEP 4: OUTPUT
# ---------------------------------------------------------------------------
if result:
    print(result.model_dump_json(indent=2))   # JSON string
    # result.model_dump()                     # plain dict (for DB / API use)
else:
    print("EXTRACTION FAILED")

print(
    f"\n\n\nTOTAL TOKEN USED: {total_token_used}"
    f"{finish_reason}"
)

if result:
    print(result.employee_name)         # "Amit Verma"
    print(result.job_role)              # "backend developer"
    print(result.expected_hike_percent) # 15.0
    print(result.achievements[0])       # lists work like normal lists
    print(result.urgency.value)         # enums need .value to get the plain string, e.g. "medium"