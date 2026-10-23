import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=key)

model = "qwen/qwen3.8-27b"
app = FastAPI()

# qwen3.8 thinks by default; turn thinking off so we get clean JSON / clean streamed text
no_think = {"reasoning_effort": "none", "reasoning_format": "hidden"}


# ---------- schemas ----------
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


resume_schema = Resume.model_json_schema()


class ChatRequest(BaseModel):
    question: str


# ---------- pdf extraction ----------
def read_pdf(file_path: Path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ---------- resume parsing (LLM -> JSON -> Resume) ----------
def parse_resume(resume_text):
    sys_mssg = {
        "role": "system",
        "content": f"""
You are an expert resume parser.
Extract information based on meaning, not only exact section headings
(Experience / Work History / Employment / Internships can all hold experience).
Skills may appear in the skills section, experience, internships or projects.

Return ONLY valid JSON matching this schema:
{resume_schema}

Rules:
1. Do not invent information.
2. If a value is not available, return null.
3. If a list has no information, return an empty list.
4. Include internships inside experiences.
5. Extract skills mentioned across the entire resume.
""",
    }
    message = {"role": "user", "content": f"Parse the following resume:\n\n{resume_text}"}
    messages = [sys_mssg, message]

    response = client.chat.completions.create(
        messages=messages,
        model=model,
        response_format={"type": "json_object"},
        extra_body=no_think,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    return Resume(**data)


# ---------- answer as the candidate (streaming) ----------
def ask_candidate(question: str, resume: Resume):
    sys_mssg = {
        "role": "system",
        "content": f"""
You are an AI assistant representing a job candidate.
Below is everything you know about the candidate.

{resume.model_dump_json(indent=2)}

Rules:
1. Answer only using this information.
2. Never hallucinate.
3. If information is unavailable, say "I don't have enough information to answer that."
4. Be professional.
5. Answer as if HR is interviewing this candidate.
""",
    }
    message = {"role": "user", "content": question}
    messages = [sys_mssg, message]

    stream = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        extra_body=no_think,
    )
    return (chunk.choices[0].delta.content or "" for chunk in stream)


# ---------- parse the resume ONCE at startup ----------
resume = parse_resume(read_pdf(Path("RESUME.pdf")))


# ---------- API ----------
@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/resume")
def download_resume():
    return FileResponse(
        "RESUME.pdf",
        media_type="application/pdf",
        filename="Nishant_Awasthi_Resume.pdf",   # makes the browser download it
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    return StreamingResponse(
        ask_candidate(request.question, resume),
        media_type="text/plain",
    )