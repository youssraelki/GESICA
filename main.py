import os
import json
import re
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from faster_whisper import WhisperModel

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from crewai import Agent, Task, Crew, LLM

# ================== CONFIG ==================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY manquant !")

PDF_DIRECTORY = "DocumentsRag"

# ================== FASTAPI ==================
app = FastAPI(title="ARM SAMU API")

class ProcessResponse(BaseModel):
    audio: str
    transcription: str
    duration_min: float
    crew_result: str
    variables: Dict[str, Any]

# ================== LOAD MODELS (1 SEULE FOIS) ==================
print("🚀 Initialisation modèles...")

whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================== RAG ==================
if os.path.exists("faiss_index"):
    print("📦 Chargement FAISS...")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("📚 Création FAISS...")
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")

retriever = vectorstore.as_retriever()

# ================== LLM ==================
crew_llm = LLM(
    model="llama3-70b-8192",
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    temperature=0.1,
)

# ================== AGENTS ==================
analyste = Agent(
    role="Analyste SAMU",
    goal="Analyser appel",
    backstory="Expert SAMU",
    llm=crew_llm,
    verbose=False
)

verificateur = Agent(
    role="Vérificateur",
    goal="Corriger analyse",
    backstory="Médecin senior",
    llm=crew_llm,
    verbose=False
)

task_analyse = Task(
    description="Analyse cet appel : {transcription}",
    agent=analyste
)

task_verif = Task(
    description="Corrige : {context}",
    agent=verificateur,
    context=[task_analyse]
)

crew = Crew(
    agents=[analyste, verificateur],
    tasks=[task_analyse, task_verif]
)

# ================== UTIL ==================
def extract_variables(text):
    data = {}
    patterns = {
        "age": r"age\s*:\s*(\d+)",
        "sexe": r"sexe\s*:\s*(\w+)",
        "priorite": r"P[0-3]"
    }

    for k, p in patterns.items():
        m = re.search(p, text)
        if m:
            data[k] = m.group(1)

    return data

# ================== CORE ==================
def process_audio_file(path: str):

    segments, info = whisper_model.transcribe(path)
    transcription = " ".join([s.text for s in segments])
    duration = round(info.duration / 60, 2)

    context_docs = retriever.invoke(transcription)
    context = "\n".join([d.page_content for d in context_docs])

    result = crew.kickoff({
        "transcription": transcription,
        "context": context
    })

    variables = extract_variables(str(result))

    return {
        "audio": os.path.basename(path),
        "transcription": transcription,
        "duration_min": duration,
        "crew_result": str(result),
        "variables": variables
    }

# ================== API ==================
@app.post("/process-audio", response_model=ProcessResponse)
async def process_audio(file: UploadFile = File(...)):

    if not file.filename.endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(400, "Format non supporté")

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "Fichier trop volumineux")

    os.makedirs("temp", exist_ok=True)
    path = f"temp/{file.filename}"

    with open(path, "wb") as f:
        f.write(content)

    try:
        result = process_audio_file(path)
    finally:
        os.remove(path)

    return result

@app.get("/health")
def health():
    return {"status": "ok"}

# ================== MAIN ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)