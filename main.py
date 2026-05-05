import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException

from faster_whisper import WhisperModel

# LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# CrewAI
from crewai import Agent, Task, Crew, LLM

# ================= CONFIG =================
app = FastAPI()

PDF_DIRECTORY = "DocumentsRag"
FAISS_INDEX_PATH = "faiss_index"

# ⚠️ IMPORTANT
WHISPER_MODEL_SIZE = "tiny"

# ================= GLOBAL (lazy load) =================
whisper_model = None
retriever = None
crew = None

# ================= LOAD MODELS =================
def load_models():
    global whisper_model, retriever, crew

    # 🔹 Whisper
    if whisper_model is None:
        print("🔄 Loading Whisper...")
        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )

    # 🔹 Embeddings + FAISS
    if retriever is None:
        print("🔄 Loading FAISS...")

        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"}
        )

        if os.path.exists(FAISS_INDEX_PATH):
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(docs)

            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 🔹 CrewAI
    if crew is None:
        print("🔄 Loading CrewAI...")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise Exception("❌ GROQ_API_KEY manquante")

        llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            temperature=0.1,
        )

        analyst = Agent(
            role="Analyste",
            goal="Analyser appel",
            backstory="Expert SAMU",
            llm=llm
        )

        task = Task(
            description="Analyse: {transcription}",
            agent=analyst
        )

        crew = Crew(agents=[analyst], tasks=[task])


# ================= PROCESS =================
def process_audio_file(file_path: str) -> Dict:
    load_models()

    # 🔹 Transcription
    segments, info = whisper_model.transcribe(file_path)
    transcription = " ".join([s.text for s in segments])

    # 🔹 RAG
    docs = retriever.invoke(transcription)
    context = " ".join([d.page_content for d in docs])

    # 🔹 LLM
    result = crew.kickoff({
        "transcription": transcription,
        "context": context
    })

    return {
        "audio": os.path.basename(file_path),
        "transcription": transcription,
        "result": str(result)
    }


# ================= API =================
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):

    if not file.filename.endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(400, "Format invalide")

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "Fichier trop volumineux")

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    temp_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        result = process_audio_file(str(temp_path))
        return result

    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}