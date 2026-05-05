import redis
import os
import json
from pathlib import Path

from main import process_audio_file  # ⚠️ adapte si nécessaire
print("🚀 Initialisation des modèles...")

whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ====================== RAG (avec persistance) ======================
if os.path.exists(FAISS_INDEX_PATH):
    print("📦 Chargement de l'index FAISS existant...")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("📚 Création d'un nouvel index FAISS...")
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    docs = loader.load()
    print(f"✅ {len(docs)} documents chargés.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✅ Index FAISS créé avec {len(chunks)} chunks.")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20}
)

# ====================== LLM & AGENTS ======================
# ====================== LLM & AGENTS ======================
GROQ_API_KEY = "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"   # ← Colle ta clé ici

crew_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    temperature=0.1,
)

r = redis.Redis.from_url(
    os.environ["REDIS_URL"],
    ssl=True,
    decode_responses=True
)

print("Worker lancé...")

while True:
    job = r.brpop("audio_queue")  # attend un job

    if job:
        data = json.loads(job[1])

        job_id = data["job_id"]
        file_path = data["file_path"]

        print(f"Traitement job {job_id}")

        try:
            # traitement long (2 min)
            result = process_audio_file(file_path)

            # sauvegarde du résultat
            r.set(f"result:{job_id}", json.dumps(result))

            print(f"Job terminé {job_id}")

        except Exception as e:
            r.set(f"result:{job_id}", json.dumps({"error": str(e)}))
            print(f"Erreur job {job_id}: {e}")

        finally:
            # supprimer fichier temporaire
            Path(file_path).unlink(missing_ok=True)