import redis
import os
import json
from pathlib import Path

# IMPORTS MANQUANTS
from faster_whisper import WhisperModel
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from crewai import Agent, Task, Crew, LLM

# IMPORT DE TON MAIN
from main import (
    process_audio_file,
    WHISPER_MODEL_SIZE,
    EMBEDDING_MODEL,
    PDF_DIRECTORY,
    FAISS_INDEX_PATH,
    VARIABLES_GUIDE,
    ROLE_ET_PIPELINE_ARM
)

print("🚀 Initialisation des modèles...")

# ====================== WHISPER ======================
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

# ====================== EMBEDDINGS ======================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ====================== RAG ======================
if os.path.exists(FAISS_INDEX_PATH):
    print("📦 Chargement FAISS...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("📚 Création FAISS...")
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20}
)

# ====================== LLM ======================
api_key = os.environ.get("GROQ_API_KEY")

crew_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
    temperature=0.1,
)

# ====================== AGENTS ======================
analyste = Agent(
    role="Analyste Expert en Régulation SAMU",
    goal="Analyser précisément l'appel en utilisant les protocoles RAG.",
    backstory="Tu es un régulateur SAMU expérimenté, rigoureux et structuré.",
    llm=crew_llm,
    verbose=False
)

verificateur = Agent(
    role="Vérificateur Médical Senior Rigoureux",
    goal="Vérifier, corriger et améliorer l'analyse.",
    backstory="Tu es très exigeant sur la précision médicale et la cohérence.",
    llm=crew_llm,
    verbose=False
)

# ====================== TASKS (tes tâches d'origine) ======================
task_analyse = Task(
    description=(
        "Analyse cet appel d'urgence de manière complète et professionnelle.\n\n"

        "==================== TRANSCRIPTION ====================\n"
        "{transcription}\n\n"

        "==================== CONTEXTE RAG ====================\n"
        "{context}\n\n"

        "==================== RÈGLES STRICTES ====================\n"
        "- La transcription est la SEULE source de vérité\n"
        "- Le RAG sert UNIQUEMENT à t'aider à raisonner sur la gravité et la décision.\n"
        "- Ne JAMAIS recopier ou mentionner le RAG\n"
        "- Ne JAMAIS écrire des exemples (P0, P1, etc.)\n"
        "- Ne PAS inventer d'informations absentes de la transcription.\n\n"

        "==================== GUIDE VARIABLES ====================\n"
        "{VARIABLES_GUIDE}\n\n"

        "==================== PIPELINE ARM ====================\n"
        "{ROLE_ET_PIPELINE_ARM}\n\n"

        "==================== FORMAT OBLIGATOIRE ====================\n\n"

        "**Raison de l'appel :**\n"
        "**Résumé :**\n"
        "**Termes médicaux :**\n"
        "**Patient / Famille :**\n"
        "**Centre d'urgences :**\n"
        "**Gravite :**\n"
        "**Decision :**\n"
        "**Orientation_patient :**\n"
        "**Conclusion :**\n"
        "**Recommendations :**\n\n"

        "==================== VARIABLES ====================\n"
        "- debut :\n"
        "- heure :\n"
        "- duree_min :\n"
        "- age :\n"
        "- sexe :\n"
        "- provenance :\n"
        "- motif_code :\n"
        "- motif_libelle :\n"
        "- priorite_ARM :\n"
        "- orientation_mru :\n"
        "- departement :\n"
        "- zone :\n"
        "- commune_dest :\n"
        "- carence_smur :\n"
        "- carence_any :\n"
        "- smur_envoye :\n"
        "- conseil_seul :\n"
        "- moyen_medicalise :\n"
        "- moyen_non_medicalise :\n"
        "- medecin_proximite :\n"
        "- modalite_transport :\n"
        "- niveau_decision :\n\n"

        "==================== RÈGLES VARIABLES ====================\n"
        "- Remplir toutes les variables\n"
        "- Si info absente → null\n"
        "- priorite_ARM cohérente avec gravité\n"
        "- smur_envoye = 1 uniquement si SMUR envoyé\n"
        "- conseil_seul = 1 si aucun moyen engagé\n"
        "- moyen_medicalise = 1 si SMUR ou équivalent\n"
        "- moyen_non_medicalise = 1 si ambulance simple\n"
        "- orientation_mru = 'Médecin régulateur' uniquement si P0 ou P1\n\n"
        "- duree_min : NE PAS estimer → laisser null (sera calculé par le système)\n"
"- age : extraire précisément depuis la transcription (ex: '2 ans et demi' → 2.5)\n"
"- sexe :\n"
"  - 'mon fils', 'garçon','homme','monsieur','père','voisin','cousin','frère','grand-père','il','petit','grand' → M\n"
"  - 'ma fille', 'femme', 'madame','mère','grande-mère','voisine','cousine','elle','petite','grande','soeur','tante' → F\n"
"- motif_libelle : court, précis (2 à 4 mots max)\n"
"- Raison de l'appel : 2 à 4 mots MAXIMUM (ex: 'détresse respiratoire enfant')\n"
"- provenance : déduire si possible (ex: domicile si appel familial)\n"
"- Patient / Famille :\n"
"  - Identifier QUI appelle (mère, père, voisin, etc.)\n"
"  - Si inconnu → 'proche'\n"
"- Centre d'urgences :\n"
"  - Lister les questions posées OU attendues par un ARM\n"
"  - Même si absentes → les ajouter selon protocole SAMU\n"
"  - Exemple : localisation, conscience, respiration, âge\n"
"- modalite_transport : TEXTE clair (ex:'ambulance','voiture','automobile','moteur','aucun')\n"
"- Orientation_patient :\n"
"  - Indiquer clairement :\n"
"    → 'Médecin régulateur'\n"
"    → OU 'Conseil médical'\n"
"    → OU 'Envoi SMUR / ambulance'\n"
"- niveau_decision :\n"
"  - 'ARM seul' ou 'Médecin régulateur'\n"
"- Les variables sont AUSSI IMPORTANTES que l'analyse\n"
"- Une réponse sans variables correctes est invalide\n"

       
        "==================== IMPORTANT ====================\n"
        "- Répondre STRICTEMENT dans ce format\n"
        "- Ne rien ajouter avant ou après\n"
        "- Le RAG doit être invisible dans la réponse\n"
    ),
    expected_output="Analyse + variables complètes sans pollution RAG",
    agent=analyste
)

task_verification = Task(
    description=(
        "Tu es un Vérificateur Médical Senior très rigoureux.\n\n"

        "Analyse à vérifier :\n"
        "{context}\n\n"

        "RÈGLES :\n"
        "- Vérifier cohérence médicale\n"
        "- Corriger uniquement si nécessaire\n"
        "- Ne pas inventer\n"
        "- Améliorer clarté et précision\n"
        "- Vérifier cohérence des VARIABLES\n"
        "- Si P0 ou P1 → orientation_mru = Médecin régulateur\n\n"

        "IMPORTANT :\n"
        "- Garder STRICTEMENT le même format\n"
        "- Ne rien ajouter avant ou après\n"
        "- Ne jamais introduire de contenu RAG\n"
    ),
    expected_output="Analyse corrigée + variables cohérentes",
    agent=verificateur,
    context=[task_analyse]
)

crew = Crew(
    agents=[analyste, verificateur],
    tasks=[task_analyse, task_verification],
    verbose=True
)
# ====================== REDIS ======================
r = redis.Redis.from_url(
    os.environ["REDIS_URL"],
    ssl=True,
    decode_responses=True
)

print("✅ Worker lancé...")

# ====================== LOOP ======================
while True:
    job = r.brpop("audio_queue")

    if not job:
        continue

    try:
        data = json.loads(job[1])

        job_id = data["job_id"]
        file_path = data["file_path"]

        print(f"🎯 Traitement job {job_id}")

        # ✅ CORRECTION MAJEURE ICI
        result = process_audio_file(
            file_path,
            whisper_model,
            retriever,
            crew
        )

        r.set(f"result:{job_id}", json.dumps(result))

        print(f"✅ Terminé {job_id}")

    except Exception as e:
        print(f"❌ Erreur: {e}")

        try:
            r.set(f"result:{job_id}", json.dumps({
                "error": str(e)
            }))
        except:
            pass

    finally:
        try:
            Path(file_path).unlink(missing_ok=True)
        except:
            pass