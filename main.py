import os
import json
import re
import warnings
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any


from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from faster_whisper import WhisperModel
import redis
import uuid

# LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# CrewAI
from crewai import Agent, Task, Crew, LLM

# ====================== CONFIG ======================
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

warnings.filterwarnings("ignore")

app = FastAPI(title="ARM SAMU - API Régulation Médicale", version="1.0")

class ProcessResponse(BaseModel):
    audio: str
    transcription: str
    duration_min: float
    crew_result: str
    ARM_Variables: Dict[str, Any]

# ====================== CONFIGURATION ======================
PDF_DIRECTORY = "DocumentsRag"
FAISS_INDEX_PATH = "faiss_index"
WHISPER_MODEL_SIZE = "small"          # tu peux passer à "tiny" pour plus de vitesse
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# ====================== CHARGEMENT AU STARTUP ======================
whisper_model = None
retriever = None
crew_llm = None
ROLE_ET_PIPELINE_ARM = """1. Le rôle de l’ARM
L’ARM (Assistant de Régulation Médicale) est la première personne qui répond aux appels du 15. 
Il joue un rôle essentiel dans la gestion des urgences. Il filtre les appels, sécurise la situation et prépare la décision médicale.

Rôles principaux :
- Décrocher rapidement
- Localiser précisément l’appelant
- Comprendre la situation
- Évaluer l’urgence
- Transmettre au médecin régulateur
- Engager des moyens si nécessaire

2. Le pipeline de traitement d’un appel

Étape 1 : Réception
Objectifs : rassurer l’appelant et obtenir les informations essentielles rapidement.
Exemple : Une personne appelle pour un malaise → l’ARM demande immédiatement l’adresse exacte et le numéro de téléphone.

Étape 2 : Priorisation
Classification par l’ARM :
- P0 : urgence vitale immédiate (ex : arrêt cardiaque, détresse respiratoire aiguë)
- P1 : urgence grave (ex : douleur thoracique, AVC suspecté)
- P2 : urgence relative
- P3 : non urgent

Exemple : Une personne inconsciente est classée P0 immédiatement.

Étape 3 : Diagnostic médical
Le médecin régulateur prend le relais et affine l’analyse en posant des questions ciblées sur les symptômes.

Étape 4 : Décision
- R1 : SMUR
- R2 : Ambulance
- R3 : Médecin de garde / proximité
- R4 : Conseil médical seul

Étape 5 : Action
L’ARM exécute la décision : envoi des secours, coordination, suivi et maintien en ligne si nécessaire.

3. Types de questions posées par l’ARM
A. Localisation : Où êtes-vous exactement ?
B. Identification : Qui appelle ? Quel est le numéro ?
C. Motif : Que se passe-t-il ?
D. Gravité : La personne est-elle consciente ? Respire-t-elle normalement ?
E. Contexte : Antécédents médicaux ? Âge ? Traitements en cours ?

4. Actions en parallèle
- Déclenchement rapide des secours si nécessaire
- Maintien de l’appelant en ligne
- Instructions téléphoniques (massage cardiaque, position latérale de sécurité, etc.)

Exemple : En cas d’arrêt cardiaque, l’ARM guide le massage cardiaque avant même l’arrivée des secours.

5. Conclusion
L’ARM est un acteur clé du système d’urgence : il est à la fois filtreur, coordinateur et garant de la sécurité des patients. 
Sans lui, le système de prise en charge des urgences serait inefficace."""

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
# ====================== SCHEMA + FONCTIONS (de ton 1er code) ======================
# ====================== SCHEMA ARM ======================
ARM_SCHEMA: Dict[str, Any] = {
    "Variables": {
        "debut": None, 
        "heure": None, 
        "duree_min": None,
        "departement": None, 
        "zone": None, 
        "commune_dest": "Non renseigné",
        "age": None, 
        "sexe": None, 
        "provenance": None,
        "motif_code": None, 
        "motif_libelle": None,
        "carence_smur": 0, 
        "carence_any": 0,
        "smur_envoye": 0, 
        "conseil_seul": 0,
        "moyen_medicalise": 0, 
        "moyen_non_medicalise": 0, 
        "medecin_proximite": 0,
        "orientation_mru": None, 
        "modalite_transport": None,
        "priorite_ARM": None, 
        "niveau_decision": None
    },
    "Raison_de_l_appel": None,
    "Resume": None,
    "Termes_medicaux": [],
    "Patient_Famille": None,
    "Centre_d_urgences": [],
    "Gravite": None,
    "Decision": None,
    "Orientation_patient": None,
    "Conclusion": None,
    "Recommendations": None
}
# ← Ton schéma complet

VARIABLES_GUIDE = """Significations exactes des variables (à respecter strictement) :

- debut              : Date et heure exacte de l'appel au format ISO datetime (ex: 2025-04-29T14:35:00)
- heure              : Heure de l'appel extraite du champ 'debut' (entier entre 0 et 23)
- duree_min          : Durée totale de l'appel en minutes (entier positif)
- age                : Âge du patient en années (entier). NaN ou -1 si inconnu.
- sexe               : Sexe du patient ("F" pour féminin, "M" pour masculin)
- provenance         : Origine de l'appel (domicile, Ehpad, voie publique, cabinet médical, etc.)
- motif_code         : Code du motif de l'appel (si disponible dans le système)
- motif_libelle      : Libellé texte du motif principal de l'appel
- priorite_ARM       : Niveau de priorité attribué par l'ARM (P0, P1, P2, P3, ...)
- orientation_mru    : Orientation de l'appel. Mettre "Médecin régulateur" uniquement si priorite_ARM est P0 ou P1.
                       Sinon, indiquer l'orientation réelle (Conseil médical, SMUR, Urgences, Médecin traitant, etc.)
- departement        : Département de l'appel (code à 2 ou 3 caractères)
- zone               : Zone géographique ou secteur de régulation
- commune_dest       : Commune de destination ou commune de l'appelant
- carence_smur       : 1 si carence SMUR, 0 sinon
- carence_any        : 1 si carence sur un moyen quelconque, 0 sinon
- smur_envoye        : 1 si un SMUR a été envoyé, 0 sinon
- conseil_seul       : 1 si seul un conseil médical a été donné, 0 sinon
- moyen_medicalise   : 1 si un moyen médicalisé a été engagé (SMUR, ambulance médicalisée, etc.)
- moyen_non_medicalise : 1 si un moyen non médicalisé a été engagé (ambulance simple, VSL, etc.)
- medecin_proximite  : 1 si un médecin de proximité a été envoyé, 0 sinon
- modalite_transport : Modalité de transport retenue (ex: Ambulancier, SMUR, VSL, personnel, etc.)
- niveau_decision    : Niveau de décision (ARM, MRU, etc.)
"""


def extract_variables_from_llm(text: str) -> Dict[str, Any]:
    """
    Extrait les variables à partir de la réponse du LLM (format markdown **variable : valeur**)
    """
    variables = {}
   
    patterns = {
        "age": r"\*\*age\s*:\*\*\s*([^\n]+)",
        "sexe": r"\*\*sexe\s*:\*\*\s*([^\n]+)",
        "provenance": r"\*\*provenance\s*:\*\*\s*([^\n]+)",
        "motif_code": r"\*\*motif_code\s*:\*\*\s*([^\n]+)",
        "motif_libelle": r"\*\*motif_libelle\s*:\*\*\s*([^\n]+)",
        "priorite_ARM": r"\*\*priorite_ARM\s*:\*\*\s*([^\n]+)",
        "orientation_mru": r"\*\*orientation_mru\s*:\*\*\s*([^\n]+)",
        "departement": r"\*\*departement\s*:\*\*\s*([^\n]+)",
        "zone": r"\*\*zone\s*:\*\*\s*([^\n]+)",
        "commune_dest": r"\*\*commune_dest\s*:\*\*\s*([^\n]+)",
        "carence_smur": r"\*\*carence_smur\s*:\*\*\s*([^\n]+)",
        "carence_any": r"\*\*carence_any\s*:\*\*\s*([^\n]+)",
        "smur_envoye": r"\*\*smur_envoye\s*:\*\*\s*([^\n]+)",
        "conseil_seul": r"\*\*conseil_seul\s*:\*\*\s*([^\n]+)",
        "moyen_medicalise": r"\*\*moyen_medicalise\s*:\*\*\s*([^\n]+)",
        "moyen_non_medicalise": r"\*\*moyen_non_medicalise\s*:\*\*\s*([^\n]+)",
        "medecin_proximite": r"\*\*medecin_proximite\s*:\*\*\s*([^\n]+)",
        "modalite_transport": r"\*\*modalite_transport\s*:\*\*\s*([^\n]+)",
        "niveau_decision": r"\*\*niveau_decision\s*:\*\*\s*([^\n]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)  # Ajout de IGNORECASE pour plus de robustesse
        if match:
            value = match.group(1).strip()
           
            if value.lower() in ["non spécifié", "null", "inconnu", "n/a", "aucun", "non renseigné"]:
                variables[key] = None
            elif value in ["0", "1"]:
                variables[key] = int(value)
            elif value.replace('.', '', 1).replace(',', '', 1).isdigit():  # pour les âges décimaux
                variables[key] = float(value.replace(',', '.'))
            else:
                variables[key] = value

    return variables
def extract_full_analysis(text: str) -> Dict[str, Any]:
    """Extrait les sections principales (Raison, Résumé, etc.)"""
    analysis = {
        "Raison_de_l_appel": None,
        "Resume": None,
        "Termes_medicaux": [],
        "Patient_Famille": None,
        "Centre_d_urgences": [],
        "Gravite": None,
        "Decision": None,
        "Orientation_patient": None,
        "Conclusion": None,
        "Recommendations": None
    }

    sections = {
        "Raison de l'appel": "Raison_de_l_appel",
        "Résumé": "Resume",
        "Termes médicaux": "Termes_medicaux",
        "Patient / Famille": "Patient_Famille",
        "Centre d'urgences": "Centre_d_urgences",
        "Gravite": "Gravite",
        "Decision": "Decision",
        "Orientation_patient": "Orientation_patient",
        "Conclusion": "Conclusion",
        "Recommendations": "Recommendations"
    }

    for section_fr, key in sections.items():
        pattern = rf"\*\*{section_fr}\s*:\*\*(.*?)(?=\*\*|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value and value.lower() not in ["null", "non spécifié", "aucun"]:
                if key in ["Termes_medicaux", "Centre_d_urgences"]:
                    analysis[key] = [v.strip() for v in value.split(",") if v.strip()]
                else:
                    analysis[key] = value

    return analysis
# ====================== POST-PROCESSING ======================
def strict_post_processing(data: Dict[str, Any], transcription: str, audio_duration: float) -> Dict[str, Any]:
    text = transcription.lower()
    v = data["Variables"]

    v["duree_min"] = round(audio_duration, 2)

    # Age
    age_match = re.search(r"(\d+[.,]?\d*)\s*(ans?|an)", text)
    if age_match:
        v["age"] = float(age_match.group(1).replace(',', '.'))

    # Sexe
    if not v.get("sexe"):
        if any(w in text for w in ["femme", "dame", "madame", "épouse", "fille", "mère", "grande-mère", "elle"]):
            v["sexe"] = "F"
        elif any(w in text for w in ["homme", "monsieur", "mari", "garçon", "père", "grand-père", "il"]):
            v["sexe"] = "M"

    # Date/heure
    now = datetime.now()
    v["debut"] = now.strftime("%Y-%m-%dT%H:%M:%S")
    v["heure"] = now.hour

    # Variables par défaut
    for key in ["debut", "heure", "duree_min", "age", "sexe", "provenance", "motif_code", 
                "motif_libelle", "priorite_ARM", "orientation_mru", "departement", "zone", 
                "commune_dest", "modalite_transport", "niveau_decision"]:
        if key not in v or v[key] == "":
            v[key] = None

    for key in ["carence_smur", "carence_any", "smur_envoye", "conseil_seul", 
                "moyen_medicalise", "moyen_non_medicalise", "medecin_proximite"]:
        if key not in v or v[key] == "":
            v[key] = 0

    # Cohérence
    if v.get("priorite_ARM") in ["P0", "P1"]:
        v["orientation_mru"] = "Médecin régulateur"
        v["niveau_decision"] = "Médecin régulateur"
        v["smur_envoye"] = 1

    if v.get("conseil_seul") == 1:
        v["smur_envoye"] = 0
        v["moyen_medicalise"] = 0
        v["moyen_non_medicalise"] = 0
        v["medecin_proximite"] = 0

    return data
# ====================== PROCESS ======================
# ====================== PROCESS ======================
# ====================== PROCESS ======================
def process_audio_file(file_path: str) -> Dict:
    # Transcription
    segments, info = whisper_model.transcribe(file_path, beam_size=5)
    transcription = " ".join(s.text for s in segments).strip()
    duration_min = round(info.duration / 60, 2)

    # RAG
    context_docs = retriever.invoke(transcription)
    context = "\n\n".join([doc.page_content for doc in context_docs])

    # CrewAI
    crew_output = crew.kickoff(inputs={
        "transcription": transcription,
        "context": context,
        "VARIABLES_GUIDE": VARIABLES_GUIDE,
        "ROLE_ET_PIPELINE_ARM": ROLE_ET_PIPELINE_ARM
    })

    # Nettoyage du texte
    final_text = str(crew_output)
    if "Final Output:" in final_text:
        final_text = final_text.split("Final Output:")[-1].strip()

    # ====================== EXTRACTION ======================
    result_dict = deepcopy(ARM_SCHEMA)
    
    # Extraction des variables
    llm_vars = extract_variables_from_llm(final_text)
    result_dict["Variables"].update(llm_vars)

    # Extraction des sections principales (Raison, Résumé, etc.)
    full_analysis = extract_full_analysis(final_text)
    result_dict.update(full_analysis)

    # Post-processing
    result_dict = strict_post_processing(result_dict, transcription, duration_min)

    return {
        "audio": os.path.basename(file_path),
        "transcription": transcription,
        "duration_min": duration_min,
        "crew_result": final_text,
        "ARM_Variables": result_dict
    }
# ====================== API ENDPOINTS ======================

# connexion Redis (Railway injecte REDIS_URL)
r = redis.Redis.from_url(
    os.environ["REDIS_URL"],
    ssl=True,
    decode_responses=True
)


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):

    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(400, "Seuls les fichiers audio sont acceptés")

    content = await file.read()

    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(400, "Fichier trop volumineux (max 15MB)")

    # dossier temporaire
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    job_id = str(uuid.uuid4())
    temp_path = temp_dir / f"{job_id}_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(content)

    # envoyer le job dans Redis
    job_data = {
        "job_id": job_id,
        "file_path": str(temp_path)
    }

    r.lpush("audio_queue", json.dumps(job_data))

    return {
        "message": "job lancé",
        "job_id": job_id
    }


@app.get("/result/{job_id}")
def get_result(job_id: str):
    data = r.get(f"result:{job_id}")

    if not data:
        return {"status": "processing"}

    return json.loads(data)


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)