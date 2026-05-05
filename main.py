
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

import redis
import uuid

# ====================== CONFIG ======================
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

warnings.filterwarnings("ignore")

app = FastAPI(title="ARM SAMU - API Régulation Médicale", version="1.0")

# ====================== RESPONSE MODEL ======================
class ProcessResponse(BaseModel):
    audio: str
    transcription: str
    duration_min: float
    crew_result: str
    ARM_Variables: Dict[str, Any]

# ====================== CONFIGURATION ======================
PDF_DIRECTORY = "DocumentsRag"
FAISS_INDEX_PATH = "faiss_index"
WHISPER_MODEL_SIZE = "small"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# ====================== ROLE ARM ======================
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



# ====================== SCHEMA ======================
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


# ====================== EXTRACTION VARIABLES ======================
def extract_variables_from_llm(text: str) -> Dict[str, Any]:
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
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        value = match.group(1).strip()

        if value.lower() in ["non spécifié", "null", "inconnu", "n/a", "aucun", "non renseigné"]:
            variables[key] = None
        elif value in ["0", "1"]:
            variables[key] = int(value)
        elif value.replace('.', '', 1).replace(',', '', 1).isdigit():
            variables[key] = float(value.replace(',', '.'))
        else:
            variables[key] = value

    return variables

# ====================== EXTRACTION ANALYSE ======================
def extract_full_analysis(text: str) -> Dict[str, Any]:
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

            if value.lower() not in ["null", "non spécifié", "aucun"]:
                if key in ["Termes_medicaux", "Centre_d_urgences"]:
                    analysis[key] = [v.strip() for v in value.split(",") if v.strip()]
                else:
                    analysis[key] = value

    return analysis

# ====================== POST PROCESS ======================
def strict_post_processing(data: Dict[str, Any], transcription: str, audio_duration: float) -> Dict[str, Any]:
    text = transcription.lower()
    v = data["Variables"]

    v["duree_min"] = round(audio_duration, 2)

    # AGE
    age_match = re.search(r"(\d+[.,]?\d*)\s*(ans?|an)", text)
    if age_match:
        v["age"] = float(age_match.group(1).replace(',', '.'))

    # SEXE
    if not v.get("sexe"):
        if any(w in text for w in ["femme", "madame", "elle"]):
            v["sexe"] = "F"
        elif any(w in text for w in ["homme", "monsieur", "il"]):
            v["sexe"] = "M"

    # DATE
    now = datetime.now()
    v["debut"] = now.strftime("%Y-%m-%dT%H:%M:%S")
    v["heure"] = now.hour

    return data

# ====================== PROCESS ======================
def process_audio_file(file_path: str, whisper_model, retriever, crew) -> Dict:
    segments, info = whisper_model.transcribe(file_path, beam_size=5)
    transcription = " ".join(s.text for s in segments).strip()
    duration_min = round(info.duration / 60, 2)

    context_docs = retriever.invoke(transcription)
    context = "\n\n".join([doc.page_content for doc in context_docs])

    crew_output = crew.kickoff(inputs={
        "transcription": transcription,
        "context": context,
        "VARIABLES_GUIDE": VARIABLES_GUIDE,
        "ROLE_ET_PIPELINE_ARM": ROLE_ET_PIPELINE_ARM
    })

    final_text = str(crew_output).split("Final Output:")[-1].strip()

    result_dict = deepcopy(ARM_SCHEMA)

    result_dict["Variables"].update(extract_variables_from_llm(final_text))
    result_dict.update(extract_full_analysis(final_text))

    result_dict = strict_post_processing(result_dict, transcription, duration_min)

    return {
        "audio": os.path.basename(file_path),
        "transcription": transcription,
        "duration_min": duration_min,
        "crew_result": final_text,
        "ARM_Variables": result_dict
    }

# ====================== REDIS ======================
REDIS_URL = os.environ.get("REDIS_URL")
if not REDIS_URL:
    raise Exception("REDIS_URL manquant")

r = redis.Redis.from_url(REDIS_URL, ssl=True, decode_responses=True)

# ====================== API ======================
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):

    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(400, "Format audio invalide")

    content = await file.read()

    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(400, "Fichier trop volumineux")

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    job_id = str(uuid.uuid4())
    temp_path = temp_dir / f"{job_id}_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(content)

    r.lpush("audio_queue", json.dumps({
        "job_id": job_id,
        "file_path": str(temp_path)
    }))

    return {"job_id": job_id}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    data = r.get(f"result:{job_id}")
    return json.loads(data) if data else {"status": "processing"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ====================== RUN ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)