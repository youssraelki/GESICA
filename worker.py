import redis
import os
import json
from pathlib import Path

from main import process_audio_file  # ⚠️ adapte si nécessaire

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