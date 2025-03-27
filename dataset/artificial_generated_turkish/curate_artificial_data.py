import openai
import requests
import os, random
import itertools
import csv
from time import sleep
from utils.log import logger

from dotenv import load_dotenv

load_dotenv()

voices = ["alloy", "ash", "coral", "fable", "onyx", "nova", "sage", "shimmer"]

# categories - disease types - outcomes - phases
# there are used to create various scenarios for the dataset
# to create prompts that will generate baseline text
medical_fields = [
    "Kardiyoloji", "Ortopedi", "Nöroşirurji", "Genel Cerrahi",
    "Gastroenteroloji Cerrahisi", "Onkolojik Cerrahi", "Üroloji",
    "Jinekoloji", "Plastik Cerrahi"
]
# # so many possibilities and no enough budget
#disease_types = ["Akut Durum", "Kronik Hastalık", "Kanser", "Travma", "Doğumsal Anomaliler"]
#outcomes = ["Başarılı Operasyon", "Başarısız Operasyon", "Komplikasyonlu Operasyon", "Kısmi Başarı"]
#phases = ["Pre-operatif Hazırlık", "Intra-operatif Durum", "Post-operatif Durum"]

client = openai.OpenAI()


# base prompt template
def format_prompt(field):
    # add disease, outcome, phase them if you have enough budget
    # and remove comments and activate lists
    # engineer the prompt
    return (
        f"{field} alanında gerçekleştirilen bir ameliyatın "
        f"Akut, kronik, travma vb. ile sonuçlandığı durumu anlatan detaylı bir ameliyat notu yaz. Eşsiz vakalar olsun."
        f"Kurgusal hasta adı, yaşı, cinsiyeti ve "
        f"ameliyat tarihi gibi bilgileri eklemeyi unutma. Hastalar Türk. Ameliyat notu en fazla 200 kelime olsun."
        f"Tarihleri açıkça yaz ve ses modeli rahat okusun."
    )

def generate_text(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()


def generate_audio(text, voice, output_path):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "wav"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Saved: {output_path}")
    else:
        logger.error(f"TTS Error: {response.status_code} - {response.text}")

def generate_dataset(split="train", start_id=1):
    os.makedirs(split, exist_ok=True)
    metadata_path = os.path.join(split, "metadata.csv")
    fieldnames = ["id", "prompt", "voice", "split"]

    # metadata
    write_header = not os.path.exists(metadata_path)
    with open(metadata_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        id_counter = start_id
        #combinations = list(itertools.product(medical_fields, disease_types, outcomes, phases))
        #selected_combos = medical_fields[:sample_count]
        selected_combos = medical_fields

        for combo in selected_combos:
            #prompt = format_prompt(*combo)
            prompt = format_prompt(combo)
            text = generate_text(prompt)
            # choose random voice
            voice = random.choice(voices)  

            file_id = f"{id_counter:05d}"
            text_path = os.path.join(split, f"{file_id}.txt")
            audio_path = os.path.join(split, f"{file_id}.wav")

            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            generate_audio(text, voice=voice, output_path=audio_path)

            writer.writerow({
                "id": file_id,
                "prompt": prompt,
                "voice": voice,
                "split": split
            })

            logger.info(f"[{split}] ID: {file_id} completed with voice: {voice}")
            id_counter += 1
            sleep(1.5)
            if id_counter == start_id + 65:
                break
        logger.info(f"[{split}] Dataset generation completed. {id_counter - start_id} files generated.")


# generate datasets
if __name__ == "__main__":
    generate_dataset(split="dataset/artificial_generated_turkish/train", start_id=19)
    generate_dataset(split="dataset/artificial_generated_turkish/dev", start_id=84)
    generate_dataset(split="dataset/artificial_generated_turkish/test", start_id=149)
    # "Eşsiz vakalar olsun." added to prompt on second run of main
    # also start_id are changed to +9 
    # each time :D (3 times)
