# Medical Notes - Speech-to-Text

## Project Overview
Medical Notes - Speech-to-Text is a system designed to streamline the documentation of surgical notes by converting spoken records into text and storing them digitally. This tool enables healthcare professionals to efficiently document critical information during or after surgeries, particularly in high-pressure environments like emergency procedures. The project aims to enhance the accuracy and accessibility of patient records while reducing the administrative burden on medical staff.

## Surgical Note Sample
[From](test/test_text_data/test_medikal_apandisit.txt)

Hasta Adı Soyadı: Hasan Demirtaş
Dosya No: 90345167
Yaş / Cinsiyet: 58 / Erkek
Tarih: 25.03.2025
Ameliyat Türü: Acil Laparotomi + Apendektomi (Perfore Apandisit)
Ameliyat Ekibi:

Operatör: Op. Dr. Esra Tuncel
* Asistan: Dr. Halil Özcan
* Anestezi Uzmanı: Uzm. Dr. Tuğçe Alkan
* Sirküle Hemşire: Sevgi Önder
* Scrub Hemşire: Ayşegül Şanlı

Anestezi Türü: Genel anestezi

Ameliyat Süresi: 00:45 – 02:20

Ameliyat Endikasyonu:
Hasta 3 gündür devam eden karın ağrısı, ateş, kusma ve halsizlik şikayetleri ile acil servise başvurdu. Yapılan batın muayenesinde yaygın defans ve rebound alınması üzerine acil abdominal BT çekildi. Görüntüleme sonucunda perfore apandisit ve diffüz peritonit bulguları saptandı. Acil cerrahi kararı verildi.

Ameliyat Tekniği:
Hasta genel anestezi altında supin pozisyonda yatırıldı. Alt orta hat insizyonla batın açıldı. Açıldığı anda yoğun miktarda purulan (irinli) sıvı boşaldı. İnce barsak ansları arası yaygın fibrin ve yapışıklıklar mevcuttu. Sağ alt kadranda nekrotik ve perfore halde apendiks tespit edildi. Apendiksin tabanı sağlam olduğundan, taban seviyesinden ligatürle bağlanarak apendektomi yapıldı.
Batın bol serum fizyolojik ile irrigasyon yapılarak yıkandı. Apseli ve nekrotik alanlar debride edildi. Sağ parakolik alana ve pelvise dren yerleştirildi. Batın katları katmanlı olarak kapatıldı.

Ameliyat Sonucu:
Ameliyat teknik olarak tamamlandı. Ancak hastada yaygın peritonit nedeniyle postoperatif dönemde sepsis riski yüksek olarak değerlendirildi. Operasyon sonrası hipotansiyon ve taşikardi devam etti. Yoğun bakım takibi önerildi.

## Surgical Notes
[Notes used for testing](test/test_text_data)
[Audio files of used notes for testing](test/test_voice_data)

## Speech-to-Text Pipelines
- [Whisper](results/whisper)

## Speech-to-Text and LLM Pipelines
- [Whisper + Deepseek](results/whisper_deepseek)
- [Whisper + Deepseek R1](results/whisper_deepseek_r1)
- [Whisper + Gemini](results/whisper_gemini)
- [Whisper + GPT](results/whisper_gpt)

## Speech-to-Text and Local LLM Pipelines
- [Whisper + Deepseek-7B-chat-Q5_K_M](speech/agent/deepseek_model.py)

## Finetuned Speech-to-Text Pipelines
Finetuned models. Whisper generally outperformed, more generalizable and less computationally costly.
Training notebooks:
* [Turkish Speech Corpus](train/artificially-generated-medical-notebooks)
* [Artificially Generated Medical Dataset](train/artificially-generated-medical-notebooks)

### Ready Dataset - Turkish Speech Corpus
- [Whisper](models/trained/results/turkish-seech-corpus-whisper)
- [Wav2Vec](models/trained/results/turkish-speech-corpus-wav2vec)
### Generated Dataset using OpenAI Audio
- [Whisper](models/trained/results/generated-med-whisper)
- [Wav2Vec](models/trained/results/generated-med-wav2vec)