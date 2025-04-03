# Medical Notes - Speech-to-Text

## Project Overview
Medical Notes - Speech-to-Text is a system designed to streamline the documentation of surgical notes by converting spoken records into text and storing them digitally. This tool enables healthcare professionals to efficiently document critical information during or after surgeries, particularly in high-pressure environments like emergency procedures. The project aims to enhance the accuracy and accessibility of patient records while reducing the administrative burden on medical staff.

## Quick Start
Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Set the Python path:
```bash
export PYTHONPATH=.
```
With default settings (uses gemini model and test/test_voice_data/test_medikal_apandisit.mp3):
```bash
python3 speech/transcribe.py
```
With custom model and audio file:
```bash
python3 speech/transcribe.py --model gemini --audio_path /to/path/audio_file.mp3
```

## Surgical Notes
- [Notes used for testing](test/test_text_data)
- [Audio files of used notes for testing](test/test_voice_data)

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