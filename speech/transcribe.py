import whisper
import openai
import os
from dotenv import load_dotenv
from utils.log import logger
from speech.agent.gpt_model import GPTModel
from speech.agent.gemini_model import GeminiModel


load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

client = openai.OpenAI()


def transcribe_audio(audio_file: str) -> str:
    """
    Transcribes the given audio file using the Whisper model.
    """
    model = whisper.load_model("large")
    
    result = model.transcribe(audio_file)
    return result["text"]

def refine_transcription(transcription: str, model: str = "gemini", language: str = "tr") -> str:
    """
    Uses the GPT API to refine the transcription with a focus on medical terminology.
    """
    # Prepare a prompt instructing GPT to ensure accurate medical transcription
    prompt = (
        f"Transcription:\n{transcription}\n\n"
        f"Refined transcription in {language} language:"
    )
    
    if model == "gemini":
        agent = GeminiModel()
    elif model == "gpt":
        agent = GPTModel()
    else:
        raise ValueError(f"Unsupported model '{model}'. Please choose 'gpt' or 'gemini'.")

    corrected_text = agent.ask(prompt, language=language)
    return corrected_text

def transcribe_and_refine(audio_file: str) -> str:
    """
    Combines transcription and GPT-based refinement for medical audio inputs.
    """
    # 1. transcribe the audio file using whisper large
    initial_transcription = transcribe_audio(audio_file)
    print("Initial Transcription:")
    print(initial_transcription)
    
    # 2. refine the transcription using LLM
    final_transcription = refine_transcription(initial_transcription)
    return final_transcription

if __name__ == "__main__":
    audio_path = "test/test_voice_data/test_medikal_apandisit.mp3"
    
    refined_transcript = transcribe_and_refine(audio_path)
    
    print("\nFinal Refined Transcript:")
    print(refined_transcript)
