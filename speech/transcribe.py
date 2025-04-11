import whisper
import openai
import os
from utils.log import logger
from speech.agent.gpt_model import GPTModel
from speech.agent.gemini_model import GeminiModel
from speech.agent.deepseek_model import DeepSeekModel
from speech.agent.deepseek_r1_model import DeepSeekR1Model
import argparse


def transcribe_audio(audio_file: str) -> str:
    """
    Transcribes the given audio file using the Whisper model.
    """
    model = whisper.load_model("large")
    
    result = model.transcribe(audio_file)
    return result["text"]

def refine_transcription(transcription: str, model: str = "gemini", language: str = "tr") -> str:
    """
    Uses API to refine the transcription with a focus on medical terminology.
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
    elif model == "deepseek":
        agent = DeepSeekModel()
    elif model == "deepseek-r1":
        agent = DeepSeekR1Model()
    else:
        raise ValueError(f"Unsupported model '{model}'. Please choose 'gpt', 'gemini', 'deepseek', or 'deepseek-r1'.")

    corrected_text = agent.ask(prompt, language=language)
    return corrected_text

def transcribe_and_refine(audio_file: str, model: str) -> str:
    """
    Combines transcription and GPT-based refinement for medical audio inputs.
    """
    # 1. transcribe the audio file using whisper large
    initial_transcription = transcribe_audio(audio_file)
    print("Initial Transcription:")
    print(initial_transcription)
    
    # 2. refine the transcription using LLM
    final_transcription = refine_transcription(initial_transcription, model=model)
    return final_transcription

def main():
    # argument parser
    parser = argparse.ArgumentParser(description="Transcribe and refine audio from a specified file path.")
    parser.add_argument(
        "--audio_path",
        type=str,
        default="test/test_voice_data/test_medikal_apandisit.mp3",
        help="Path to the audio file (default: test/test_voice_data/test_medikal_apandisit.mp3)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemini", "gpt", "deepseek", "deepseek-r1"],
        default="gemini",
        help="Model for refining transcription (choices: gemini, gpt, deepseek, deepseek-r1; default: gemini)"
    )
    
    # parse arguments
    args = parser.parse_args()
    
    # transcribe
    print(f">>> transcribe_and_refine('{args.audio_path}', model='{args.model}')")
    refined_transcript = transcribe_and_refine(args.audio_path, model=args.model)
    
    print("\nFinal Refined Transcript:")
    print(refined_transcript)

if __name__ == "__main__":
    main()
