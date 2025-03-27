from dataclasses import dataclass, field
from speech.agent.base_agent import BaseModel
from utils.log import logger
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa


@dataclass
class Wav2VecModel(BaseModel):
    model_path: str = field(default="models/wav2vec")
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """Load the fine-tuned model and processor after initialization."""
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_path).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        self.model.eval()

    def ask(self, audio_path: str, language: str = "tr") -> str:
        """Transcribe audio file using the fine-tuned Wav2Vec2 model."""
        try:
            # Load and preprocess audio
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
            input_values = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(pred_ids)[0]
            
            return transcription.strip()
        
        except Exception as e:
            logger.error(f"Wav2Vec2 inference error: {e}")
            return "An error occurred during inference."

if __name__ == "__main__":
    model = Wav2VecModel()
    audio_path = "test/test_voice_data/test_medikal_apandisit.mp3" 
    transcription = model.ask(audio_path)
    print(transcription)