from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from speech.agent.base_agent import BaseModel
from utils.log import logger
from speech.prompts import get_system_prompt


@dataclass
class DeepSeekLocalModel(BaseModel):
    model_path: str = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")  # change to 33b if needed
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()

    def ask(self, user_prompt: str, language: str = "tr") -> str:
        try:
            system_prompt = get_system_prompt(language)
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Strip system/user prompt to isolate assistant response
            return response.split("<|assistant|>")[-1].strip()

        except Exception as e:
            logger.error(f"Local DeepSeek error: {e}")
            return "An error occurred during local inference."
