from dataclasses import dataclass, field
from speech.agent.base_agent import BaseModel
from utils.log import logger
from speech.prompts import get_system_prompt_for_deepseek
import ollama  

# local client
client = ollama.Client(host="http://localhost:11434")


@dataclass
class DeepSeekR1Model(BaseModel):
    # firstly get model from ollama
    # ollama pull deepseek-r1
    model: str = field(default="deepseek-r1")  # Adjust this based on your Ollama model name

    def ask(self, user_prompt: str, language: str = "tr") -> str:
        try:
            messages = [
                {"role": "system", "content": get_system_prompt_for_deepseek(language=language)},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                }
            )
            
            deepseek_response = response["message"]["content"]
            finish_reason = response.get("done_reason", "stop")  # Ollama's equivalent
            
            if finish_reason == "length":
                logger.error(f"Ollama API Error: Max tokens exceeded for prompt: {user_prompt}")

            # remove thinking part - we need answers only not thinking part
            # TODO: store thinking part and analyze it
            if "</think>" in deepseek_response:
                deepseek_response = deepseek_response.split("</think>")[-1].strip()
            return deepseek_response.strip()
        
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "An error occurred during inference."


if __name__ == "__main__":
    model = DeepSeekR1Model()
    response = model.ask("Merhaba, nasılsın?")
    print("DeepSeek Response:\n", response)