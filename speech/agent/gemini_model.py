import os
from dataclasses import dataclass, field
from google.generativeai import GenerativeModel, configure 
from speech.agent.base_agent import BaseModel
from utils.log import logger
from speech.prompts import get_system_prompt

from dotenv import load_dotenv
load_dotenv()

configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@dataclass
class GeminiModel(BaseModel):
    model_name: str = field(default="models/gemini-2.0-flash")

    def ask(self, user_prompt: str, language: str = "tr") -> str:
        try:
            model = GenerativeModel(self.model_name)

            system_prompt = get_system_prompt(language=language)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                }
            )

            return response.text

        except Exception as e:
            logger.error(f"GEMINI API Error: {e} | Prompt: {user_prompt}")
            return "An error occurred while processing the Gemini response."

# check alive
if __name__ == "__main__":
    print(GeminiModel().ask("Hello, how are you?"))
