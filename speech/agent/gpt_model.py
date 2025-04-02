from dataclasses import dataclass
from dataclasses import field
from speech.agent.base_agent import BaseModel
from utils.log import logger
from speech.prompts import get_system_prompt
import os
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

client = openai.OpenAI()

@dataclass
class GPTModel(BaseModel):
    model: str = field(default="gpt-4o")

    def ask(self, user_prompt: str, language: str = "tr") -> str:
        messages = [
            {"role": "system", "content": get_system_prompt(language=language)},
            {"role": "user", "content": user_prompt}
        ]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        gpt_response = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        if finish_reason == 'length':
            logger.error(f"OPENAI API Error: Max tokens exceeded for prompt: {user_prompt}")
        return gpt_response

# check alive
if __name__ == "__main__":
    print(GPTModel().ask("Hello, how are you?"))
