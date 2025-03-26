from dataclasses import dataclass, field
from llama_cpp import Llama
from speech.agent.base_agent import BaseModel
from speech.prompts import get_system_prompt_for_deepseek
from utils.log import logger


@dataclass
class DeepSeekModel(BaseModel):
    model_path: str = field(default="models/deepseek/deepseek-llm-7b-chat.Q5_K_M.gguf")
    n_ctx: int = field(default=4096)
    n_threads: int = field(default=4)

    def __post_init__(self):
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            use_mlock=True,
            use_mmap=True,
            verbose=False
        )

    def ask(self, user_prompt: str, language: str = "tr") -> str:
        try:
            self.temperature = 0.3
            self.top_p = 0.8
            system_prompt = get_system_prompt_for_deepseek(language)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # get chat template and stop tokens using metadata
            if hasattr(self.llm, 'metadata'):
                chat_template = self.llm.metadata.get('chat_template', None)
                stop_tokens = self.llm.metadata.get('stop_tokens', ["<|end|>"])
            else:
                chat_template = None
                # default stop tokens
                stop_tokens = ["<|end|>", "\n"]
            
            # create chat template if exists
            if chat_template:
                prompt = self.llm.apply_chat_template(messages, chat_template)
            else:
                prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: "
            
            output = self.llm(
                prompt,
                temperature=self.temperature,  
                top_p=self.top_p,              
                max_tokens=self.max_tokens,    
                repeat_penalty=1.2,            
                stop=stop_tokens               
            )
            
            response = output["choices"][0]["text"].strip()
            return response
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return "Sorry, something went wrong!"


if __name__ == "__main__":
    model = DeepSeekModel()
    response = model.ask("Karın ağrısı ne ile ilişkilidir?")
    print("\nModel Response:\n", response)
