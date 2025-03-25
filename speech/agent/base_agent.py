from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class BaseModel(ABC):
    system_prompt: str = field(default="")
    temperature: float = field(default=0.7)
    max_tokens: int = field(default=1024)
    top_p: float = field(default=1.0)
    frequency_penalty: float = field(default=0.0)
    presence_penalty: float = field(default=0.0)

    @abstractmethod
    def ask(self, user_prompt: str) -> str:
        pass
