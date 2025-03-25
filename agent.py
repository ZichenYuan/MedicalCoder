from pydantic import BaseModel
from config import groq_client, openai_client
#, azure_client #, openai_client
from langfuse.decorators import observe
from typing import List, Optional

class Agent(BaseModel):
    ai_provider: str
    model: str
    max_token: int
    response_model: Optional[type[BaseModel]] = None

    def __init__(self, ai_provider, model, max_token, response_model=None):
        super().__init__(ai_provider=ai_provider, model=model, max_token=max_token, response_model=response_model)

    def inference(self, message: str, system_prompt: str):
        client = self._get_client(sync=True)
        return self._perform_inference(client, message, system_prompt, self.max_token, self.response_model)

    def _get_client(self, sync=True):
        if sync:
            if self.ai_provider == "azure_client":
                return azure_client
            if self.ai_provider == "openai_client":
                return openai_client
            if self.ai_provider == "groq_client":
                return groq_client
            else:
                raise ValueError(f"Invalid AI provider: {self.ai_provider}")
        else:
            raise ValueError(f"Invalid AI provider: {self.ai_provider}")

    @observe()
    def _perform_inference(self, client, message: str, system_prompt: str, max_token: int, response_model):
        try:
            if response_model:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": message
                    }],
                    temperature=0,
                    max_tokens=max_token,
                    response_model=response_model,
                    max_retries=2,
                )
                response_dict = response.dict() if isinstance(response, BaseModel) else response
                validated_response = response_model(**response_dict) if response_model else response_dict
                return validated_response
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": message
                    }],
                    temperature=0,
                    max_tokens=max_token,
                    max_retries=2,
                )
                response_dict = response.dict() if isinstance(response, BaseModel) else response
                return response_dict

        except Exception as e:
            raise RuntimeError(f"Inference failed with {self.model}: {str(e)}")



class ExtractModel(BaseModel):
    diagnosis: List[str]
    evidence: List[str]
