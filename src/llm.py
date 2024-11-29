from abc import ABC, abstractmethod
import google.generativeai as genai
import torch
from openai import OpenAI
from transformers import OpenAIGPTConfig, OpenAIGPTModel, pipeline
from src.structures import ModelName

class BaseLLM(ABC):
    """Abstract base class for LLMs."""

    def __init__(self, system_prompt: str):
        self._system_prompt = system_prompt

    @abstractmethod
    def __call__(self, query: str) -> str:
        pass
    
    @property
    @abstractmethod
    def name(self) -> ModelName:
        pass
    
    @property
    def system_prompt(self) -> str:
        return self._system_prompt


class LlamaMini(BaseLLM):
    def __init__(
        self, system_prompt: str, temperature: float = 0.7, do_sample: bool = True
    ):
        """Load the Llama-3.2-1B-Instruct model and set the system prompt.

        Args:
            temperature (float, optional): Defaults to 0.7.
            do_sample (bool, optional): Defaults to True.

            If you want the response to be more for more precise tasks,
            set temperature to 0.0 and do_sample to False.
            For more creative tasks, set temperature to 0.7 and do_sample to True.
        """
        super().__init__(system_prompt)

        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_new_tokens = 256 # TODO: What is a good value for this?

    @property
    def device(self):
        return self.pipe.device

    def __call__(self, query: str) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]

        response = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,  # Adjust as needed
            temperature=self.temperature,
            do_sample=self.do_sample,
        )[0]["generated_text"]

        # TODO: do i need to return response[-1]['content']
        return response

    @property
    def name(self) -> ModelName:
        return ModelName.LLAMA


# class GPTLLM(BaseLLM):
#     def __init__(self):
#         self.configuration = OpenAIGPTConfig()
#         self.model = OpenAIGPTModel(self.configuration)
#         self.configuration = self.model.config

#     def generate_prompt(self, query):
#         return f"<s>[INST] {query} [/INST]"


class GPTMini(BaseLLM):
    def __init__(self, system_prompt: str):
        super().__init__(system_prompt)

        self.client = OpenAI(api_key="")
        self.model_id = "gpt-4o-mini"
        # https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/

    def __call__(self, query: str) -> str:
        messages = [{"role": "system", "content": self._system_prompt}, {"role": "user", "content": query}]
        chat = self.client.chat.completions.create(
            model=self.model_id, messages=messages
        )
        reply = chat.choices[0].message.content
        # self.messages.append({"role": "assistant", "content": reply})   # if we are keeping track of the history for subsequent calls we can use this
        return reply

    @property
    def name(self) -> ModelName:
        return ModelName.GPT


class Gemini(BaseLLM):
    def __init__(self, system_prompt: str):
        super().__init__(system_prompt)

        genai.configure(api_key="")
        
        # Set system instruction as the system prompt (https://ai.google.dev/gemini-api/docs/system-instructions?lang=python)
        self.model = genai.GenerativeModel("gemini-1.5-flash",  system_instruction=system_prompt)

    def __call__(self, query: str) -> str:
        response = self.model.generate_content(query)
        return response.text
    
    @property
    def name(self) -> ModelName:
        return ModelName.GEMINI

def create_llm(model_name: str, system_prompt: str) -> BaseLLM:
    if model_name.lower() not in llm_models:
        raise ValueError(f"Model {model_name} not supported")
    return llm_models[model_name.lower()](system_prompt)
