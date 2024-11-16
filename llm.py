from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseLLM(ABC):
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )  # TODO: see if padding_side="right" is better
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, load_in_4bit=True
        ).to(device)
        # TODO: figure out what bit model to load ^
        self.device = device

    @abstractmethod
    def generate_prompt(self, query):
        # TODO: add docstring later
        pass

    def __call__(self, query):
        prompt = self.generate_prompt(query)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            # TODO: figure out what hyperparameters to use here. just used some default values
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        response = generated_text[len(prompt) :].strip()
        return response

    def __repr__(self):
        return f"BaseLLM(model_name={self.model_name}, device={self.device})"


class LlamaLLM(BaseLLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-2-7b-chat-hf")

    def generate_prompt(self, query):
        return f"[INST] {query} [/INST]"


class MistralLLM(BaseLLM):
    def __init__(self):
        super().__init__("mistralai/Mistral-7B-v0.1")

    def generate_prompt(self, query):
        return f"<s>[INST] {query} [/INST]"
