"""Single seeded Gemma instance shared by the Preseason Agent and the ABA."""
import json
import re
import logging
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from agri_scholar.seeding import derive_seed

log = logging.getLogger("agri_sigra")


class LocalLLM:
    def __init__(self, model_id: str, gpu_id: int, max_new_tokens: int = 1500,
                 temperature: float = 0.3, seed: int = None):
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=self.device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self._call_count = 0

    def generate_text(self, prompt: str) -> str:
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        # Reseed per call so successive calls don't replay the same draw.
        if self.seed is not None:
            self._call_count += 1
            call_seed = derive_seed(self.seed, self._call_count)
            torch.manual_seed(call_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(call_seed)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                temperature=self.temperature, do_sample=True,
            )
        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    @staticmethod
    def extract_json(response: str):
        """Returns the parsed dict from the model output, or None."""
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError as e:
            log.warning(f"LLM JSON parse error: {e}")
            return None
