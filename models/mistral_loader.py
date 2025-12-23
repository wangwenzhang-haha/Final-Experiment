"""加载本地量化 Mistral-7B-Instruct 模型的工具。"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class MistralLLM:
    """对 4-bit Mistral 模型的轻量封装，用于生成文本。"""

    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        # Configure bitsandbytes to keep the footprint small enough for demos.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        print("🚀 正在加载模型，请稍等...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ 模型加载完成！")

    def generate(self, prompt, max_tokens=150):
        """根据提示生成文本，并移除模型回显的提示部分。"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, top_p=0.95)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")
