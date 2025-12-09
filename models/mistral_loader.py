# 文件: models/mistral_loader.py
# 功能: 加载本地 Mistral-7B-Instruct 模型（4bit 量化）

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class MistralLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        print("🚀 正在加载模型，请稍等...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ 模型加载完成！")

    def generate(self, prompt, max_tokens=150):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, top_p=0.95)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")
