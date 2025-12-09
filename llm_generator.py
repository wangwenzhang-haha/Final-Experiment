# 文件: llm_generator.py
# 功能: 构造 Prompt + 调用 Mistral 生成推荐解释

from models.mistral_loader import MistralLLM

class LLMGenerator:
    def __init__(self):
        self.llm = MistralLLM()

    def build_prompt(self, user_profile, item_desc, path_texts):
        prompt = f"""
你是一个推荐系统助手，需要为用户解释推荐某商品的原因。
用户偏好：{user_profile}
商品信息：{item_desc}
推荐证据：
"""
        for i, p in enumerate(path_texts):
            prompt += f"{i+1}. {p}\n"
        prompt += "\n请结合以上信息，生成一段自然语言的推荐理由，语气自然亲切："
        return prompt

    def generate_explanation(self, user_profile, item_desc, path_texts):
        prompt = self.build_prompt(user_profile, item_desc, path_texts)
        return self.llm.generate(prompt)
