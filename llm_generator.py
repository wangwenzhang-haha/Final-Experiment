"""构造提示词并调用本地 Mistral 模型生成文本的工具。"""

from models.mistral_loader import MistralLLM


class LLMGenerator:
    """封装本地 Mistral 模型，基于证据生成解释。"""

    def __init__(self):
        # 惰性加载量化模型，保持下游脚本轻量。
        self.llm = MistralLLM()

    def build_prompt(self, user_profile, item_desc, path_texts):
        """拼接包含检索证据的提示词。"""
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
        """生成提示并调用底层模型完成解释。"""
        prompt = self.build_prompt(user_profile, item_desc, path_texts)
        return self.llm.generate(prompt)
