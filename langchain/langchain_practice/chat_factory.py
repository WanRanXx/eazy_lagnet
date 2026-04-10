import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama

# 全局加载环境变量（仅一次）
load_dotenv()
# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class LLMConfigError(Exception):
    """自定义配置异常：环境变量缺失或为空"""
    pass


class LLMFactory:
    """
    🔥 生产级 LLM 工厂
    ✅ 所有 key / url 从环境变量读取
    ✅ 缺失配置自动抛错
    ✅ 支持 DeepSeek / OpenAI / 通义千问 / 文心 / 讯飞 / Ollama
    """
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 4096

    # ===================== 环境变量校验工具 =====================
    @staticmethod
    def _check_env(var_name: str):
        """检查环境变量是否存在且不为空"""
        value = os.getenv(var_name, "").strip()
        if not value:
            raise LLMConfigError(f"❌ 环境变量缺失或为空：{var_name}")
        return value

    # ===================== 获取模型实例 =====================
    @classmethod
    def get_llm(cls, model_type: str = "deepseek", **kwargs):
        temperature = kwargs.get("temperature", cls.DEFAULT_TEMPERATURE)
        max_tokens = kwargs.get("max_tokens", cls.DEFAULT_MAX_TOKENS)
        model_name = kwargs.get("model_name")

        # 模型路由 + 环境变量校验
        model_routes = {
            "openai": cls._get_openai,
            "deepseek": cls._get_deepseek,
            "qwen": cls._get_qwen,
            "wenxin": cls._get_wenxin,
            "spark": cls._get_spark,
            # "ollama": cls._get_ollama,
        }

        if model_type not in model_routes:
            raise ValueError(f"仅支持模型：{list(model_routes.keys())}")

        return model_routes[model_type](
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_name
        )

    # ===================== 各模型实现 =====================
    @staticmethod
    def _get_openai(**kwargs):
        api_key = LLMFactory._check_env("OPENAI_API_KEY")
        base_url = LLMFactory._check_env("OPENAI_BASE_URL")
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=kwargs.get("model_name") or "gpt-3.5-turbo",
            temperature=kwargs["temperature"],
            max_tokens=kwargs["max_tokens"]
        )

    @staticmethod
    def _get_deepseek(**kwargs):
        api_key = LLMFactory._check_env("DEEPSEEK_API_KEY")
        base_url = LLMFactory._check_env("DEEPSEEK_BASE_URL")
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=kwargs.get("model_name") or "deepseek-chat",
            temperature=kwargs["temperature"]
        )

    @staticmethod
    def _get_qwen(**kwargs):
        api_key = LLMFactory._check_env("QWEN_API_KEY")
        base_url = LLMFactory._check_env("QWEN_BASE_URL")
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=kwargs.get("model_name") or "qwen-turbo",
            temperature=kwargs["temperature"]
        )

    @staticmethod
    def _get_wenxin(**kwargs):
        api_key = LLMFactory._check_env("WENXIN_API_KEY")
        base_url = LLMFactory._check_env("WENXIN_BASE_URL")
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=kwargs.get("model_name") or "ernie-3.5-8k",
            temperature=kwargs["temperature"]
        )

    @staticmethod
    def _get_spark(**kwargs):
        api_key = LLMFactory._check_env("SPARK_API_KEY")
        base_url = LLMFactory._check_env("SPARK_BASE_URL")
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=kwargs.get("model_name") or "generalv3.5",
            temperature=kwargs["temperature"]
        )

    # @staticmethod
    # def _get_ollama(**kwargs):
    #     # Ollama 本地模型，无需 API Key
    #     base_url = LLMFactory._check_env("OLLAMA_BASE_URL")
    #     # return ChatOllama(
    #     #     base_url=base_url,
    #     #     model=kwargs.get("model_name") or "llama3.1",
    #     #     temperature=kwargs["temperature"]
    #     # )