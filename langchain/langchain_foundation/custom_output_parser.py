"""
StrOutputParser、JsonOutputParser、PydanticOutputParser 等，本质上都是 BaseOutputParser 的子类。
BaseOutputParser 是 LangChain 中所有输出解析器的抽象基类，核心作用是定义统一的解析器接口规范，
所有具体解析器都必须实现它的抽象方法，同时它也是我们实现“自定义解析器”的核心基础。
"""
import os
from typing import Dict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import BaseOutputParser

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")

if not deepseek_api_key:
    print("api key不存在，请检查环境变量设置")
if not deepseek_api_base:
    print("api base 不存在，请检查环境变量设置")

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=deepseek_api_key,
    base_url=deepseek_api_base,
    max_tokens=1024,
    streaming=True
)

# 自定义输出解析器，解析成一个字符串列表
class CustomOutputParser(BaseOutputParser):

    def parse(self, text: str) -> Dict[str, str]:
        """将模型输出按 '工具名@核心功能@学习难度' 解析为字典"""
        text = text.strip().replace("\n", "").replace(" ", "")
        parts = text.split("@")
        if len(parts) != 4:
            raise ValueError(f"输出格式错误！需满足「@代表作1@代表作2@代表作3」，当前输出：{text}")
        return {
            "tool_name": parts[0].strip(),
            "masterpiece1": parts[1].strip(),
            "masterpiece2": parts[2].strip(),
            "masterpiece3": parts[3].strip(),
        }


    def get_format_instructions(self) -> str:
        """提示词，让大模型按照提示词输出"""
        return "请严格按照以下格式输出：姓名@代表作1@代表作2@代表作3， 示例：成龙@A计划@警察故事@红番区"


parser = CustomOutputParser()
prompt = PromptTemplate(
    template="请介绍1个好莱坞明星。{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt | llm | parser
result = chain.invoke({})

print("自定义解析器解析结果：")
print(result)
print("解析结果类型：", type(result))