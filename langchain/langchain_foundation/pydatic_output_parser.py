"""
使用Pydantic模型作为输出解析器的实现。它将LLM的输出解析为Pydantic模型实例。适合复杂场景的工程化输出解析，提供了更强的类型检查和数据验证功能。
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")
chat_model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=deepseek_api_key,
    base_url=deepseek_api_base,
    max_tokens=1024,
    streaming=True
)

class ToolInfo(BaseModel):
    """
    工具模型
    """
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述, 30字以内")
    args: str = Field(..., description="工具参数")
    return_value: str = Field(..., description="工具返回值")
    difficulty: str = Field(..., description="学习难度，分为easy, medium, hard三个等级")

# 利用工具模型创建一个输出解析器
parser = PydanticOutputParser(pydantic_object=ToolInfo)


prompt = PromptTemplate(
    template="请介绍langchain的{user_input}工具，严格按照格式输出\n，{format_instructions}",
    input_variables=["user_input"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | chat_model | parser

# 测试输出解析器
result = chain.invoke({"user_input": "一个网页解析"})

print(result)
print(type( result))

print("字段校验 difficulty：", result.difficulty)

print("转化为字典：", result.model_dump())
