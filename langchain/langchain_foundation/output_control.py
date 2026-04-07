from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv
import os

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

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="请介绍langchain的常用工具， 输出工具名和核心功能，{format_instructions}",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt| llm | parser

result = chain.invoke({})
print(result)
print(type(result))
print(f"获取单个工具名：{result['tools'][0]['name']}")
