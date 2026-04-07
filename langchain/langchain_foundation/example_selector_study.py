from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import BaseExampleSelector, LengthBasedExampleSelector
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
print(deepseek_api_key)
deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")
print(deepseek_api_base)

if not deepseek_api_key:
    print("api key不存在，请检查环境变量设置")
if not deepseek_api_base:
    print("api base 不存在，请检查环境变量设置")

chat_model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=deepseek_api_key,
    base_url=deepseek_api_base,
    max_tokens=1024,
    streaming=True
)

# 工程化示例管理，从json文件中加载示例
with open(r"D:\data\langchain\eazy_Iagent\langchain\langchain_foundation\example.json", "r", encoding="utf-8") as f:
    examples = json.load(f)


# 当需要根据用户输入特征选择示例时，可以自定义一个ExampleSelector
class DifficultyExampleSelector(BaseExampleSelector):
    """根据用户输入的 difficulty 字段筛选样本"""
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def add_example(self, example: Dict[str, Any]) -> None:
        self.examples.append(example)
    
    def select_examples(self, input_variables: Dict[str, Any]) -> List[Dict[str, str]]:
        # 获取用户输入的难度等级，如果没有提供则默认为 'easy'
        target_difficulty = input_variables.get("difficulty", "easy")
        # 过滤出匹配难度的所有示例
        return [ex for ex in self.examples if ex.get("difficulty") == target_difficulty]

    
example_selector = DifficultyExampleSelector(examples)

# 5. 构建工程化少样本模板 

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["subject", "difficulty", "method"],
        template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"
    ),
    example_separator="\n---\n",
    prefix="请根据以下示例提供适合的学习方法：\n",
    suffix="用户输入的学科：{subject}\n用户输入的难度：{difficulty}\n请提供适合的学习方法："
)

# 6. 动态生成不同难度的提示词
# 场景1：生成入门级LangChain学习方法
formatted_prompt_easy = few_shot_prompt.format(
    subject="LangChain",
    difficulty="easy"
)
print("入门级少样本提示词：")
print(formatted_prompt_easy)
result_easy = chat_model.invoke([{"role": "user", "content": formatted_prompt_easy}])
print("\n入门级学习方法：")
print(result_easy.content)

# 场景2：生成进阶级LangChain学习方法
formatted_prompt_hard = few_shot_prompt.format(
    subject="LangChain",
    difficulty="hard"
)
print("\n进阶级少样本提示词：")
print(formatted_prompt_hard)
result_hard = chat_model.invoke([{"role": "user", "content": formatted_prompt_hard}])
print("\n进阶级学习方法：")
print(result_hard.content)
