"""
进阶实践：多输入多输出复杂线性任务
基础实践的 | 串联适合单输入单输出场景，若任务更复杂（比如“输入产品介绍和目标人群，先提取卖点，再根据卖点和人群写营销话术”），
就需要用到 RunnableSequence 结合 RunnablePassthrough 实现多输入多输出。
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnablePassthrough, RunnableMap
from langchain_core.prompts import PromptTemplate

load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=deepseek_api_key,
    base_url=deepseek_api_base,
    max_tokens=1024,
    streaming=True
)

# 定义prompt
sell_point_prompt = PromptTemplate(
    input_variables=["product_intro"],
    template="从以下产品介绍中提取3个核心卖点，简洁列出：{product_intro}"
)

marketing_prompt = PromptTemplate(
    input_variables=["sell_points", "target_audience"],
    template="针对{target_audience}，结合以下核心卖点，写一段朋友圈营销话术：{sell_points}"
)

# 3. 多输入多输出线性链（教学标准版）
overall_chain = (
    # Step 1：生成卖点 + 透传原始输入
    RunnableMap({
        "sell_points": sell_point_prompt | llm | (lambda x: x.content),
        "target_audience": RunnablePassthrough(),       # 透传原始输入，可以传入原始输出的人群
    })
    # Step 2：营销话术生成
    | marketing_prompt
    | llm
)

# 4. 执行
input_data = {
    "product_intro": "这款无线耳机采用蓝牙5.3芯片，连接稳定无延迟...",
    "target_audience": "大学生群体（喜欢运动、预算有限、注重性价比）"
}

result = overall_chain.invoke(input_data)

print("营销话术：")
print(result.content)