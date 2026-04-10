"""
单输入输出线性流转
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

# 加载环境变量
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
# =========================
# 3. 组件1：提取核心卖点
# =========================
sell_point_prompt = PromptTemplate(
    template="请从以下产品介绍中提取3个核心卖点，用简洁的语言列出：{product_intro}",
    input_variables=["product_intro"]
)

sell_point_chain = sell_point_prompt | llm
# =========================
# 4. 中间结果结构化（LangChain 风格）
# =========================

extract_sell_points = RunnableLambda(
    func=lambda msg: {"sell_points": msg.content}
)

# =========================
# 5. 组件2：生成营销话术
# =========================
marketing_prompt = PromptTemplate(
    input_variables=["sell_points"],
    template="请根据以下产品核心卖点，写一段吸引消费者的营销话术（适合朋友圈发布）：{sell_points}"
)

marketing_chain = marketing_prompt | llm

# =========================
# 6. 线性串联（Sequential Runnable）
# =========================
overall_chain = (
    sell_point_chain
    | extract_sell_points
    | marketing_chain
)

# =========================
# 7. 执行
# =========================
product_intro = """这款无线耳机采用蓝牙5.3芯片，连接稳定无延迟，支持高清通话；续航长达30小时，充电10分钟可使用2小时；机身采用亲肤硅胶材质，佩戴舒适，防水防汗，适合运动使用。"""

result = overall_chain.invoke({"product_intro": product_intro})

print("\n最终营销话术：")
print(result.content)