"""
结合记忆和工具使用的示例，展示如何在LangChain中同时利用这两者来增强模型的能力。
实现一个带记忆的对话机器人
"""

import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_experimental.tools import PythonREPLTool    # 数学计算工具
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent



load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_API_BASE")
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=api_key,
    base_url=base_url,
    max_tokens=1024,
    streaming=True
)

# 初始化科学计算器
calc_tool = PythonREPLTool()

# 定义对话窗口大小
WINDOW_SIZE = 5
# 定义提示词模板，适配记忆窗口与提示词工具调用

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", f"""你是一个智能助手，具有以下功能：\n
        1.记忆：你能记住用户之前的{WINDOW_SIZE}轮对话，用简洁明确的语言回答用户问题。\n
        2.计算：如果需要科学计算，先调用计算工具得到结果，再用自然语言解释先调用计算工具得到结果，再用自然语言解释。\n
        3.回答：非计算问题直接回答，记得结合历史对话上下文。\n
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ]
)

# 工具调用逻辑
def judge_and_calc(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    核心逻辑：
    1. 检测用户问题是否包含数学计算需求
    2. 是：调用PythonREPLTool计算，再结合LLM生成回答
    3. 否：直接用LLM回答
    """
    user_input = inputs["user_input"]
    chat_history = inputs["chat_history"]

    # 简单的计算意图检测（可根据需求扩展）
    calc_pattern = r"(\+|\-|\×|\*|÷|/|=|计算|求和|求差|平方|立方|多少|等于)"
    is_calc_needed = bool(re.search(calc_pattern, user_input))
    if is_calc_needed:
        # 步骤1：调用计算工具执行运算
        try:
            # 提取计算表达式（简化版：取数字和运算符部分）
            calc_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", user_input)
            if not calc_expr:
                calc_result = "未识别到可计算的表达式"
            else:
                calc_result = calc_tool.run(calc_expr)
        except Exception as e:
            calc_result = f"计算出错：{str(e)}"

        # 步骤2：构造包含计算结果的提示，让LLM生成自然语言回答
        enhanced_input = f"""
            用户问题：{user_input}
            计算过程/结果：{calc_result}
            请结合计算结果，用简单易懂的语言回答用户问题，同时参考历史对话：{chat_history}
            """
        inputs["input"] = enhanced_input
    return inputs

window_memory_store = {}

def get_window_history(user_id: str) -> Dict[str, Any]:
    """
    获取用户的窗口记忆历史，返回最近的WINDOW_SIZE轮对话
    """
    if user_id not in window_memory_store:
        window_memory_store[user_id] = InMemoryChatMessageHistory()
    history = window_memory_store[user_id]
    if len(history.messages) > 2 * WINDOW_SIZE:
        history.messages = history.messages[-2 * WINDOW_SIZE:]
    return history
# 核心链
chain = (
    RunnableLambda(judge_and_calc) |  # 判断是否需要计算并执行
    prompt_template |
    llm

)

# 定义带记忆和工具的指令链
memory_tool_chain = RunnableWithMessageHistory(
    runnable=chain,
    input_messages_key="user_input",
    output_messages_key="output",
    get_session_history=get_window_history,
    history_messages_key="chat_history"
)

# 多轮对话测试

if __name__ == "__main__":
    session_id = "student_123"
    print("===== 带窗口记忆的数学计算智能助手 =====")
    print("支持：多轮对话、仅保留最近2轮记忆、自动数学计算")
    print("输入'退出'结束对话\n")

    while True:
        user_input = input("你：")
        if user_input in ["退出", "quit", "q"]:
            print("助手：再见！有问题随时问我～")
            break

        # 调用带窗口记忆的智能体
        response = memory_tool_chain.invoke(
            {"user_input": user_input, "window_size": WINDOW_SIZE},
            config={"configurable": {"session_id": session_id}}
        )

        # 输出回答（并将对话存入记忆）
        print(f"助手：{response.content}\n")
