"""
窗口记忆组件:
- 该组件使用一个固定大小的窗口来存储最近的对话历史。
- 当新的对话加入时，旧的对话会被移除，以保持窗口的大小不变。
- 适用于需要关注最近对话内容的场景，如实时聊天应用。
"""

import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI


load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")

if not deepseek_api_key:
    print("api key不存在，请检查环境变量设置")
if not deepseek_api_base:
    print("api base 不存在，请检查环境变量设置")

# 初始化 llm，使用 deepseek-chat 模型，设置温度为 0.3，启用流式输出，最大 token 数为 1024
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=deepseek_api_key,
    base_url=deepseek_api_base,
    max_tokens=1024,
    streaming=True
)

WINDOW_SIZE = 2
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '你是一个聊天机器人，需要基于最近的对话进行回答问题'),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ]
)

base_chain = prompt | llm
window_memory_store = {}

def get_window_history(session_id: str) -> BaseChatMessageHistory:
    """根据会话id获取历史记录，不存在则创建会话"""
    if session_id not in window_memory_store:
        window_memory_store[session_id] = InMemoryChatMessageHistory()
    history = window_memory_store[session_id]
    if len(history.messages) > 2 * WINDOW_SIZE:
        history.messages = history.messages[-2 * WINDOW_SIZE:]

    return history


# 构建带有窗口记忆的会话链

windows_memory_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_window_history,
    input_messages_key="user_input",
    history_messages_key="chat_history",

)


# 测试多轮对话（指定session_id=user_001，隔离不同用户）
config: RunnableConfig = {
    "configurable": {
        "session_id": "user_001"
    }
}


# 模拟5轮对话，验证窗口记忆的截断效果
inputs = [
    "我叫小红",
    "我喜欢画画",
    "我来自上海",
    "我是一名学生",
    "我刚才说我来自哪里？",  # 第5轮：询问第3轮的信息，验证窗口截断
    "我叫什么名字？"  # 第6轮：询问第1轮的信息，验证窗口记忆
]

for i, user_input in enumerate(inputs, 1):
    response = windows_memory_chain.invoke({"user_input": user_input}, config=config)
    print(f"\n第{i}轮 - 助手回复：", response.content)

# 查看窗口记忆的最终历史（仅保留最近2轮）
print("\n窗口记忆的最终对话历史（最近2轮）：")
for msg in get_window_history("user_001").messages:
    print(f"{msg.type}: {msg.content}")


