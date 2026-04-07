"""
摘要记忆模块：
该模块实现了一个基于摘要的记忆系统，能够在对话过程中总结和提取关键信息，以便在后续对话中使用。它通过使用语言模型来生成对话的摘要，并将这些摘要存储在内存中，以便在需要时进行检索和使用。这种方法可以帮助提高对话系统的效率和响应质量，特别是在处理长时间对话或复杂信息时。

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


summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是对话摘要助手，负责总结对话中的关键信息（包括用户身份信息，爱好，偏好，关键问题与回答等），不超过50字。"),
        ("human", "历史对话：{chat_history_text}\n请总结关键信息")
    ]

)

# 构建基础对话链，使用 prompt 和 llm

summary_chain = summary_prompt | llm

summary_memory_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '你是一个聊天机器人，基于对话摘要回答问题, 摘要内容跟是对话的关键信息'),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ]
)

def generate_chat_summary(x: dict) -> str:
    """生成对话摘要"""
    chat_history = x.get("chat_history", [])
    if len(chat_history) > 0:
        chat_history_text = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in chat_history]
        )
    else:
        return "暂无历史对话"
    summary_result = summary_chain.invoke(
        {
            "chat_history_text": chat_history_text
        }
    )
    return summary_result.content

summary_base_chain = (
    RunnablePassthrough.assign(
        chat_summary=generate_chat_summary
    )
    | summary_memory_prompt
    | llm
)

summary_history_store = {}

def get_summary_history(session_id: str) -> BaseChatMessageHistory:
    """根据会话id获取历史记录，不存在则创建会话"""
    if session_id not in summary_history_store:
        summary_history_store[session_id] = InMemoryChatMessageHistory()
    return summary_history_store[session_id]


# 7. 构建带摘要记忆的对话链
summary_memory_chain = RunnableWithMessageHistory(
    runnable=summary_base_chain,
    get_session_history=get_summary_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"  # 传入完整历史用于生成摘要
)


# 测试多轮对话（session_id=user_003）
config: RunnableConfig = {"configurable": {"session_id": "user_003"}}

# 多轮对话输入
inputs = [
    "我叫小李，是一名产品经理",
    "我负责一款电商APP的迭代",
    "最近在优化用户下单流程",
    "遇到了用户流失率高的问题",
    "你能给我一些优化建议吗？"
]

for i, user_input in enumerate(inputs, 1):
    response = summary_memory_chain.invoke({"user_input": user_input}, config=config)
    print(f"\n第{i}轮 - 助手回复：", response.content)

# 查看完整历史与最终摘要
history = get_summary_history("user_003")
print("\n摘要记忆的完整对话历史：")
for msg in history.messages:
    print(f"{msg.type}: {msg.content}")

# 单独生成最终摘要验证
final_summary = summary_chain.invoke({
    "chat_history_text": "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])
}).content
print(f"\n最终对话摘要：{final_summary}")
# 输出示例：摘要：小李，产品经理，负责电商APP迭代，优化下单流程时遇用户流失率高问题，寻求建议。