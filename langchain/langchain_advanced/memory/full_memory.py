"""
langchain 全量记忆实现，包含了所有的记忆类型，适用于需要全面记忆功能的场景。
核心组件:
InMemoryHistory: 用于存储对话历史的内存实现，支持文本和向量两种存储方式。
MemoryManager: 负责管理不同类型的记忆，包括文本记忆、向量记忆和工具记忆，提供统一的接口进行操作。
MemoryRetrieval: 实现了基于文本和向量的记忆检索功能，支持多种检索算法
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


"""
使用InMemoryChatMessageHistory 存储完整对话历史，每次调用时自动注入所有历史消息到提示词中，适用于对话轮数少、需要完整上下文的场景。
"""

# full_memory_prompt = ChatPromptTemplate.from_messages(
#     [
#         ('system', '你是一个聊天机器人，需要基于历史完整对话进行回答问题'),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{user_input}")
#     ]
# )

full_memory_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '你是一个聊天机器人，需要基于历史完整对话进行回答问题'),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ]
)



base_chain = full_memory_prompt | llm

# 3. 会话历史存储（内存模式，生产环境可替换为数据库存储）
full_memory_store = {}

def get_full_memory_history(session_id: str) -> BaseChatMessageHistory:
    """根据会话id获取历史记录，不存在则创建会话"""
    if session_id not in full_memory_store:
        full_memory_store[session_id] = InMemoryChatMessageHistory()
    return full_memory_store[session_id]


# 构建带全量会话历史记忆的对话链

full_memory_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_full_memory_history,
    input_messages_key="user_input",
    history_messages_key="chat_history",

)

# 测试多轮对话（指定session_id=user_001，隔离不同用户）
config : RunnableConfig = {"configurable": {"session_id": "user_001"}}

# 第一轮对话
response1 = full_memory_chain.invoke({"user_input": "我叫小明，喜欢编程"}, config=config)
print("助手回复1：", response1.content)
# 输出示例：你好小明！编程是一项很有创造力的技能，你平时常用什么编程语言呢？

# 第二轮对话（验证记忆：询问历史信息）
response2 = full_memory_chain.invoke({"user_input": "我刚才说我喜欢什么？"}, config=config)
print("助手回复2：", response2.content)
# 输出示例：你刚才说你喜欢编程呀～

# 查看完整历史记录
print("\n全量记忆的对话历史：")
for msg in get_full_memory_history("user_001").messages:
    print(f"{msg.type}: {msg.content}")