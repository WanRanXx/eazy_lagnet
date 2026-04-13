from chat_factory import LLMFactory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate


llm = LLMFactory.get_llm("deepseek")


summery_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "请总结以下文本，不超过50字"),
        ("human", "{text}"),
    ]
)

# 基础链
base_chain = summery_prompt | llm | StrOutputParser()


# 重试链
retry_chain = base_chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,  # 指数退避 + 抖动（推荐）
    retry_if_exception_type=(
        OutputParserException,
        TimeoutError,
        ConnectionError,
        Exception,
    )
)

# 4️⃣ 调用
try:
    result = retry_chain.invoke({
        "text": "LangChain是一个用于构建大模型应用的框架，提供了丰富的Runnable组件，支持重试、降级等工程化能力。"
    })
    print("总结结果：", result)

except OutputParserException as e:
    # ❗ 解析错误通常是逻辑问题，不建议重试
    print("输出解析失败：", e)

except Exception as e:
    print("最终失败（已达到最大重试次数）：", e)