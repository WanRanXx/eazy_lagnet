"""
异常捕获学习
"""
from chat_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.exceptions import OutputParserException


llm = LLMFactory.get_llm("deepseek")

marketing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个营销机器人，根据产品卖点和目标人群写一套营销话术"),
        ("human", "产品卖点：{sell_points}，目标人群：{target_audience}")
    ]
)

marketing_chain: Runnable = marketing_prompt | llm | StrOutputParser()

inputs = {
    "sell_points": "无线耳机续航30小时",
    # target_audience 故意缺失，测试异常捕获
    "target_audience": "运动爱好者"
}

try:
    result = marketing_chain.invoke(inputs)
    print("营销话术", result)
except KeyError as e:
    missing_var = str(e).strip("`\"")
    print(f"错误提示：缺少必要的参数{missing_var}, 请检查您的输入")
except OutputParserException as e:
    print(f"解析失败，请检查你的prompt, 失败原因：{str(e)}")
except Exception as e:
    print(f"未知错误：错误原因：{e}")
