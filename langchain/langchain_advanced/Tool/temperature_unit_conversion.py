import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
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

# ======================
# 2. 参数模型
# ======================

class TemperatureConvertInput(BaseModel):
    """温度转换输入参数模型"""
    temperature: float = Field(description="需要转换的温度值，例如37.0")
    from_unit: str = Field(description="原始温度单位，只能是celsius或fahrenheit")

@tool(args_schema=TemperatureConvertInput)
def temperature_converter(temperature: float, from_unit: str) -> str:
    """温度单位转换工具，支持摄氏度和华氏度之间的转换"""
    if from_unit.lower() == "celsius":
        converted = temperature * 9 / 5 + 32
        return f"{temperature}°C 等于 {converted:.2f}°F"
    elif from_unit.lower() == "fahrenheit":
        converted = (temperature - 32) * 5 / 9
        return f"{temperature}°F 等于 {converted:.2f}°C"
    else:
        return "不支持的温度单位，请使用celsius或fahrenheit"


tools = [temperature_converter]

system_prompt = """你是一个专业的温度转换助手，能够将摄氏度和华氏度之间进行转换。当用户提供一个温度值和原始单位时，你会调用temperature_converter工具来完成转换，并返回结果。请确保用户输入的单位是celsius或fahrenheit，否则提示用户输入正确的单位。
"""

# ======================
# 4. 创建 Agent
# ======================
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
    debug=True
)

if __name__ == "__main__":
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="将37摄氏度摄氏度转换为华氏度")
            ]
        }
    )
    print("\n最终回答：")
    print(response["messages"][-1].content)
