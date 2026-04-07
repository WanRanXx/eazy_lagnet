"""
简单工具调用示例
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

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


@tool
def weather_query(city: str) -> str:
    """查询指定城市天气"""
    weather_data = {
        "北京": "北京今日天气：晴，-2~8℃",
        "上海": "上海今日天气：多云，5~12℃",
        "广州": "广州今日天气：小雨，18~25℃",
    }
    return weather_data.get(city, f"暂无 {city} 数据")

tools = [weather_query]

# 创建一个agent，调用工具并开启debug模式

agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,
)

# ======================
# 4. 运行
# ======================
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "北京今天的天气怎么样？"}
    ]
})

print("\n最终回答：")
print(response["messages"][-1].content)