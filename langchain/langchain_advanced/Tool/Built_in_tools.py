import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import FileManagementToolkit
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
# 2. 创建工具
# ======================

toolkit = FileManagementToolkit(root_dir=".")
tools = toolkit.get_tools()

agent = create_react_agent(
    model=llm,
    tools=tools,
    debug=True
)


if __name__ == "__main__":
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="在当前目录下创建一个名为test.txt的文件，并写入Hello World!")
            ]
        }
    )
    print("\n最终回答：")
    print(response["messages"][-1].content)
