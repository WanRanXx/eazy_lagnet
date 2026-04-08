"""
初次实战，利用eazy_lagent案例3.3.2 实践2：带记忆的文件夹操作助手，创建一个实现github readme.md内容自动补充助手, tavily作为搜索引擎
"""

import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_API_BASE")
tavily_api_key = os.getenv("TAVILY_API_KEY")
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=api_key,
    base_url=base_url,
    max_tokens=1024,
    streaming=True
)

search_tool = TavilySearchResults(api_key=tavily_api_key, max_results=5, description="检索相关项目文档，信息与最佳实践")

WINDOW_SIZE = 5
memory_store = {}

def get_window_history(user_id: str) -> BaseChatMessageHistory:
    if user_id not in memory_store:
        memory_store[user_id] = InMemoryChatMessageHistory()
    history = memory_store[user_id]
    if len(history.messages) > WINDOW_SIZE * 2:
        history.messages = history.messages[-WINDOW_SIZE * 2:]
    return history

# =========================
# 4. 定义工具（@tool）
# =========================

root_path = r"D:\data\langchain\eazy_Iagent"

@tool(description="查看指定文件目录下的文件列表")
def list_files(path: str = ".") -> str:
    try:
        if not os.path.exists(path):
            return f"目录 {path} 不存在"
        files = os.listdir(path)
        if not files:
            return "该目录下没有文件"
        results = []
        for file in files:
            full = os.path.join(path, file)
            if os.path.isfile(full):
                results.append(f"文件：{file} ({os.path.getsize(full)}字节)")
            else:
                results.append(f"文件夹：{file}")
        return "\n".join(files)
    except Exception as e:
        return f"发生错误：{str(e)}"


@tool(description="查看指定文件夹的文件树结构，可设定层级深度")
def view_file_tree(path: str = ".", max_depth: int = 2) -> str:
    """
    生成指定路径的文件树结构

    参数:
        path: 要查看的文件夹路径，默认为当前目录
        max_depth: 最大层级深度，默认为2层

    返回:
        格式化的文件树字符串
    """

    def _build_tree(current_path: str, current_depth: int, prefix: str = "") -> list:
        if current_depth > max_depth:
            return []

        tree_lines = []
        try:
            items = sorted(os.listdir(current_path))
        except PermissionError:
            return [f"{prefix}[权限受限]"]
        except Exception as e:
            return [f"{prefix}[错误: {str(e)}]"]

        # 过滤隐藏文件和常见忽略目录
        ignore_items = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.idea', '*.pyc'}
        items = [item for item in items if item not in ignore_items and not item.startswith('.')]

        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "

            full_path = os.path.join(current_path, item)

            if os.path.isdir(full_path):
                tree_lines.append(f"{prefix}{connector}📁 {item}/")
                extension = "    " if is_last else "│   "
                subtree = _build_tree(full_path, current_depth + 1, prefix + extension)
                tree_lines.extend(subtree)
            else:
                tree_lines.append(f"{prefix}{connector}📄 {item}")

        return tree_lines

    try:
        # 转换为绝对路径
        abs_path = os.path.abspath(path)

        # 检查路径是否存在
        if not os.path.exists(abs_path):
            return f"错误：路径 {path} 不存在"

        # 检查是否为目录
        if not os.path.isdir(abs_path):
            return f"错误：{path} 不是一个目录"

        # 构建文件树
        folder_name = os.path.basename(abs_path)
        tree_lines = [f"📦 {folder_name}/"]
        tree_content = _build_tree(abs_path, 1)
        tree_lines.extend(tree_content)

        result = "\n".join(tree_lines)

        # 统计信息
        total_dirs = sum(1 for line in tree_lines if '📁' in line)
        total_files = sum(1 for line in tree_lines if '📄' in line)

        result += f"\n\n总计: {total_dirs} 个目录, {total_files} 个文件 (最大深度: {max_depth}层)"

        return result

    except Exception as e:
        return f"生成文件树时出错：{str(e)}"

@tool(description="创建文件，并写入初始内容")
def create_file(path: str, content:str ="") -> str:
    """创建文件，并写入初始内容"""
    try:
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'w', encoding="utf-8") as f:
            f.write(content)
        return f"已创建文件：{path}"
    except Exception as e:
        return f"无法创建文件：{str(e)}"

@tool(description="读取文件内容")
def read_file(path: str) -> str:
    """读取文件内容"""
    try:
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            return f"文件不存在：{path}"
        with open(path, 'r', encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"无法读取文件：{str(e)}"

@tool(description="写入文件内容,支持追加内容或覆盖写入")
def write_file(path:str, content:str, append: bool=True) -> str:
    try:
        if not os.path.exists(path):
            return f"文件夹不存在：{os.path.dirname(path)}"
        mode = 'a' if append else 'w'
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        return f"文件写入成功"
    except Exception as e:
        return f"无法写入文件：{str(e)}"

@tool(description="删除文件或空文件夹")
def delete_file(path: str) -> str:
    """删除文件或空文件夹"""
    try:
        if not os.path.exists(path):
            return f"路径不存在：{path}"

        if os.path.isfile(path):
            os.remove(path)
            return f"文件已删除：{path}"

        if os.path.isdir(path):
            if os.listdir(path):
                return "文件夹非空，无法删除"
            os.rmdir(path)
            return f"文件夹已删除：{path}"

        return "无效路径"
    except Exception as e:
        return f"删除失败：{e}"

prompt = ChatPromptTemplate(
    [
        ("system", """
        你是一个**计划执行智能体**，严格按照以下流程工作：
        1. 理解用户需求 → 拆解成清晰的执行步骤（计划）；\n
        2. 自动调用工具完成每一步（文件操作/联网搜索）；\n
        3. 执行完成后给出总结；\n
        4. 保留对话历史，多轮对话保持上下文连贯；\n
        5. 不需要工具时直接回答;\n
    
        可用工具：
        - 文件管理：列出文件、查看目录树、创建/读取/写入/删除文件;\n
        - 联网搜索：检索文档、资料、最佳实践; \n
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ]
)

tools = [list_files, view_file_tree, create_file, read_file, write_file, delete_file, search_tool]

# =========================
# 6. 构建 Tool-Calling Agent
# =========================

chain = prompt | llm.bind_tools(tools)

plan_execute_agent = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_window_history,
    history_messages_key="chat_history",
    input_messages_key="user_input",
    output_messages_key="output",
    debug=True
)

# =========================
# 4. 使用示例（对话入口）
# =========================
if __name__ == "__main__":
    session_id = "tool_agent_demo"

    print("===== 🧠 Tool Calling 文件 Agent =====")
    print("示例：")
    print(" - 查看当前文件夹")
    print(" - 创建文件 test.txt 内容 Hello")
    print(" - 写入文件 test.txt 内容 World 追加")
    print(" - 删除文件 test.txt")
    print("输入 q 退出\n")

    while True:
        user_input = input("你：")
        if user_input.lower() in ["q", "quit", "退出"]:
            print("助手：再见 👋")
            break

        # ===== 第一次：模型思考 =====
        result = plan_execute_agent.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        history = get_window_history(session_id)

        print("\n🧠【模型输出】")
        if result.content:
            print(result.content)

        # ===== 模型决定调用工具 =====
        if isinstance(result, AIMessage) and result.tool_calls:
            print("\n🔧【模型决定调用工具】")
            for call in result.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]

                print(f"➡️ 工具名：{tool_name}")
                print(f"➡️ 参数：{tool_args}")

                tool_func = next(t for t in tools if t.name == tool_name)
                observation = tool_func.invoke(tool_args)

                print("\n📦【工具执行结果】")
                print(observation)

                history.add_message(
                    ToolMessage(
                        tool_call_id=call["id"],
                        content=str(observation)
                    )
                )

            print("\n✅【本轮结束：工具执行完成】\n")
            continue  # 回到 while True 等用户输入

        # ===== 最终回答（没有工具调用） =====
        print("\n✅【最终回答】")
        print(result.content, "\n")