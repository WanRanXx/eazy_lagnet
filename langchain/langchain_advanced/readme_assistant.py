"""
初次实战，利用eazy_lagent案例3.3.2 实践2：带记忆的文件夹操作助手，创建一个实现github readme.md内容自动补充助手, tavily作为搜索引擎
使用 ReAct (Reasoning and Acting) 范式实现智能体
"""

import os
import re
import json
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
tavily_api_key = os.getenv("TAVILY_API_KEY")
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=api_key,
    base_url=base_url,
    max_tokens=2048,
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

# =========================
# 5. 构建 ReAct Agent
# =========================

# ReAct 提示词模板
react_prompt = ChatPromptTemplate(
    [
        ("system", """
你是一个遵循 ReAct (Reasoning and Acting) 范式的智能助手。

工作流程：
1. **Thought (思考)**: 分析当前情况，推理下一步应该做什么
2. **Action (行动)**: 选择合适的工具并执行
3. **Observation (观察)**: 获取工具执行结果
4. 循环执行直到得出最终答案

可用工具：
- list_files: 查看指定目录下的文件列表
- view_file_tree: 查看文件夹的树形结构
- create_file: 创建文件并写入内容
- read_file: 读取文件内容
- write_file: 写入文件内容（支持追加或覆盖）
- delete_file: 删除文件或空文件夹
- search_tool: 联网搜索文档和资料

输出格式：
Thought: [你的思考过程]
Action: [工具名称]
Action Input: {{"参数名": "参数值"}}

当得出最终答案时，使用：
Final Answer: [最终回答]

注意：
- 每次只执行一个工具
- 仔细分析工具执行结果后再决定下一步
- 如果不需要工具，直接给出 Final Answer
- 保持对话历史连贯性
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ]
)

tools = [list_files, view_file_tree, create_file, read_file, write_file, delete_file, search_tool]

# 创建工具映射字典
tool_map = {tool.name: tool for tool in tools}

# ReAct Agent 链
react_chain = react_prompt | llm

# 带记忆的 ReAct Agent
react_agent = RunnableWithMessageHistory(
    runnable=react_chain,
    get_session_history=get_window_history,
    history_messages_key="chat_history",
    input_messages_key="user_input",
    output_messages_key="output",
    debug=True
)

# =========================
# 6. ReAct 解析和执行函数
# =========================

def parse_react_output(text: str):
    """
    解析 ReAct 格式的输出
    返回: (thought, action, action_input, final_answer)
    """
    thought = None
    action = None
    action_input = None
    final_answer = None
    
    # 提取 Thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)', text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    
    # 提取 Action 和 Action Input
    action_match = re.search(r'Action:\s*(.+?)\n', text)
    if action_match:
        action = action_match.group(1).strip()
        
        # 提取 Action Input (JSON 格式)
        action_input_match = re.search(r'Action Input:\s*(\{.+?\})', text, re.DOTALL)
        if action_input_match:
            try:
                action_input = json.loads(action_input_match.group(1))
            except json.JSONDecodeError:
                action_input = {}
    
    # 提取 Final Answer
    final_answer_match = re.search(r'Final Answer:\s*(.+?)$', text, re.DOTALL)
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
    
    return thought, action, action_input, final_answer

def execute_react_loop(user_input: str, session_id: str, max_iterations: int = 5):
    """
    执行 ReAct 循环
    """
    history = get_window_history(session_id)
    
    for iteration in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"🔄 迭代 {iteration + 1}/{max_iterations}")
        print(f"{'='*50}")
        
        # 调用模型
        result = react_agent.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        print("\n🧠【模型输出】")
        print(result.content)
        
        # 解析 ReAct 输出
        thought, action, action_input, final_answer = parse_react_output(result.content)
        
        # 如果有最终答案，返回
        if final_answer:
            print("\n✅【最终答案】")
            print(final_answer)
            return final_answer
        
        # 如果没有 action，直接返回模型输出
        if not action:
            print("\n✅【直接回答】")
            return result.content
        
        # 执行工具
        print(f"\n🔧【执行工具】")
        print(f"Thought: {thought}")
        print(f"Action: {action}")
        print(f"Action Input: {action_input}")
        
        if action in tool_map:
            try:
                tool_func = tool_map[action]
                observation = tool_func.invoke(action_input or {})
                
                print(f"\n📦【观察结果】")
                print(observation)
                
                # 将观察结果添加到历史
                observation_message = f"Observation: {observation}"
                history.add_message(AIMessage(content=f"Action: {action}\nAction Input: {action_input}"))
                history.add_message(HumanMessage(content=observation_message))
                
                # 更新 user_input 为观察结果，继续循环
                user_input = observation_message
                
            except Exception as e:
                error_msg = f"工具执行错误: {str(e)}"
                print(f"\n❌【错误】{error_msg}")
                user_input = f"Observation: {error_msg}"
        else:
            error_msg = f"未知工具: {action}"
            print(f"\n❌【错误】{error_msg}")
            user_input = f"Observation: {error_msg}"
    
    print("\n⚠️【达到最大迭代次数】")
    return "达到最大迭代次数，未能完成任务"

# =========================
# 7. 使用示例（对话入口）
# =========================
if __name__ == "__main__":
    session_id = "react_agent_demo"

    print("===== 🧠 ReAct 文件 Agent =====")
    print("ReAct 范式: Thought → Action → Observation → 循环")
    print("\n示例：")
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

        # 执行 ReAct 循环
        final_result = execute_react_loop(user_input, session_id)
        print(f"\n{'='*50}\n")
