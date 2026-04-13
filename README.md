## 基于Datawhale的eazy-langent项目进行学习

## 项目简介
这是基于Datawhale的eazy-langent项目进行学习
Datawhale eazy-langent项目地址 https://datawhalechina.github.io/easy-langent

## 项目结构

```
📦 eazy_lagent/
├── 📄 README.md                    # 项目说明文档
├── 📁 langchain/                   # LangChain相关学习代码
│   ├── 📁 langchain_advanced/      # LangChain高级功能
│   │   ├── 📁 Tool/                # 工具模块
│   │   ├── 📄 __init__.py          # 包初始化文件
│   │   ├── 📁 memory/              # 记忆模块
│   │   ├── 📄 memory_and_tools.py  # 记忆与工具使用示例
│   │   └── 📄 readme_assistant.py  # README助手示例
│   └── 📁 langchain_foundation/    # LangChain基础功能
│       ├── 📄 custom_output_parser.py      # 自定义输出解析器
│       ├── 📄 example.json                 # 示例数据
│       ├── 📄 example_selector_study.py    # 示例选择器学习
│       ├── 📄 output_control.py            # 输出控制
│       └── 📄 pydatic_output_parser.py     # Pydantic输出解析器
├── 📁 langgraph/                   # LangGraph相关代码
├── 📄 main.py                      # 主程序入口
├── 📄 pyproject.toml               # Python项目配置
├── 📁 tools/                       # 工具模块
│   ├── 📄 __init__.py              # 包初始化文件
│   └── 📄 model_list.py            # 模型列表管理
└── 📄 uv.lock                      # 依赖锁定文件

总计: 7 个目录, 14 个文件
```

## 学习内容
- LangChain基础：链、提示模板、输出解析器
- LangChain高级：工具使用、记忆管理、智能体
- LangGraph：状态机、工作流管理
- 实际应用：构建智能助手、自动化任务处理

## 核心实现：ReAct 智能体

### ReAct (Reasoning and Acting) 范式

本项目采用 **ReAct 范式** 实现智能助手，这是一种更接近人类思维模式的智能体架构。

#### ReAct 工作流程

```
用户输入 → Thought (思考) → Action (行动) → Observation (观察) → 循环 → Final Answer
```

#### 核心特点

1. **Thought (思考)**: 模型分析当前情况，推理下一步应该做什么
2. **Action (行动)**: 选择合适的工具并执行
3. **Observation (观察)**: 获取工具执行结果
4. **循环执行**: 根据观察结果继续思考，直到得出最终答案

#### 与 Tool-Calling 的区别

| 特性 | Tool-Calling | ReAct |
|------|--------------|-------|
| 思考过程 | 隐式 | 显式 (Thought) |
| 执行方式 | 自动调用工具 | 显式循环 (Action → Observation) |
| 可解释性 | 较低 | 高 (每步都有明确推理) |
| 调试难度 | 较难 | 容易 (可追踪每一步) |
| 适用场景 | 简单任务 | 复杂多步骤任务 |

#### 实现文件

- `langchain/langchain_advanced/readme_assistant.py`: ReAct 智能体实现
  - 文件操作工具：列出文件、查看目录树、创建/读取/写入/删除文件
  - 联网搜索工具：Tavily 搜索引擎
  - 记忆管理：滑动窗口记忆机制
  - ReAct 循环：自动解析和执行 Thought-Action-Observation

#### 使用示例

```python
# 运行 ReAct 智能体
python langchain/langchain_advanced/readme_assistant.py

# 示例对话
你：查看当前文件夹
🧠【模型输出】
Thought: 用户想查看当前文件夹的内容，我需要使用 list_files 工具
Action: list_files
Action Input: {"path": "."}

🔧【执行工具】
Action: list_files
Action Input: {"path": "."}

📦【观察结果】
README.md
langchain
langgraph
main.py
...
```