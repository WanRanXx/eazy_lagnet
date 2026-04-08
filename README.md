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