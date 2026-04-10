"""
路由链实现，采用新版LangChain写法（基于Runnable范式，使用ChatOpenAI模型）。
"""
from chat_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# =============== 1. 一行获取 DeepSeek 模型 ===============
llm = LLMFactory.get_llm("deepseek")

# 2. 定义各场景的提示词模板与目标链（新版用RunnableSequence组合Prompt+LLM+Parser）
# 场景1：查订单链
order_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个销售人员，负责解决用户订单查询问题"),
    ("human", "用户问题：{query}， 请引导用户提供订单号，并告知查询流程：1. 提供订单号；2. 系统验证；3. 反馈订单状态。"),
])

order_chain = order_prompt | llm | StrOutputParser()

# 场景2：退货链
refund_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个销售人员，负责解决用户退货问题"),
    ("human", "用户问题：{query}， 请引导用户提供订单号和退货原因，并告知退货流程：请说明退款流程：1. 申请退款（订单页面点击退款）；2. 等待审核（1-3个工作日）；3. 退款到账（原路返回，3-5个工作日）。如果用户问退款进度，引导提供退款申请单号。"),
])
refund_chain = refund_prompt | llm | StrOutputParser()

# 3.售后链
warranty_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智能客服，负责解答产品保修政策问题。"),
    ("human", "用户问题：{query}\n请说明保修政策：本产品保修期限为1年，保修范围包括质量问题（非人为损坏），保修流程：1. 联系客服；2. 提供购买凭证；3. 寄回检测维修。")
])
warranty_chain = warranty_prompt | llm | StrOutputParser()
# 3. 定义路由判断逻辑（大模型解析需求，输出场景标识）
# 路由提示词：让大模型输出标准化的场景名称，用于后续分支匹配
# 3. 定义路由判断逻辑（大模型解析需求，输出场景标识）
# 路由提示词：让大模型输出标准化的场景名称，用于后续分支匹配
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是路由选择器，需根据用户问题判断所属场景，仅输出以下标准化标识之一：
- order：订单查询相关（含订单状态、订单号）
- refund：退货款相关（含退款进度、退款申请）
- warranty：保修相关（含维修、售后保障）
- default：以上均不匹配
无需输出任何其他内容，仅返回标识字符串。
"""),
    ("human", "用户问题：{query}")
])

# 路由解析链
router_chain = router_prompt | llm | StrOutputParser()

# 4.定义默认链
default_prompt = PromptTemplate(
    input_variables=["query"],
    template="抱歉，我无法解答你的问题'{query}'。请你重新描述问题，或者联系人工客服（工作时间：9:00-18:00）。"
)
default_chain = default_prompt | llm | StrOutputParser()

# 5. 构建完整路由链（核心：RunnableBranch实现条件分发）
# 逻辑：先通过router_chain获取场景标识，再由RunnableBranch分发到对应目标链

full_router_chain = RunnableLambda(lambda x:x ) | (
    # 分支场景分发
    RunnableBranch(
        (lambda x: x["scene"] == "order", order_chain),
        (lambda x: x["scene"] == "refund", refund_chain),
        (lambda x: x["scene"] == "warranty", warranty_chain),
        default_chain  # 默认链放在最后，作为兜底
    )

).with_config(
    run_name="full_router_chain"
)

# 6. 封装调用函数（整合场景解析与路由分发）
def process_query(query: str):
    # 第一步：获取场景标识
    scene = router_chain.invoke({"query": query})
    # 第二步：将query和scene传入完整路由链，执行分发处理
    return full_router_chain.invoke({"query": query, "scene": scene})

# 7. 测试不同场景输入
test_queries = [
    "我的订单什么时候发货？",
    "怎么申请退款呀？",
    "这个产品保修多久？",
    "你们家有什么新品？"  # 无法匹配，触发默认链
]

for query in test_queries:
    print(f"\n用户问题：{query}")
    print("客服回复：", process_query(query))



