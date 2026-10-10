import json
from openai import OpenAI

# ========= 1. 初始化大模型客户端 =========
# 替换 api_key，base_url可以换成deepseek、glm等
client = OpenAI(
    api_key="、",
    base_url="https://api.deepseek.com"
)

# ========= 2.【本地真实工具函数】Python函数，模型看不见这个代码 =========
def get_weather(city: str):
    """查询城市天气，真实执行的逻辑"""
    data = {
        "武汉": "晴，26℃，微风",
        "北京": "多云，20℃，北风3级",
        "上海": "小雨，22℃，东南风"
    }
    return data.get(city, f"没有查询到 {city} 的天气")

# ========= 3.【Tool Schema 工具说明书】传给大模型，模型只看这个！ =========
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市实时天气。用户询问某个城市天气的时候调用这个工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名字，例如武汉、北京"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 函数映射表：模型返回函数名字符串，找到本地对应的python函数
func_map = {
    "get_weather": get_weather
}

# ========= 4. 主逻辑：Function Calling完整流程 =========
def run_agent(user_question):
    # 对话消息列表，保存所有对话历史
    messages = [
        {"role": "system", "content": "你是天气助手，可以调用get_weather查询天气。"},
        {"role": "user", "content": user_question}
    ]

    # 限制最大循环次数，防止无限死循环
    max_loop = 3
    loop_count = 0

    while loop_count < max_loop:
        loop_count += 1
        # 调用大模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"  # auto：模型自己判断要不要调用工具
        )

        msg = response.choices[0].message

        # 分支1：不需要调用工具，直接返回回答，结束
        if not msg.tool_calls:
            return msg.content

        # 分支2：模型要求调用工具！
        # 把assistant的tool_call消息加入上下文
        messages.append(msg)

        # 遍历所有要调用的工具（支持并行调用多个）
        for tool_call in msg.tool_calls:
            call_id = tool_call.id
            func_name = tool_call.function.name
            # arguments是JSON字符串，必须解析！
            args = json.loads(tool_call.function.arguments)

            # 执行本地函数、
            target_func = func_map[func_name]
            tool_result = target_func(**args)

            # 【关键】把工具执行结果包装成tool消息，放回对话上下文
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": tool_result
            })

    # 如果循环耗尽，返回提示
    return "多次调用工具仍然无法获取答案"


# ========= 入口，运行测试 =========
if __name__ == "__main__":
    user_input = input("请提问：")
    result = run_agent(user_input)
    print("AI最终回答：", result)
