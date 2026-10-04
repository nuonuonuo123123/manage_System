import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from openrouter.components import AssistantMessage

# 1.第一种不使用langchain的message调用
load_dotenv(override=True)
DEEPSEEK_KEY = os.getenv('DEEPSEEK_KEY')
BASE_URL = os.getenv('BASE_URL')

client = OpenAI(
    api_key = DEEPSEEK_KEY,
    base_url = BASE_URL
)

messages = [
    {"role": "system", "content": "你是一个Python程序员，代码尽量简短。"},
    {"role": "user", "content": "写一个求斐波那契函数"},
    {"role": "assistant", "content": "```python\ndef fib(n):\n    if n<=2: return 1\n    return fib(n-1)+fib(n-2)\n```"},
    {"role": "user", "content": "改成迭代版本"}
]

#create必须包含两个 一个是model 一个是message
responses = client.chat.completions.create(
    model = 'deepseek-flash',
    messages = messages
)

#context是上下文 content是内容
#resp.choices[0].message.content = 拿到 API 返回的第一条 AI 回答文本
answer = responses.choices[0].message.content
print(answer)



#2.使用langchain调用 ChatPromptTemplate模板写法
model = ChatOpenAI(
    model = 'deepseek-flash',
    api_key = DEEPSEEK_KEY,
    base_url = BASE_URL
)

messages = [
    SystemMessage(content="你是一个数学家"),
    HumanMessage(content="请解释勾股定理"),
]

resp1 = model.invoke(messages)
print(resp1)

#可以接着添加对话
messages.append(AIMessage(content=resp1.content))
messages.append(HumanMessage(content="举一个例子"))
resp2 = model.invoke(messages)
print(resp2)
