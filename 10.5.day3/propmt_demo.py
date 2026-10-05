import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
DEEPSEEK_KEY = os.getenv('DEEPSEEK_KEY')
BASE_URL = os.getenv('BASE_URL')

#1.zero-shot
model = ChatOpenAI(
    model = 'deepseek-flash',
    api_key = DEEPSEEK_KEY,
    base_url = BASE_URL
)

messages1  = [
    SystemMessage(content="""
    你是一个计算器，需要你严格计算
    输出要求：只需要输出JSON格式，不需要任何解释
    JSON格式：{"formula":"算式","result":"答案"}
    禁止输出JSON以外的任何内容
    """),
    HumanMessage(content='1+2+3*1=?')
]

print(model.invoke(messages1))

#2.fow-shot
messages2 = [
    SystemMessage(content="""
    你是一个计算器，需要你严格计算
    输出要求：严格按照下面案例的输出格式
    实例1
    输入：2+2
    输出：{"formula":"2+2","result":"4"}    
    输入：2+6
    输出：{"formula":"2+6","result":"8"}
    禁止输出JSON以外的任何内容
    """),
    HumanMessage(content='2*3*2')
]

print(model.invoke(messages2))

#3.Cot
message3 = [
    SystemMessage(content="""
    规则：
    1. 拿到题目后，**分步写出推理思考过程**
    2. 最后单独一行输出最终JSON结果
    3. 思考过程是自然文字，结果严格使用 {"formula":"算式","result":数字}
    """),
    HumanMessage(content='2*3*2')
]
print(model.invoke(message3))