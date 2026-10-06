import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
DEEPSEEK_KEY = os.getenv('DEEPSEEK_KEY')
BASE_URL = os.getenv('BASE_URL')

llm = ChatOpenAI(
    model = "deepseek-flash",
    base_url = BASE_URL,
    api_key = DEEPSEEK_KEY,
)

resp1 = llm.invoke("简单介绍LangChain")
print("完整返回对象：", resp1)
print("\n最终文本：", resp1.content)



print("===========================")

# =========流式输出 stream =========
print("流式输出：")
full_text = ""
# stream返回迭代器，循环读取每一小块chunk
for chunk in llm.stream("简单介绍LangChain"):
    # chunk是AIMessageChunk，里面content是增量文本
    piece = chunk.content
    if piece:
        full_text += piece
        print(piece, end="", flush=True) # flush实时打印，不缓存
print("\n\n拼接完成的全文：", full_text)