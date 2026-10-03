import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
DEEPSEEK_KEY =os.getenv('DEEPSEEK_KEY')
BASE_URL = os.getenv('BASE_URL')

model = ChatOpenAI(
    model = 'deepseek-flash',
    api_key=DEEPSEEK_KEY,
    base_url=BASE_URL
)

print(model.invoke("你是谁家的助手"))