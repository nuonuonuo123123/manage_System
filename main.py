from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True) # 加载当前目录的.env
DEEPSEEK_KEY =os.getenv('DEEPSEEK_KEY')
BASE_URL = os.getenv('BASE_URL')

client = OpenAI(
api_key=DEEPSEEK_KEY,
    base_url=BASE_URL
)

# create必须要有两个配置一个是model一个是messages
respond = client.chat.completions.create(
    model ="deepseek-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "1+1=?"},
    ]
)

print(respond.choices[0].message.content)
