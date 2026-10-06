import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

# 加载.env环境变量
load_dotenv(override=True)

DEEPSEEK_KEY = os.getenv('DEEPSEEK_KEY')
BASE_URL = os.getenv('BASE_URL')

# 文本信息提取器
# 1.定义输出结构
class Person(BaseModel):
    name : str
    age : int
    city : str
    job : str
    skill_list : list[str]

# 2.初始化DeepSeek模型，extra_body关闭思考模式！！
llm = ChatOpenAI(
    model='deepseek-flash',
    api_key=DEEPSEEK_KEY,
    base_url=BASE_URL,
    # DeepSeek独有参数，关闭思考模式，解决tool_choice冲突
    extra_body={
        "thinking": {"type": "disabled"}
    }
)

# 使用function_calling结构化输出，删掉strict=True（deepseek不支持strict）
extractor = llm.with_structured_output(Person, method="function_calling")

text = """
张明，今年27岁，定居在杭州。
职业是后端开发工程师，熟悉Python、LangChain、MySQL、FastAPI。
"""

try:
    result = extractor.invoke(text)
    print("=== 抽取结果（Pydantic对象）===")
    print(f"姓名：{result.name}")
    print(f"年龄：{result.age}")
    print(f"城市：{result.city}")
    print(f"工作：{result.job}")
    print(f"技能：{result.skill_list}")

except ValidationError as e:
    print("本地校验失败：", e.errors())
except Exception as e:
    print("调用异常：", e)
