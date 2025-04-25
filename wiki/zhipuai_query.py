import time
from zhipuai import ZhipuAI

#  API Key
client = ZhipuAI(api_key="8f9297e04ae34c2ab4604e3dc79fc2b1.U3z0Z7fVtp4sn9dz")


def query_china_capital():
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "system",
                 "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
                {"role": "user", "content": "中国的首都是什么？"}
            ]
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        capital = response.choices[0].message.content
        return capital, elapsed_time
    except Exception as e:
        print(f"查询出错: {e}")
        return None, None


capital, time_taken = query_china_capital()
if capital:
    print(f"中国的首都是: {capital}")
    print(f"查询耗时: {time_taken} 秒")
else:
    print("查询失败。")
