import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from zhipuai import ZhipuAI
import requests
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import streamlit as st
from modelscope import snapshot_download

# 侧边栏中创建标题和链接
with st.sidebar:
    st.markdown("## InternLM LLM")
    "[InternLM](https://github.com/InternLM/InternLM.git)"
    "[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm.git)"
    "[![小红书美妆文案生成导师](https://github.com/codespaces/badge.svg)](https://github.com/Star-cre/Creation_XHS)"

# 设置标题和副标题
st.title("💬 : 文字转图片")
st.caption("🚀 A streamlit APP powered by 智谱AI")


def blog_outline(topic):
    api_key = "d93d034e042e186b6cd605c6bb6fd31f.ApkLkpE8XiFAFkF2"
    client = ZhipuAI(api_key=api_key)  # 请填写您自己的APIKey
    response = client.images.generations(
        model="cogview-3",  # 填写需要调用的模型名称
        #     prompt=f'请根据以下文案生成产品图：{prompt},参考提示词：{short_prompt}',
        prompt=topic,
    )
    # 得到图片的URL
    url_of_image = response.data[0].url
    generated_image = get_image_from_url(url_of_image)
    st.image(generated_image, caption="Generated Image", use_column_width=True)

def get_image_from_url(image_url):
    # 发送 HTTP 请求并下载图片
    response = requests.get(image_url)
    # 检查响应状态码
    if response.status_code == 200:
        # 从响应中获取图像数据
        image_data = response.content
        # 将图像数据转换为 PIL 图像对象
        image = Image.open(BytesIO(image_data))
        # 显示图像
        # image.show()
        return image
    else:
        print("Failed to download image. Status code:", response.status_code)


with st.form("myform"):
    topic_text = st.text_input("Enter prompt:", "")
    submitted = st.form_submit_button("Submit")
    if submitted:
        blog_outline(topic_text)

