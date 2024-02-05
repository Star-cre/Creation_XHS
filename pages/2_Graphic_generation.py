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
    system_prompt = st.text_input(
        "System_Prompt", """
        你是一个小红书文案生成器，请按照以下规则生成小红书文案：\n
        - 主题/产品：xx（在这里填写具体的美妆产品名称或类别）\n
        - 需求：撰写一篇关于xx的小红书爆款文案，突出其特点和使用体验\n
        - 风格：口语化、生动活泼，使用Emoji表情图标，吸引读者注意\n
        - 限制：文案长度控制在500字以内，避免连续性标题结构，主要以中文思维方式撰写\n
        请不要输出多余的文字，主输出文案本体\n
        下边的[]内给出需要生成的小红书文案主题/产品\n
        """)

# 设置标题和副标题
st.title("💬 Chatbot: 小红书图文生成")
st.caption("🚀 A streamlit chatbot powered by InternLM LLM")

mode_name_or_path = '/root/xhs_tuner/Creation_XHS'
# mode_name_or_path = '/root/xhs_tuner/internlm2-chat-20b-4bits'
# mode_name_or_path = '/root/share/model_repos/internlm2-chat-20b-4bits'
# mode_name_or_path = 'aitejiu/xhs_createation'

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(
        mode_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mode_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return tokenizer, model


tokenizer, model = get_model()

# 如果session_state中没有"messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []


for msg in st.session_state.messages:
    st.chat_message("user").write(msg[0])
    st.chat_message("assistant").write(msg[1])


if prompt := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)
    prompt = '[' + prompt + ']'
    response, history = model.chat(
        tokenizer, prompt, meta_instruction=system_prompt, history=st.session_state.messages)
    st.session_state.messages.append((prompt, response))
    st.chat_message("assistant").write(response)