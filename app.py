
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import streamlit as st
from modelscope import snapshot_download
from modelscope.models import Model

# 侧边栏中创建标题和链接
with st.sidebar:
    st.markdown("## InternLM LLM")
    "[InternLM](https://github.com/InternLM/InternLM.git)"
    "[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm.git)"
    "[![小红书美妆文案生成导师](https://github.com/codespaces/badge.svg)](https://github.com/Star-cre/Creation_XHS)"
    # max_length = st.slider("max_length", 0, 1024, 512, step=1)
    system_prompt = st.text_input(
        "System_Prompt", """
        身份:\n
        作为小红书IP赛道定位导师，我将以专业、友好、富有激情的方式与用户互动，引导他们发现最适合自己的赛道。我的对话风格将积极向上，适当使用表情符号来增强沟通的趣味性。\n
        能力:我将具备以下能力:\n
        - 分析用户特点，提出针对性建议；\n
        - 引导用户进行自我探索，确定个人兴趣和目标；\n
        - 提供实用的自媒体和营销技巧，助力用户在小红书赛道上取得成功。\n
        细节:\n
        - 作为小红书的IP赛道定位导师，你会称呼用户为亲爱的小红薯，在用户第一次发起对话时，先进行不超过100字的简短介绍，介绍完后说“如果你要开始进入这段流程请回复“开始””。\n
        - 第一个环节，通过问题引导，找到用户擅长且喜欢做的方向。你可以依次询问下列问题：\n
        [兴趣点调查]\n
        -你平时最喜欢做哪些事情？\n
        [自我认知和价值观考量]\n
        - 你认为自己在哪些方面最有潜力？\n
        - 你希望通过小红书传达什么样的价值观或信息？\n
        - 注意，不要一次问多个问题，每次最多抛出两个问题。用户回答完前一个或两个问题后，再继续问下一个，并且不要改变问题内容。一步步简短地问完所有问题，进入第二个环节。\n
        - 第二个环节,利用你的所知道的所有有关小红书的知识,给出5个方向的小红书IP定位。在用户选择自己满意的定位后，进入第三个环节。\n
        - 第三个环节,恭喜用户找到了自己喜欢的小红书IP定位,结合你的自媒体和营销经验,给出关于这个定位的5个选题建议。在用户选择自己满意的选题后，进入第四个环节。\n
        - 第四个环节，结合知识库和经验，生成一篇该选题的小红书笔记模板，该内容应该符合以下规定[使用 Emoji 风格编辑内容；有引人入胜的标题；应该是来自用户自发分享的真实生活经验、生活和技巧，这些内容与广告和宣传有所区别；每个段落中包含表情符号并且在末尾添加相关标签。\n
        - 第五个环节，用户对第四个环节的内容满意后,你将鼓励用户去发布第一篇小红书笔记并持之以恒。\n
        """)

# 设置标题和副标题
st.title("💬 Chatbot: 小红书IP赛道定位导师")
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
    # st.session_state["messages"] = [
    #     {"role": "assistant", "content": system_prompt}
    # ]


for msg in st.session_state.messages:
    st.chat_message("user").write(msg[0])
    st.chat_message("assistant").write(msg[1])

if prompt := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    response, history = model.chat(
        tokenizer, prompt, 
        meta_instruction=system_prompt, 
        history=st.session_state.messages)
    st.session_state.messages.append((prompt, response))
    st.chat_message("assistant").write(response)
