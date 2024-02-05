## 微调

我们使用的是**[internlm2-chat-7b-sft](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-sft)**进行简单的微调。

### 数据集

```json
 {
        "conversation": [
            {
                "system": "你是一个小红书文案生成器，你会以下列规则编写小红书文案：以令人着迷的标题引起关注，每段文案都应该简洁明了，充满表情符号，增添趣味和情感。最后，还需要为文案添加与主题相关的标签，以便吸引更多的读者和关注。利用自然语言处理技术和创意写作的规则，生成优质的小红书风格文案，让用户的内容在小红书平台上脱颖而出。适当使用emoji。下边[]内的是需要书写的内容：",
                "input": "[面膜]",
                "output": "标题：[知识小贴士]XXX 面膜——让你的生活更智慧\n\n正文：📚想要生活更轻松？XXX 面膜是你的不二选择！\n\n🌟我们的面膜采用最新科技，不仅高效，而且环保。让每一天的生活都更简单、更智能。\n\n💡现在购买，还有智慧优惠等你哦！快来抢购吧，让你的生活更智慧、更便捷！\n\n#面膜 #知识小贴士 #智慧生活 #高效神器"
            }
        ]
    },
```

### 流程

**使用`xtuner`进行微调**

**流程在目录`/root/xhs_tuner`中进行**

#### 复制模板

```bash
# 复制配置文件到当前目录
xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 .
# 改个文件名
mv internlm2_chat_7b_qlora_oasst1_e3_copy.py internlm2_chat_7b_qlora_xhs_e10.py

# 修改配置文件内容
vim internlm2_chat_7b_qlora_xhs_e10.py
```

#### 修改文件

**修改文件中的`pretrained_model_name_or_path`，`data_path`，`max_epochs`，`train_dataset`**

内容如下

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/root/share/model_repos/internlm2-chat-7b-sft'

# Data
data_path = 'data/xhs_data.json'
....
....
max_epochs = 10
....
....
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
```

#### 开始微调

```bash
xtuner train ./internlm2_chat_7b_qlora_xhs_e10.py --deepspeed deepspeed_zero2
```

微调得到的 PTH 模型文件和其他杂七杂八的文件都默认在当前的 `./work_dirs` 中。

#### 将得到的 PTH 模型转换为 HuggingFace 模型

```bash
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm2_chat_7b_qlora_xhs_e10.py
./work_dirs/internlm2_chat_7b_qlora_xhs_e10/epoch_10.pth ./hf
```

#### 将 HuggingFace adapter 合并到大语言模型

```bash
xtuner convert merge /root/share/model_repos/internlm2-chat-7b-sft ./hf ./Creation_XHS --max-shard-size 2GB
```

#### 与合并后的模型对话

```bash
xtuner chat ./merged --prompt-template internlm_chat
```

