# Creation_XHS
构建小红书图文创作应用的目的：
- 提高效率：通过大模型的语言生成能力，帮助用户快速生成高质量的图文内容，节省用户的时间和精力，同时提高内容创作的效率。
- 个性化推荐：利用大模型对用户喜好和行为的分析，可以根据用户的偏好生成个性化推荐的内容，提升用户体验，增加用户粘性。
- 降低门槛：对于不擅长创作或没有创作经验的用户，自动化图文生成应用可以降低创作门槛，让更多人参与到内容创作和分享中来。
- 增加多样性：通过大模型的丰富知识库和创作能力，可以生成多样性的内容，丰富平台上的图文信息，满足用户多样化的需求。

设计模块：
- 个人赛道定位模块：通过开放式问题引导用户，帮助用户找到喜欢的方向并确定选题。最后生成小红书笔记（采用Prompt模板来引导模型完成）（模型使用xutuner微调过的InternLM2-7b模型：internlm2-chat-7b-sft）
![image](https://github.com/Star-cre/Creation_XHS/assets/95208730/3bca08c6-8119-4a22-9f55-83420fa7195b)
- 图片生成模块：根据个人赛道定位模块最终生成的小红书笔记，使用智谱Ai图片接口来生成相应的图片。（后续可以考虑使用stableDiffusion/mj来实现）
![image](https://github.com/Star-cre/Creation_XHS/assets/95208730/8d6d4316-b70f-44a6-aea7-0521f7061451)


项目成员：
- [Star-cre](https://github.com/Star-cre)
- [Aitejiu](https://github.com/Aitejiu)
- [2404589803](https://github.com/2404589803)
- [Wly0910](https://github.com/Wly0910)
- [Durian-1111](https://github.com/Durian-1111)
