from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
import os
import config_
os.environ['OPENAI_API_KEY'] = config_.openai_key

llm = OpenAI(model_name='text-davinci-003',
             temperature=0.9,
             max_tokens=1024)

# 首先创建一个短语示例,该示例包含两组输入和输出，每输入一个词语,LLM就会输出一个对应的反义词
examples = [
    {"输入": "AI+能源数字化与机器人专场：AI+矿山/电力，助推能源改革丨迎客松-产业高峰论坛暨国金证券中期策略会", "输出": "能源数字化"},
    {"输入": "西部郑宏达｜商汤 - 中国目前做的最好的端侧大模型公司", "输出": "端侧大模型"},
    {"输入": "国盛通信 | 谷歌OCS光交换机：光通信的新思路", "输出": "OCS光交换机"},
    {"输入": "申万宏源军工 | 2024电话会议第16场：星载T/R芯片核心民企，星座建设加速带动业绩高增 - 铖昌科技深度推荐", "输出": "星载T/R芯片"},
    {"输入": "民生建材新材料｜如何看待光伏玻璃价格弹性？", "输出": "光伏玻璃"},
    {"输入": "西部汽车｜Sora对智能驾驶落地的影响", "输出": "Sora"},
    {"输入": "西部电新｜N型电池片快速转型，低空经济迎发展东风 - 电新行业周度数据及周观点更新电话会", "输出": "低空经济"},
    {"输入": "东北军工百日谈 | Day7：我国军用无人机都有哪些？谈航天彩虹", "输出": "军用无人机"},
    {"输入": "招商证券 | 宏观周周谈：再议中国式QE", "输出": "中国式QE"},
    {"输入": "华创交运 · 低空经济活动周系列 | 第二期：eVTOL的市场在哪里？", "输出": "低空经济"},
    {"输入": "中泰计算机｜再谈Sora对AI Infra（算力）带来怎样的影响？", "输出": " Sora"},
    {"输入": "西部电新 | 深度报告解读第77期 -宇邦新材：光伏焊带龙头，新技术迭代提振未来业绩", "输出": "光伏焊带"},
    {"输入": "民生计算机 | 国内多模态和算力上市公司如何应对sora的横空出世 - Sora最佳受益方向系列电话会", "输出": "sora"},
    {"输入": "华创计算机 | 中科星图 - 低空经济高峰论坛", "输出": "低空经济"},
    {"输入": "国信计算机 | 国产大模型Kimi带动产业链革新", "输出": "Kimi"},
    {"输入": "东吴传媒互联网 | Sora系列电话会议3：再看Sora模型的意义、节奏及应用前景", "输出": "Sora"},
    {"输入": "长江计算机 | 树”说计算机（第86期）:  从KIMI看国产大模型发展", "输出": "KIMI"},
    {"输入": "痛钝性发阵间时短门肛", "输出": "错误"},
    {"输入": "国投焦娟 | MR+系列会议三：以Apple VIsion Pro为基准，空间计算设备的定性分析与定量比较", "输出": "Apple VIsion Pro"},
    {"输入": "中金公司 | 低空飞行业务更新及卧龙电驱推荐", "输出": "低空飞行"},
    {"输入": "华福AI互联网 | “Kimi VS 阶跃星辰＂测评", "输出": "Kimi,阶跃星辰"},
    {"输入": "华鑫传媒 | 专家鑫访谈：AI系列|对话Sora专家", "输出": "Sora"},
    {"输入": "民生电子｜MR深度：Vision Pro革新未来", "输出": "Vision Pro"},
    {"输入": "长江传媒互联网 | MR超声波（6）- 2023年XR硬件销量增速放缓，内容生态成破局关键", "输出": "MR超声波"},
]

#创建一个prompt模板，
example_prompt = PromptTemplate(
    input_variables=["输入", "输出"],
    template="\n输入: {输入}\n输出: {输出}\n",
)

# 最后我们创建一个短语prompt模板对象
few_shot_prompt = FewShotPromptTemplate(
    # 这些是我们要插入到prompt中的示例
    examples=examples,
    # 将示例插入prompt时，格式化示例的方式。
    example_prompt=example_prompt,
    # 输入变量是用户直接输入的变量
    input_variables=["input"],
    # 前缀变量
    prefix=f"假如你是一个专业的金融领域分析师，请学习我给你的案例,并提取文本中的新兴的专业词汇",
    # 后缀变量
    suffix="输入: {input}\n输出:",
    # 用来连接前缀、示例和后缀的字符串。
    example_separator="\n",
)

# 测试一下短语模板对象
# print(few_shot_prompt.format(input="小腹刺痛"))

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=few_shot_prompt)
print(chain.run("国盛医药｜脑机接口如何为大脑通信另辟蹊径，科幻照进现实？ - 专家会【第19期】 '经济"))
