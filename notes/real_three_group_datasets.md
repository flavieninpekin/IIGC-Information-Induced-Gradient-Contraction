# 真实三组异质数据集短名单

> 最后更新：2026-08-28。目标是寻找“同一输入由多个真实群体分别评价”，从而能够
> 计算 group-conditioned gradients，并检验共享模型与条件化模型的折衷。
> 许可判断只依据当前公开数据卡和仓库说明；正式发布前仍需保存数据版本和完整条款。

## 1. 选择标准

候选数据集需要尽量满足：

1. 同一 item/input 被多个参与者或评审者看到；
2. 每条标注能关联到真实的 group/rater 属性；
3. 至少有三个样本量足够的群体；
4. 群体之间有目标、偏好或判断差异，而不仅是输入域不同；
5. 可以在不把 group 放进模型输入的情况下构造条件损失；
6. 有独立的 item-level 测试划分，避免同一输入泄漏到训练和测试；
7. 数据许可允许研究使用，并能遵守隐私和敏感属性要求。

## 2. 推荐顺序

| 优先级 | 数据集 | 输入/目标 | 群体信息 | 可用性判断 |
|---|---|---|---|---|
| A | **DICES-350** | 350 个对话，安全/伤害多项标注 | race/ethnicity、gender、age、education | 当前最适合做第一版真实梯度实验 |
| A- | **World Values Survey / WorldValuesBench** | 同一价值问题的个人问卷答案 | country/continent、urban/rural、education | 最贴合“同一问题、不同群体目标”，但原始数据有非再分发限制 |
| B+ | **POPQUORN** | 邮件/评论的礼貌、冒犯、QA 标注 | annotator race、age、gender、education | 条件结构非常好，需先核实数据许可和字段完整性 |
| B | **DICES-990** | 990 个对话，多评审安全标注 | US/India、gender | 比 DICES-350 输入更多，但群体设计较不适合三种族对照 |
| C+ | **PRISM Alignment** | LLM 对话、评分和细粒度反馈 | participant demographics、country、survey preference | LLM 相关性最高，但每个参与者自行选择 prompt，输入与群体混杂 |
| C | **GlobalOpinionQA** | 跨国调查问题和群体答案分布 | country | 只有聚合分布，没有个人标注，适合输出评估，不适合直接算梯度 |
| C | **CivilComments/WILDS** | 评论毒性分类 | identity mention 属性 | 适合 subpopulation shift，不是“不同人群对同一输入的目标不同” |

## 3. 首选：DICES-350

### 公开信息

- 仓库：<https://github.com/google-research-datasets/dices-dataset>
- 数据文件：`350/diverse_safety_adversarial_dialog_350.csv`
- 规模：350 个对话、123 名评审者、43,050 条评审记录；每名评审者评价全部对话。
- 群体：两种 gender、五种 race/ethnicity、三种 age group、两种 education。
- 标签：`Yes/No/Unsure` 的安全判断，以及 harmful content、bias、misinformation、
  policy 等细粒度项目。
- 许可：仓库 README 明确写为 CC BY 4.0；数据仓库在 2025-09-18 已归档，但仍可读。
- 论文：Aroyo et al., “DICES Dataset: Diversity in Conversational AI Evaluation for
  Safety”, NeurIPS Datasets and Benchmarks 2023/2024 proceedings，
  <https://arxiv.org/abs/2306.11247>。

### 为什么适合本项目

这是目前最接近理想测量基的数据：输入 conversation 完全相同，只有评审者的群体
属性和主观安全判断变化。可以选 `rater_race` 的三个群体作为 A/B/C，并把 group
从模型输入中隐藏。

建议第一版使用：

- 输入：`context + response`；
- 目标：`Q_overall` 或一个单独的 Q2/Q3/Q6 子项；
- 条件：`rater_race` 的三个群体；
- 模型：冻结小型文本编码器或随机初始化的浅层分类器；
- 损失：三分类 cross-entropy 或对 `Yes/No` 的 binary cross-entropy；
- 评估：平均 loss、每组 loss、worst-group loss、group-conditioned gradient、
  `kappa_mixture_ref`。

### 必须避免的陷阱

- 按 `item_id` 划分 train/test，不能按行随机划分；否则同一对话会同时出现在两边。
- 同时报告全部 123 名评审者和排除低质量评审者后的结果。
- 不要把专家 `safety_gold` 或聚合字段作为输入。
- 先验证三组的每 item 标注率和差异，再决定使用 race、age 还是 gender。
- DICES 评估的是安全感知，不等于真实的客观危害；论文中应使用“群体条件安全判断”。

### 最小实验

1. 统计每个 item 在三个群体中的标签分布和 Krippendorff alpha。
2. 用冻结编码器计算三个群体的条件梯度。
3. 比较共享模型、group-conditioned head 和 group-specific adapter。
4. 训练时隐藏 group，评估时分别报告三组性能。
5. 只在 group-conditioned gradient 差异显著的 item 子集上检验 κ 预测。

## 4. 第二选择：WVS / WorldValuesBench

### 公开信息

- WVS 官方数据：<https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp>
- WorldValuesBench 仓库：<https://github.com/Demon702/WorldValuesBench>
- 相关论文：<https://aclanthology.org/2024.lrec-main.1539/>
- WorldValuesBench 使用 WVS Wave 7，约 93,278 名参与者、240 个价值问题，构造超过
  2,000 万条 `(demographic attributes, value question) -> answer` 样本。
- 可按 country/continent、urban/rural、education 构造群体。
- WVS 官方数据免费，但下载需要注册并接受 non-redistribution data use license；原始
  CSV 不应提交到本仓库。

### 为什么有价值

它比 DICES 更贴近“同一个问题，不同群体给出不同价值目标”。每个问题由很多参与者
回答，适合计算每个群体的 ordinal/categorical loss 和梯度，也适合测试平均 reward
与 worst-group reward 的折衷。

### 推荐使用方式

- 先只选 3 个样本量充足的国家或 3 个大洲，而不是直接使用全部 48 个交叉人口组；
- 只使用共同问题 ID，过滤国家特有问题和缺失严重的问题；
- 按参与者 ID 划分 train/validation/test；
- 对 ordinal answer 使用 ordinal regression 或分类，不要未经说明地当作连续 reward；
- group hidden 时，输入只包含问题文本；group visible 时加入 country/region token；
- 以每题的 group-level answer distribution、平均 Wasserstein loss 和 worst-group loss
  作为行为指标。

## 5. POPQUORN

- 仓库：<https://github.com/Jiaxin-Pei/Potato-Prolific-Dataset>
- 论文：<https://arxiv.org/abs/2306.06826>
- 规模：约 45,000 条标注、1,484 名评审者；包含 offensiveness、QA、text rewriting
  和 politeness rating 四类任务。
- 同一评论或邮件由多个评审者标注，并保留评审者 demographics。
- 论文报告 Black、Asian 等群体对同一评论/邮件的判断存在系统差异。
- 优点是与 DICES 类似的重复标注结构，且任务不局限于安全。
- 风险是当前仓库 README 没有看到明确的数据 license 文件；论文的 CC BY 不自动等同于
  原始数据许可。使用前需联系作者或核对数据条款，不能直接假定可再发布。

## 6. PRISM、GlobalOpinionQA 与其他候选

### PRISM Alignment

PRISM 包含约 1,500 名参与者、75 个出生国家、8,011 个对话和 68,371 条评分，并把
参与者调查、人口属性和对话反馈连接起来。人类 prompt 为 CC BY 4.0，模型回复为
CC BY-NC 4.0 且受原模型提供方条款约束。

它最接近 LLM alignment，但参与者自行选择 opening prompt，导致 group 与 input
distribution 混杂；不适合作为第一版的“同一输入、不同群体目标”实验。更适合后续检验
群体条件 reward model 或个性化模型。

### GlobalOpinionQA

仓库：<https://huggingface.co/datasets/Anthropic/llm_global_opinions>。

它提供约 2,560 个调查问题及按国家聚合的答案分布，许可标为 CC BY-NC-SA 4.0。由于
没有个人级回答或条件标注，不能直接估计个人样本的 group-conditioned gradient，但可
作为 WVS 结果的外部分布评估集。

### CivilComments/WILDS

CivilComments 的 identity 字段描述评论中是否提到某个身份，而不是评审者属于哪个
群体。它适合研究 group shift 和 worst-group generalization，不适合直接支撑“群体
目标冲突”的论断。其数据卡标为 CC0 1.0，规模很大，可作为边界/对照数据。

## 7. 推荐决策

### 第一阶段

先用 **DICES-350**。它最小、结构最干净、每个 item 有完整评审覆盖、许可信息最明确，
而且无需下载 WVS 原始微数据。

### 第二阶段

如果 DICES 中三组梯度差异明显，再使用 **WVS/WorldValuesBench** 做规模和价值任务
复现。WVS 更可能产生真实的多条件目标差异，但数据清洗和许可管理成本更高。

### 不建议现在使用

- PRISM：先解决 prompt 与 demographic confounding；
- GlobalOpinionQA：只能做分布级外部评估；
- CivilComments：条件含义不匹配；
- SWE-smith：条件标签需要自行定义，留给独立的 coding-agent 项目。
