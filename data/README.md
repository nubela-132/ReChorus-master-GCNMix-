# Dataset

We include the public [Amazon dataset](http://jmcauley.ucsd.edu/data/amazon/links.html) (*Grocery_and_Gourmet_Food* category, 5-core version with metadata), [MIND-Large dataset](https://msnews.github.io/), and [MovieLens-1M](https://grouplens.org/datasets/movielens/) as our built-in datasets.
The pre-processed version of Amazon dataset can be found in the `./Grocery_and_Gourmet_Food` dir, which supports Top-k recommendation tasks.
You can also download and process MIND and MovieLens datasets for CTR prediction, Top-k recommendation, and re-ranking tasks in the corresponding notebooks in `MIND_Large` and `MovieLens_1M` datasets.

Our framework can also work with other datasets easily. We describe the required file formats for each task and the format for context information below:

## Top-k Recommendation Task

**train.csv**
- Format: `user_id \t item_id \t time`
- All ids **begin from 1** (0 is reserved for NaN), and the followings are the same.
- Need to be sorted in **time-ascending order** when running sequential models.

**test.csv & dev.csv**

- Format: `user_id \t item_id \t time \t neg_items`
- The last column is the list of negative items corresponding to each ground-truth item (should not include unseen item ids beyond the `item_id` column in train/dev/test sets).
- The number of negative items need to be the same for a specific set, but it can be different between dev and test sets.
- If there is no `neg_items` column, the evaluation will be performed over all the items by default.

## CTR Prediction Task

**train.csv & test.csv & dev.csv**
- Format: `user_id \t item_id \t time \t label`
- Labels should be 0 or 1 to indicate the item is clicked or not.
- Need to be sorted in **time-ascending order** when running sequential models.

## Impression-based Ranking/Reranking Task

**train.csv & test.csv & dev.csv**
- Format: `user_id \t item_id \t time \t label \t impression_id`
- All interactions with the same impression id will be grouped as a candidate list for training and evaluations.
- If there is no `impression_id` column, interactions will grouped by `time`.
- Labels should be 0 or 1 to indicate the item is clicked or not.
- Need to be sorted in **time-ascending order** when running sequential models.


## Context Information

**item_meta.csv** (optional)

- Format: `item_id \t i_<attribute>_<format> \t ... \t r_<relation> \t ...`
- Optional, only needed for context-aware models and some of the knowledge-aware models (CFKG, SLRC+, Chorus, KDA).
- For context-aware models, an argument called `include_item_features` is used to control whether to use the item metadata or not.
- `i_<attribute>_<format>` is the attribute of an item, such as category, brand, price, and so on. The features should be numerical. The header should start with `i_` and the <format> is set to `c` for categorical features and `f` for dense (float) features.
- `r_<relation>` is the relations between items, and its value is a list of items (can be empty []). Assume `item_id` is `i`, if `j` appears in `r_<relation>`, then `(i, relation, j)` holds in the knowledge graph. Note that the corresponding header here must start with "r_" to be distinguished from attributes.

**user_meta.csv** (optional)

- Format: `user_id \t u_<attribute>_<format> \t ...`
- Optional, only needed for context-aware models, where an argument called `include_user_features` is used to control whether to use the user metadata or not.
- `u_<attribute>_<format>` is the attribute of a user, such as gender, age, and so on. The header should start with `u_` and the <format> is set to `c` for categorical features and `f` for dense (float) features.

**situation metadata** (optional)
- Situation features are appended to each line of interaction in **train.csv & test.csv & dev.csv**
- Format: `user_id \t item_id \t time \t ... \t c_<attribute>_<format> \t ...`
- Optional, only needed for context-aware models, where an argument called `include_situation_features` is used to control whether to use the sitaution metadata or not.
- `c_<attribute>_<format>` is the attribute of a situation, such as day of week, hour of day, activity type, and so on. The header should start with `c_` and the <format> is set to `c` for categorical features and `f` for dense (float) features.

↓ Examples of different data formats

![data format](../docs/_static/data_format.png)
我们内置了三个公开数据集，分别是亚马逊数据集（“食品与美食” 类别，含元数据的 5-core 版本）、MIND-Large 数据集和MovieLens-1M 数据集。亚马逊数据集的预处理版本位于./Grocery_and_Gourmet_Food目录下，支持 Top-k 推荐任务。你也可以通过MIND_Large和MovieLens_1M目录下对应的笔记本，下载并处理 MIND 和 MovieLens 数据集，用于点击率预测、Top-k 推荐和重排序任务。
我们的框架还可轻松适配其他数据集。以下将说明各任务所需的文件格式及上下文信息格式：
Top-k 推荐任务
train.csv
格式：用户ID \t 物品ID \t 时间
所有 ID 均从 1 开始（0 预留用于表示缺失值），后续所有文件遵循此规则。
运行序列模型时，需按时间升序排序。
test.csv & dev.csv
格式：用户ID \t 物品ID \t 时间 \t 负样本物品
最后一列是每个真实物品对应的负样本物品列表，不得包含训练集、验证集、测试集中物品 ID 列之外的未见过物品 ID。
同一数据集内的负样本数量需保持一致，验证集和测试集的负样本数量可不同。
若不存在 “负样本物品” 列，默认将对所有物品进行评估。
点击率预测任务
train.csv & test.csv & dev.csv
格式：用户ID \t 物品ID \t 时间 \t 标签
标签取值为 0 或 1，分别表示物品未被点击或已被点击。
运行序列模型时，需按时间升序排序。
基于曝光的排序 / 重排序任务
train.csv & test.csv & dev.csv
格式：用户ID \t 物品ID \t 时间 \t 标签 \t 曝光ID
相同曝光 ID 对应的所有交互记录将被分组为一个候选列表，用于训练和评估。
若不存在 “曝光 ID” 列，将按时间对交互记录进行分组。
标签取值为 0 或 1，分别表示物品未被点击或已被点击。
运行序列模型时，需按时间升序排序。
上下文信息
item_meta.csv（可选）
格式：物品ID \t i_<属性>_<格式> \t ... \t r_<关系> \t ...
可选文件，仅上下文感知模型和部分知识感知模型（CFKG、SLRC+、Chorus、KDA）需要。
对于上下文感知模型，可通过include_item_features参数控制是否使用物品元数据。
i_<属性>_<格式>表示物品的属性（如类别、品牌、价格等），特征需为数值型。列名需以i_开头，<格式>中c代表分类特征，f代表稠密（浮点型）特征。
r_<关系>表示物品间的关系，取值为物品列表（可为空列表 []）。假设物品 ID 为i，若j出现在r_<关系>中，则知识图谱中存在三元组(i, 关系, j)。注意此列列名必须以 “r_” 开头，以与属性区分。
user_meta.csv（可选）
格式：用户ID \t u_<属性>_<格式> \t ...
可选文件，仅上下文感知模型需要，可通过include_user_features参数控制是否使用用户元数据。
u_<属性>_<格式>表示用户的属性（如性别、年龄等）。列名需以u_开头，<格式>中c代表分类特征，f代表稠密（浮点型）特征。
场景元数据（可选）
场景特征需追加到训练集、测试集、验证集的每一行交互记录中。
格式：用户ID \t 物品ID \t 时间 \t ... \t c_<属性>_<格式> \t ...
可选文件，仅上下文感知模型需要，可通过include_situation_features参数控制是否使用场景元数据。
c_<属性>_<格式>表示场景的属性（如星期几、一天中的时段、活动类型等）。列名需以c_开头，<格式>中c代表分类特征，f代表稠密（浮点型）特征。