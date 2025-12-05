## GCNMix 技术路线（Top-K 推荐）

GCNMix 位于 `src/models/general/GCNMix.py`，在 LightGCN 结构上加入 IMix 训练策略，核心包含：
- **图构建与归一化**：用户-物品交互构成二部图邻接矩阵，经 \(D^{-1/2}AD^{-1/2}\) 归一化，可选 `--self_loop` 加自环。
- **图传播与聚合**：初始化用户/物品嵌入后，进行 `n_layers` 次稀疏矩阵乘法，收集各层输出并取平均（LightGCN 风格）作为最终表征。
- **IMix 正负混合**：训练阶段以概率 `mix_prob` 将正样本嵌入替换为 `mix_alpha*pos + (1-mix_alpha)*rand_neg`，提升对难负样本的鲁棒性。
- **预测与训练**：用户/物品表征做内积得分；与框架内的 BPR 训练和 Top-K 评测流程协同。
- **主要参数**：`emb_size`、`n_layers`、`mix_alpha`、`mix_prob`、`self_loop`。

### 典型运行命令（Grocery_and_Gourmet_Food）
```bash
python src/main.py --model_name GCNMix --dataset Grocery_and_Gourmet_Food --path ../data/ \
  --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-6 --mix_alpha 0.5 --mix_prob 0.3 \
  --gpu 0 --num_workers 0
```

