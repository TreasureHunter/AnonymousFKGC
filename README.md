# ℹ CARE: Context-Augmented Relation Generation for Few-Shot Knowledge Graph Completion

> Code Coming soon! ⏰

🛠 In this paper, we summarize two main challenges in the Few-shot Knowledge Graph Completion (FKGC) task:  *(1) Deep Context Aggregation. (2) Key Context Identification .*

🔬 To address these challenges, we employ **Key Path-aware GNN (KPGNN)** to aggregate multi-hop neighbors information to extract abundant structural knowledge for relation generation. Besides, we implement **Context Augmentation (CA)** to refine the few-shot relation representations by emphasizing critical context-level information.

🖼️ The figures below illustrates the challenges and our proposed framework.

![challenges](./fig/challenges.png)

![framework](./fig/framework.png)

## 📚 Environment

```
- python 3.9
- Ubuntu 22.04
- RTX4090/A100
- Memory 32G/80G
```

## 💡 Requirements

```
pip install -r requirements.txt
```

## 📑 Data Preparation

Download [Nell-One and Wiki-One dataset and the pre-trained embeddings.](https://github.com/xwhan/One-shot-Relational-Learning)

Download [Fb15k-237 dataset and the pre-trained embeddings.](https://github.com/SongW-SW/REFORM)

Put the pre-trained embedding files in the `emb` subfolder of each dataset directory, e.g., `Nell-One/emb/`.
