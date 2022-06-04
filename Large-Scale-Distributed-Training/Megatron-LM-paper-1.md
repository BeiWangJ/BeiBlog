# Megatron-LM: Training Multi-Billion Parameter Language Models Using

## Abstract

- 主要工作内容
  - In this work, we present our techniques for training very large transformer models and implement a simple, efficient intra-layer model parallel approach that enables training transformer models with billions of parameters.
  - 在这项工作中，我们提出了我们的训练大型transformer模型的技术，并实现了一个简单、有效的层内模型并行方法，使训练具有数十亿参数的transformer模型。

- 方法便捷性亮点
  - Our approach does not require a new compiler or library changes, is orthogonal and complimentary to pipeline model parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch.
  - 我们的方法不需要新的编译器或库的更改，它是与pipeline模型并行性正交和互补的，并且可以通过在本机PyTorch中插入一些通信操作来完全实现。

- 在模型参数扩展上的改进
  - We illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the entire application with 76% scaling efficiency when compared to a strong single GPU baseline that sustains 39 TeraFLOPs, which is 30% of peak FLOPs.
  - 我们通过使用512个gpu收敛基于transformer的高达83亿个参数的模型来说明这种方法。我们在整个应用程序中维持了15.1个PetaFLOPs，具有76%的扩展效率，而它维持了39个TeraFLOPs，占峰值FLOPs的30%。

- 精度效果
  - o demonstrate that large language models can further advance the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9 billion parameter model similar to BERT.We show that careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy of 89.4%).
  - 为了证明大型语言模型可以进一步推进最先进的技术(SOTA)，我们训练了一个类似于GPT-2的83亿个参数转换器语言模型和一个类似于BERT的39亿个参数模型。我们表明，在类似bert的模型中，仔细注意层标准化的放置对于提高随着模型大小增长的性能至关重要。使用GPT-2模型，我们在维基文本103(10.8SOTA困惑为15.8)和SOTA(66.5%SOTA准确性为63.2%)数据集上获得了SOTA结果。我们的BERT模型在RACE数据集上获得了SOTA的结果(90.9%，而SOTA的准确率为89.4%)。
```

## Introduction

- 展示扩展效果，虚线是
![](./imgs/p1-f1.jpg)















