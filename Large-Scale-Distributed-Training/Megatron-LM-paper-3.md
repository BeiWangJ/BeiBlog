# Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

- paper: https://arxiv.org/pdf/2205.05198
- github: https://github.com/NVIDIA/Megatron-LM
- 本文就上篇文章的中的一个点--Activation Recomputation展开了研讨，提出了两种新技术--sequence parallelism 
& selective activation recomputation，从而在优化显存占用的同时，也提高了设备的利用率，最终提高了30%的吞吐
