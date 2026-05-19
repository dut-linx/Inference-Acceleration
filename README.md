# Awesome Dynamic Inference Acceleration for AIGC, LLM/MLLM, and VLA

![Survey](https://img.shields.io/badge/Survey-Dynamic%20Inference-blue)
![Topics](https://img.shields.io/badge/Topics-AIGC%20%7C%20LLM%2FMLLM%20%7C%20VLA-green)
![Status](https://img.shields.io/badge/Status-Curated%20Reading%20List-orange)

本项目整理 2025--2026 年 AIGC 视频生成、动态大模型、MLLM/Video-LLM 以及 Vision-Language-Action Model 推理加速方向的代表性论文。核心关注点包括缓存复用、动态 token 剪枝、视觉 token early-exit、稀疏注意力、扩散蒸馏、动作 tokenization、action chunking、VLA cache 与端侧部署。

> Note: 部分 2026 年论文仍处于 arXiv / OpenReview 状态，正式录用信息请以会议官网或作者主页为准。

## Contents

- [1. Overview](#1-overview)
- [2. Taxonomy](#2-taxonomy)
- [3. Paper List](#3-paper-list)
- [4. Research Gaps](#4-research-gaps)
- [5. Suggested Research Direction](#5-suggested-research-direction)
- [6. Suggested Repository Structure](#6-suggested-repository-structure)

## 1. Overview

现有推理加速工作大多围绕一个共同假设展开：大模型推理过程中存在大量可复用、可剪枝或可提前退出的冗余计算。对于视频扩散模型，冗余主要来自相邻 denoising timestep、相邻 DiT block 或相邻 layer feature 的高相似性。对于 LLM/MLLM，冗余主要来自长上下文 token、视觉 token 和视频 token 的不均匀重要性。对于 VLA，冗余则来自连续机器人观测帧、动作序列结构以及控制任务中不同 action step 的重要性差异。

这批论文已经显著推动了推理效率提升，但尚未同时解决高速、低损、通用、硬件友好和真实部署稳定性的问题。尤其在 AIGC 与 VLA 场景中，速度提升不能只看 FLOPs 或 token 数，还必须考虑生成质量、时序一致性、动作稳定性、真实 latency 和任务成功率。

## 2. Taxonomy

| Category | Representative Papers | Core Idea | Suitable Research Entry Point |
| --- | --- | --- | --- |
| Cache Reuse | TeaCache, AdaCache, EasyCache, BWCache, SmoothCache, VLA-Cache, EfficientVLA | 相邻 timestep、layer、block 或机器人帧存在冗余，可复用中间结果。 | Runtime / plug-in style acceleration. |
| Dynamic Token Pruning | SDTP, SlimInfer, ATP-LLaVA, DyCoke, DTP | 不同 token 的贡献不同，可按输入、层或时间动态保留关键 token。 | Difficulty-aware pruning with quality constraints. |
| Early Exit / Layer Skipping | DyVTE, DySL-VLA, Early-exit LLM, Runaway is Ashamed | 不同样本、不同视觉 token 或不同动作步骤不一定需要完整计算。 | Tail latency, batch throughput, and adaptive depth. |
| Sparse Attention | PASA | 视频 DiT 的 attention 计算量大，但稀疏化需要避免闪烁和时序不稳定。 | Structured acceleration for video generation. |
| Few-step Distillation / Distribution Matching | RMD, DisCa | 减少 diffusion sampling steps，或让缓存策略与蒸馏兼容。 | Large speedup with higher training cost. |
| Action Representation Optimization | FAST, OpenVLA-OFT, SmolVLA, Knowledge Insulation, Stable-FAST | 将高频连续动作转化为更短、更稳定、更易生成的 action token、action chunk 或 continuous action head。 | Low-latency VLA deployment. |

## 3. Paper List

### 3.1 AIGC / Video Diffusion Acceleration

| Technique | Paper | Year / Status | Main Contribution | Why It Matters |
| --- | --- | --- | --- | --- |
| Timestep-aware cache | TeaCache: Timestep Embedding Tells: It’s Time to Cache for Video Diffusion Model | CVPR 2025 | 使用 timestep embedding 估计相邻去噪步的输出差异，并决定何时复用缓存。 | 入门必读，适合理解视频扩散模型为什么可以按 timestep 缓存。 |
| Adaptive cache | AdaCache: Adaptive Caching for Faster Video Generation with Diffusion Transformers | ICCV 2025 | 根据不同视频的生成难度自适应设计 cache schedule，而不是使用固定缓存间隔。 | 适合写“样本难度感知推理加速”。 |
| Runtime-adaptive cache | EasyCache: Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching | 2025 arXiv | 运行时动态复用 transformation vectors，不依赖离线 profiling 或大量参数搜索。 | 工程落地价值较高，适合作为插件式加速参考。 |
| Block-wise cache | BWCache: Accelerating Video Diffusion Transformers through Block-Wise Caching | 2025 arXiv / OpenReview | 发现 DiT block 特征变化呈 U 型，提出 block-level 动态缓存。 | 适合关注“层级缓存”和“模块级动态计算”。 |
| Layer-wise feature cache | SmoothCache: A Universal Inference Acceleration Technique for Diffusion Transformers | 2024 arXiv | 根据 layer-wise representation error 设计静态缓存策略。 | 可作为 TeaCache / BWCache 类方法的前置参考。 |
| Diffusion distillation | RMD: Cross-Resolution Distribution Matching for Diffusion Distillation | 2026 arXiv | 用 cross-resolution distribution matching 缓解多分辨率少步蒸馏中的分布差异。 | 适合写“few-step distillation + multi-resolution inference”。 |
| Sparse attention | PASA: Ride the Wave: Precision-Allocated Sparse Attention for Smooth Video Generation | 2026 arXiv | 根据生成轨迹动态分配 attention 预算，并缓解 video flickering。 | 说明视频加速不能只快，还要保持时间平滑。 |
| Learnable cache + distillation | DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching | 2026 arXiv / CVPR 2026 status to verify | 使用轻量 predictor 学习高维 feature 演化，使 caching 与 distillation 更兼容。 | 适合研究 training-free cache 之后的 learnable cache。 |

### 3.2 Dynamic LLM / MLLM / Video-LLM Acceleration

| Technique | Paper | Year / Status | Main Contribution | Why It Matters |
| --- | --- | --- | --- | --- |
| LLM token pruning | SDTP: Saliency-driven Dynamic Token Pruning for Large Language Models | 2025 arXiv | 逐层估计 token 重要性并动态剪枝，减少长序列推理计算。 | LLM 动态 token pruning 的代表性工作。 |
| Long-context pruning | SlimInfer: Accelerating Long-Context LLM Inference via Dynamic Token Pruning | AAAI 2026 | 基于 information diffusion 现象动态剪掉非关键 prompt token，并管理 KV cache。 | 适合长上下文推理与 KV cache 优化方向。 |
| Visual-token early exit | DyVTE: Accelerating Multimodal Large Language Models via Dynamic Visual-Token Exit | NeurIPS 2025 | 当文本 token 已获得足够视觉信息后，让视觉 token 动态退出后续计算。 | 可衔接 MLLM 与 VLA 中的视觉 token 冗余问题。 |
| Adaptive visual token pruning | ATP-LLaVA: Adaptive Token Pruning for Large Vision Language Models | CVPR 2025 | 提出 instance-wise、layer-wise adaptive visual token pruning。 | 多模态视觉 token 剪枝代表方法。 |
| Problem analysis | Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem? | ACL Findings 2025 | 系统反思 attention score、语言信息、随机选择、评估协议等问题。 | 适合放在综述的 limitation / reflection 部分。 |
| Video-LLM token compression | DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models | CVPR 2025 | 在 decoding 阶段动态压缩 temporal 与 spatial redundancy。 | 适合连接视频理解大模型与动态 token 压缩。 |
| Early-exit algorithm | Accelerating Large Language Model Inference via Early-Exiting Algorithms | 2025 arXiv | 分析 early-exit 在 batch inference 中的系统瓶颈。 | 说明“算法少算”不一定等于“系统更快”。 |
| Embodied agent early exit | Runaway is Ashamed, But Helpful: On the Early-Exit Behavior of LLM-based Agents in Embodied Environments | EMNLP Findings 2025 | 在具身环境中减少冗余动作，并评估 progress degradation。 | 适合任务级 early-exit 与 embodied agent 加速。 |

### 3.3 VLA / Vision-Language-Action Acceleration

| Technique | Paper | Year / Status | Main Contribution | Why It Matters |
| --- | --- | --- | --- | --- |
| Action tokenization | FAST: Efficient Action Tokenization for Vision-Language-Action Models | 2025 arXiv | 使用 DCT 将连续动作序列压缩为 frequency-space action tokens。 | VLA 动作 token 化必读。 |
| Action chunking / parallel decoding | OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success | 2025 arXiv | 引入 parallel decoding、action chunking、continuous action representation 和 L1 regression。 | 直接对应 VLA 的速度与成功率协同优化。 |
| Adaptive token caching | VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation | 2025 arXiv / OpenReview | 识别连续帧中变化小的视觉 token，并通过 KV-cache 复用计算结果。 | VLA cache acceleration 的直接参考。 |
| Compact VLA / async inference | SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics | 2025 arXiv | 设计小型 VLA，并使用异步推理栈解耦感知、动作预测与动作执行。 | 适合端侧、低成本机器人部署。 |
| Knowledge insulation | Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better | 2025 arXiv | 研究 action expert 与 VLM backbone 的耦合，减少连续动作专家对语义知识的破坏。 | 适合理解动作头高效化与语义知识保持。 |
| Visual token pruning | DTP: A Simple yet Effective Distracting Token Pruning Framework for Vision-Language Action Models | 2026 arXiv / OpenReview | 动态剪掉与任务无关的 distracting image tokens。 | 适合研究动作相关的视觉 token pruning。 |
| Dynamic layer skipping | DySL-VLA: Efficient VLA Inference via Dynamic-Static Layer-Skipping | 2026 arXiv / DAC 2026 page | 区分 informative layers 与 incremental layers，并根据 action importance 动态跳层。 | 适合研究 VLA dynamic depth。 |
| Structured training-free compression | EfficientVLA: Training-Free Acceleration and Compression for VLA Models | 2025 arXiv / OpenReview | 联合 language layer pruning、visual token selection 与 diffusion action head caching。 | 适合做 VLA full-pipeline acceleration 代表。 |
| Inference stabilization | Stable-FAST: Stabilizing Inference of Autoregressive VLA Models | 2026 OpenReview, status to verify | 改进 action tokenization 的推理稳定性，降低动作不稳定和执行时间。 | 说明 VLA 加速还必须关注动作平滑与控制稳定性。 |

## 4. Research Gaps

### 4.1 Real-time inference is still not fully solved

EasyCache 虽然报告 2.1--3.3x speedup，但距离真正实时视频生成仍有明显差距。VLA-Cache 的加速也主要体现为约 1.7x CUDA latency reduction 与一定程度的控制频率提升，说明机器人控制仍需要更强的系统级优化。

### 4.2 Speed and quality remain in conflict

缓存、剪枝、稀疏注意力和蒸馏都会引入误差累积。DisCa 指出，当 feature caching 压缩更强时，语义和细节可能下降；视频 step distillation 也容易带来时序退化。因此，未来加速方法不能只最大化压缩率，还需要显式建模质量损失与误差传播。

### 4.3 Dynamic scoring is not sufficiently reliable

MLLM token pruning 方向已经开始质疑 attention-based scoring 是否可靠、语言信息是否真正有帮助、随机选择为什么有时表现很强，以及评估协议是否公平。这说明动态剪枝不是简单设置阈值，而是需要可信的 token importance estimation。

### 4.4 Removed or skipped information is hard to recover

ATP-LLaVA 中被剪掉的视觉 token 后续不可恢复。DyVTE 也指出 full visual-token exit 不一定适合所有任务，因为后层可能仍需要部分视觉信息。未来值得研究 recoverable pruning、soft caching、token recall 或 uncertainty-triggered recomputation。

### 4.5 FLOPs reduction does not always mean latency reduction

很多工作报告 FLOPs、token 数或理论 speedup，但真实系统中的 batch synchronization、KV cache 管理、FlashAttention 兼容性、GPU 并行效率会显著影响最终 latency。动态 early-exit 在 batched inference 中甚至可能形成新的系统瓶颈。

### 4.6 VLA acceleration cannot ignore control stability

VLA 的速度问题不能脱离机器人控制稳定性。OpenVLA-OFT 暴露了多模态 demonstration、pretraining 适用性和语言 grounding 稳定性等问题。Stable-FAST 进一步说明，自回归 VLA 的 action instability 本身就是影响真实部署的关键因素。

## 5. Suggested Research Direction

一个更有价值的研究切入点不是简单写“提高推理速度”，而是聚焦于：

> Reliable Dynamic Inference Acceleration for AIGC and VLA: introducing recoverable mechanisms, task-sensitive error control, and hardware-friendly scheduling into caching, pruning, layer skipping, and action generation.

可以进一步拆成三个方向：

1. **Recoverable Dynamic Computation**: 在 token pruning、visual-token exit、layer skipping 中加入可恢复机制，避免早期误判造成不可逆信息损失。
2. **Task-sensitive Error Control**: 不仅约束图像质量或语言指标，还引入动作稳定性、检测/分割性能、机器人成功率等任务反馈。
3. **Hardware-friendly Scheduling**: 将动态策略与 KV cache、batching、GPU parallelism、FlashAttention 和端侧部署机制联合设计。

## 6. Suggested Repository Structure

```text
.
├── README.md
├── papers/
│   ├── aigc-video-diffusion.md
│   ├── dynamic-llm-mllm.md
│   └── vla-acceleration.md
├── notes/
│   ├── research-gaps.md
│   ├── taxonomy.md
│   └── reading-roadmap.md
├── figures/
│   └── taxonomy.png
└── assets/
    └── paper_links.bib
```

## Reading Roadmap

建议按照下面顺序阅读：

1. **缓存复用入门**: SmoothCache → TeaCache → AdaCache → EasyCache → BWCache。
2. **动态 token 方法**: SDTP → ATP-LLaVA → DyVTE → DyCoke → Token Pruning in MLLMs。
3. **VLA 加速主线**: FAST → OpenVLA-OFT → VLA-Cache → EfficientVLA → Stable-FAST。
4. **研究空白总结**: 从“实时性、质量损失、动态判断、可恢复性、硬件友好、控制稳定性”六个角度组织调研报告。

## Citation

If this repository is useful for your survey or project, please consider citing the original papers listed above. This README is a curated research note and should be updated with official paper links, code links, and venue information when available.
