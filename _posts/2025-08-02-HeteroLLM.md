---
title: "HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators"
date: 2025-08-02 22:26:00 +0900
categories: [Papers, Inference Acceleration]
tags: [cs, ai, on-device ai]
---

해당 논문은 2025년 1월에 arxiv에 올라온 논문으로, 교수님께서 공유해주신 [Awesome-On-Device-AI-Systems](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems/blob/main/README.md)에 소개되어 있어서 읽어보게 되었다. 이 글에서는 단순히 해당 논문의 내용과 주장을 정리하기 때문에, 해당 내용이 사실인지는 별도의 검증과 조사가 필요하다.

## Abstract
Privacy와 response latency 등의 측면을 개선하기 위해, 현재 ai를 mobile system 등에서 on-device로 돌리는 시도가 이루어지고 있다. 현재의 mobile SoC(System on Chip)에서는 이에 따른 computational demand를 만족시키기 위해, GPU, NPU 등 다양한 ai accelerator를 포함하고 있다. 하지만 현존하는 design들은 단일 ai accelerator에 대한 것으로, computation 및 memory bandwidth 측면에서 이런 heterogeneous processor들을 잘 고려해서 최적화하지는 못한다.

<!-- 실제로 NPU가 최신 SoC에 많이 들어가고 있는 추세인가? 빠지고 있는 추세는 아닌가? 현존하는 design들이 이걸 잘 못한다는 것이 사실인가? -->

이에 따라 논문에서는 우선 mobile SoC에 대해 성능적 특징을 살펴본다. 이후 저자는 해당 관찰을 통해 1. prefill phase와 decoding phase 각각에서의 요구사항에 따른 partition strategy와, 2. mobile SoC에서의 빠른 synchronization 기법을 활용하는 inference engine인 HeteroLLM을 제시한다. HeteroLLM은 layer-level과 tensor-level 모두에 대해서 heterogeneous execution을 지원한다고 한다.

## Introduction


## Background & Related Work


## Performance Characteristic


## Deisgn


## Evaluation


## Discussion


## Conclusion


