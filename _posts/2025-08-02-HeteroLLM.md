---
title: "HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators"
date: 2025-08-02 22:26:00 +0900
categories: [Papers, Inference Acceleration]
tags: [cs, ai, on-device ai]
---
해당 논문은 2025년 1월에 arxiv에 올라온 논문으로, 교수님께서 공유해주신 [Awesome-On-Device-AI-Systems](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems/blob/main/README.md)에 소개되어 있어서 읽어보게 되었다. 이 글에서는 단순히 해당 논문의 내용과 주장을 정리하기 때문에, 해당 내용이 사실인지는 별도의 검증과 조사가 필요하다.

## Abstract

Privacy와 response latency 등의 측면을 개선하기 위해, 현재 ai를 mobile system 등에서 on-device로 돌리는 시도가 이루어지고 있다. 현재의 mobile SoC(System on Chip)에서는 이에 따른 computational demand를 만족시키기 위해, GPU, NPU 등 다양한 ai accelerator를 포함한다. 하지만 현존하는 design들은 단일 ai accelerator에 대한 것으로, computation 및 memory bandwidth 측면에서 이런 heterogeneous processor들을 잘 고려해서 최적화하지는 못한다.

<!-- 실제로 NPU가 최신 SoC에 많이 들어가고 있는 추세인가? 빠지고 있는 추세는 아닌가? 현존하는 design들이 이걸 잘 못한다는 것이 사실인가? -->

이에 따라 논문에서는 우선 mobile SoC에 대해 성능적 특징을 살펴본다. 이후 저자는 해당 관찰을 통해 **1. prefill phase와 decoding phase 각각에서의 요구사항에 따른 partition strategy**와, **2. mobile SoC에서의 빠른 synchronization 기법을 활용**하는 inference engine인 **HeteroLLM**을 제시한다. HeteroLLM은 layer-level과 tensor-level 모두에 대해서 heterogeneous execution을 지원한다고 한다.

## Introduction

### 해당 연구의 필요성

앞에서 언급한 것처럼 현재 ai를 스마트폰과 같은 mobile system에서 돌리려는 시도가 이루어지고 있고, 이에 따라 SoC 제조사들은 GPU, NPU와 같이 matrix/vecotr multiplication에서 이점을 가지는 다양한 ai accelerator들을 칩 안에 통합시켜왔다. 하지만 heterogeneous processor들을 활용하는 inference engine에 대한 선행 연구들에서는 아래와 같은 이유로 현재의 mobile platform에 적합하지 않다.

- GPU들과 NPU들을 위한 기존의 synchronization 기법은 LLM inference에 대한 overhead가 크다. 특히 각 kernel의 실행 시간이 수백 ms인 decoding phase에서 overhead가 커진다.

<!-- 이런 processor들은 SoC 안에서 하나의 physical memory를 동시에 활용하게 되기도 한다. -->

- NPU는 GPU보다 상당히 높은 성능을 보인다. 예륻 들어, Qualcomm 8 Gen 3에서 GPU는 실제로 2.8 TFLOPS를 보이는데, NPU는 10 TFLOPS를 보인다. 이에 따라 NPU와 GPU를 기존의 방식대로 병렬적으로 활용하는 경우 성능 하락이 발생할 수 있다.

<!-- 왜인지 모르겠다. 기존의 방식이 어떤 것인지를 알아야 이해할 수 있을 것 같다. [19. 20. 25]가 해당 논문이라고 한다. -->

- 일부 연구에서는 CPU와 NPU에 대해 mixed-precision을 적용해 sparsity를 활용하는데, 이는 activation/weight에 sparsity가 실제로 존재해야 유의미하다. 반면 최근 연구에서는 LLM이 dense하다는(sparsity가 적다는) 결과를 보인다.

이에 따라 heterogeneous processor들을 고려한 efficient inference engine은 여전히 중요한 과제로 남아있다.

### Mobile SoC의 특징

Mobile SoC에 대한 분석 결과, 그 하드웨어 architecture적인 특징으로는 아래와 같은 것들이 있다.

- NPU 성능은 tensor-sensitive하다.  
최적의 조건인 경우에 NPU는 tensor 연산에 대해 GPU보다 높은 성능을 보인다. 하지만 tensor의 order, size, shape 등이 NPU의 하드웨어 architecture에 적합하지 않다면 그 성능은 GPU 수준으로 떨어진다.

<!-- tensor의 order란? NPU의 weight-stall이란? -->

- 생성 비용이 높은 static NPU graph  
현재의 mobile NPU는 static computation graph만을 지원하는데, 이는 LLM의 dynamic한 workload와 호환되지 않는다. 또한 NPU의 architecture적인 특성에 따라 최적의 graph를 계산하는 것이 GPU에 비해 더 복잡하고, 런타임에 이를 계산하는 것은 overhead가 크다.

<!-- static computation graph란? 왜 NPU의 graph를 계산하는 것이 GPU에 비해 더 복잡한 것일까? -->

- 단일 processor의 memory bandwidth restriction  
단일 processor만 활용하는 경우 SoC의 memory bandwidth를 만족시키기에는 충분하지 않다.

<!-- SoC의 memory bandwidth란.. 그러면 이건 그냥 시스템의 성능이 충분하지 않다는 말인 거 같다. -->


## Background & Related Work

## Performance Characteristic

## Deisgn

## Evaluation

## Discussion

## Conclusion
