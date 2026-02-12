---
title: "[Paper Survey] A Survey on Distributed Inference of Vision-Language Models"
date: 2026-2-12 15:00:00 +0900
categories: [Papers, Efficient VLM]
tags: [ai, VLM]
math: true
---

<!-- 
## Introduction
### Abstract & Motivation
[Efficient VLA survey 논문](https://arxiv.org/abs/2510.24795)
![](/assets/img/posts/2026-1-6-OpenVLA/rt2 arch.png){: width="800"} 
-->

기존에 조사하던 Efficent VLA는 그 평가 방식이 연구실에서 실험하기에 적절하지 않았다. 특히 여러 VLA survey, efficient VLA와 관련된 논문들을 확인해 본 결과 robot arm을 사용하는 연구의 경우 실물 robot arm을 활용하는 것이 적절해 보였다. 32개 정도의 주요 논문 중 5개의 논문에서는 simulation만을 사용해 평가하지만, 나머지 논문들에서는 실물 robot arm만을 활용해 평가하거나 실물 robot arm과 simulation을 모두 사용해 평가했다. Robot arm에 대한 VLA뿐만 아니라, 자율주행 관련 논문들에 대해서도 조사했는데, simulation을 사용하는 경우 이를 위한 별도의 VRAM과 추가적인 세팅 과정이 필요하다는 문제점이 있었다.

이에 따라 VLA보다는 VLM에 대한 최적화를 알아보기로 했고, 교수님께서 추천해주신 대로 distributed VLM에 대한 선행 연구들을 조사했다.

## Distributed VLMs

### VLM이란?

VLM(Vision-Language Model) 또는 vision task에 대한 MLLM(Multimodal Large Language Model)은 LLM에 vision processing을 결합한 모델로, 다음과 같이 vision encoder, vision-language projector, LLM으로 구성된다.

![](/assets/img/posts/2026-2-12-distributed-vlm/MLLM arch.png){: width="800"}

### Distirbuted VLM이란?

### 참고한 논문들

## Paper 1

## Paper 2

## Paper 3

## 결론

