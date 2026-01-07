---
title: "[Paper Review] OpenVLA: An Open-Source Vision-Language-Action Model"
date: 2026-1-6 16:00:00 +0900
categories: [Papers, Efficient VLA]
tags: [ai, VLA]
math: true
---

지난 번 읽어본 [Efficient VLA survey 논문](https://arxiv.org/abs/2510.24795)에 이어, 이번에는 해당 survey에서 소개하는 여러 논문 중 거의 가장 인용수가 높으면서 그 이름대로 코드와 모델이 오픈소스인 [**OpenVLA**](https://arxiv.org/abs/2406.09246)라는 논문을 읽어봤다. 이 논문은 2024년 9월 초에 arxiv에 올라온 논문으로, 26.1.6. 기준 인용수가 1433회이다.

## Introduction

### Abstract & Motivation

기존 robotics 관련 모델들의 주요 한계는 LLM 등에 비교했을 때 데이터셋 자체가 너무 작아서 generalization, robustness가 부족하다는 것이다. 하지만 internet-scale dataset으로 pretrain된 vision-language foundation model들은 높은 generalization 능력을 가지고 있다. 이에 따라 **pretrained vision-language foundation model들을 core block으로 사용하는 시도**가 이루어지고 있다.

하지만 앞선 연구에서 제안하는 최신 모델들은 **1) 오픈소스가 아니고, 2) 새로운 환경/하드웨어에 deploy하는데에 있어 best practice가 아니라는 한계**가 있다. 이런 배경에서 저자는 generalization 능력을 갖춘 VLA가 기존의 오픈소스 language model들과 같이, 오픈소스여야 하고 효율적인 fine-tuning을 지원해야 한다고 주장한다.

이에 따라 이 논문에서 제안하는 VLA인 [**OpenVLA**](https://openvla.github.io/)(arxiv 2024, 1435회 인용)는 다음과 같은 특징을 가지고, 기존의 SOTA였던 RT-2-X, pretrained model인 Octo를 outperform했다.

1. **pretrained vision-language foundation model을 backbone으로 하고, Open X-Embodiment dataset에서 fine-tuning해 generalization 능력을 갖췄다.**
2. **오픈소스로 배포된 VLA 모델이다.**
3. **LoRA와 quantization을 활용한 효율적인 fine-tuning 및 inference를 지원한다.**

### VLM and VLA

**Vision-Language Model(VLM)**의 주요 architecture는 pretrained vision encoder와 pretrained language model을 사용해 모델을 구성하는 형태이다. 이때 vision feature를 tokenize하여 language model의 space로 projection하는 식으로 두 모델을 연결해 사용한다.

robototics에서는 로봇의 조작을 위해 이런 VLM을 적용하려는 시도가 계속 이루어지고 있었는데, 최근 많은 연구는 pretrained VLM을 robot action을 예측하도록 fine-tuning하는 방법을 사용한다. 본 논문에서는 이런 구조를 가지는 VLM을 **Vison-Language-Action Model(VLA)**이라고 한다. 즉, VLA는 VLM의 추론 능력과 학습 방식을 활용해 로봇 조작을 수행한다.

### Baselines

최근 robotics의 트렌드는 multi-task에 대한 처리가 가능한 generalist를 만드는 것이다. OpenVLA 논문에서 언급하는 baseline 모델은 Octo와 RT-2-X가 있다.

- [**Octo**](https://arxiv.org/abs/2405.12213)(arxiv 2024, 831회 인용)는 93M 크기의 오픈소스 VLA이다. 그 아키텍처는 다음 그림과 같다. pretrained language tokenizer로 T5-base를 사용했고, image tokenizer로 CNN을, backbone 모델로는 ViT와 동일한 크기의 transformer를, action decoder로는 diffusion process를 수행하는 action head를 사용했다. 학습 시에는 language tokenizer는 freeze하고 나머지 부분은 pretrain했고, dataset으로는 OXE dataset 중 일부를 선별해 활용했다. 이때 readout token은 action head의 입력으로 사용되는 learnable token으로, BERT의 CLS token과 같은 역할을 가진다. 또한 새로운 로봇에 대한 fine-tuning 시에는 추가적인 observation과 readout/action head를 추가할 수 있다.

    Octo는 다른 기기로의 적용과 fine-tuning을 고려하고 있지만, 추가적인 component를 pretrain하여 사용한다. 반면 OpenVLA는 pretrained VLM을 사용하여 더 단순하게 더 좋은 성능을 달성했다.

![](/assets/img/posts/2026-1-6-OpenVLA/octo arch.png){: width="800"}

- [**RT-2-X**](https://robotics-transformer2.github.io/)(PMLR 2023, 2390회 인용)는 55B 크기(더 작은 버전도 존재)의 오픈소스가 아닌(closed) VLA로, OpenVLA 이전의 SOTA이다. 그 아키텍처는 다음 그림과 같다. RT-2-X는 VLM인 PaLI-X와 PaLM-E을 backbone으로 사용했고, OpenVLA와 유사하게 기존 token들 중 256개를 action token으로 활용했다. 학습 시에는 RT-1-X에서 사용했던 robot data와 internet-scale dataset인 WebLI dataset을 함께 사용했다.

    RT-2-X는 하나의 로봇이나 simulation에서의 학습과 평가에만 집중하고, 새로운 로봇에 대한 효율적인 fine-tuning을 지원하지 않는다. 또한 오픈소스가 아니다. 반면 OpenVLA는 성능이 더 좋고, 효율적인 fine-tuning을 지원하면서, 오픈소스이다.

![](/assets/img/posts/2026-1-6-OpenVLA/rt2 arch.png){: width="800"}

## OpenVLA

### Architecture

**OpenVLA**는 다음 그림과 같이 pretrained VLM인 Prismatic-7B VLM을 backbone으로 사용하는 architecture를 가진다. 또한 RT-2-X처럼 기존 token들 중 256개를 action token으로 해, 다음 토큰을 autoregressive하게 예측하도록 Prismatic-7B VLM을 fine-tuning한다. 즉, discrete output language token과 continuous robot action을 mapping한 것으로 볼 수 있다.

이때 robot action을 256개의 bin(구간)으로 나눠 활용한다. bin width는 백분위수를 사용해 제1백분위수와 제99백분위수 사이를 uniform하게 나누는 값으로 지정한다. (RT-2와 달리 outlier를 제외한 것.)

N차원의 robot action은 N개의 0~255사이 값으로 구성되는 action token을 사용하게 된다. Llama tokenizer는 100개의 speical token만을 가지므로, least used token들을 overwrite해 사용한다.


![](/assets/img/posts/2026-1-6-OpenVLA/OpenVLA arch.png){: width="700"}

[**Prismatic VLMs paper**](https://openreview.net/forum?id=6FXtu8clyp)(ICML 2024, 236회 인용)에서는 VLM이 가지는 여러 가지 design decision들이 존재하지만 최적의 design이 under-explore된 것에 착안하여, VLM을 평가하는 unified framework를 정의하고 여러 model checkpoint를 제공한다. 또한 VLM design에 대한 평가 결과를 바탕으로 다음 그림과 같이 **Prismatic-7B VLM**을 제안한다. Prismatic-7B VLM는 OpenVLA architecture 그림에 나와있는 것처럼, vision encoder로 SigLIP와 DinoV2를, language model로 Llama2를 사용하며, MLP projector로 vision feature를 language embedding space로 projection하는 architecture를 가진다.

참고로 visual encoder는 SigLIP와 DinoV2를 함께 사용하는 것이 spatial reasoning에 더 유리하다고 한다.

각각 넣은 뒤 두 embedding을 concat한다.

MLP projector는 그냥 vision embedding 크기를 language embedding 크기로 바꾸는..

![](/assets/img/posts/2026-1-6-OpenVLA/prism result.png){: width="400"}





### Training

해당 component로 구성한 뒤 fine-tuning한 것이라고 한다.

OpenVLA는 next-token prediction을 수행하도록 학습된다.

OpenVLA training dataset으로는 Open X-Embodiment dataset을 변형해 활용했다. 변형된 dataset은 1) input&output space가 일관적이어야 하고 2) 다양한 ~들이 최종 training mixture에 포함되어 있어야 한다.

카메라에 적어도 하나의 3rd person이 있으면서, single arm end-effector control만 포함하고, Octo의 방식을 사용해 data mixture를 구성했다.

A100 GPU 64개로 14일간 학습을 돌렸고, 추론에는 VRAM 15GB가 필요하다. 물론 추후 소개할 quantization을 적용해 성능 저하 없이 더 줄일 수 있다.

### Design Decisions

최종 training을 수행하기 전에, BridgeData V2 데이터셋에서 여러 design decision을 수행했다.

VLM backbone으로는 LLaVA, IDEFICS-1도 사용해봤는데, IDEFICS-1 < LLaVA < Prismatic VLM 이었다. 특히 multi-object일 때 그랬는데, 아마 spatial reasoning 능력 때문일 것이라고 한다. 또한 codebase가 쓰기도 더 쉽다.

여러 VLM bench에서 image resolution을 높이면 성능이 개선되지만, OpenVLA에 대해 실험했을 때는 성능 차이가 없어서, 더 적은 resolution을 사용했다. resolution이 높아질수록 vision token 증가에 따른 context length 증가로 computation이 늘어나므로 적은 걸 선택했다.

VLM 관련 기존 연구에선 vision encoder를 freeze하는 게 기존 지식이 보존되어 더 높은 성능을 보였지만, OpenVLA 학습 시에는 함께 fine-tuning하는 것이 성능이 더 좋았다. vision encoder 자체가 robotic control 자체에 대한 지식을 가지고 있지 않았어서 그런 것으로 보인다.

LLM/VLM 학습 시의 전형적인 fine-tuning epoch은 1~2 정도이지만, OpenVLA 학습 시에는 epoch을 늘려도 성능이 계속 올라가서 27까지 했다.


leaning rate는 실험적으로 [44]에서의 학습에서와 동일한 2e-5가 최적이었다고 한다.

## Experiments

실험의 목표는 1) 기존 방식과 비교해서 OpenVLA가 multi-robot과 다양한 task에 대해서 성능이 좋은지, 그리고 2) fine-tuning해 새로운 robot task로서 사용되기에 좋은지, 마지막으로 3) PEFT나 quantization을 적용해 더 accessible하게 할 수 있는지를 보는 것이다.

### Out-of-the-Box

우선 out-of-the-box 성능을 평가했다.

BridgeData V2와 Google robot를 활용해 evalutation을 수행했고, visual, motion, physical, semantic과 관련된 generatliztion task를 사용해 평가했다.

RT-1-X, RT-2-X, Octo와의 비교를 수행했다(model size가 꽤 차이난다.). RT-1-X, Octo는 Open-X로 pretrain된 모델이고, RT-2-X는 internet-pretrain된 backbone을 활용해 fine-tuning한 모델이다.

RT-1-X, Octo는 generalization 능력이 부족해 distractor 객체가 존재할 때 구분하는 걸 잘 못한다. 반면 RT-2-X와 OpenVLA는 distractor가 존재해도 잘 동작했고, 이 둘을 outperform했다.

OpenVLA는 RT-1-X보다 훨씬 작음에도 dataset에 따라 비슷하거나 더 좋은 성능을 보였다. 데이터셋의 규모와 품질, fused vision encoder를 사용했기 때문이다.

하지만 semantic generalization에 대해서는 RT-2-X가 더 좋았다.

### Adaptation to New Robot Setups

다른 robot setup에 적용해서 성능을 비교해봤다. (아마도 비교하는 다른 모델들도 해당 논문에서 사용하지 않았던 기기였을 거 같다. 추후 확인해보자.)

diffusion policy, Octo, OpenVLA를 비교했다. narrower single-instruction task, dexterous task(정확하고 부드러운 조작 필요)에서는 diffusion policy가, multiple object가 있거나 language conditioning이 필요한 경우 generalist policy가 더 좋았다. OpenX로 학습시켰어서 그렇다고 한다. OpenVLA가 평균적으로는 제일 좋다.

OpenVLA에서도 diffusion policy의 일부 방식을 활용하면 더 세밀하고 정확한 조작이 가능할 것이라고 한다.

### PEFT

PEFT를 위해 full fine-tuning, last layer only, frozen vision, sandwich fine-tuning, LoRA를 비교했다.

vision encoder를 학습시키지 않는 방식은 성능이 크게 떨어졌고, LoRA의 성능이 가장 좋았다. 이때 rank는 상관이 거의 없었다.

### Quantization

Octo와 비교하면 OpenVLA가 VRAM이 많이 필요하다 보니, quantization을 적용해봤다.

modern LLM quantization 기법인 [27, 88]을 사용해 INT4, INT8로 precision을 낮춰봤고, GPU별 speed와 precision별 bridge success?를 확인해봤다. INT4가 좋았다.

또한 quantization 연산 overhead에 의해 INT8의 speed가 더 느렸고, speed가 느려지다 보니 system dynamics와 맞지 않아 성능도 떨어졌다.

## 결론

### Pros and Cons

### 향후 방향

느낌을 볼 수 있었고.. VLA와 관련하여 edge-server 협응..?


