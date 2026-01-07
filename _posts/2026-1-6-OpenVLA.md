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

기존 robotics 관련 모델들의 주요 한계는 LLM 등에 비교했을 때 dataset 자체가 너무 작아서 generalization, robustness가 부족하다는 것이다. 하지만 internet-scale dataset으로 pretrain된 vision-language foundation model들은 높은 generalization 능력을 가지고 있다. 이에 따라 **pretrained vision-language foundation model들을 core block으로 사용하는 시도**가 이루어지고 있다.

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

- [**RT-2-X**](https://robotics-transformer2.github.io/)(PMLR 2023, 2390회 인용)는 55B 크기(더 작은 버전도 존재)의 오픈소스가 아닌(closed) VLA로, OpenVLA 이전의 SOTA이다. 그 아키텍처는 다음 그림과 같다. RT-2-X는 internet-scale에서 pretrain된 VLM인 PaLI-X와 PaLM-E을 backbone으로 사용했고, OpenVLA와 유사하게 기존 token들 중 256개를 action token으로 활용했다. 학습 시에는 RT-1-X에서 사용했던 robot data와 internet-scale dataset인 WebLI dataset을 함께 사용했다.

    RT-2-X는 하나의 로봇이나 simulation에서의 학습과 평가에만 집중하고, 새로운 로봇에 대한 효율적인 fine-tuning을 지원하지 않는다. 또한 오픈소스가 아니다. 반면 OpenVLA는 성능이 더 좋고, 효율적인 fine-tuning을 지원하면서, 오픈소스이다.

![](/assets/img/posts/2026-1-6-OpenVLA/rt2 arch.png){: width="800"}

## OpenVLA

### Architecture

**OpenVLA**는 다음 그림과 같이 pretrained VLM인 Prismatic-7B VLM을 backbone으로 사용하는 architecture를 가진다. 

OpenVLA는 RT-2-X처럼 기존 token들 중 256개를 action token으로 해, action token을 예측하도록 Prismatic-7B VLM을 fine-tuning한다. 즉, discrete output language token과 continuous robot action을 mapping한 것으로 볼 수 있다. 이때 action token에는 robot action을 256개의 bin(구간)으로 나눠 mapping한다. bin width는 training data에서 action 값에 대한 백분위수를 사용해 제1백분위수와 제99백분위수 사이를 uniform하게 나누는 값으로 지정한다(RT-2-X와 달리 outlier를 제외한 것.). 이에 따라 LLM token 중 총 256개를 action token으로 사용해야 하는데, Llama tokenizer는 100개의 speical token만을 가지므로 least used token들을 overwrite해 활용한다(이는 가장 마지막 256개의 token이다.).

이후 N차원의 action 값이 필요한 로봇에 대해 예측을 수행한다고 가정하면, autoregressive하게 action token을 하나씩 생성하다가 0~255사이의 정수 값으로 구성되는 총 N개의 action token이 모였을 때 해당 값들로 로봇을 조작한다. 예를 들어, $\Delta x, \Delta y, \Delta z, \Delta roll, \Delta pitch, \Delta yaw, gripper$로 차원이 7인 action 값이 필요한 로봇의 경우 총 해당 값들이 모두 모일 때까지 추론을 반복한다.

![](/assets/img/posts/2026-1-6-OpenVLA/OpenVLA arch.png){: width="700"}

OpenVLA에서 사용하는 Prismatic-7B VLM는 [**Prismatic VLMs paper**](https://openreview.net/forum?id=6FXtu8clyp)(ICML 2024, 236회 인용)의 VLM이다. 해당 논문에서는 VLM이 가지는 여러 가지 design decision들이 존재하지만 최적의 design이 under-explore된 것에 착안하여, VLM을 평가하는 unified framework를 정의하고 여러 model checkpoint를 제공한다. 또한 VLM design에 대한 평가 결과를 바탕으로 다음 그림과 같이 **Prismatic-7B VLM**을 제안한다. Prismatic-7B VLM는 OpenVLA architecture 그림에 나와있는 것처럼, vision encoder로 SigLIP와 DinoV2를, language model로 Llama2와 같이 internet-scale에서 pretrain된 모델을 사용하며, MLP projector로 vision feature를 language embedding space로 projection하는 architecture를 가진다.

이때 visual encoder로 SigLIP와 DinoV2를 함께 사용할 때는 입력 이미지의 각 patch를 두 encoder에 각각 넣어 출력된 두 embedding을 단순히 concat해 사용한다. 이런 encoder 구조가 하나의 encoder만 사용하는 것보다 spatial reasoning에 더 유리하다고 한다. 또한 projector는 단순히 vision embedding을 입력으로 받아 language embedding 크기로 변환하는 MLP이다.

![](/assets/img/posts/2026-1-6-OpenVLA/prism result.png){: width="400"}

### Training

OpenVLA는 action token을 출력하도록 Prismatic-7B VLM를 full fine-tuning한 것이다.

training dataset으로는 [**Open X-Embodiment(OXE)**](https://robotics-transformer-x.github.io/) dataset을 변형해 활용했다. 변형된 dataset은 1) input&output space가 일관적이어야 하고 2) 다양한 embodiment, task, scene들이 training mixture에 포함되어 있어야 한다. 이에 따라 적어도 하나의 3rd person camera가 있으면서, single arm end-effector control만 포함하고, diversity가 낮은 dataset은 제외하는 등 Octo에서 사용한 방식을 사용해 data mixture를 구성했다.

A100 GPU 64개로 14일간 학습을 돌렸고, 학습된 모델은 추론에 VRAM 15GB가 필요하다고 한다. 물론 추후 소개할 quantization을 적용하면 더 적은 VRAM으로 돌릴 수 있다.

![](/assets/img/posts/2026-1-6-OpenVLA/OpenVLA dataset.png){: width="600"}

### Design Decisions

최종 training을 수행하기 전에, 비교적 작은 dataset인 BridgeData V2 dataset에서 다음과 같이 여러 design decision을 수행했다.

- VLM backbone으로는 LLaVA, IDEFICS-1도 사용해봤는데, IDEFICS-1 < LLaVA < Prismatic VLM 이었다. 특히 multi-object일 때 그랬는데, 아마 spatial reasoning 능력 때문일 것이라고 한다. 또한 모델 코드를 활용하기도 더 쉬웠다고 한다.

- 여러 VLM bench에서 image resolution을 높이면 성능이 개선되지만, OpenVLA에 대해 실험했을 때는 성능 차이가 없어서 더 적은 resolution을 사용했다. resolution이 높아질수록 vision token 증가에 따른 context length 증가로 computation이 늘어나므로 적은 걸 선택했다.

- VLM 관련 기존 연구에선 vision encoder를 freeze하는 게 기존 지식이 보존되어 더 높은 성능을 보였지만, OpenVLA 학습 시에는 함께 fine-tuning하는 것이 성능이 더 좋았다. vision encoder 자체가 robotic control 자체에 대한 지식을 가지고 있지 않았어서 그런 것으로 보인다고 한다.

- LLM/VLM 학습 시의 전형적인 fine-tuning epoch은 1~2 정도이지만, OpenVLA 학습 시에는 epoch을 늘려도 성능이 계속 올라가서 27까지 했다고 한다.

- leaning rate는 Prismatic VLMs paper에서의 학습에서와 동일한 2e-5가 실험적으로 최적이었다고 한다.

## Experiments

본 논문에서 진행한 실험의 목표는 다음과 같은 사항들을 확인하는 것이다.

1. 기존 방식과 비교해서 OpenVLA가 multi-robot과 다양한 task에 대해서 성능이 좋은가?
2. fine-tuning해 새로운 robot task로서 사용되기에 좋은가? 
3. PEFT나 quantization을 적용해 더 accessible하게 할 수 있는가?

### Out-of-the-Box

OpenVLA의 out-of-the-box 성능은 OpenVLA, RT-1-X, RT-2-X, Octo간의 비교를 수행했다. 사용한 로봇은 WidowX(BridgeData V2 dataset의 로봇)와 Google robot(RT-1-X, RT-2-X에서 평가에 사용한 로봇)이다. 평가에는 몇 가지 generatliztion task를 정의해 각 세부 task별로 여러 번의 trial을 실제로 로봇을 사용해 시도하며 성공 횟수를 측정했다.

![](/assets/img/posts/2026-1-6-OpenVLA/exp1.png){: width="800"}

![](/assets/img/posts/2026-1-6-OpenVLA/exp2.png){: width="400"}

**대부분의 task에서, 그리고 평균값에서 OpenVLA가 가장 높은 성공률을 보였다.** internet-scale 데이터로 학습되지 않은 RT-1-X, Octo는 generalization 능력이 부족해 distractor 객체가 존재할 때 구분하는 걸 잘 못했다. 반면 internet-scale로 pretrain된 VLM을 사용해 만든 RT-2-X와 OpenVLA는 distractor가 존재해도 잘 동작했다. 또한 OpenVLA(7B)는 RT-1-X(55B)보다 훨씬 작음에도 dataset에 따라 비슷하거나 더 좋은 성능을 보였다. 더 크고 좋은 규모의 dataset(OXE 기반 dataset)을 사용해 훈련했고, fused vision encoder(SigLIP와 DinoV2)를 사용했기 때문이라고 한다. 하지만 semantic generalization에 대해서는 RT-2-X가 더 좋았는데, 이는 RT-2-X가 학습 시에 robot data와 internet data를 섞어서 사용해서 pretraining knowledge를 보존했기 때문이라고 한다.

### Adaptation to New Robot Setups

fine-tuning을 통해 새로운 robot setup에 적용하는 실험은 OpenVLA(full fine-tuning을 한 버전), OpenVLA(scratch)(fine-tuning을 하지 않은 버전), diffusion policy, diffusion policy(matched), Octo(fine-tuning을 한 버전) 간의 비교를 수행했다. RT-2-X는 오픈소스가 아니고 API에서 fine-tuning을 지원하지 않는다고 한다. 사용한 로봇은 Franka Emika Panda 7-DoF robot arm이고, Franka-Tabletop과 Franka-DROID에서 평가했다.

[Diffusion Policy](https://journals.sagepub.com/doi/full/10.1177/02783649241273668)(IJRR 2024, 1566회 인용)는 robot action을 transformer 기반의 diffusion model로 수행하는 모델로, 이미지와 로봇의 현재 물리적 상태를 입력으로 받아, action의 chunk를 예측한다. 이 실험에서 비교에 사용하는 diffusion policy(matched)는 이미지만 입력으로 사용하고, chunk 대신 하나의 출력을 사용하도록 해 입출력을 OpenVLA와 맞춘 것이다. diffusion policy와 diffusion policy(matched)는 scratch부터 pretrain되었다고 한다.

![](/assets/img/posts/2026-1-6-OpenVLA/exp3.png){: width="800"}

**평균적으로 OpenVLA가 제일 좋았다.** 하지만 narrower single-instruction task, dexterous task(정확하고 부드러운 조작 필요)에서는 diffusion policy가 좋았는데, 이는 diffusion policy의 아키텍처적 특성에 의한 것이라고 한다. 반면 multiple object가 있거나 language conditioning이 필요한 경우 generalist policy가 더 좋았고, 이는 OXE dataset으로 학습시켰어서 그렇다고 한다. OpenVLA에서도 diffusion policy의 방식을 일부 활용하면 더 세밀하고 정확한 조작이 가능할 것이라고 언급한다.

### PEFT

OpenVLA에 대한 full fine-tuning은 8개의 A100 GPUs를 사용했을 때 task에 따라 5~15시간이 걸린다고 한다. 이에 따라 Parameter Efficient Fine-Tuning(PEFT)가 가능한지를 확인하기 위해 full fine-tuning, last layer only(transformer block의 last layer와 token embedding matrix만 학습), frozen vision(vision encoder만 freeze), sandwich fine-tuning(last layer only에 vision encoder까지 학습), LoRA(모든 linear layer에 사용)를 비교했다.

![](/assets/img/posts/2026-1-6-OpenVLA/exp4.png){: width="400"}

**vision encoder를 학습시키지 않는 방식은 성능이 크게 떨어졌고, LoRA의 성능이 가장 좋았다. 이때 rank는 상관이 거의 없었다.**

### Quantization

Octo(93M)와 비교하면 OpenVLA(7B)가 inferece에 VRAM이 많이 필요하다 보니, quantization을 적용하는 실험도 수행했다.

modern LLM quantization 기법인 [QLoRA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)(NeurIPS 2023, 4802회 인용)와 [LLM.int8()](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html)(NeurIPS 2022, 1734회 인용)을 사용해 INT4, INT8로 precision을 낮춰서 비교했고, GPU별 speed와 precision별 BridgeData V2 dataset 중 8가지 task에서의 성공 횟수를 확인했다.

![](/assets/img/posts/2026-1-6-OpenVLA/exp5.png){: width="700"}

INT4가 좋았다. 또한 quantization 연산 overhead에 의해 INT8의 speed가 더 느렸고, speed가 느려지다 보니 system dynamics와 맞지 않아 성능도 떨어졌다.

## 결론

### Pros and Cons

### 향후 방향

느낌을 볼 수 있었고.. VLA와 관련하여 edge-server 협응..?


