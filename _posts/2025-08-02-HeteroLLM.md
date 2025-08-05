---
title: "HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators"
date: 2025-08-02 22:26:00 +0900
categories: [Papers, Inference Acceleration]
tags: [cs, ai, on-device ai]
math: true
---
해당 논문은 2025년 1월에 arxiv에 올라온 논문으로, 교수님께서 공유해주신 [Awesome-On-Device-AI-Systems](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems/blob/main/README.md)에 소개되어 있어서 읽어보게 되었다. 이 글에서는 단순히 해당 논문의 내용과 주장을 정리하기 때문에, 해당 내용이 사실인지는 별도의 검증과 조사가 필요하다.

## Abstract

Privacy와 response latency 등의 측면을 개선하기 위해, 현재 ai를 mobile system 등에서 on-device로 돌리는 시도가 이루어지고 있다. 현재의 mobile SoC(System on Chip)에서는 이에 따른 computational demand를 만족시키기 위해, GPU, NPU 등 다양한 ai accelerator를 포함한다. 하지만 현존하는 design들은 단일 ai accelerator에 대한 것으로, computation 및 memory bandwidth 측면에서 이런 heterogeneous processor들을 잘 고려해서 최적화하지는 못한다.

<!-- 실제로 NPU가 최신 SoC에 많이 들어가고 있는 추세인가? 빠지고 있는 추세는 아닌가? 현존하는 design들이 이걸 잘 못한다는 것이 사실인가? -->

이에 따라 논문에서는 우선 mobile SoC에 대해 성능적 특징을 살펴본다. 이후 저자는 해당 관찰을 통해 **1. prefill phase와 decoding phase 각각에서의 요구사항에 따른 partition strategy와, 2. mobile SoC에서의 fast synchronization 기법을 활용하는 inference engine인 HeteroLLM을 제시한다.** HeteroLLM은 layer-level과 tensor-level 모두에 대해서 heterogeneous execution을 지원한다고 한다.

## Introduction

### 해당 연구의 필요성

앞에서 언급한 것처럼 현재 ai를 스마트폰과 같은 mobile system에서 돌리려는 시도가 이루어지고 있고, 이에 따라 SoC 제조사들은 GPU, NPU와 같이 matrix/vecotr multiplication에서 이점을 가지는 다양한 ai accelerator들을 칩 안에 통합시켜왔다. 하지만 heterogeneous processor들을 활용하는 inference engine에 대한 선행 연구들은 아래와 같은 이유로 현재의 mobile platform에 적합하지 않다.

- GPU들과 NPU들을 위한 기존의 synchron  ization 기법은 LLM inference에 대한 overhead가 크다. 특히 각 kernel의 실행 시간이 수백 ms인 decoding phase에서 overhead가 커진다.

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

- 생성 비용이 높은 static NPU graph.

    현재의 mobile NPU는 static computation graph만을 지원하는데, 이는 LLM의 dynamic한 workload와 호환되지 않는다. 또한 NPU의 architecture적인 특성에 따라 최적의 graph를 계산하는 것이 GPU에 비해 더 복잡하고, 런타임에 이를 계산하는 것은 overhead가 크다.

<!-- static computation graph란? graph란 무엇인가? 왜 NPU의 graph를 계산하는 것이 GPU에 비해 더 복잡한 것일까? LLM은 decoding phase에서 동적으로 예측을 수행하기 때뭉네 dynamic하다는 것인가? -->

- 단일 processor의 memory bandwidth restriction.

    단일 processor만 활용하는 경우 SoC의 memory bandwidth를 만족시키기에는 충분하지 않다.

<!-- SoC의 memory bandwidth란.. 그러면 이건 그냥 시스템의 성능이 충분하지 않다는 말인 거 같다. -->

이런 성능적 특징을 고려했을 때, 특정 상황에서 NPU의 성능이 저하되는 경우 이를 GPU로 보완하는 GPU-NPU parallelism 기법이 유효할 수 있다.

### HeteroLLM

논문에서 제시하는 inference engine인 HeteroLLM는 mobile SoC에서의 heterogeneous processsing을 지원한다. HeteroLLM에서 CPU는 synchronization, GPU kernel scheduling을 수행하고, NPU는 primary computing unit으로 기능하고, GPU는 NPU의 lower bound를 개선하기 위한 secondary computing unit으로 동작한다. 또한 layer-level과 tensor-level 모두에 대해 GPU-NPU parallelism을 구현하기 위해 HeteroLLM에서는 아래의 기법을 활용한다.

<!-- 또한 HeteroLLM이 stage performance, order-sensitive performance, shape-sensitive performance를 고려한다고 하는데, 이게 tensor factor인 것 같다. 근데 구체적으로 뭔지는 잘 모르겠다. -->

- tensor-level heterogeneous execution을 위해 prefill phase와 decoding phase 각각에 대해 다른 tensor partition 전략을 활용한다.
- kernel waiting time을 기반으로 한 fast synchronization 기법을 활용한다.
- hardware profiler와 runtime decider를 활용하는 tensor partition solver를 활용한다.

<!-- kernel waiting time이란? -->

<!-- tensor partition solver는 partition된 걸 합치는 부분인가? -->

저자는 HeteroLLM을 Arm CPU/GPU/NPU를 사용하는 Qualcomm 8 Gen 3 SoC에 구현했다고 한다. 또한 CPU/GPU를 활용하는 SOTA LLM inference engine인 PPL을 기반으로 구현했고, NPU를 통합하기 위해 Qualcomm의 QNN을 사용했다. 이때 activation에 대한 quantization과 sparsity 기법은 orthogonal하다고 판단하여 실험에서 제외했다.

결과적으로 HeteroLLM은 billion-scale mobile LLM에 대해 prefill에서 float 연산을 사용하면서 1000 token/s의 성능을 냈고, prefill phase와 decoding phase 모두에서 성능 향상을 보였다.

<!-- Qualcomm 8 Gen 3 SoC, PPL, QNN에 대해 알아보자. -->

## Background & Related Work

### LLM Inference

LLM inference는 prefill phase와 decoding phase로 나뉜다.

- Prefill Phase

    input 전체를 하나의 batch로 처리하여 첫 번째 token을 생성하는 phase. 이에 따라 matrix-matrix multimplication이 수행되고, computation intensive하다.

- Decoding Phase

    autoregressive로 sequential하게 한 번에 하나의 token을 생성하는 phase. 이에 따라 matrix-vector multimplication이 수행되고, memory-intensive하다.

<!-- prefill과 decoding이 각각 구체적으로 어떻게 계산되는지 수식을 써보자. memory intensive하다는 것은.. 뭐 당연하긴 하다. -->

mobile에서의 LLM inference의 latency는 TTFT(Time to First Token)와 TPOT(Time per Output Token)으로 구분될 수 있는데, 각각 prefill phase와 decoding phase의 속도에 의해 정해진다.

### Mobile-side Heterogeneous SoC

앞에서 언급한 것처럼 priviacy와 security, latency 등의 이유로 데이터를 cloud service로 전송하는 대신 LLM을 local device에서 돌리려는 시도가 이루어지고 있다. 이에 따라 주요 제조사들은 CPU, GPU, NPU를 포함하는 heterogenous SoC를 개발하고 있다. ([모바일 SoC 성능 순위 - Nanoreview](https://nanoreview.net/en/soc-list/rating)) 또한 mobile-side에서 이런 processor들은 하나의 physical memory를 공유해 사용하는 경우가 많다.

### Mobile-side Inference Engine

mobile-side inference engine으로는 ONNX Runtime, Llama.cpp, MNN, PPL 등 여러 가지가 있고, 이는 대체로 ONNX format으로 입력을 받아 optimization을 수행하는 식으로 동작한다. 또한 mobile accelerator들을 CPU, GPU, NPU 등의 백엔드로 추상화하고, accelerator의 instruction set과 programming language를 활용해 대응되는 low-level operator를 구현한다.

하지만 기존의 inference engine들은 heterogeneous processer들을 활용하면서 accuracy 하락이 있거나, tensor granularity로는 활용하지 못하는 등의 한계가 존재한다. 또한 GPU-NPU parallelism은 구현하지 못했다.

<!-- CPU, GPU, NPU 등의 백엔드로 추상화한다는 것이 무슨 의미인가.. -->

## Performance Characteristic

design을 살펴보기 전에 우선 각 accelerator의 architecture적인 특성을 알아보자.

### GPU Characteristics

SIMT instuction, on-chip shared memory, SM(Streaming Multiprocessor)을 활용한다는 점에서 mobile GPU는 desktop GPU와 동일하다. 하지만 mobile GPU는 system memmory와 독립된 memory를 활용하는 desktop GPU와는 달리, system memory에 통합되어 있는 UMA(Unified Memory Address Space)를 활용한다.

그러나 OpenCL 등 desktop GPU를 상정하는 framework들은 이런 구조를 반영하고 있지 않아 mobile GPU에 대한 redundancy가 존재한다. 예를 들어, mobile GPU는 UMA를 활용하므로 CPU memory와 GPU memory 사이의 데이터 데이터 전송이 불필요하다.

<!-- GPU가 SIMT instruction을 어떻게 구현하고 있나? 어떤 연산이 존재하나? -->

mobile GPU의 주요 특징들로는 아래와 같은 것들이 있다.

- Linear Performance

    연산하는 tensor의 크기에 따른 GPU의 성능(TFLOPS)은 아래의 그래프와 같다. tensor size가 작을 때 TFLOPS가 linear하게 증가하므로, 연산이 memory-bound이다. 이후 특정 threshold 이후에는 TFLOPS가 더 이상 증가하지 않으며 연산이 computation-bound이다.

<!-- computation/memory bound라는 것은 computation/memory가 해당 상황에서 성능을 결정짓는 병목이라는 뜻이다. -->

![](/assets/img/posts/2025-08-02-HeteroLLM/gpu_linear_performance.png)

- High-cost Synchronization

    GPU에 대한 synchronization은 비용이 높은데, 이는 1. 현재의 GPU framework는 mobile GPU에 대해서도 desktop GPU에서와 같이 API를 호출하도록 구현되어 있고 이에 따라 data size에 관계없이 400 ms 정도의 latency가 발생하기 때문이다. 또한, 2. GPU는 기본적으로 asynchronous model로 설계되어 queue에 kernel을 저장해 두고 연산하는데, synchronization을 적용하면 queue가 비워질 때까지 기다리기 때문에 50~100 ms의 추가적인 latency가 발생하게 된다.

<!-- 두 번째가 문제가 되는 이유가, 원래는 GPU가 asynchronous하게 kernel을 queue에 저장해 두면서 연산을 하는데, synchronization을 적용하는 순간 queue가 완전히 비워지는 것을 기다리게 되고, queue가 완전히 비워진 뒤에 kernel submission을 받으면 submission에 의한 latency를 해당 시점마다 매번 기다려야 되기 때문인가? -->

### NPU Characteristics

NPU에서 가장 중요한 component는 matrix computation unit(ex. systolic array)로, 아래의 그림은 가장 기본적인 systolic array의 구조이다. NPU의 연산 과정에서는 우선 computation 이전에 PE(Processing Element)에 weight가 preload되고, 이후 weight stall(고정) 상태로 input/activation이 계산된다. 이후 최종 결과는 on-chip SRAM에 저장되거나 다음 systolic array unit에 전달된다. 이런 과정을 통해 NPU는 weight/activation에 대한 load/store 연산과 cycle 수를 줄인다.

<!-- load/store 연산이 구체적으로 어떻게 줄어든다는 것인지 궁금하다. -->
<!-- systolic array의 동작은 https://deep-math.tistory.com/29 를 참고했다. -->

![](/assets/img/posts/2025-08-02-HeteroLLM/systolic array.png)

위와 같은 architecture에 따라, NPU의 성능은 아래와 같이 3가지 특징을 가진다.

- Stage performance

    NPU의 hardward computing array(ex. systolic arary)의 크기가 고정되어 있기 때문에, matrix multiplication을 수행해야 하는 tensor의 크기가 computing unit의 크기와 align되지 않으면 해당 unit이 inefficient하게 활용되게 된다. computing unit들을 최적으로 활용하려면 compiler가 tensor를 computing unit 크기에 맞도록 잘 나눠줘야 하고, tensor를 쪼갠 뒤 남는 부분이 생길 경우 연산을 위해 padding을 추가힌다.
    
    이에 따라 tensor의 크기에 의해 NPU 성능이 아래의 그래프와 같이 결정된다. 이런 misalignment result를 Stage Performance라고 한다.

![](/assets/img/posts/2025-08-02-HeteroLLM/stage performance.png)

- Order-sensitive performance

    $M \times N$ matrix와 $N \times K$ matrix의 multiplication, 그리고 $K \times N$ matrix와 $N \times M$ matrix의 multiplication은 모두 실제 연산량이 $2MNK$로 동일하다. 하지만 NPU에서는 아래의 그래프와 같이 weight tensor가 클수록 성능 저하가 발생한다(오른쪽 matrix가 weight). 이는 weight tensor가 클수록 단일 computing unit으로 처리하는 대신, 해당 computing unit에 값을 load/store하는 것을 반복하며 연산해야 하기 때문이다.
    
    NPU는 weight stall에 따른 load/store 연산을 줄여 성능을 향상시키므로 이 경우 성능이 떨어지는 것인데, 이를 Order-sensitive Performace라고 한다.

![](/assets/img/posts/2025-08-02-HeteroLLM/order sensitive.png)

- Shape-sensitive performance

    NPU의 성능은 input tensor의 shape에 의해서도 결정된다. input tensor의 row보다 column이 클수록 성능이 저하되는데, 이는 matrix multiplication이 수행되므로 column의 크기가 weight tensor의 크기에 영향을 미치기 때문이다(오른쪽 matrix가 weight).

<!-- 비율이 왜 중요하다는 것인지 잘 모르겠다..? 이 부분은 다시 알아보자. 
TPU에 값을 하나씩 넣으므로... 그런건가...?
-->

### SoC Memory Bandwidth

Qualcomm Snapdragon 8 Gen 3에서 실험한 결과, 아래 그래프와 같이 decoding phase에서 단일 processor만 사용하는 경우 SoC의 최대 memory bandwidth를 충분히 활용하지 못한다. 즉, NPU-GPU parallelism으로 memory bandwidht를 더욱 활용함으로써 성능 향상을 기대할 수 있다.

<!-- 이런 것은 실험을 진행한 Qualcomm Snapdragon 8 Gen 3에서만 적용되는 결과일 수 있겠다. 또는 실제로 이렇게 Soc가 설계되는 것인가.-->

![](/assets/img/posts/2025-08-02-HeteroLLM/soc memory bandwidth.png)

## Deisgn

앞에서 언급한 것처럼 HeteroLLM에서 CPU는 synchroniztaion, GPU kernel scheduling, non-compute intensive task를 처리하는 control plane으로 활용한다. CPU는 LLM task를 처리하기에 적합하지 않고, 다른 application도 처리해야 하므로 모든 power를 사용하지 않는 것으로 했다. 반편 NPU는 특정 상황을 제외하면 LLM task에 대해 GPU보다 성능이 뛰어나므로 primary computing unit으로, GPU는 NPU의 lower bound를 보완하는 secondary computing unit으로 활용한다.

HeteroLLM에서는 GPU-NPU parallelism에 따른 heterogeneous execution을 아래와 같이 layer-level과 tensor-level로 구현한다.

- Layer-level apporach에서는 1. 각 연산을 가장 적절한 backend(GPU or NPU)에 할당한다. 예를 들어, Matrix multiplication은 NPU에, RMSNorm과 SwiGLU는 GPU에 할당한다. 그리고 2. 전형적인 LLM에서는 input에 비해 weigt가 크므로 transpose하여 오른쪽 matrix(weight 위치)에 input(더 작은 쪽)이 오도록 한다.

<!-- RMSNorm과 SwiGLU란? -->

- Tensor-level approach에서는 backend에 따른 partition 전략을 활용하고, solver를 사용하여 최적의 partition solution을 결정하도록 한다.

<!-- solution이란? -->

또한 이 두 approach 모두에 대해 새로운 fast synchronization 기법을 적용하여 GPU, NPU에서의 synchronization overhead를 줄인다.

전반적인 실행 흐름은 아래 그림과 같다.

![](/assets/img/posts/2025-08-02-HeteroLLM/heterollm 실행구조.png)

### Tensor Partition Strategy

HeteroLLM에서 활용하는 partition strategy는 아래와 같은 것들이 있다. 이는 1. tensor shape에 따른 NPU 성능 저하, 2. 고비용의 static computation graph, 3. SoC memory bandwidth를 고려한 기법들이다.

#### Partition during Prefill Phase

prefill phase에서 적용 가능한 partition strategy로는 이런 것들이 있다.

<!-- 여기에서의 전략들을 prefill phase에서 computation이 병목임을 고려하고 다시 판단해보자. -->

- Row-cutting

    activation-weight multiplication을 전치해서 $W^T A^T$를 연산한다고 하자. 이 경우 만약 sequence length($A^T$의 column)가 짧아지면 stage performance에 의해 NPU의 computational resource를 충분히 활용하지 못한다. 또한 FFN-down은 dimension을 줄이는 layer이므로 $W^T$는 row에 비해 column이 더 큰 matrix이므로, shape-sensitive performance에 의해 성능 저하가 발생한다.

    이런 경우 NPU의 성능은 GPU 수준 또는 그 이하로 떨어지게 되는데, 논문에서는 첫 번째 matrix($W^T$)를 row dimension에 대해 partition하는 Row-cutting을 적용하여 일부는 NPU에서, 다른 일부는 GPU에서 연산하도록 한다. 또한 partition함에 따라 다른 layer의 연산 이전에 GPU/NPU 각각에 의해 연산되는 모든 부분이 완료되었음을 보장하기 위해 synchronizaiton point를 명시적으로 설정한다.

![](/assets/img/posts/2025-08-02-HeteroLLM/row cutting.png)

<!-- 
그냥 성능이 떨어지니까 나눠서 일부를 GPU에서 연산하도록 한 것인가? 왜냐하면 NPU-3에서는 row에 비해 column이 커지면 성능이 떨어진다고 했는데, 여기에서는 column이 아니라 row를 잘랐다. 물론 의미적으로 그렇게 하는 게 적절할 거 같기는 한데, 그러면 NPU에서의 성능은 여전히 떨어지는 거 아니냐?

또한 이런 partition 비율은 어떻게 결정되나? 
뒤에 나오는 sovler는 decoding phase에서의 partition 비율을 결정하는 거 같다..?

load/store 연산을 고려해보자.
-->

- Sequence-length cutting

    현재의 mobile NPU에서는 dataflow graph compilation을 주로 활용하는데, 이에 따라 mobile NPU에서는 static graph execution만을 지원한다. 즉, kernel initialization 시에 tensor의 size와 shape이 결정될 수 있어야 한다. 또한 아래의 그래프에 따르면 tensor의 크기가 커질수록 kernel optimization에 대한 search space가 넓어져, optimization 비용이 높아진다.

    반면 GPU는 다양한 tensor shape을 처리할 수 있는 여러 kernel implementation을 지원하므로, 임의의 shape을 가지는 tensor를 dynamic하게 처리할 수 있다.

    NPU에서 dynamic input shape을 처리할 수 있도록 하는 기본적인 방법은, predefined tensor shape의 집합을 미리 정의해 두고 input tensor에 padding을 붙여 해당 shape으로 맞춘 뒤 연산하는 것이다. 하지만 이 경우 padding에 의한 추가적인 computational overhead가 발생한다. 예를 들어, sequence length가 130인 경우 256에 맞추기 위해 padding을 126만큼 추가하고 이를 연산해야 한다.

    이에 따라 논문에서는 아래 두 번째 그림과 같이 Sequence-length cutting을 적용하여, NPU에서는 fixed-size tensor를 연산하도록 하고, GPU에서는 dynamic-shape tensor를 연산하도록 한다.

![](/assets/img/posts/2025-08-02-HeteroLLM/optimization and shape.png)

<!-- 왜 mobile NPU에서는 static computation graph만을 사용하나? dataflow graph compilation[1, 7, 52] 때문이라는데, 잘 모르겠다.
이게 뭐임?

inference engine에서 이런 방식을 차용하는 거 같다.
tensor shape과 sequence length에 따라 kerenl optimization 비용이 높아진다고 한다. 이건 graph 생성에 포함된 과정인가?
search space가 넓어져서 그렇다는데? 작은 거보다는 아무래도.. 최적화 경우의 수가 많아지니까..?
어떤 원리로 graph가 생성되는지 모르겠다. 그래서 왜 비용이 커지는지도 모르겠다.
또한 왜 predefined된 graph를 활용하는 것에 이점이 있는지도 잘 모르겠다. 해당 크기에 대한 최적화 방식이 정해져 있는 것인가.

NPU inference engine에서 이런 방식을 활용하고 있고, 구체적인 최적화 방식까지 알기는 어려울 거 같다.

gpu에서는 여러 kernel implementation을 제공한다고 하는데, 어떤 거길래 다양한 shape에 대해 유효한건가?
이것 또한 GPU inference engine에 대한 내용 같다.
-->

![](/assets/img/posts/2025-08-02-HeteroLLM/partition.png)

- Multi-sequence-lenght cutting

    linear performance에서 확인했듯이, GPU에서도 sequence length가 threshold를 넘어가면 computation이 병목이 된다. 
    
    이에 따라 위 그림과 같이 input tensor를 한 번이 아니라 여러 번 predefined shape으로 쪼개는 Multi-sequence-length cutting을 활용할 수 있다. 이 경우 여러 개의 predefined shape으로 tensor를 쪼개 TPU에서 연산하도록 하고, 임의의 shape을 가지는 마지막 tensor를 GPU에서 연산하도록 한다.

- Hybrid-cutting

    위 그림과 같이 row-cutting과 sequence-length cutting을 함께 활용하는 Hybrid-cutting도 가능하다. 이 경우 row-cutting에 의해 나눠진 GPU 연산 부분에서는 단순히 연산하면 되고, NPU 연산 부분에서는 padding을 추가해 연산한다.

<!-- 근데 의문이, GPU를 secondary하게 활용하는데, 좀 더 적극적으로 활용하면 성능이 더 좋은 거 아닌가? 
그리고, GPU 활용에 따른 overhead는 존재하지 않는가? -->

#### Partition during Decoding Phase

prefill phase와는 달리 decoding phase는 memory bandwidth가 병목이다. 앞서 본 관찰에서 알 수 있듯이, 단일 processor를 활용하는 경우 SoC의 전체 memory bandwidth를 충분히 활용하지 못하므로 NPU와 GPU를 모두 사용하도록 한다. 이때 input token의 sequence length는 기본적으로 1(물론 speculative decoding 등을 사용하면 n)로 고정되므로, NPU computation graph를 predefine할 수 있다.

이에 따라 NPU computation graph에 대한 overhead는 고려할 필요가 없고, row-cutting을 적용해 TPU/GPU에 의한 memory bandwidth 활용을 최대화한다. 여기에서의 row-cutting은 prefill에서와 같이 computation 측면에서의 최적화라기보단, memory bandwidth 측면의 최적화로서 기능한다.

<!-- 마지막 문장에 대해 잘 설명할 수 있을지 다시 생각해보자. -->

또한 아래에서 설명할 partition solver를 사용하여 최적의 partition 정도를 계산한다.

### Fast Synchronization

GPU-NPU parallelism은 실행 시간을 줄이지만 synchronization에 따른 overhead가 추가로 발생할 수 있다. 특히 단일 연산의 실행 시간이 짧은 decoding phase에 그 overhead가 두드러진다. 이에 따라 HeteroLLM은 아래의 두 가지 synchronization 기법을 활용한다.

- mobile SoC에서는 unified memory를 사용하므로 추가적인 data transfer가 필요하지 않다. 그래서 transfomer의 각 연산에 대한 input/output tensor들을 저장하기 위한 dedicated memory pool을 reserve해 활용한다. LLM의 각 layer는 동일한 decoder block 구조를 사용하므로 이 memory를 재사용될 수 있다.

- LLM은 대체로 동일한 연산을 수행하므로 각 layer에 대한 waiting time을 예측할 수 있다. synchronization 대상인 thread는 해당 waiting time만큼 sleep했다가 polling mechanism(실행해도 되는지 반복 확인하는 방식)을 통해 다시 실행하도록 한다. 이때 mobile SoC의 usleep이 가지는 granularity는 80~100 ms 수준으로 정확한 synchronization에는 적합하지 않다. 이에 따라 thread의 sleep이 끝나면 CPU core를 활용해 output tensor가 준비되었는지 flag bit(output tensor가 준비되면 set되도록 구현했다.)에 대한 polling으로 확인한다.

이런 synchronization은 아래 그림과 같이 prefill phase와 decoding phase에서 각각 다른 양상을 보인다. prefill phase에서는 NPU가 computaiton 측면에서 dominant하고 GPU는 덜 활용되므로, 다음 GPU kerenl에 대한 submission은 NPU 연산을 기다려야 한다. 이에 따른 submission overhead가 존재하기는 하는데, 10 ms 정도로 무시할 수 있다고 한다. 반면 decoding phase에서는 GPU가 dominant하고, GPU submission overhead도 적다.

<!-- GPU kernel implementation이 더 안정적이고 memory bandwidth에서 efficient하기 때문에 decoding phase에서 GPU 사용이 dominant하다고 하는데, 이게 잘 이해가 안된다. -->

<!-- NPU는 queue를 쓰고.. 이런 게 없는 것인가? 행렬 곱을 위한 단순한 구조이기 때문? -->

<!-- memory bandwidth를 잘 활용해서 ai 성능을 높이는 것은 좋은데, 시스템의 memory bandwidth를 너무 많이 써버리면 다른 application의 실행은 보장되기 어려울 수도 있겠다. 스마트폰같은 mobile device에서는 하나의 프로그램만 돌리는 게 아닐 거 같은데. -->

### Putting It All Together


## Evaluation

## Discussion

## Conclusion


