# T5X Retrieval

T5X Retrieval is a JAX implementation of T5 (Text-to-Text Transfer Transformer) optimized for retrieval applications.
It is built on top of T5 on JAX, aka [T5X](https://github.com/google-research/t5x).
This is targeted at Natural Language Understanding researchers as well as application developers who are aiming to use the latest T5-based Transformer models for search, retrieval and ranking applications, but in the JAX framework as opposed to TensorFlow.

T5X Retrieval is an efficient training and evaluation framework that supports trasformer-based neural retrieval and ranking models such as sentence encoders and dense retrieval models. It supports multi-pod large model training, large cross-batch negatives and the capability to initialize from any pre-trained model trained using T5X.

This launch open sources the training and inference code, including references to TFDS for training data, actual model training code (Python JAX & Flaxformer), pre-trained models and basic inference example code. This end-to-end example model code is meant to accompany the SentenceT5 and Generalizable T5 Retrieval models that includes the implementation and performance on relevant benchmarks.


# What's here

-   configs/\*.gin - Model configurations
-   tasks.py - Task definitions that generate the dataset.
-   feature_converters.py - Converters that transform the task features from the dataset to model features
-   models.py - High-level models, such as DualEncoderDecoderModel, that take the feature converters ouputs as inputs.

For more details about the training pipeline and task definitions, you can check out [T5X](https://github.com/google-research/t5x) and [Seqio](https://github.com/google/seqio).

# Models

## Sentence encoders
**SentenceT5** is a family of high performing sentence encoders trained using T5X Retrieval. The sentenceT5 models encode text into high-dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language processing tasks.

SentenceT5 models are built on top of the Text-To-Text Transfer Transformer (T5). It is trained on a variety of data sources and initialized from pre-trained T5 models with different model sizes as described in [1]. The input is variable-length English text and the output is a 768-dimensional vector. Note that there's no hard length limit for T5 (i.e., no 512 tokens limit as in BERT), but that it's been trained to produce good embeddings for approximately sentence length text.

### Metrics

*   We evaluate this model on the
    [SentEval](https://github.com/facebookresearch/SentEval) sentence
    representation benchmark.

    Transfer tasks                                                | MR   | CR   | SUBJ | MPQA | SST  | TREC | MRPC | Average
    :------------------------------------------------------------ | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ------:
    **ST5-Base**                                                  | 85.8 | 92.1 | 94.6 | 90.9 | 91.8 | 96.4 | 75.2 | 89.5
    [ST5-Large](https://tfhub.dev/google/sentence-t5/st5-large/1) | 88.9 | 93.5 | 95.4 | 91.5 | 94.2 | 96.2 | 77.1 | 91.0
    [ST5-3B](https://tfhub.dev/google/sentence-t5/st5-3b/1)       | 89.9 | 94.1 | 95.9 | 91.6 | 94.8 | 96.2 | 77.9 | 91.5
    [ST5-11B](https://tfhub.dev/google/sentence-t5/st5-11b/1)     | 90.8 | 94.4 | 96.3 | 91.7 | 94.8 | 95.4 | 77.9 | 91.6

    <br/>

    STS tasks                                                     | STS12 | STS13 | STS14 | STS15 | STS16 | STSb | SICK-R | Average
    :------------------------------------------------------------ | ----: | ----: | ----: | ----: | ----: | ---: | -----: | ------:
    **ST5-Base**                                                  | 78.1. | 85.8  | 82.2  | 87.5  | 84.0  | 86.0 | 79.8   | 83.3
    [ST5-Large](https://tfhub.dev/google/sentence-t5/st5-large/1) | 79.1  | 87.3  | 83.2  | 88.3  | 84.4  | 86.7 | 79.8   | 84.1
    [ST5-3B](https://tfhub.dev/google/sentence-t5/st5-3b/1)       | 79.0  | 88.8  | 84.3  | 88.9  | 85.3  | 86.3 | 79.5   | 84.6
    [ST5-11B](https://tfhub.dev/google/sentence-t5/st5-11b/1)     | 80.1  | 88.8  | 84.7  | 88.9  | 85.2  | 86.8 | 80.4   | 85.0

More details about the evaluations can be found in the paper [1].

## Dense retrieval models
The **Generalizable T5 Retrieval** models are dual encoders that encode two pieces of text into two dense
vectors respectively [2]. This is typically used to encode a query and a document to
compute their similarity for dense retrieval.

GTR models are built on top of [T5](https://arxiv.org/pdf/1910.10683.pdf) (i.e.
the Text-To-Text Transfer Transformer). The GTR-Base model employs a 12-layer
transformer architecture, which is the same as the T5 base model. The model is
first initialized from the pre-trained T5 checkpoint. It is then further
pre-trained with a set of community question-answer pairs we collected. Finally,
the model is fine-tuned on the [MS Marco](https://microsoft.github.io/msmarco/)
dataset.

The two encoders are shared so the GTR model functions as a single text encoder.
The input is variable-length English text and the output is a 768-dimensional
vector.

### Metrics

We evaluate on the [BEIR](https://github.com/UKPLab/beir) benchmark and report the Recall@100.

Dataset \ Model  | **GTR-Base** | [GTR-Large](https://tfhub.dev/google/gtr/gtr-large/1) | [GTR-XL](https://tfhub.dev/google/gtr/gtr-xl/1) | [GTR-XXL](https://tfhub.dev/google/gtr/gtr-xxl/1)
---------------- | ------------ | ----------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------
MS MARCO         | 0.898        | 0.908                                                 | 0.911                                           | 0.916
Trec-Covid       | 0.411        | 0.434                                                 | 0.457                                           | 0.407
BioASQ           | 0.441        | 0.490                                                 | 0.483                                           | 0.483
NFCorpus         | 0.275        | 0.298                                                 | 0.318                                           | 0.300
NQ               | 0.893        | 0.930                                                 | 0.936                                           | 0.946
HotpotQA         | 0.676        | 0.725                                                 | 0.739                                           | 0.752
FiQA-2018        | 0.670        | 0.742                                                 | 0.755                                           | 0.780
Signal-1M        | 0.263        | 0.261                                                 | 0.268                                           | 0.268
Trec-News        | 0.475        | 0.525                                                 | 0.512                                           | 0.544
Robust04         | 0.324        | 0.365                                                 | 0.364                                           | 0.372
ArguAna          | 0.974        | 0.978                                                 | 0.980                                           | 0.983
Touché-2020      | 0.281        | 0.282                                                 | 0.297                                           | 0.301
Quora            | 0.996        | 0.996                                                 | 0.997                                           | 0.997
DBPedia-entity   | 0.418        | 0.480                                                 | 0.480                                           | 0.494
SCIDOCS          | 0.340        | 0.358                                                 | 0.358                                           | 0.366
Fever            | 0.923        | 0.941                                                 | 0.944                                           | 0.947
Climate-Fever    | 0.522        | 0.552                                                 | 0.569                                           | 0.556
SciFact          | 0.872        | 0.899                                                 | 0.911                                           | 0.900
CQADupStack      | 0.681        | 0.714                                                 | 0.729                                           | 0.740
Avg              | 0.596        | 0.625                                                 | 0.632                                           | 0.634
Avg w/o MS MARCO | 0.580        | 0.609                                                 | 0.616                                           | 0.619


# Released Model Checkpoints

We have released the following checkpoints for SentenceT5 and GTR pre-trained models:

* **SentenceT5-Base** ([config](configs/models/de_t5_base.gin), 110M parameters): [gs://t5-data/pretrained_models/t5x/retrieval/st5_base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/retrieval/st5_base)
* **SentenceT5-Large** ([config](configs/models/de_t5_large.gin), 335M parameters): [gs://t5-data/pretrained_models/t5x/retrieval/st5_large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/retrieval/st5_large/)
* **SentenceT5-XL** ([config](configs/models/de_t5_3B.gin), 1.24B parameters): [gs://t5-data/pretrained_models/t5x/retrieval/st5_xl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/retrieval/st5_xl/)
* **SentenceT5-XXL** ([config](configs/models/de_t5_11B.gin), 4.8B parameters): [gs://t5-data/pretrained_models/t5x/retrieval/st5_xxl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/retrieval/st5_xxl/)
* **GTR-Base** ([config](configs/models/de_t5_base.gin), 110M parameters): [gs://t5-data/pretrained_models/t5x/retrieval/gtr_base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/retrieval/gtr_base/)
* **GTR-Large** ([config](configs/models/de_t5_large.gin), 335M parameters): [gs://t5-data/pretrained_models/t5x/retrieval/gtr_large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/retrieval/gtr_large/)
* **GTR-XL** ([config](configs/models/de_t5_3B.gin), 1.24B parameters): [gs://t5-data/pretrained_models/t5x/retrieval/gtr_xl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/retrieval/gtr_xl/)
* **GTR-XXL** ([config](configs/models/de_t5_11B.gin), 4.8B parameters): [gs://t5-data/pretrained_models/t5x/retrieval/gtr_xxl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/retrieval/gtr_xxl/)


# References

[1] Jianmo, Ni, Gustavo Hernández Ábrego, Noah Constant, Ji Ma, Keith B. Hall,
Daniel Cer, Yinfei Yang.
[Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models.](https://arxiv.org/abs/2108.08877)
ACL 2022.

[2] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego,
Ji Ma, Vincent Zhao, Yi Luan, Keith B. Hall, Ming-wei Chang, Yinfei Yang.
[Large Dual Encoders Are Generalizable Retrievers.](https://arxiv.org/abs/2112.07899)
December 2021.

This is not an officially supported Google product.
