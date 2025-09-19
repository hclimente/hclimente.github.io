---
layout: post
title: Data-centric machine learning
date: 2023-05-08 11:59:00-0000
description: Notes from the MIT course
tags: machine_learning
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

# Introduction. Data-Centric ML vs. Model-Centric ML

The commonly taught approach to AI is **model-centric**: the data is assumed to be fixed and perfect, and the challenge is producing the best model. However this contrast with the real world application, in which the data is messy, non-curated and low quality. **Data-centric** AI aims to produce the highest quality dataset. This is challenging when **the datasets are massive**.

Data-centric AI often takes one of two forms:

- Algorithms that understand the data and leverage that to improve the models. E.g., train on easy data first (curriculum learning).
- Algorithms that modify the data. E.g., train on filtered datasets in which mislabeled items have been removed (confident learning).

**Model-centric AI**

- Given a dataset, produce the best model
- Change the model to improve performance (hyperparameters, loss function, etc.)

**Data-centric AI**

- Given a model, improve the training set
- Systematically/algorithmically change the dataset

A motivating example is the kNN algorithm. It has no loss function, the prediction is simply a majority vote, and the quality of the prediction depends on the quality of the data. Another one is DALL-E, whose training required handling many mislabeled examples in the massive training dataset.

Note that data-centric AI is not about hand-picking good datapoints or getting more data. Some examples are:

- Outlier detection and removal: handling abnormal examples
- Error detection and correction: handling incorrect examples
- Data augmentation: adding examples that encode prior knowledge
- Feature engineering/selection: manipulating how the data is represented
- Establish consensus labels: determine the true label from crowdsourced annotations
- Active learning: select the most informative data to label next
- Curriculum learning: order the examples from easiest to hardest

# Confident learning: handling label errors

## Assumptions on the nature of the noise

Each training example in our dataset consists on a feature vector $$\mathbf x$$, and an observed (noisy) label $$\tilde y$$. Note that this label might not equal the latent true label $$y^\ast$$. This can happen in multiple ways:

1. Correctable mis-classification: $$\tilde y$$ is wrong, but there is only one $$y^\ast$$.
2. Multi-label: multiple $$y^\ast$$ are correct.
3. Neither (potentially _out of distribution_): the $$y^\ast$$ is/are unknowable
4. Non-agreement (_hard_): multiple $$y^\ast$$ are possible

We will tackle the first case using the **confident learning** framework. This label noise can occur in three ways:

- Uniform: $$p\left(\tilde{y}=i \mid y^*=j\right)=\epsilon, \forall i \neq j$$
- Systematic: $$p\left(\tilde{y}=i \mid y^*=j\right)$$ follows any valid distribution
- Instance-dependent: $$p\left(\tilde{y}=i \mid y^*=j, \boldsymbol{x}\right)$$ depends on the data

The first case can be by-passed by good modeling, given enough data. However, it is hardly realistic. On the other hand, second form is a compromise between what is realistic and what is tractable, and will be our focus. In other words, we assume that $$p\left(\tilde{y}=i \mid y^\ast =j, \boldsymbol{x}\right) = p\left(\tilde{y}=i \mid y^\ast =j\right)$$, and hence the label noise only depends on the original class. For example, a _fox_ is much more likely to be mislabeled as a _dog_ than as a _banana_, but any _fox_ is equally likely to be mislabeled as a _dog_. This is a core assumption of confident learning.

<aside>
💡 **Confident learning** is a data-centric and model-agnostic framework. In other words, it allows to use **any** model’s predicted probability to find label errors. It is implemented in the Python library `cleanlab`, in the function `find_label_issues`.

</aside>

Confident learning aims to estimate $$p\left(\tilde{y} \mid y^\ast \right)$$ and $$p\left(y^\ast\right)$$. It does so in one step by estimating the joint distribution $$p\left(\tilde{y} , y^\ast \right)$$ as a contingency table that counts the frequency of samples with each label, correct or incorrect, assigned to each true label. In the absence of **both** model errors and label errors, this would produce a diagonal matrix. And in the absence of label errors only, off-diagonals would correspond to the mislabeled examples. However, in the presence of both types of errors, they need to be disambiguated.

## Confident learning

### Overview

<aside>
💡 A naive way to find label errors is to sort by the loss function (assuming no overfitting), i.e., the examples with the highest loss are likely to be label errors. However, how do we pick an appropriate threshold?

</aside>

In order to estimate $$p\left(\tilde{y}, y^*\right)$$, confident learning requires two inputs: the noisy labels $$\tilde y$$ and the (out-of-sample) predicted probabilities $$\hat p\left(\tilde{y} = i; \boldsymbol{x}, \boldsymbol{\theta}\right)$$.

The difficulty lies in assigning latent labels to the examples. How to know which examples are mislabeled? The key insight is finding a threshold for the algorithm’s confidence on the prediction. That is, if the algorithm is very confident on its prediction, we will assume the prediction was the true label all along. A sensible **per-class** threshold is:

$$
t_j=\frac{1}{\left|\boldsymbol{X}_{\tilde{y}=j}\right|} \sum_{\boldsymbol{x} \in \boldsymbol{X}_{\tilde{y}=j}} \hat{p}(\tilde{y}=j ; \boldsymbol{x}, \boldsymbol{\theta})
$$

In words, if a class is very confidently predicted on average, we will request every example to be predicted with a very high confidence.

For each example in the **out-of-sample** data, we will take the highest probability class, and compare it to its respective threshold $$t_j$$. If the model is not that confident in the class, we will move on to the next most likely label, and so on. That way we will have an estimate of the joint distribution, and have the list of label errors in the dataset. Then, we will re-train exclusively on the clean data!

<aside>
💡 This computation is robust to outliers: examples that are not confidently predicted for any of the classes will be dropped from the contingency table.

</aside>

Confident learning consists of three steps:

1. Estimate noise via counting
2. Clean the data: rank & prune
3. Re-train with the errors removed

### Counting to estimate the noise

Our ultimate goal is to estimate the joint distribution of true and noisy labels, $$\boldsymbol{ Q_{\tilde y, y^\ast}}$$.

The first step is computing a **confident joint** table $$\boldsymbol{ C_{\tilde y, y^\ast} } \in \mathbf N^{m \times m}$$. Diagonal elements count likely correct labels, and non-diagonals count likely label errors. The formal definition is the following:

$$
\boldsymbol{C}_{\tilde{y}, y^*}[i][j]:=\left|\hat{\boldsymbol{X}}_{\tilde{y}=i, y^*=j}\right|
$$

where

$$
\hat{\boldsymbol{X}}_{\tilde{y}=i, y^*=j}:=\left\{\boldsymbol{x} \in \boldsymbol{X}_{\tilde{y}=i}: \hat{p}(\tilde{y}=j ; \boldsymbol{x}, \boldsymbol{\theta}) \geq t_j, \underset{l \in[m]: \hat{p}(\tilde{y}=l ; \boldsymbol{x}, \boldsymbol{\theta}) \geq t_l}{j=\arg \max } \hat{p}(\tilde{y}=l ; \boldsymbol{x}, \boldsymbol{\theta})\right\}
$$

and the threshold is the expected self-confidence for each class:

$$
t_j=\frac{1}{\left|\boldsymbol{X}_{\tilde{y}=j}\right|} \sum_{\boldsymbol{x} \in \boldsymbol{X}_{\tilde{y}=j}} \hat{p}(\tilde{y}=j ; \boldsymbol{x}, \boldsymbol{\theta})
$$

Note that this goes against the common assumption that the true label is $$\tilde y_k = \operatorname*{argmax}_{i \in [m]} \hat{p}(\tilde{y}=j ; \boldsymbol{x}, \boldsymbol{\theta})$$. Instead we compare each prediction to the expected predicted probability. In consequence, this version is more robust. For instance, it accounts for class imbalances. Even if a model is very confident in the majority class prediction, the minority class has a chance. It also excludes ambiguous examples with low predicted probabilities across classes.

The next step is using $$\boldsymbol{ C_{\tilde y, y^\ast} }$$ to estimate $$\boldsymbol{ Q_{\tilde y, y^\ast} }$$:

$$
\hat{\boldsymbol{Q}}_{\tilde{y}=i, y^*=j}=\frac{\frac{\boldsymbol{C}_{\tilde{y}=i, y^*=j}}{\sum_{j \in[m]} \boldsymbol{C}_{\tilde{y}=i, y^*=j}} \cdot\left|\boldsymbol{X}_{\tilde{y}=i}\right|}{\sum_{i \in[m], j \in[m]}\left(\frac{\boldsymbol{C}_{\tilde{y}=i, y^*=j}}{\sum_{j^{\prime} \in[m]} \boldsymbol{C}_{\tilde{y}=i, y^*=j^{\prime}}} \cdot\left|\boldsymbol{X}_{\tilde{y}=i}\right|\right)}
$$

### Clean the data: rank & prune

Given $$\boldsymbol{ C_{\tilde y, y^\ast} }$$ and $$\boldsymbol{ Q_{\tilde y, y^\ast} }$$, any rank and prune approach can be used to clean the data.

# Dataset creation and curation

A data-related issue in machine learning is **selection bias**. This happens when the training and the real-world distributions do not match. This can happen for several reasons.

| **Reason**                                          | **Validation set**         |
| --------------------------------------------------- | -------------------------- |
| **Time:** the training data only considers the past | Most **recent** data       |
| **Overfitting:** e.g., we “over-curate” our data    |                            |
| **Rare** events that do not make it to our dataset  | **Oversample** rare events |
| Collecting some datapoints is **inconvenient**      |                            |

Selection bias can lead to **spurious correlations**, in which our models pick up “shortcuts”, rather than important features. To avoid this, we will pick the **validation data** which most closely resembles the real-world distribution.

## Estimating the amount of data required

Say we want to achieve 95% classification accuracy. What is the required sample size? To find out, we can train our model on increasingly lager subsamples to study how performance improves, then extrapolate to the desired accuracy. Do multiple samples per sample size to account for sampling. However, extrapolation is hard when the required number of samples is much larger than the studied sample size. However, a log-linear transformation usually works well in practice.

## Annotator bias

Our labels will often come from a team of annotators. Hence, each training example might have multiple, sometimes contradicting annotations. Moreover, not every annotator sees every sample. We can decide how much to trust each label and each annotator. _The best solution is to make the annotators label a few gold standards, for which we know the true label._

Given such a multi-annotated dataset, our goals are:

- Consensus label: pick the single, best label
- Confidence in the consensus label: how confident we are in that label. That will depend on:
  - The number of annotations for that example
  - The disagreements between the annotations
  - The quality of each annotator
- Quality of each annotator: the overall accuracy of their labels

### Majority vote

Simplest solution: majority vote.

Confidence score: fraction of annotators agreeing on the majority label. Out of those who emitted a label for that training example.

Quality of the annotator: fraction of the annotations that match the majority label.

Problems: it won’t handle well examples with a single annotation examples (low confidence); it does not account for ties, or for the quality of the annotators.

### Classifier as a new annotator

Solution: train a classifier, which predicts each class, and treat it as a separate annotator. We learn on the majority vote annotator. This solves:

- Ties: are broken by the classifier
- Single-annotations: believe the label if the model is also very confident in it

### Crowdlab

Solution: learn the probability distribution of what the true label should be given the example and all the annotations. This will be a weighted average of all the annotations from both the annotators and the classifiers. We will give a lower weight to bad annotators. This is similar to confidence learning. Then, we will prune out the bad annotations, compute majority vote again.

# Data-centric evaluation of models

The focus or data-centric ML is to improve models by improving the data. Crucially, this requires identifying the relevant evaluation metric(s) to improve. Such evaluation metrics (or, simply, metrics) measure prediction of the model on an _unseen_ example by comparing it to its given label. (Although it is ideal to use multiple metrics, multi-objective problems are hard to tackle in practice.) The metric might use the **most likely** predicted class (e.g., accuracy, balanced accuracy, precision, recall); or the predicted **probabilities** for each class (e.g., log loss, auROC, calibration error).

In practice, we will compute this metric for multiple unseen examples, then aggregate them by:

- Averaging the metrics obtained over all the examples
- Averaging the metrics obtained over all the examples _in each class_ (e.g., per-class accuracy)
- Computing the confusion matrix. Although attractive, since we access more granularity, it is hard to compare the matrices produced by different models.

> 💡 Invest as much time thinking about model evaluation as about the models themselves. It has **huge** impact in real world applications. I list some common pitfalls below.

## Common pitfalls in model evaluation

- Data leakage: we fail to use truly held-out data. This happens, for instance when we use data from multiple data sources which contain the same training examples.
- Misspecified metric: reporting only the average loss can under-represent severe failures on certain subgroups.
- Selection bias: the validation data is not representative of the deployment setting.
- Annotation error: some labels are incorrect.

### Underperforming in subpopulations

A **data slice** is a subset of the data that shares a common characteristic (a.k.a., cohort, subpopulation or subgroup), like race, gender, age or socioeconomics. Although it might be tempting to exclude such slicing features, doing that would make the model learn it anyway from other correlated features. Hence, keeping them allows to at least asses how the model performs on the different slices.

Ideally, model predictions should not change across slices. When that does not happen, we can tackle that in different ways:

1. Use a more flexible model, that can capture more complex relationships
2. Over-sample/up-weight the examples from the minority group receiving poor predictions
3. Collect additional data from the group with poor performance, probably after some experiments showing how more data is helpful
4. Measure or create additional features that allow the model to perform better on the slice

In any way, the first step is to find out the subpopulations on which we are underperforming.

# Further reading

- [MIT: Introduction to Data-Centric AI](https://dcai.csail.mit.edu/)
