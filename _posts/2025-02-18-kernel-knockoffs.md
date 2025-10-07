---
layout: post
title: Knockoffs
date: 2025-02-18 12:00:00-0000
description: FDR-controlled feature selection
tags: fdr knockoffs feature_selection
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

In many scientific applications, the goal is to discover which features are truly associated with an outcome. The [false discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate) (FDR) is defined as the expected proportion of false positives among the selected features. Controlling the FDR is less conservative than controlling the [family-wise error rate](https://en.wikipedia.org/wiki/Family-wise_error_rate), often leading to more discoveries.

When dealing with statistical tests, Benjamini–Hochberg and Benjamini–Yekutieli are common procedures to keep the FDR below a level $$\alpha$$. However, such strategies rely on certain assumptions; for instance, that P-values are well-calibrated or that tests have certain correlation structures. If these are not met, the statistical guarantees on FDR control are also out the window. Furthermore, they require having P-values to work with; in many cases we just want to control the FDR of selected features, but do not have a well-characterized null hypothesis. For instance, given a set of active features in Lasso, how can we make sure the fraction of non-explanatory features is controlled? In such cases, _knockoffs_ can be helpful.

# Knockoffs

The **knockoff** filter is a procedure to perform feature selection while keeping the FDR controlled. Given an outcome $$\mathbf{Y}$$ and a feature matrix $$X$$, the goal is to select a subset of features $$X_S$$ such that

$$
Y \perp X_{-S} \mid X_S
$$

The procedure computes and leverages a new matrix $$\tilde{X}$$, with the same dimensions as $$X$$, containing "knockoff" copies of the original features. Each original variable $$\mathbf{X_i}$$ has its own knockoff $$\mathbf{\tilde{X}_i}$$. These knockoffs are engineered to mimic the correlation structure of the original features: for any $$i \neq j$$, $$\rho(\mathbf{X_i}, \mathbf{X_j}) = \rho(\mathbf{X_i}, \mathbf{\tilde{X}_j}) = \rho(\mathbf{\tilde{X}_i}, \mathbf{\tilde{X}_j})$$. Also, knockoffs are created without using $$\mathbf{Y}$$. Hence, conditional on $$X$$, $$Y$$ is independent of $$\tilde{X}$$.

There are two paradigms to model knockoffs: Model-X and Fixed-X.

The **Model-X** paradigm assumes that the explanatory variables are random variables with a known joint distribution. Although theoretically appealing, this assumption can be impractical for real-world data, since we do not know the data generating function $$F_X$$. For that reason, I will ignore it for the remainder of this discussion.

The **Fixed-X** paradigm makes no assumptions on the distribution of the explanatory variables. Instead, they can be treated as fixed quantities. This makes it more applicable in practice. However, it imposes three important restrictions:

- $$F_{Y \mid X}$$ must be linear and homoscedastic
- The problem must be low dimensional (number of samples $$>$$ number of features)
- The statistics $$D(X_i, Y)$$ and $$D(\tilde{X}_i, Y)$$ must satisfy additional requirements (see references)

# The knockoff procedure

Intuitively, by comparing the association measure computed for each original feature against its knockoff, one can determine which features provide true signals. Specifically, the knockoff-based feature selection consists of four steps.

## 1. Generate knockoffs

Generating features that are independent of $$Y$$ is easy: simply generate them without looking at the outcome. Generating features that, additionally, __look like__ the original features is considerably more challenging. The two most important families of methods are Gaussian and deep learning methods.

In a nutshell, **Gaussian** methods assume that $$X$$ comes from a multivariate Gaussian distribution. If we can parametrize this distribution, we can sample from it.

Create synthetic copies of the features that retain the original correlation structure without any outcome information. An obvious question is how to synthesize such knockoff copies.

## 2. Compute association measures

For each feature, calculate the association measure $$D(\mathbf{Y}, \mathbf{X_k})$$ and its counterpart $$D(\mathbf{Y}, \tilde{\mathbf{X}}_k)$$ on the knockoff.

Kernel-based measures are powerful tools for detecting complex, non-linear dependencies:

- **HSIC (Hilbert-Schmidt Independence Criterion):** Computes the covariance between kernel-transformed versions of the feature and the outcome, capturing a broad range of dependency structures.
- **Conditional MMD (cMMD):** Assesses the difference between the conditional distribution of a feature given the outcome and its marginal distribution. This measure is particularly useful when dealing with categorical outcomes.
- **TR Measure:** A linear combination of Kendall’s τ and Spearman’s ρ, designed to effectively capture associations in both continuous and discrete data.

These measures satisfy the sure independence screening property under bounded kernel conditions—meaning that, with high probability, the truly active features are recovered when a proper threshold is used.

> A potential limitation of kernel knockoffs is its sometimes overly conservative nature. To keep the FDR low, the procedure may end up selecting very few—or even no—features. This suggests that the chosen association measure might not be sufficiently sensitive. One possible remedy is to explore alternative kernel choices or optimize feature screening steps before applying knockoff filtering.

## 3. Compute the knockoff statistic

Define the statistic as $$w_k = D(Y, X_k) - D(Y, \tilde{X}_k)$$. A larger $$w_k$$ indicates stronger evidence that the original feature is associated with the outcome.

## 4. Select a threshold and select features

Identify the smallest threshold $$t$$ such that $$\frac{\#\{w_k \le -t\}}{\#\{w_k \ge t\}} \le \alpha$$ where $$\alpha$$ is the desired FDR level. Retain all features with $$w_k \ge t$$.

# Screening in High Dimensions

A notable challenge arises when the number of features $$p$$ is large compared to the sample size $$n$$ (i.e., $$2p>n$$). In such high-dimensional settings, constructing knockoffs directly is infeasible. A common workaround is to:

- Pre-screen Features: Use a subset of the data to rank and reduce the feature set.
- Apply the knockoff filter: With the reduced set of features, generate knockoffs using the remaining samples (ensuring $$m > 2d$$, where $$d$$ is the number of features after screening).

This two-step approach helps maintain statistical power while ensuring robust FDR control.

# References

- [Variable Selection with Knockoffs](https://web.stanford.edu/group/candes/knockoffs/)
- [B. Poignard, P. J. Naylor, H. Climente-González, M. Yamada, in International Conference on Artificial Intelligence and Statistics (PMLR, 2022), pp. 1935–1974.](https://proceedings.mlr.press/v151/poignard22a.html)
