---
layout: post
title: Fixed effects
date: 2025-04-26 11:59:00
description: TBD
tags: statistics genetics
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

**Metaanalyses** are a powerful tool for scientists. They allow us to aggregate the results from multiple studies on the same topic and reach a more robust conclusion. In genetics, metaanalyses are commonly applied to **genome-wide association studies (GWAS)**. GWASs look for associations between millions of genetic variants spread across the genome and a trait. In their most basic form, they do so by fitting a linear regression model to each variant:

$$
\mathbf{y} = \beta_0 + \beta_1 \mathbf{x} + \epsilon
$$

where $\mathbf{y}$ is the trait, $\beta_0$ is the intercept, $\beta_1$ is the effect size of the variant, $\mathbf{x}$ is the genotype of the variant, and $\epsilon$ is the error term. The model assumes that the error term follows a normal distribution with mean 0 and variance $\sigma^2$. Typically, the genotype is coded as 0, 1, or 2, depending on the number of minor alleles carried by the individual. The effect size $\beta_1$ is estimated using maximum likelihood estimation (MLE) or restricted maximum likelihood estimation (REML).

# Fixed effects

$$
y_{ij} = \beta_{0j} + \beta_1 x_{ij} + \epsilon_{ij}
$$

# Random effects

$$
y_{ij} = \beta_0 + \beta_1 x_{ij} + u_j + \epsilon_{ij}
$$
