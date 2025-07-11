---
layout: post
title: Cross-entropy
date: 2025-07-10 11:59:00-0000
description: The secret sauce of machine learning
tags: information_theory machine_learning
giscus_comments: true
related_posts: false
---

In popscience, **entropy** is considered a measure of _disorder_: a system has high entropy when it is disordered (e.g., my college bedroom), and low entropy when it is ordered (e.g., a chocolate box). This understanding probably comes from thermodynamics, where my college bedroom was evidence of the universe getting ever closer to its [heat death](https://en.wikipedia.org/wiki/Heat_death_of_the_universe).

In this article I will focus on _Shannon's_ entropy a property of probability distributions. For a distrete random variable:

$$
H(X) = \sum_{i=0}^n p_i \log_2 \frac{1}{p_i}.
$$

As stated above, using the binary logarithm, it is measured in [bits](https://en.wikipedia.org/wiki/Bit); when using the natural logarithm instead, the unit of measure are nats. For the remaining of this post, I will only use the binary logarithm, and omit the subindex.
