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

(As stated above, using the binary logarithm, it is measured in [bits](https://en.wikipedia.org/wiki/Bit); when using the natural logarithm instead, the unit of measure are nats. For the remaining of this post, I will only use the binary logarithm, and omit the subindex.)

# Interpreting entropy

Entropy is just an average of a outcome-related quantity ($$\log \frac{1}{p_i}$$), in which each outcome is weighted by its probability. This "outcome-related quantity" tries to capture the degree of "surprise" we experience from observing that outcome being realized: if an outcome is rare ($$p_i$$ is small), observing it should be quite surprising ($$\log \frac{1}{p_i}$$ is large); if it is very common, the surprise should be low.

<!-- Why the logarithm? Probabilities multiply, surprises add up (?) -->

A more accurate interpretation of entropy links it to the _encoding_ of a message. Imagine we want to encode the outcome of a probability distribution. We observe an outcome, and want to let our colleague know in a unequivocal fashion. For instance, let's say the weather in my city follows the following probability distribution:

| Weather | Probability |
| ------- | ----------- |
| Cloudy  | 0.5         |
| Rainy   | 0.4         |
| Sunny   | 0.1         |

(Yes, I am based in the UK.)

Every morning, I will look out the window at 9am exactly, and give my colleague the weather report. Of course, we could just text them "cloudy", "rainy" or "sunny" as appropriate. If we encode it using [ASCII](https://en.wikipedia.org/wiki/ASCII):

| Weather | Binary string                                    |
| ------- | ------------------------------------------------ |
| Cloudy  | 011000110110110001101111011101010110010001111001 |
| Rainy   | 0111001001100001011010010110111001111001         |
| Sunny   | 0111001101110101011011100110111001111001         |

In the long term, the average message will have taken $$0.1 \dot 40 + 0.5 \dot 48 + 0.4 \dot 40 = 44$$ bits. Not a big deal I guess... but we can do much better! For instance, we can associate each sting to an integer (8 bits), or even better, to a binary string. To this end, we need to generate

The [Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding) is a common solution to this problem which significantly shortens our average message:

| Weather | Binary string |
| ------- | ------------- |
| Cloudy  | 00            |
| Rainy   | 10            |
| Sunny   | 01            |

This is much better: we have gone from 44 bits to $$0.1 \dot 2 + 0.5 \dot 2 + 0.4 \dot 2 = 2$$ bits on average.

Are we satisfied yet? Well... no, and this is where entropy kicks in. Note that our message length will be the same on rainy days and on sunny days. However, rainy days are 4 times more common! Wouldn't it make sense to associate common outcomes to shorter strings and rarer outcomes to longer strings? Of course, for this to be possible, both our colleague and us need to _know_ what the true weather distribution is. Otherwise, our bit allocation won't be optimal. But, if you grant me that, this is the encoding I would suggest:

| Weather | Binary string |
| ------- | ------------- |
| Cloudy  | 0             |
| Rainy   | 1             |
| Sunny   | 00            |

This encoding will result in an average message length of $$0.5 \dot 1 + 0.4 \dot 1 + 0.1 \dot 2 = 1.1$$ bits.

Now let's compute the entropy of this distribution:

$$
0.5 log2 (1/0.5) + 0.4 log2 (1/0.4) + 0.1 log2 (1/0.1) = 1.36
$$

Wait, this is actually higher! How come? I thought the entropy is the average length if the optimal encoding.

# Cross-entropy

The entropy

# Further readings

- [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)
