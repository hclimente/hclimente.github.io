---
layout: post
title: Cross-entropy
date: 2025-07-14 11:59:00-0000
description: The secret sauce of machine learning
tags: information_theory machine_learning
giscus_comments: true
related_posts: false
---

In popscience, **entropy** is considered a measure of _disorder_: a system has high entropy when it is disordered (e.g., my college bedroom), and low entropy when it is ordered (e.g., a chocolate box). This meaning probably has its roots in thermodynamics, where my college bedroom was evidence of the universe getting ever closer to its [heat death](https://en.wikipedia.org/wiki/Heat_death_of_the_universe).

In this article I will focus on _Shannon's_ entropy is a property of probability distributions. For a discrete random variable:

$$
H(P) = \sum_{i=0}^n p_i \log_2 \frac{1}{p_i}.
$$

(As stated above, using the binary logarithm, entropy is measured in [bits](https://en.wikipedia.org/wiki/Bit); when using the natural logarithm instead, the unit of measure is nats. For the remainder of this post, I will only use the binary logarithm, and omit the subscript.)

# What does entropy _really_ mean?

Entropy is the expected value of a quantity ($$\log \frac{1}{p_i}$$) that reflects how surprising an outcome is: if an outcome is rare ($$p_i$$ is small), observing it should be quite surprising ($$\log \frac{1}{p_i}$$ is large); if it is very common, the surprise should be low.

<!-- Why the logarithm? Probabilities multiply, surprises add up (?) -->

A more tangible interpretation of entropy links it to the _encoding_ of a message. Imagine we want to encode the outcome of a probability distribution. We observe an outcome, and want to let our colleague know in a unambiguous fashion. For instance, let's say the weather in my city follows the following probability distribution:

| Weather | Probability |
| ------- | ----------- |
| Cloudy  | 0.5         |
| Rainy   | 0.4         |
| Sunny   | 0.1         |

(Yes, I live in the UK.)

For easier computations, let's also assume that probabilities remain independent and constant over time. Which, again, it's not too far from my reality.

Every morning, I will look out the window exactly at 9am, and send my colleague the weather report. Of course, we could just text them "cloudy", "rainy" or "sunny" as appropriate. If we encode it using [ASCII](https://en.wikipedia.org/wiki/ASCII):

| Weather | Probability | Binary string (length)                                |
| ------- | ----------- | ----------------------------------------------------- |
| Cloudy  | 0.5         | 011000110110110001101111011101010110010001111001 (48) |
| Rainy   | 0.4         | 0111001001100001011010010110111001111001 (40)         |
| Sunny   | 0.1         | 0111001101110101011011100110111001111001 (40)         |

In the long term, the average message will have taken $$0.5 \times 48 + 0.4 \times 40 + 0.1 \times 40 = 44$$ bits. Not a big deal I guess... But we can do much better! For instance, we can associate each string to an integer (8 bits), or even better, to a binary string. To this end, we need to generate a [prefix code](https://en.wikipedia.org/wiki/Prefix_code).

The [Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding) is a common solution to this problem which significantly shortens our average message:

| Weather | Probability | Binary string (length) |
| ------- | ----------- | ---------------------- |
| Cloudy  | 0.5         | 0 (1)                  |
| Rainy   | 0.4         | 10 (2)                 |
| Sunny   | 0.1         | 11 (2)                 |

This is much better: we have gone from 44 bits to $$0.5 \times 1 + 0.4 \times 2 + 0.1 \times 2 = 1.5$$ bits on average. Of course, for this to be possible, both our colleague and us need to _know_ what the true weather distribution is. Otherwise, our bit allocation won't be optimal.

Are we satisfied yet? Well... no, and this is where entropy kicks in. Note that our message length will be the same on rainy days and on sunny days. However, rainy days are 4 times more common! Wouldn't it make sense to associate common outcomes to shorter strings and rarer outcomes to longer strings? But here's the problem: our strings are already quite short. In fact, it's impossible to create an even shorter prefix code if we intend to send one weather report each morning.

However, it is possible to go lower if we are willing to instead send our weather reports in batches. Imagine we only want to batch-send the weather report every 10 days. There are $$3^{10}$$ possible outcomes and hence messages to send. The most likely one is a streak of one 10 cloudy days, which will occur with probability $$0.5^{10}$$. The Huffman coding will assign a much shorter codeword to this string than to the most unlikely string, a streak of one 10 sunny days:

| 10-day weather | Probability | Length of the binary string |
| -------------- | ----------- | --------------------------- |
| CCCCCCCCCC     | 9.77e-04    | 11                          |
| CCCCCCCCCR     | 7.81e-04    | 11                          |
| CCCCCCCCRC     | 7.81e-04    | 11                          |
| ...            | ...         | ...                         |
| SSSSSSSSSR     | 4.00e-10    | 32                          |
| RSSSSSSSSS     | 4.00e-10    | 33                          |
| SSSSSSSSSS     | 1.00e-10    | 33                          |

The average length of this code is 14.6 bits, or 1.46 bits per day. Slightly better! And it's easy to see how, as we keep batching more and more days together, each single day requires less and less bits. And this brings us to the key point: entropy represents the lower bound for the average message length required to encode each outcome of a random process. Even with the optimal encoding and incredibly large batches, we can't do better than entropy. In the case of our distribution:

$$
H(P) = 0.5 \log \frac 1 {0.5} + 0.4 \log \frac 1 {0.4} + 0.1 \log \frac 1 {0.1} = 1.36
$$

The Huffman encoding was doing pretty well after all!

# Cross-entropy

We just saw how knowing the underlying probability distribution gave us an edge in encoding the outcomes efficiently. However, here in the real world we rarely have access to _true_ probability distributions, if such a thing even exists. At most, we have access to our best guess of what the true probability distribution is. For instance, we rely on very complex models to accurately predict the weather. But let's leave these aside, and let's use my mental model of how weather works as learnt in Barcelona, Spain:

| Weather | Probability |
| ------- | ----------- |
| Cloudy  | 0.2         |
| Rainy   | 0.1         |
| Sunny   | 0.7         |

As you can imagine, after moving to London, my mental model of weather was not that useful. While I was developing my own model, I often experienced _surprise_, as outcomes that should be rare were quite common, and viceversa. Entropy quantified our average surprise when observing a distribution's outcomes while knowing the true distribution. Similarly, the **cross-entropy** measures our surprise when observing a distribution's outcomes while only having a _model_ of the true distribution. If $$p$$ is the true distribution and $$q$$ is our model of the world:

$$
H(p, q) = \sum_x p(x) \log \frac{1}{q(x)}.
$$

Just like entropy, $$\log \frac{1}{q(x)}$$ measures the degree of surprise (higher is more surprising), which is weighted by the frequency with which we observe the outcome. It is also measured in bits. And it can also be interpreted as the average number of bits required to encode the outcome from $$p$$ while using a code generated using $$q$$, leveraging probability in the same ways we saw above.

> Note that order matters! $$H(p, q) \neq H(q, p)$$.

This is the cross-entropy of my weather model right after landing in London:

$$
H(\text{weather}_\text{London}, \hat{\text{weather}}_\text{Barcelona}) = 0.5 \log 1 0.2 + 0.4 \log 1 0.1 + 0.1 \log 1 0.7 = 2.54.
$$

However, after a few years in London my model became quite accurate, to the extent that $$\hat{\text{weather}}_\text{London} = \text{weather}_\text{London}$$:

$$
H(\text{weather}_\text{London}, \hat{\text{weather}}_\text{London}) = 0.5 \log 1 0.5 + 0.4 \log 1 0.4 + 0.1 \log 1 0.1 = 1.36.
$$

This is an important result: $$H(p, q) \geq H(p, p) = H(p)$$.

# Why this all matters?

We have talked a great deal about random distributions, processes, encodings and so on. However one of the most important applications of cross-entropy is machine learning. Many machine learning algorithms aim at learning a probability distribution. For instance, discriminative models aim to learn the probability of an outcome given some input data ($$P(Y \mid X)$$), e.g., what's the likelihood of a disease given that some blood test came positive?. Generative models aim to learn the probability distribution of the data itself ($$P(X, Y)$$), e.g., a language models like GPT can generate novels, answer questions, software or legal documents. During model _training_ these models aim to learn a distribution that minimizes the cross-entropy between their model of the data, and the data itself.

# Further readings

- [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)
