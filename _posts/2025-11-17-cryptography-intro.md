---
layout: distill
title: Digital security
date: 2025-11-17 11:59:00 +0000
description: Cocktail parties make me anxious
tags:
  - algorithms
  - computer_science
  - cryptography
giscus_comments: true
related_posts: false
---

If you've ever read anything about online security and privacy, you'll have quickly felt drowning in a soup of letters in which "S"s are definitely overrepresented. Often there are also numbers, which are sometimes, but not always, powers of 2. You will also have heard many words that seem kind of the same, but also kind of different: keys, certificates, signatures, passkeys. No wonder most of us give up before even trying.

In reality, modern cryptography is built around a few, well-trusted algorithms. Everything else are wrappers to adapt them to specific applications (or legacy algorithms we should ditch as soon as possible!). Each application tries to address one or more of these goals:

- __Confidentiality__: our data and communications remain private
- __Authentication__: we are who we say we are
- __Integrity__: our data and communications are not tampered with
- __Non-repudiation__: we cannot deny having authored a message

Let's see how each of them applies to several day-to-day online activities:

| Scenario              | Confidentiality | Authentication | Integrity | Non-repudiation |
|-----------------------|-----------------|----------------|-----------|-----------------|
| Web browsing          | ✓               |                | ✓         |                 |
| Login/email           | ✓               | ✓              | ✓         |                 |
| Local storage         | ✓               | ✓              |           |                 |
| Software downloads    |                 | ✓              | ✓         |                 |
| File transfers        |                 |                | ✓         |                 |
| Online payments       | ✓               | ✓              |           | ✓               |
| SSH access            | ✓               | ✓              | ✓         |                 |
| `git commit`          |                 |                | ✓         | ✓               |
| `git push`            | ✓               | ✓              | ✓         | ✓               |

In these series, I go over the main algorithms behind each goal, and how I use them to stay safe. If you don't care about the theory, simply skip to the TL;DR of each section. This won't protect you from either state actors or _wrench cryptanalysis_, but should be more than enough for 99% of us.

![](https://imgs.xkcd.com/comics/security.png)
<div class="caption" align="center">
    From <a href="https://xkcd.com/538">xkcd</a>.
</div>

> Throughout the series, I'll be using _message_ to mean _data_ or _information_. This should bring to mind a more concrete picture, but the contents go well beyond bantering on WhatsApp.

Our heroes in this story will be [Alice and Bob](https://en.wikipedia.org/wiki/Alice_and_Bob). Alice and Bob just want to talk to each other without being snooped in by their evil counterparts, Eve and Mallory.

# Prelude: finite field arithmetic

Digital cryptography works with __integers__, not with real numbers. That's because cryptography often relies on a fixed series of calculations to reliably reaching the same conclusion. Each floating point operation incurs in a precision cost, and hence they cannot guarantee that.

However, to make the maths work, we need to rely on __finite fields__. Let's unpack what that means.

First, the fields we work with are __finite__, that is, they operate on a bounded set of items. In our case a set of integers. The size of the set is called the _order_ of the field.

Second, they are __fields__, that is, they have 4 binary operators (multiplication, addition, subtraction and division) satisfying the field axioms. An important one is _closure_, i.e., the result of the operation must also be in the field.

The classical example is the set of integers modulo $$p$$, with $$p$$ being a prime:

$$
\mathbb{Z} / p \mathbb{Z} = \{ \bar{a}_p | a \in \mathbb{Z} \}
$$

where $$\bar{a}_p$$ represents the entire set of integers that produce the same remainder as $$a$$ when divided by $$p$$.
