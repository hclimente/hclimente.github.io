---
layout: post
title: Random walks and Markov chains
date: 2025-01-27 11:59:00-0000
description: PageRank, MCMC, and others
tags: graphs
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

# Random walk

A **random walk (RW)** is a [stochastic process](https://en.wikipedia.org/wiki/Stochastic_process) in which a walker, located in one of the graph's vertices, moves to one of the current vertex's neighbors at each timestep. Often the transition probability between vertices is represented by the matrix $$A_{rw}$$, a normalized version of the [adjacency]({% post_url 2025-01-23-graphs-linear-algebra %}#adjacency-matrix) in which the weight of all outbound edges adds up to 1. The outcome of a single random walk is a [walk]({% post_url 2025-01-23-graphs-glossary %}#walk) of length $$t$$, where $$t$$ represents the number of timesteps. We can consider the outcome of infinitely many random walkers departing from a given vertex. In that case, the outcome at time $$t$$ is a vector $$n \times 1$$ $$\pi_t$$, in which $$\pi_{ti}$$ represents the probability of the walker starting at a given vertex and being on vertex $i$ at time $t$. The probability distribution at time $$t+1$$ is computed as:

$$
\pi_{t+1} = A_{rw} \pi_t
$$

In the case in which the graph has a single [component]({% post_url 2025-01-23-graphs-glossary %}#component), if we allow the RW to run indefinitely, the probability of being at any given vertex is uniform (TODO Why?):

$$
\lim_{t \to \inf} \pi_{t} = \frac 1 n
$$

Because of this property, the root vertex is not important in the long run.

In the **random walk with restart (RWR)**, the walker can return to its root vertex with a restart probability $$r \in [0, 1]$$. In this case:

$$
\pi_{t+1} = r \pi_0 + (1 - r) A_{rw} \pi_t
$$

where $$\pi_0$$ represents the probability of starting at each vertex. If $$r = 0$$, the walker will never be teleported back to the root, and a RW is equivalent to a RWR. If $$r = 1$$, the walker will not be allowed to move out of the root, and $$\pi_t = \pi_0$$. But, for certain values of $r$, the walker is allowed to explore the root's neighborhood before teleporting back. If the root is part of a [module]({% post_url 2025-01-23-graphs-glossary %}#module), the walk will mostly happen within that module. If the root is very central, the walker will explore many parts of the network. Importantly, the RWR has a stationary distribution $$\pi$$ which is not necessarily uniform:

$$
\lim_{t \to \inf} \pi_{t} = \pi
$$
