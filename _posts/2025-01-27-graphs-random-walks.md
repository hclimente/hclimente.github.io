---
layout: post
title: Random walks and Markov chains
date: 2025-01-27 11:59:00-0000
description: PageRank, MCMC, and others
tags: graphs random_walks linear_algebra
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

# Random walk

A **random walk (RW)** is a [stochastic](https://en.wikipedia.org/wiki/Stochastic_process), discrete process. At a given time step the walker is located in one of the graph's vertices; in the next time step, it will move to one of the current vertex's neighbors. Often the transition probability between vertices is represented by the **transition** matrix $$P$$, a normalized version of the [adjacency]({% post_url 2025-01-23-graphs-linear-algebra %}#adjacency-matrix) in which the weights of all outbound edges add up to 1:

$$
P = D^{-1} A
$$

The outcome of a single random walk is a [walk]({% post_url 2025-01-23-graphs-glossary %}#walk) of length $$t$$, where $$t$$ is the number of steps.

We can consider the outcome of infinitely many random walkers departing from a given vertex. In that case, the outcome at time $$t$$ is a vector $$\mathbf{\pi}_t$$ of dimension $$n \times 1$$, in which $$\pi_{ti}$$ represents the probability of the walker starting at a given vertex and being on vertex $i$ at time $t$. The probability distribution at step $$t+1$$ is computed as:

$$
\mathbf{\pi}_{t+1} = P \mathbf{\pi}_t
$$

In the case in which the graph has a single [component]({% post_url 2025-01-23-graphs-glossary %}#component), if we allow the RW to run indefinitely, the probability of being at any given vertex is uniform. Because of this property, the initial vertex is not important in the long run.

To understand why this happens, let's expand what happens at each step of a random walk starting at vertex $$i$$:

- At step 0, $$\mathbf{\pi}_0 = (0, 0, \cdots, 1, \cdots, 0)^\intercal$$, i.e., a $$0$$-vector almost everywhere, with a $$1$$ at position $$i$$.
- At step 1, $$\mathbf{\pi}_{1} = P \mathbf{\pi}_0$$
- At step 2, $$\mathbf{\pi}_{2} = P \mathbf{\pi}_1 = P (P \mathbf{\pi}_0) = A^2_{rw} \mathbf{\pi}_0$$
- At step 3, $$\mathbf{\pi}_{3} = P \mathbf{\pi}_2 = P (A^2_{rw} \mathbf{\pi}_0) = A^3_{rw} \mathbf{\pi}_0$$
- …
- At step $$t$$, $$\mathbf{\pi}_{t} = A^t_{rw} \mathbf{\pi}_0$$

When taking powers of a matrix, it is useful to use its [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix). After computing the eigenvectors ($$\mathbf{u}_1, \cdots, \mathbf{u}_n$$) and the eigenvalues ($$\lambda_1, \cdots, \lambda_n$$) of $$P$$, we first expand $$\mathbf{\pi}_0$$ in the eigenbasis:

$$\mathbf{\pi}_0 = c_1 \mathbf{u}_1 + c_2 \mathbf{u}_2 + \cdots + c_n \mathbf{u}_n$$

Then, for an arbitrary step $$t$$:

$$
\begin{multline*}
  \mathbf{\pi}_{t} = A^t_{rw} \mathbf{\pi}_0 \\
  = A^t_{rw} (c_1 \mathbf{u}_1 + \cdots + c_n \mathbf{u}_n) \\
  = c_1 A^t_{rw} \mathbf{u}_1 + \cdots + c_n A^t_{rw} \mathbf{u}_n \\
  = c_1 \lambda^t_1 \mathbf{u}_1 + \cdots + c_n \lambda^t_n \mathbf{u}_n
\end{multline*}
$$

Note that the [Laplacian]({% post_url 2025-01-23-graphs-linear-algebra %}#normalized-laplacian-matrices) and the transition matrices are deeply related:

$$
L_{rw} = D^{-1}L = D^{-1}(D - A) = I - P
$$

In fact, their eigenvalues are connected: $$\lambda_i(L_{rw}) = 1 - \lambda_i(P)$$. Since the [smallest eigenvalue of $$L_{rw}$$ is 0]({% post_url 2025-01-23-graphs-linear-algebra %}#connectivity-of-the-graph), the largest eigenvalue of $$P$$ is $$1$$, corresponding to $$\mathbf{u}_1 = \left(\frac {1} {\sqrt{1}}, \cdots, \frac 1 {\sqrt{1}} \right)$$ (TODO Why?). Similarly, the remaining eigenvalues are positive and strictly less than 1. Hence,

$$
\lim_{t \to \infty} \mathbf{\pi}_{t} = \lim_{t \to \infty} c_1 \lambda^t_1 \mathbf{u}_1 + \cdots + c_n \lambda^t_n \mathbf{u}_n = c_1 \mathbf{u}_1 = \left(\frac {1} {\sqrt{1}}, \cdots, \frac 1 {\sqrt{1}} \right) \blacksquare
$$

There are several remarks we can do:

- This result holds regardless of what the starting node is. In fact, $$\pi_0$$ could be a probability distribution over the nodes.
- The *speed* at which the distribution converges depends on the eigenvalues of $$P$$. Specifically, if $$\lambda_2$$ is close to 1, the convergence will be slow.

# Random walk with restart

In the **random walk with restart (RWR)**, the walker can return to its root vertex with a restart probability $$r \in [0, 1]$$:

$$
\mathbf{\pi}_{t+1} = r \mathbf{\pi}_0 + (1 - r) P \mathbf{\pi}_t
$$

where $$\mathbf{\pi}_0$$ represents the probability of starting at each vertex. If $$r = 0$$, the walker will never be teleported back to the root, and a RW is equivalent to a RWR. If $$r = 1$$, the walker will not be allowed to move out of the root, and $$\mathbf{\pi}_t = \mathbf{\pi}_0$$. But, for certain values of $r$, the walker is allowed to explore the root's neighborhood before teleporting back. If the root is part of a [module]({% post_url 2025-01-23-graphs-glossary %}#module), the walk will mostly happen within that module. If the root is very central, the walker will explore many parts of the network.

Importantly, the RWR has a stationary distribution $$\pi$$ which is not necessarily uniform:

$$
\lim_{t \to \infty} \mathbf{\pi}_{t} = \pi
$$

# Further reading

- [Full title: The Unreasonable Effectiveness of Spectral Graph Theory: A Confluence of Algorithms, Geometry, and Physics](https://www.youtube.com/watch?v=8XJes6XFjxM)