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

A **random walk (RW)** is a [stochastic](https://en.wikipedia.org/wiki/Stochastic_process), discrete process. At a given time step, the walker, located in one of the graph's vertices, picks one of its neighbors at random and moves to it. Often the transition probability between vertices is represented by the **transition** matrix $$P$$, a normalized version of the [adjacency]({% post_url 2025-01-23-graphs-linear-algebra %}#adjacency-matrix) in which the weights of all outbound edges add up to 1:

$$
P = D^{-1} A
$$

The outcome of a single random walk is a [walk]({% post_url 2025-01-23-graphs-glossary %}#walk) of length $$t$$, where $$t$$ is the number of steps.

We can consider the outcome of infinitely many random walkers departing from a given vertex. In that case, the outcome at time $$t$$ is an $$n$$-dimensional column vector $$\mathbf{\pi}_t$$ in which $$\pi_{ti}$$ represents the probability of the walker starting at a given vertex and being on vertex $i$ at time $t$. The probability distribution at step $$t+1$$ is computed as:

$$
\mathbf{\pi}_{t+1} = \mathbf{\pi}_t P
$$

If we allow the RW to run indefinitely, the probability of ending up at each vertex reaches a stationary distribution $$\pi$$ such that the probability of being in vertex $$i$$

$$
\pi_i = \frac {d_i} {\sum_j d_j}.
$$

That is, high [degree]({% post_url 2025-02-09-graph-properties %}#regular) vertices are more likely to be visited. If the graph is [regular]({% post_url 2025-01-23-graphs-glossary %}#regular), the stationary distribution is uniform. Because of this property, the initial vertex is not important in the long run: if we allow the RW to run indefinitely, the probability of being at any given vertex is uniform.

> ***Lazy* random walks:** A stationary distribution does not always exists. For instance, consider the case of a random walk on a [bipartite]({% post_url 2025-01-23-graphs-glossary %}#bipartite) graph: at step $$t$$ the walker will be on one side or another, depending on the initial vertex and the parity of $$t$$. Such cases have a stationary distribution under the [*lazy* random walk](https://ocw.mit.edu/courses/18-409-topics-in-theoretical-computer-science-an-algorithmists-toolkit-fall-2009/100377025e8520aab9f61d8585e71cc5_MIT18_409F09_scribe4.pdf), in which the walker has a probability $$\frac 1 2$$ of remaining at the current vertex, and a probability $$\frac 1 2$$ of leaving it.

{% details Proof %}

To understand why this happens, let's expand what happens at each step of a random walk starting at vertex $$i$$:

- At step 0, $$\mathbf{\pi}_0 = (0, 0, \cdots, 1, \cdots, 0)^\intercal$$, i.e., a $$0$$-vector almost everywhere, with a $$1$$ at position $$i$$.
- At step 1, $$\mathbf{\pi}_{1} = \mathbf{\pi}_0 P$$
- At step 2, $$\mathbf{\pi}_{2} = \mathbf{\pi}_1 P = (\mathbf{\pi}_0 P) P = \mathbf{\pi}_0 P^2 $$
- At step 3, $$\mathbf{\pi}_{3} = \mathbf{\pi}_2 P = (\mathbf{\pi}_0 P^2) P = \mathbf{\pi}_0 P^3$$
- …
- At step $$t$$, $$\mathbf{\pi}_{t} = \mathbf{\pi}_0 P^t$$

When taking powers of a matrix, it is useful to use its [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix). After computing the eigenvectors ($$\mathbf{u}_1, \cdots, \mathbf{u}_n$$) and the eigenvalues ($$\lambda_1, \cdots, \lambda_n$$) of $$P$$, we first expand $$\mathbf{\pi}_0$$ in the eigenbasis:

$$\mathbf{\pi}_0 = c_1 \mathbf{u}_1 + c_2 \mathbf{u}_2 + \cdots + c_n \mathbf{u}_n$$

Then, for an arbitrary step $$t$$:

$$
\begin{multline*}
  \mathbf{\pi}_{t} = \mathbf{\pi}_0 P^t \\
  = (c_1 \mathbf{u}_1 + \cdots + c_n \mathbf{u}_n) P^t \\
  = c_1 \mathbf{u}_1 P^t + \cdots + c_n \mathbf{u}_n P^t \\
  = c_1 \lambda^t_1 \mathbf{u}_1 + \cdots + c_n \lambda^t_n \mathbf{u}_n
\end{multline*}
$$

Note that the [Laplacian]({% post_url 2025-01-23-graphs-linear-algebra %}#normalized-laplacian-matrices) and the transition matrices are deeply related:

$$
L_{rw} = D^{-1}L = D^{-1}(D - A) = I - P
$$

In fact, their eigenvectors and eigenvalues are connected. If $$\mathbf{u}$$ is an eigenvector of $$P$$, with eigenvalue $$\lambda$$:

$$
\mathbf{u} L_{rw} = \mathbf{u} (I - P) = \mathbf{u} - \mathbf{u} P = (1 - \lambda) \mathbf{u}
$$

That is, $$P$$ and $$L_{rw}$$ have the same eigenvectors, and the eigenvalues are related as $$\lambda_i(L_{rw}) = 1 - \lambda_i(P)$$. Since the [smallest eigenvalue of $$L_{rw}$$ is 0]({% post_url 2025-01-23-graphs-linear-algebra %}#connectivity-of-the-graph), the largest eigenvalue of $$P$$ is $$1$$. The corresponding eigenvector $$\pi$$ is one in which

$$
\pi P = \pi.
$$

Hence

$$
\pi_i = \sum_j P_{ij} \pi_j =  \sum_j \frac {A_ij} {d_j} \pi_j.
$$

, corresponding to $$\mathbf{u}_1 = \left(\frac 1 n, \cdots, \frac 1 n \right)$$. Similarly, the remaining eigenvalues are positive and strictly less than 1. Hence,

$$
\lim_{t \to \infty} \mathbf{\pi}_{t} = \lim_{t \to \infty} c_1 \lambda^t_1 \mathbf{u}_1 + \cdots + c_n \lambda^t_n \mathbf{u}_n = c_1 \mathbf{u}_1 = \left(\frac 1 n, \cdots, \frac 1 n \right) \blacksquare
$$

There are several remarks we can do:

- This result holds regardless of what the starting vertex is. In fact, $$\pi_0$$ could be a probability distribution over the vertices.
- The _speed_ at which the distribution converges depends on the eigenvalues of $$P$$. Specifically, if $$\lambda_2$$ is close to 1, the convergence will be slow.

{% enddetails %}

# Random walk with restart

In the **random walk with restart (RWR)**, the walker can return to its root vertex with a restart probability $$r \in [0, 1]$$:

$$
\mathbf{\pi}_{t+1} = r \mathbf{\pi}_0 + (1 - r) \mathbf{\pi}_t P
$$

where $$\mathbf{\pi}_0$$ represents the probability of starting at each vertex. If $$r = 0$$, the walker will never be teleported back to the root, and a RW is equivalent to a RWR. If $$r = 1$$, the walker will not be allowed to move out of the root, and $$\mathbf{\pi}_t = \mathbf{\pi}_0$$. However, for certain values of $r$, the walker is allowed to explore the root's neighborhood before teleporting back. If the root is part of a [module]({% post_url 2025-01-23-graphs-glossary %}#module), the walk will mostly happen within that module. If the root is very central, the walker will explore many parts of the network.

Importantly, the RWR has a stationary distribution $$\pi$$ which is not necessarily uniform:

$$
\lim_{t \to \infty} \mathbf{\pi}_{t} = \pi
$$

# Markov chains

A **Markov chain** is a sequence of events in which the probability of each event only depends on the state attained in the previous event. A random walk is a Markov chain: the probability of visiting a vertex only depends on what the neighbors of the current vertex are, and what is the probability of visiting each of them. We can describe some of the properties of a Markov chain by describing the underlying graph:

- *Time reversibility*
- *Symmetry*: a Markov chain is symmetric when the underlying graph is [regular]({% post_url 2025-01-23-graphs-glossary %}#regular).

In the context of Markov chains, the transition matrix $$P$$ is known as the **right stochastic matrix**.

{% details Types of stochastic matrices %}

- ***Row/right* stochastic matrix**: square matrix with non-negative entries where each row sums to $$1$$.
- ***Column/left* stochastic matrix**: square matrix with non-negative entries where each column sums to $$1$$.
- ***Doubly* stochastic matrix**: square matrix with non-negative entries where each row and column sum to $$1$$.

{% enddetails %}

# Further reading

- [Full title: The Unreasonable Effectiveness of Spectral Graph Theory: A Confluence of Algorithms, Geometry, and Physics](https://www.youtube.com/watch?v=8XJes6XFjxM)
