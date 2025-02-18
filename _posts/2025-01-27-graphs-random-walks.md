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

A **random walk (RW)** is a [stochastic](https://en.wikipedia.org/wiki/Stochastic_process), discrete process. At every time step a walker, located in one of the graph's vertices, picks one of its neighbors at random and moves to it. Often the transition probability between vertices is represented by the **transition** matrix $$P$$, a normalized version of the [adjacency]({% post_url 2025-01-23-graphs-linear-algebra %}#adjacency-matrix) in which the weights of all outbound edges add up to 1:

$$
P = D^{-1} A
$$

Note that $$P$$ corresponds to a [row stochastic matrix](#markov-chains). The outcome of a single random walk is a [walk]({% post_url 2025-01-23-graphs-glossary %}#walk) of length $$t$$, where $$t$$ is the number of steps. Let's see how a random walk starting at vertex $$i$$ plays out:

- At step 0, $$\mathbf{\pi}_0 = (0, 0, \cdots, 1, \cdots, 0)$$. That is, $$\pi_0$$ is an $$n$$-dimensional row vector that is $$0$$ almost everywhere, with a $$1$$ at component $$i$$.
- At step 1, $$\mathbf{\pi}_{1} = \mathbf{\pi}_0 P$$
- At step 2, $$\mathbf{\pi}_{2} = \mathbf{\pi}_1 P = (\mathbf{\pi}_0 P) P = \mathbf{\pi}_0 P^2 $$
- At step 3, $$\mathbf{\pi}_{3} = \mathbf{\pi}_2 P = (\mathbf{\pi}_0 P^2) P = \mathbf{\pi}_0 P^3$$
- …
- At step $$t$$, $$\mathbf{\pi}_{t} = \mathbf{\pi}_0 P^t$$

$$\pi_t$$ is an $$n$$-dimensional row vector $$\mathbf{\pi}_t$$ in which $$\pi_{tj}$$ represents the probability of the walker starting at vertex $$i$$ and being on vertex $j$ at time $t$.

We might be interested in what happens if we let the random walk run indefinitely:

$$
\lim_{t \to \infty} \mathbf{\pi}_{t} = \mathbf{\pi}_0 P^t
$$

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

The eigenvalues of a stochastic matrix are always less than or equal to 1 in absolute value. When the random walk is *ergodic* (see below), $$P$$ has an eigenvalue of 1 with an eigenvector $$\pi$$ such that:

$$
\pi_i = \frac {d_i} {\sum_j d_j}.
$$

{% details Proof %}

The degree row-vector $$\mathbf{d} = ({d_1}, \cdots, d_n )$$ is a left eigenvector of $$P$$:

$$
\mathbf{d} P = \mathbf{d} D^{-1} A = \mathbf{1} A = \mathbf{d}
$$

where $$\mathbf{1}$$ represents the row vector of all ones. That is, $$\mathbf{d}$$ is an eigenvector with eigenvalue 1 and non-negative entries. In order to transform it into a valid probability distribution, we need to make sure that $$\sum_i \pi_i = 1$$:

$$
\pi = \frac 1 {\sum_i d_i} \mathbf{d}
$$

{% enddetails %}

This is the stationary distribution of the random walk. It formalizes the intuitive result that high [degree]({% post_url 2025-02-09-graph-properties %}#degree) vertices are more likely to be visited. If the graph is [regular]({% post_url 2025-01-23-graphs-glossary %}#regular), the stationary distribution is uniform. Note that this is a property of the matrix, and not of $$\pi_0$$. This implies, that the initial distribution is not important in the long run: if the random walk is allowed to run indefinitely, the probability of ending up in each vertex will converge to $$\pi$$.

> **Ergodicity and _lazy_ random walks:** A unique stationary distribution does not always exists. A random walk is *ergodic* if a stationary distribution exists and is the same for any $$\pi_0$$. For the random walk to be ergodic, the graph needs to be connected and non [bipartite]({% post_url 2025-01-23-graphs-glossary %}#bipartite). If the graph has multiple components, starting in different components will produce different stationary distributions. If the graph is bipartite, at step $$t$$ the walker will be on one side or another, depending on the initial vertex and the parity of $$t$$. Bipartite graphs have a ergodic [_lazy_ random walk](https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture11.pdf), in which the walker has a probability $$\frac 1 2$$ of remaining at the current vertex and a probability $$\frac 1 2$$ of leaving it.

{% details Connection to the Laplacian %}

The [Laplacian]({% post_url 2025-01-23-graphs-linear-algebra %}#normalized-laplacian-matrices) and the transition matrices are deeply related:

$$
L_{rw} = D^{-1}L = D^{-1}(D - A) = I - P
$$

In fact, their eigenvectors and eigenvalues are connected. If $$\mathbf{u}$$ is an eigenvector of $$P$$, with eigenvalue $$\lambda$$:

$$
\mathbf{u} L_{rw} = \mathbf{u} (I - P) = \mathbf{u} - \mathbf{u} P = (1 - \lambda) \mathbf{u}
$$

That is, $$P$$ and $$L_{rw}$$ have the same eigenvectors, and the eigenvalues are related as $$\lambda_i(L_{rw}) = 1 - \lambda_i(P)$$. Since the [smallest eigenvalue of $$L_{rw}$$]({% post_url 2025-01-23-graphs-linear-algebra %}#connectivity-of-the-graph) is 0, corresponding to the eigenvector $$\mathbf{1}$$, $$P$$ has an eigenvalue of $$1$$ corresponding to that same eigenvector.

{% enddetails %}

There are several remarks we can do:

- This result holds regardless of what the starting vertex is. In fact, $$\pi_0$$ could be a probability distribution over the vertices.
- The _speed_ at which the distribution converges depends on the eigenvalues of $$P$$. Specifically, if $$\lambda_2$$ is close to 1, the convergence will be slow.

# Random walk with restart

In the **random walk with restart (RWR)**, the walker can return to its root vertex with a restart probability $$r \in [0, 1]$$:

$$
\mathbf{\pi}_{t+1} = r \mathbf{\pi}_0 + (1 - r) P \mathbf{\pi}_t
$$

where $$\mathbf{\pi}_0$$ represents the probability of starting at each vertex. If $$r = 0$$, the walker will never be teleported back to the root, and a RW is equivalent to a RWR. If $$r = 1$$, the walker will not be allowed to move out of the root, and $$\mathbf{\pi}_t = \mathbf{\pi}_0$$. However, for certain values of $r$, the walker is allowed to explore the root's neighborhood before teleporting back. If the root is part of a [module]({% post_url 2025-01-23-graphs-glossary %}#module), the walk will mostly happen within that module. If the root is very central, the walker will explore many parts of the network.

Importantly, the RWR also has a stationary distribution $$\pi$$:

$$
\lim_{t \to \infty} \mathbf{\pi}_{t} = \pi
$$

{% details Personalized PageRank %}

TODO

{% enddetails %}

# Markov chains

A **Markov chain** is a sequence of events in which the probability of each event only depends on the state attained in the previous event. A random walk is a Markov chain: the probability of visiting a vertex depends only on the current vertex's neighbors and the corresponding transition probabilities. We can describe some of the properties of a Markov chain by describing the underlying graph:

- _Time reversibility_
- _Symmetry_: a Markov chain is symmetric when the underlying graph is [regular]({% post_url 2025-01-23-graphs-glossary %}#regular).

In the context of Markov chains, the transition matrix $$P$$ is known as the **right stochastic matrix**.

{% details Types of stochastic matrices %}

- **_Row/right_ stochastic matrix**: square matrix with non-negative entries where each row sums to $$1$$.
- **_Column/left_ stochastic matrix**: square matrix with non-negative entries where each column sums to $$1$$.
- **_Doubly_ stochastic matrix**: square matrix with non-negative entries where each row and column sum to $$1$$.

{% enddetails %}

# Further reading

- [Full title: The Unreasonable Effectiveness of Spectral Graph Theory: A Confluence of Algorithms, Geometry, and Physics](https://www.youtube.com/watch?v=8XJes6XFjxM)
