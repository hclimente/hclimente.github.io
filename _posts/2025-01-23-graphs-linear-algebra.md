---
layout: post
title: Graphs and Linear Algebra
date: 2025-01-23 11:59:00-0000
description: Matrices associated to graphs and their properties
tags: comments
categories: graphs linear_algebra
giscus_comments: true
related_posts: false
mermaid:
    enabled: true
toc:
    sidebar: left
---

In this article I discuss matrices associated to graphs. As we will see, a graph can be represented as a matrix without any information loss. Hence, the properties of these matrices describe properties of the underlying graph.

# Matrices associated to graphs

A graph $$G = (V, E)$$ s.t. $$V = \{v_1, \dots, v_n\}$$ and $$E = \{e_1, \dots, e_m \}$$ has several important associated matrices. I consider case in which edges can have weights $$w_{ij} \geq 0$$. For convenience, I often refer to vertex $$v_i$$ simply by its index ($$i$$), and to an edge by the vertices it links (e.g., $$ij$$).

I will show examples on the following graph, named $$G_1$$:


```mermaid
---
config:
  layout: elk
  look: handDrawn
---
graph LR
    node_1((1))
    node_2((2))
    node_3((3))
    node_4((4))

    node_1 === node_2
    node_1 === node_3
    node_1 === node_4
    node_2 === node_3
```


## Degree matrix

The **degree** matrix $$D$$ is a diagonal $$n \times n$$ matrix such that $$D_{ii} = \sum_{j=1}^n w_{ij}$$. For instance, for $$G_1$$:

$$
D = \begin{bmatrix}
3 & 0 & 0 & 0 \\
0 & 2 & 0 & 0 \\
0 & 0 & 2 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

## Incidence matrix

[Incidence](../graphs-glossary#incidence) is used to define the **incidence** matrix $$Q$$, a $$n \times m$$ matrix such that $$Q_{ij}$$ equals:

- If $$G$$ is *directed*:
    - $$0$$ if vertex $$i$$ and edge $$e_j$$ are not incident
    - $$\sqrt{w_{ij}}$$ if edge $$e_j$$ originates at vertex $$i$$
    - $$-\sqrt{w_{ij}}$$ if edge $$e_j$$ terminates at vertex $$i$$
- If $$G$$ is *undirected*:
    - If $$Q$$ is *unoriented*:
        - $$0$$ if vertex $$i$$ and edge $$e_j$$ are not incident
        - $$\sqrt{w_{ij}}$$ otherwise
    - If $$Q$$ is *oriented*: we pick an [orientation](../graphs-glossary#orientation) of the graph, and use the incidence matrix of the resulting directed graph.

## Adjacency matrix

[Adjacency](../graphs-glossary#adjacency) is used to define the **adjacency** matrix $$A$$, a matrix $$n \times n$$ such that the $$A_{ij}$$ equals:

- $$0$$ if vertices $$i$$ and $$j$$ are not adjacent (note that in simple graphs vertices are not self-adjacent)
- $$w_{ij}$$ otherwise

For $$G_1$$:

$$
A = \begin{bmatrix}
0 & 1 & 1 & 1 \\
1 & 0 & 1 & 0 \\
1 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
\end{bmatrix}
$$

The adjacency matrix relates to the concept of [**paths**](../graphs-glossary#path) in an unweighted graph: $$(A^k)_{ij}$$ represents the number of paths of length $$k$$ from vertex $$i$$ to vertex $$j$$. In a weighted graph, it represents the sum of products of weights. For instance, if edge weights represent transition probabilities, $$(A^k)_{ij}$$ represents the probability of starting a walk at node $$i$$ and ending at node $$j$$ after $$k$$ steps.

The adjacency matrix has some important properties:

- If $$G$$ is undirected, $$A$$ is symmetric

## Laplacian matrix

The **Laplacian** matrix $$L$$ is a $$n \times n$$ matrix such that the $$L_{ij}$$ equals::

- For $$i \neq j$$:
    - $$0$$ if vertex $$i$$ and edge $$j$$ are not adjacent
    - $$-w_{ij}$$ otherwise
- For $$i = j$$, the degree of $$i$$.

More concisely, $$L = D - A$$. Or, given any oriented incidence matrix $$Q(G)$$, $$L = QQ^T$$.

For $$G_1$$:

$$
L = D - A = \begin{bmatrix}
3 & -1 & -1 & -1 \\
-1 & 2 & -1 & 0 \\
-1 & -1 & 2 & 0 \\
-1 & 0 & 0 & 1 \\
\end{bmatrix}
$$

The Laplacian relates to the connectedness of a graph, giving rise to [spectral graph theory](#spectral-graph-theory). It also is connected to [*flows*](../graphs-glossary#flow). The diagonal entries represent the total outflow capacity from a vertex, while off-diagonal entries encode pairwise connection strengths.

The Laplacian matrix has some important properties:

- If $$G$$ is undirected, $$L$$ is symmetric and positive semi-definite.
- $$L$$ has $$n$$ non-negative, real-valued eigenvalues.

## Normalized Laplacian matrices

$$L_\text{sym}$$ is a symmetric matrix derived from $$L$$ as follows:

$$L_\text{sym} = D^{-1/2}LD^{-1/2}$$

$$L_\text{rw}$$ is a matrix closely related to random walks that is derived from $$L$$ as follows:

$$L_\text{rw} = D^{-1}L$$

# Spectral graph theory

**Spectral graph theory** study how the eigenvalues and eigenvectors of a graph's associated matrices relate to its properties. Specifically, the eigenvalues of the Laplacian are closely related to the connectivity of the associated graph.

## Number of connected components

A simple, but ultimately insightful property of $$L$$ is that, for an undirected graph, the sum over the rows or the columns equals 0. In other words, multiplying $$L$$ by an all-ones vector $$\mathbf{1}$$ results in the zero vector. This tells us that $$L$$ has an eigenvalue of 0, corresponding to the eigenvector $$\mathbf{1}$$. Separately, linear algebra tells us that since $$L$$ is real and symmetric, it has *real* eigenvalues and *orthogonal* eigenvectors. And since $$L$$ is positive semi-definite, its eigenvalues are *non-negative*. As we have just seen, the [first eigenvalue](../graphs-glossary#first-k-eigenvectors), $$\lambda_1$$, of $$L$$ is 0, corresponding to the $$\mathbf{1}$$ eigenvector. If a vector has multiple [components](../graphs-glossary#component), $$L$$ is block diagonal. This makes it easy to see that the indicator vectors, representing the membership of each vertex to one of the components, are eigenvectors with an eigenvalue of 0. This highlights another important property of the Laplacian: given an undirected graph, the multiplicity of the eigenvalue 0 of $$L$$ equals the number of [components](../graphs-glossary#component). Conversely, for a [connected](../graphs-glossary#connected-graph) graph, $$\lambda_2 > 0$$. (The second smallest eigenvalue is sometimes called the Fiedler eigenvalue.)

## Spectral clustering

The goal of **spectral clustering** is find a partition of the graph into $$k$$ groups such that the are densely/strongly connected with each other, and sparsely/weakly connected to the others. (If we consider [random walks](#random-walks-and-markov-chains), spectral clustering seeks a partition of the graph such that a random walker tends to stay within each partition, rarely shifting between disjoint sets.)

An spectral clustering algorithm, in which seek to find *k* clusters, looks as follows:

1. Compute the [first *k* eigenvectors](../graphs-glossary#first-k-eigenvectors) $$u_1, \dots, u_k$$ of $$L$$. Store them in the columns of matrix $$U \in R^{n \times k}$$.
2. Cluster the rows $$1, \dots, n$$ of $$U$$ using k-means clustering into clusters $$C_1, \dots, C_k$$.

## Graph partitioning

TODO

# Random Walks and Markov Chains

TODO Relationship with graph connectivity and stationary distributions.

# Further reading

- [A Tutorial on Spectral Clustering](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf)
- [Four graph partitioning algorithms](https://mathweb.ucsd.edu/~fan/talks/mlg.pdf)