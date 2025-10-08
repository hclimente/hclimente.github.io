---
layout: distill
title: Evaluating the output of a clustering algorithm
date: 2025-10-06 11:59:00 +0000
description: Confronting my demons
tags:
  - machine_learning
giscus_comments: true
related_posts: false
---

It is fair to say that I am not a fan of clustering algorithms.

Clustering can have multiple goals:

# Problems with clustering

## Algorithm selection

In other words, how to pick the best clustering algorithm for a task? How to pick the best hyperparameters? Different choices lead to significantly different results.

Visualization: In high-dimensional spaces, we can't even visualize the results.

Definition: we want "similar points" to be in the same clusters, non-similar points in different clusters. But often meeting both criteria is impossible. That's because similarity is not transitive: if A is similar to B, and B is similar to C, A may not be similar to C. Hence, either all A, B, C are in the same cluster (violating non-similarity), or some of them are in different clusters (violating similarity).

Balancing conflicting requirements: often we can only meet one of the following.

- Cluster sizes are balanced, e.g., k-means (+ having dissimilar points in different clusters), min-sum
- Similar points are in the same cluster, e.g., single linkage
- Non-similar points are in different clusters, e.g., max linkage
- Outliers do not affect the clusters too much

Different applications call for different trade-offs.

In practice, how are algorithms picked?

- No need to tune parameter
- The algorithm is free
- It worked for my friend (for a different problem?)
- It's fast

To turn clustering into a well-defined task, one needs to add some bias, expressing some prior domain knowledge. Ideally, we could come up with a set of properties for each clustering algorithm, and then pick the one that best matches our needs.

## Computational complexity

# How can we evaluate clustering algorithms?
