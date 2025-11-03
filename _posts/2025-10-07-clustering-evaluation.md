---
layout: distill
title: What constitutes good clustering?
date: 2025-10-07 11:59:00 +0000
description: Confronting my demons
tags:
  - machine_learning
giscus_comments: true
related_posts: false
---

It is fair to say that I am not a fan of clustering algorithms. This plot summarizes my feelings quite well:

{% include
  figure.liquid path="assets/python/2025-10-07-clustering-evaluation/img/clustering_on_toy_datasets.webp"
  class="img-fluid"
%}

<div class="caption">
    Different clustering algorithms. Adapted from <a href="https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html">the scikit-learn examples</a>: "With the exception of the last dataset, the parameters of each of these dataset-algorithm pairs has been tuned to produce good clustering results."
</div>

These are simple 2-dimensional datasets. We can have a look at them and immediately see the clusters or lack thereof. However the different algorithms give very different results, in many cases clearly wrong. Yet, they all maximize some objective function. Now, in data science, we often have to deal with high-dimensional data, where we cannot visualize the results, and hence we similarly rely on maximizing some metric. How can we trust the quality of the clustering? How can we make sure we are not just seeing patterns in noise or cherry-picking the results that we want to see?

In this post, I challenge my own assumptions about clustering, and try to understand how I can include it in my toolbox.

# Why is clustering hard?

Clustering is often defined as the art of grouping similar points together, and dissimilar points apart. And here lies the first problem: how do we define similarity? How similar is similar enough? On one extreme, we can imagine a very loose definition such that every point is similar to every other point, and hence we have one big cluster. On the other end, the case in which every point is unique in its own way.

The second problem is that similarity is not [transitive](https://en.wikipedia.org/wiki/Transitive_relation). Let's say we have three points, A, B and C. A is to B, and B to C, but A are very different. If we group all the points into one cluster, we satisfy the similarity requirement, but violate the dissimilarity one. If we put them all in different clusters, it's the other way around.

And here lies the problem of clustering: it is an ill-defined task. There is no single correct answer, and different applications call for different trade-offs. As data scientists, our goal is to pick an algorithm and tune its hyperparameters in the way that best nagivates this trade-off. So how do we make the right choice?

# How do we evaluate clusterings?

When faced with a clustering task, we need to pick an algorithm and its hyperparameters.

# Problems with clustering

## Algorithm selection

In other words, how to pick the best clustering algorithm for a task? How to pick the best hyperparameters? Different choices lead to significantly different results.

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
