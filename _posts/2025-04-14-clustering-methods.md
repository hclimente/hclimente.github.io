---
layout: post
title: Clustering algorithms
date: 2025-04-14 11:59:00-0000
description: K-means, hierarchical clustering
tags: comments
categories: clustering
giscus_comments: true
related_posts: false
toc:
  sidebar: left
images:
  compare: true
  slider: true
---

**Clustering** consists on grouping sets of objects based on their similarity. The resulting groups (_clusters_) should contain objects that are very similar to each other, and different to objects in other clusters.

<style>
  .colored-slider {
    --divider-color: rgba(0, 0, 0, 0.5);
    --default-handle-color: rgba(0, 0, 0, 0.5);
    --default-handle-width: clamp(40px, 10vw, 200px);
  }
</style>
<img-comparison-slider class="colored-slider">
  {% include figure.liquid path="assets/python/2025-04-14-clustering-methods/img/pca_plain.jpg" class="img-fluid rounded z-depth-1" slot="first" %}
  {% include figure.liquid path="assets/python/2025-04-14-clustering-methods/img/pca_colored.jpg" class="img-fluid rounded z-depth-1" slot="second" %}
</img-comparison-slider>

> It becomes obvious that in order to cluster, one needs a way to quantify similarity between pairs of objects. That, in itself, is a topic for another study. For the remainder of this article, we will assume that we have access to all pairwise similarities.

Clustering algorithms can in turn ne broadly clustered into several groups. The main ones are: hierarchical, partitional, density-based and model-based.

# Hierarchical clustering

# Partitional clustering

## Example: K-means

# Density-based clustering

## Example: DBSCAN

DBSCAN (density-based spatial clustering with additive noise)

# Model-based clustering

## Example: Gaussian Mixture Models and the EM model

# Further reading

- [The Burden of Demonstrating Statistical Validity of Clusters](https://www.fharrell.com/post/cluster/)
- [A Tutorial on Spectral Clustering](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf)
