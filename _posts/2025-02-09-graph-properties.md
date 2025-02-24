---
layout: post
title: Properties of Graphs
date: 2025-01-24 11:59:00-0000
description: Multiscale ways to talk about graphs
tags: graphs
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

At the most fundamental level, graphs are just entities and connections between them. Yet, the network topology gives rise to emergent properties. For instance, how information flows through a social network is partly a function who posts the message and how they are connected to the rest of the network, with their immediate connections being likely more important. In this section, I review three levels at which networks operate: [local](#local-properties), [mesoscale](#mesoscale-properties) and [global](#global-properties). They refer, respectively, to properties of the nodes, properties of parts of the network and properties of the whole network.

# Local properties

## Degree

In an undirected network, the **degree** of a vertex $$u$$ ($$\deg u$$) refers to the number of edges that are incident on $$u$$. In a directed network, this concept is split between _indegree_ ([$$\deg^- u$$], the number of edges that have $$u$$ as their destination) and _outdegree_ ([$$\deg^+ u$$], number of edges that have $$u$$ as their source). Weighted graphs extend this concept to _weighted_ degree, in which $$\deg u = \sum_{i} w(e_{ui})$$.

## Local clustering coefficient

The **(local) clustering coefficient** _of a vertex_ measures the probability that its [neighbors]({% post_url 2025-01-23-graphs-glossary %}#neighborhood) are connected. It is computed as the ratio between number of [triangles]({% post_url 2025-01-23-graphs-glossary %}#triangle-graph) involving a vertex, and the number of [triplets]({% post_url 2025-01-23-graphs-glossary %}#triplet) involving that same vertex.

[Often](https://igraph.org/r/doc/transitivity.html), the clustering coefficient of a directed graph is computed without considering the direction of the edges.

# Mesoscale properties

## Modularity

The **modularity** measures how well a graph can be divided into [modules]({% post_url 2025-01-23-graphs-glossary %}#modules). Given a partition of a graph into $$k$$ modules, the modularity $$Q$$ is computed as

$$Q = \sum_{i=1}^k (e_{ii} - {a_i^2})$$

where $$e_{ii} = \frac {\| \{\{u, v\} \mid u \in V_i, v \in V_i, \{u, v\} \in E \} \|} {\|E\|} $$,$$a*i = \frac {\| \{\{u, v\} \mid u \in V_i, \{u, v\} \in E \} \|} {\|E\|} $$ and $$V_i$$ is the set of vertices in module $$i$$. $$e*{ii}$$ is the fraction of edges within module $$i$$ and $$a_i$$ is the fraction of edges incident with one vertex in module $$i$$. $$Q$$ will be large when the fraction of edges within the module is much larger than expected by chance.

## Within-module degree

The **within-module degree** of a vertex is the module version of the [degree](#degree). It is often normalized as a z-score; the z-score for node $$i$$, mapped to module $$k$$:

$$Z_i = \frac {\kappa_i - \bar \kappa_k} {\sigma_{\kappa_k}}$$

where $$\kappa_i$$ is within-module degree (the number of edges between $$i$$ and other vertices in module $$k$$); $$\bar \kappa_k$$ is the average within-module degree; and $$\sigma_{\kappa_k}$$ is the standard deviation of the within module degrees.

{% comment %}
## Participation

The **participation coefficient** of a vertex... TODO
It is a mesoscale measure of [centrality](#centrality).
{% endcomment %}

# Global properties

## Radius and diameter

The radius and the diameter measure how easy it is to traverse a graph. They both are quantities based on the maximum [distance]({% post_url 2025-01-23-graphs-glossary %}#distance) between any two vertices found in the graph. Specifically, the **radius** is the minimum maximum distance; the **diameter** is the maximum distance.

## Global clustering coefficient

The **global clustering coefficient** _of a graph_ is the ratio between closed and open [triplets]({% post_url 2025-01-23-graphs-glossary %}#triplet) in that graph. Or, equivalently:

$$C = \frac {3 \times \text{triangles}} {\text{triplets}}$$

[Often](https://igraph.org/r/doc/transitivity.html), the clustering coefficient of a directed graph is computed without considering the direction of the edges.

## Centrality

**Centrality** assigns a score or a ranking to every vertex in the graph, which represents its importance in the network according to some metric. [Degree](#degree) and [participation](#participation) are examples of such metrics, but there are others.

WIP
