---
layout: post
title: Introduction to Graphs
date: 2025-01-23 11:59:00-0000
description: Basic definitions
tags: comments
categories: graphs
giscus_comments: true
related_posts: false
toc:
    sidebar: left
---

# Intro to graphs

TODO

# Mathematical notation

Importantly, graphs are mathematical objects. Let's define a graph $$G$$ as

$$G = (V, E)$$

Where $$V$$ denotes the set of vertices and $$E$$ the set of edges (pairs of vertices). Sometimes, graphs are defined as triples $$G = (V, E, \phi)$$, which includes the incidence $$\phi$$ (mapping edges to pairs of vertices). This is to allow for [*multigraphs*](../graphs-glossary#multigraph), in which multiple edges between the same pair of vertices are allowed. In this series we will ignore multigraphs and focus on [*simple*](../graphs-glossary#simple-graph) graphs, which have at most one edge between any pair of vertices and no loops. This notation allows to concisely define multiple types of graph:

- Undirected graph: $$E \subseteq \{ \{u, v\} | u, v \in V \}$$
- Directed graph: $$E \subseteq \{ (u, v) | u, v \in V \}$$
- *Simple*, undirected graph: $$E \subseteq \{ \{u, v\} | u, v \in V, u \neq v \}$$

