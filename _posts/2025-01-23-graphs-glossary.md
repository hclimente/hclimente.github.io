---
layout: post
title: Graph Glossary
date: 2025-01-23 11:59:00-0000
description: Definitions of frequent graph terms
tags: graphs
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

# Parts of a graph

## Component

In an [undirected](#undirected-graph) graph, a [connected](#connected-graph) [subgraph](#subgraph) that is not part of a larger connected subgraph.

## Circuit

A [trail](#trail) in which the first and last vertices are equal. In contrast to the [cycle](#cycle), any vertex can be repeated.

## Cycle

A [trail](#trail) in which _only_ the first and last vertices are equal. Except for the tails and in contrast to the [circuit](#circuit), vertices cannot be repeated.

## Euler circuit

A [circuit](#circuit) that visits every edge of the graph.

## Euler trail

A [trail](#trail) that visits every edge of the graph.

## Flow

An example of a flow is a heat diffusion process across a graph. In such processes, each vertex starts with a certain amount of heat and, at each time point, exchanges heat with its [neighbors](#neighborhood) (gains heat from its hotter neighbors; loses it to its colder neighbors).

## Module

A [subgraph](#subgraph) whose vertices are densely connected to each other, and loosely to the rest of the graph.

## Orientation

An orientation of an [undirected](#undirected-graph) graph is the [directed](#directed-graph) graph resulting of assigning a direction to each of its vertices. A [directed](#directed-graph) graph is oriented if no two vertices form a 2-cycle.

## Path

A [walk](#walk) with no repeated _vertices_.

## Spanning graph

A subgraph $$G' = (V', E')$$ of $$G = (V, E)$$ is spanning if $$V' = V$$.

## Subgraph

A graph resulting from subsetting vertices from a larger graph, as well as a subset of the edges connecting them.

### Induced subgraph

A [subgraph](#subgraph) containing _all_ the edges connecting the vertices in the original graph.

## Trail

A [walk](#walk) with no repeated _edges_.

## Triplet

A set of 3 vertices and at least 2 edges between them, none of which are redundant or loops. _Open_ triplets have exactly 2 edges; _closed_ triplets have exactly 3.

## Walk

A walk _on a graph_ is an alternating sequence of vertices and edges, such that every vertex is [incident](#incidence) with the previous and the following edge (if any).

# Properties of vertices

## Adjacency

A vertex is adjacent with _another vertex_ if they are connected by an edge. $$u \sim v$$ denote that $$u$$ and $$v$$ are adjacent.

## Degree

The degree of a vertex in a (simple) [undirected](#undirected-graph) graph is the number of edges [incident](#incidence) with that vertex. In a (simple) [directed](#directed-graph) graph we distinguish the indegree (number of edges with the vertex as their [destination](#destination)) and the outdegree (number of edges with the vertex as their [source](#source)).

## Destination

In a [directed](#directed-graph) graph, the destination _of an edge_ is the vertex at the head of the edge.

## Distance

The distance _between two vertices_ is the shortest [path](#path) between them.

## Hub

A vertex with a high [degree](#degree).

## Neighborhood

The neighborhood of vertex $$v$$ is the [induced subgraph](#induced-subgraph) containing all the vertices [adjacent](#adjacency) to $$v$$.

## Incidence

A vertex is incident _with an edge_ if the vertex is one of the two vertices the edge connects.

## Source

In a [directed](#directed-graph) graph, the source _of an edge_ is the vertex at the tail of the edge.

# Types of graphs

## Acyclical graph

A graph without [cycles](#cycle).

## Bipartite graph

A [acyclical](#acyclical-graph) graph whose vertices can be divided into two sets such that no pair of vertices in the same set are [adjacent](#adjacency). Often, each of these sets are referred to as colors, and so we say that "there is no edge between two vertices of the same color."

## Complete graph

A simple, [undirected](#undirected-graph) graph in which every pair of vertices are connected by an edge. Complete graph are usually denoted by letter $$K$$ with a subindex that indicates the total number of vertices. For instance, $$K_6$$ represents the complete graph with 6 vertices.

## Connected graph

A graph in which a [path](#path) exists between every pair of vertices.

## Digraph

A [directed](#directed-graph) graph.

## Directed graph

See [Introduction to Graphs]({% post_url 2025-01-23-graphs-basics %}).

## Forest

An [undirected](#undirected-graph) graph in which any two vertices are connected by at most one path. That is, a disjoint union of [trees](#tree).

## Multigraph

A graph which can have multiple edges between the same pair of vertices.

## Navigable

A graph in which we can find a path between any two nodes via a greedy strategy that choses the neighbor closest according to a distance function.

## Regular

A graph in which every vertex has the same degree.

## Simple graph

A graph with at most one edge between any pair of vertices and no loops.

## Tree

An [undirected](#undirected-graph) graph in which there is only one [path](#path) between every pair of nodes.

## Triangle graph

A [triplet](#triplet) with 3 edges. It consists of _three_ closed triplets, each centered around each of the vertices.

## Undirected graph

See [Introduction to Graphs]({% post_url 2025-01-23-graphs-basics %}).

# Spectral graph theory

## First _k_ eigenvectors

Eigenvectors associated with the _k_ smallest eigenvalues.
