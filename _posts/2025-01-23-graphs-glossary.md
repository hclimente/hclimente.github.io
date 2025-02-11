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

In an undirected graph, a [connected](#connected-graph) [subgraph](#subgraph) that is not part of a larger connected subgraph.

## Cycle

A [trail](#trail) in which only the first and last vertices are equal.

## Flow

An example of a flow is a heat diffusion process across a graph. In such processes, each vertex starts with a certain amount of heat and, at each time point, exchanges heat with its [neighbors](#neighborhood) (gains heat from its hotter neighbors; loses it to its colder neighbors).

## Module

A [subgraph](#subgraph) whose vertices are densely connected to each other, and loosely to the rest of the graph.

## Orientation

An orientation of an undirected graph is the directed graph resulting of assigning a direction to each of its vertices. A directed graph is oriented if no two vertices form a 2-cycle.

## Path

A [walk](#walk) with no repeated *vertices*.

## Spanning graph

A subgraph $$G' = (V', E')$$ of $$G = (V, E)$$ is spanning if $$V' = V$$.

## Subgraph

A graph resulting from subsetting vertices from a larger graph, as well as a subset of the edges connecting them.

### Induced subgraph

A [subgraph](#subgraph) containing *all* the edges connecting the vertices in the original graph.

## Trail

A [walk](#walk) with no repeated *edges*.

## Triplet

A set of 3 vertices and at least 2 edges between them, none of which are redundant or loops. *Open* triplets have exactly 2 edges; *closed* triplets have exactly 3.

## Walk

A walk *on a graph* is an alternating sequence of vertices and edges, such that every vertex is [incident](#incidence) with the previous and the following edge (if any).

# Properties of vertices

## Adjacency

A vertex is adjacent with *another vertex* if they are connected by an edge. $$u \sim v$$ denote that $$u$$ and $$v$$ are adjacent.

## Degree

The degree of a vertex in a (simple) undirected graph is the number of edges [incident](#incidence) with that vertex. In a (simple) directed graph we distinguish the indegree (number of edges with the vertex as their [destination](#destination)) and the outdegree (number of edges with the vertex as their [source](#source)).

## Destination

In a directed graph, the destination *of an edge* is the vertex at the head of the edge.

## Distance

The distance *between two vertices* is the shortest [path](#path) between them.

## Neighborhood

The neighborhood of vertex $$v$$ is the [induced subgraph](#induced-subgraph) containing all the vertices [adjacent](#adjacency) to $$v$$.

## Incidence

A vertex is incident *with an edge* if the vertex is one of the two vertices the edge connects.

## Source

In a directed graph, the source *of an edge* is the vertex at the tail of the edge.

# Types of graphs

## Acyclical graph

A graph without [cycles](#cycle).

## Complete graph

A simple, undirected graph in which every pair of vertices are connected by an edge.

## Connected graph

A graph in which a [path](#path) exists between every pair of vertices.

## Digraph

A directed graph.

## Multigraph

A graph which can have multiple edges between the same pair of vertices.

## Simple graph

A graph with at most one edge between any pair of vertices and no loops.

### Triangle graph

A [triplet](#triplet) with 3 edges. It consists of *three* closed triplets, each centered around each of the vertices.

# Spectral graph theory

## First *k* eigenvectors

Eigenvectors associated with the *k* smallest eigenvalues.