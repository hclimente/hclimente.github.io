---
layout: post
title: Graph Glossary
date: 2025-01-23 11:59:00-0000
description: Definitions of frequent graph terms
tags: comments
categories: graphs
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

An example of a flow is a heat diffusion process across a graph. In such processes, each vertex starts with a certain amount of heat and, at each time point, exchanges heat with its neighbors (gains heat from its hotter neighbors; loses it to its colder neighbors).

## Module

A [subgraph](#subgraph) whose vertices are densely connected to each other, and loosely to the rest of the graph.

## Subgraph

A graph resulting from subsetting vertices from a larger graph, as well as a subset of the edges connecting them.

## Path

A [walk](#walk) with no repeated *vertices*.

## Trail

A [walk](#walk) with no repeated *edges*.

## Walk

A walk *on a graph* is an alternating sequence of vertices and edges, such that every vertex is [incident](#incidence) with the previous and the following edge (if any).

# Properties of vertices

## Adjacency

A vertex is adjacent with *another vertex* if they are connected by an edge.

## Incidence

A vertex is incident *with an edge* if the vertex is one of the two vertices the edge connects.

# Types of graphs

## Acyclical graph

A graph without [cycles](#cycle).

## Complete graph

A simple, undirected graph in which every pair of vertices are connected by an edge.

## Connected graph

A graph in which a [path](#path) exists between every pair of vertices.

## Multigraph

A graph which can have multiple edges between the same pair of vertices.

## Simple graph

A graph with at most one edge between any pair of vertices and no loops.

# Spectral graph theory

## First *k* eigenvectors

Eigenvectors associated with the *k* smallest eigenvalues.