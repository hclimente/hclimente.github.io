---
layout: distill
title: Compendium of biological networks
date: 2025-08-01 11:59:00 +0000
description: And comments.
tags:
  - graphs
  - bioinformatics
giscus_comments: true
related_posts: false
---

# Ligand-receptors & cell-cell communication

Cell-type specific genes are enriched in receptor, plasma membrane and secreted proteins. Interestingly, younger proteins are more likely to fall in these categories; nuclear or cytoplasmic proteins are more likely to be old. While ligand-receptor networks are interesting, something potentially more interesting are cell-cell networks, in which cells are nodes and edges are message passing actions between them, in the form of ligand-receptor interactions. That helps us understand how cells work together.

Many studies leverage the transcriptome of different cells to infer cell-cell communication [Ramilowski et al. (2015), Dimitrov et al. (2022)]. However, and particularly when studied from the transcriptome, most cells express around 140 receptors and 140 ligands. Understanding which message is meant for which listener can be complicated. There heuristics have been applied to constrain the solution space.

A family of solutions involve focusing on the most expressed signaling pairs and grouping the cell by lineages (hematopoietic, ephitelial, etc.) it can be seen that most communications occur within-lineage. In fact, most receptors and ligands expressed by a cell affect ligands and receptors expressed in the same cell, suggesting a role in maintaining the cell state and communicating it to the neighbors.

Another option is to restrict the analyses to cell pairs that collocalize, or focus downstream analyses on the most unique cell-cell interactions.

Problems of using transcriptomics-derived ligand-receptor networks:

- We use gene expression as proxy for the right proteoform's expression
- We are limited to observing the expression of ligand-receptor pairs, missing important details like phyisiological responses, affinities, receptor recycling or whether the functional oligomer is formed
- Most analyses are limited to the studied population of cells, and hence neglect long-distance signalling events

Multiple databases exist.

# Further reading

- Ramilowski, J., Goldberg, T., Harshbarger, J. et al. A draft network of ligand–receptor-mediated multicellular signalling in human. Nat Commun 6, 7866 (2015). https://doi.org/10.1038/ncomms8866
- Dimitrov, D., Türei, D., Garrido-Rodriguez, M. et al. Comparison of methods and resources for cell-cell communication inference from single-cell RNA-Seq data. Nat Commun 13, 3224 (2022). https://doi.org/10.1038/s41467-022-30755-0
