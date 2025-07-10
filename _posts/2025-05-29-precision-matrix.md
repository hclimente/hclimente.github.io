---
layout: post
title: The precision matrix
date: 2025-05-29 11:59:00
description: A Swiss Army Knife for data science
tags: linear_algebra graphs
giscus_comments: true
related_posts: false
mermaid:
  enabled: true
---

Imagine we have a set of 3 variables ($$U$$, $$X$$, and $$Y$$), with with one of them being upstream of the other two ($$X \leftarrow U \rightarrow Y$$):

$$
U \sim N(0, 1)
$$

$$
X = U + \varepsilon_X
$$

$$
Y = U + \varepsilon_Y
$$

$$
\varepsilon_X, \varepsilon_Y \sim N(0, 0.25)
$$

We want to discover this structure from observational data. Since both $$X$$ and $$Y$$ are caused by $$U$$, a correlation is not very enlightening:

{% include figure.liquid loading="eager" path="assets/python/2025-05-29-precision-matrix/img/correlations.webp" class="img-fluid" %}

<div class="caption">
    Scatter plots of each pair of variables on 200 random trios. On top of each graph, I show the <strong>correlation</strong> between the variables, and the associated P-value.
</div>

A sensible way of going about it is to study the relationship between each pair of variables while adjusting for the remaining variable. If we assume all relationships are linear, these are called **partial correlations**. Here is a possible implementation:

```python
def pcorr_residuals(X: np.ndarray) -> np.ndarray:
    """
    Compute the matrix of partial correlations from the residuals of linear regression models.

    Parameters
    ----------
    X : np.ndarray
        The input data matrix.

    Returns
    -------
    np.ndarray
        The matrix of partial correlations.
    """

    n, p = X.shape

    # dictionary to store the centered residuals
    # keys are tuples of (target, excluded) and values are the residuals
    residuals = np.empty((p, p), dtype=object)

    for i in range(p):
        for j in range(i + 1, p):
            idx_covars = [k for k in range(p) if k != i and k != j]
            X_covars = X[:, idx_covars]

            for target, excluded in [(i, j), (j, i)]:

                y = X[:, target]

                # fit a linear model
                beta, *_ = np.linalg.lstsq(X_covars, y, rcond=None)
                y_pred = X_covars @ beta

                # compute and center the residuals
                r = y - y_pred
                residuals[(target, excluded)] = r - r.mean()

    pcorr = np.eye(p)

    for i in range(p):
        for j in range(i + 1, p):

            res_1 = residuals[(i, j)]
            res_2 = residuals[(j, i)]
            corr = np.dot(res_1, res_2) / (np.linalg.norm(res_1) * np.linalg.norm(res_2))

            pcorr[i, j] = pcorr[j, i] = corr

    return pcorr
```

Partial correlations correctly identify that $$U$$ is correlated to both $$X$$ and $$Y$$, and in turn that those are not correlated once we account for the effect of $$U$$:

{% include figure.liquid loading="eager" path="assets/python/2025-05-29-precision-matrix/img/partial_correlations.webp" class="img-fluid" %}

<div class="caption">
    Scatter plots of the residuals of each pair of variables on 200 random trios. On top of each graph, I show the <strong>partial correlation</strong> between the variables, and the associated P-value.
</div>

Note that while $$\hat{\rho}_{X, U} \approx \rho_{X, U} = 0.8944$$, $$\hat{\rho}_{X, U \mid Y} \neq \rho_{X, U}$$. This is because $$Y$$ contains an additional noise term that makes the adjustment imperfect. Note as well that we have identified the **structure** of the data ($$X - U - Y$$), but not its **causal** structure ($$X \rightarrow U \rightarrow Y$$, $$X \leftarrow U \leftarrow Y$$ or $$X \rightarrow U \rightarrow Y$$).

A downside of this approach is its computational complexity. For a $$n \times p$$ input matrix:

- Memory complexity: $$\mathcal{O}(np^2)$$, dominated by storing $${p \choose 2} = \mathcal{O}(p^2)$$ residuals, each of length $$n$$.
- Time complexity: $$\mathcal{O}(np^4)$$, dominated by computing $${p \choose 2} = \mathcal{O}(p^2)$$ least squares problems, each of complexity $$\mathcal{O}(np^2)$$.

Can we do better? Enter the **precision matrix**, a nice mathematical object to do this at scale.

# The precision matrix

Let's start with some basic definitions. The **variance** of a random variable $$X$$ is defined as

$$
\sigma_X^2 = \mathbf{E}(X - \mathbf{E}(X))^2
$$

The variance takes values in $$(0, \infty)$$, and measures how disperse the outcomes of the RV are from its mean. Notably, the **(scalar) precision** is defined as $$\frac 1 \sigma_X^2$$, so high variance equals low precision and vice versa.

The **covariance** between two random variables, $$X_1$$ and $$X_2$$, is defined as:

$$
\text{Cov}(X_1, X_2) = \mathbf{E}((X_1 - \mathbf{E}(X_1))(X_2 - \mathbf{E}(X_2)))
$$

Note that if $$X_1 = X_2$$, $$\sigma_{X_1}^2 = \sigma_{X_2}^2 = \text{Cov}(X_1, X_2)$$.

The covariance takes values in $$(-\sigma_{X_1} \sigma_{X_2}, \sigma_{X_1} \sigma_{X_2})$$, and measures the degree to which two random variables are linearly related. The **correlation** $$\rho$$ normalizes the covariance, rescaling it to the $$[-1, 1]$$ range:

$$
\rho_{X_1, X_2} = \frac {\text{Cov}(X_1, X_2)} {\sigma_{X_1} \sigma_{X_2}}
$$

The **covariance matrix** of a set of random variables ties these quantities together. If $$\mathbf{X}$$ is a column vector such that

$$
\mathbf{X} = \begin{pmatrix}
  X_1 \\
  X_2 \\
  \vdots \\
  X_n
\end{pmatrix}
$$

then the covariance matrix $$\mathbf{\Sigma}$$ is

$$
\mathbf{\Sigma} = \begin{pmatrix}
    \sigma_{X_1}^2        & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\
    \text{Cov}(X_2, X_1)  & \sigma_{X_2}^2       & \cdots & \text{Cov}(X_2, X_n) \\
    \vdots                & \vdots               & \ddots & \vdots               \\
    \text{Cov}(X_n, X_1)  & \text{Cov}(X_n, X_2) & \cdots & \sigma_{X_n}^2
\end{pmatrix}
$$

Since always $$\text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i)$$, $$\mathbf{\Sigma}$$ is _symmetric_. It is, in fact, _positive semi-definite_ ([proof](https://statproofbook.github.io/P/covmat-psd.html)).

By normalizing the covariance matrix by dividing each item $$\mathbf{\Sigma}_{ij}$$ by $$\sigma_{X_i} \sigma_{X_j}$$, we obtain the **correlation matrix**:

$$
P = \begin{pmatrix}
    1               & \rho_{X_1, X_2} & \cdots & \rho_{X_1, X_n} \\
    \rho_{X_2, X_1} & 1               & \cdots & \rho_{X_1, X_n} \\
    \vdots          & \vdots          & \ddots & \vdots          \\
    \rho_{X_n, X_1} & \rho_{X_n, X_2} & \cdots & 1
\end{pmatrix}
$$

Finally, the **precision matrix** $$\mathbf{\Sigma}^{-1}$$ is the inverse of the covariance matrix, i.e., $$\mathbf{\Sigma} \mathbf{\Sigma}^{-1} = \mathbf{I}$$. Note that $$\mathbf{\Sigma}$$ is not guaranteed to be invertible, and hence $$\mathbf{\Sigma}^{-1}$$ might not exist. Let's ignore this case for now, and jump to where things start getting interesting. $$\mathbf{\Sigma}^{-1}$$ can be decomposed as follows:

$$
\mathbf{\Sigma}^{-1} =
U
\begin{pmatrix}
    1                                           & -\rho_{X_1, X_2 \mid X_3, \dots, X_n}          & \cdots & -\rho_{X_1, X_n \mid X_2, \cdots, X_{n-1}}      \\
    -\rho_{X_2, X_1 \mid X_3, \cdots, X_n}      & 1                                              & \cdots & -\rho_{X_2, X_n \mid X_1, X_3, \cdots, X_{n-1}} \\
    \vdots                                      & \vdots                                         & \ddots & \vdots                                          \\
    -\rho_{X_n, X_1 \mid X_2, \cdots, X_{n-1}}  & -\rho_{X_n, X_2 \mid X_1, X_3 \cdots, X_{n-1}} & \cdots & 1
\end{pmatrix}
U
$$

where $$U$$ is a normalization matrix:

$$
U =
\begin{pmatrix}
    \frac 1 {\sigma_{X_1 \mid X_2, \cdots, X_n}} &                                                   &        & 0 \\
                                                 & \frac 1 {\sigma_{X_2 \mid X_1, X_3, \cdots, X_n}} &        &   \\
                                                 &                                                   & \ddots &   \\
    0                                            &                                                   &        & \frac 1 {\sigma_{X_n \mid X_1, \cdots, X_{n-1}}}
\end{pmatrix}
$$

The entries $$-\rho_{X_., X_. \mid \dots}$$ in the middle matrix are **partial correlations**. Partial correlations quantify the correlation between two variables after removing the linear effects of the remaining variables. Partial correlations live in $$[-1, 1]$$.

# Estimating the precision matrix

Let's revisit our motivating example now. I computed the correlations in the toy example above by fitting a linear model for each pair of variables and computing the correlation between the residuals. However, we can also use our knowledge of the precision matrix to compute the same result more efficiently:

```python
def pcorr_linalg(X: np.ndarray) -> np.ndarray: # (n, p) -> (p, p)
    """
    Compute the matrix of partial correlations from the covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        The input data matrix.

    Returns
    -------
    np.ndarray
        The matrix of partial correlations.
    """

    n, p = X.shape

    # could be replaced bt covariance = np.cov(X, rowvar=False)
    centered_X = X - X.mean(axis=0)
    covariance = np.dot(centered_X.T, centered_X) / (n - 1)
    precision = np.linalg.inv(covariance)

    normalization_factors = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial_correlations = precision / normalization_factors

    return partial_correlations
```

This implementation is not only more compact, but has a much better complexity:

- Memory complexity: $$\mathcal{O}(p^2)$$, dominated by the intermediate matrices.
- Time complexity: $$\mathcal{O}(np^2 + p^3)$$, dominated by the computation of the covariance matrix and the matrix inversion.

Furthermore, this implementation is [vectorized]({% post_url 2024-02-04-python-vectors %}) which further improves the speed. As a quick benchmark, on a random $$1000 \times 100$$ matrix, the original `pcorr_residuals` took 40.85 seconds; the updated `pcorr_linalg` took only 0.0007 seconds.

Two disclaimers are needed here. One is that this code will not work when $$\Sigma$$ is [non-invertible](https://en.wikipedia.org/wiki/Singular_matrix). In those cases, [the **pseudoinverse** matrix](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) could take its place, although we would be moving away from the theory and hence we should be careful interpreting the results. The other is that the matrix inversion will be inaccurate when $$\Sigma$$ is [ill-conditioned](https://en.wikipedia.org/wiki/Condition_number). In those cases, [regularization](https://scikit-learn.org/stable/modules/covariance.html) can help.

# Regularized estimation

Let's scale up our toy example to 20 variables.

<!-- TODO regularization: https://scikit-learn.org/stable/modules/covariance.html -->

# Application: the Titanic dataset
