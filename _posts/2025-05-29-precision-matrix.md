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

I will be providing snippets of code along with the text. If you are still curious about the nitty-gritty, all the code is available [on Github](https://github.com/hclimente/hclimente.github.io/blob/main/assets/python/2025-05-29-precision-matrix/).

# Warm up: refreshing our stats

Let's start with some basic definitions. The **variance** of a random variable $$X$$ is defined as

$$
\sigma_X^2 = \mathbf{E}(X - \mathbf{E}(X))^2
$$

It takes values in $$(0, \infty)$$ that measures how disperse the outcomes of the RV are from its mean. Notably, the **(scalar) precision** is defined as $$\frac 1 \sigma_X^2$$. Note that this ties to our usual intuition of precision: if a variable has low variance, it will be easier for us to guess the value TODO?

The **covariance** between two random variables $$X_1$$ and $$X_2$$ is defined as:

$$
\text{Cov}(X_1, X_2) = \mathbf{E}((X_1 - \mathbf{E}(X_1))(X_2 - \mathbf{E}(X_2)))
$$

Note that if $$X_1 = X_2$$, $$\sigma_{X_1}^2 = \sigma_{X_2}^2 = \text{Cov}(X_1, X_2)$$.

It takes values in $$(-\sigma_{X_1} \sigma_{X_2}, \sigma_{X_1} \sigma_{X_2})$$ and measures the degree to which two random variables are linearly related. The **correlation** $$\rho$$ normalizes the covariance, taking it to the $$[-1, 1]$$ range:

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

By normalizing the covariance matrix by dividing item $$\mathbf{\Sigma}_{ij}$$ by $$\sigma_{X_i} \sigma_{X_j}$$, we obtain the **correlation matrix**:

$$
P = \begin{pmatrix}
    1               & \rho-{X_1, X_2} & \cdots & \rho-{X_1, X_n} \\
    \rho-{X_2, X_1}  & 1              & \cdots & \rho-{X_1, X_n} \\
    \vdots          & \vdots         & \ddots & \vdots         \\
    \rho-{X_n, X_1}  & \rho-{X_n, X_2} & \cdots & 1
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

Let's bring this point home with a toy example on 3 variables: $$U$$, $$X$$, and $$Y$$. $$X$$ and $$Y$$ are related to $$U$$ as follows:

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

These are the correlations and partial correlations of 200 random trios:

{% include figure.liquid loading="eager" path="assets/python/2025-05-29-precision-matrix/img/partial_correlations.webp" class="img-fluid" %}

<div class="caption">
    <strong>Top row:</strong> original values. <strong>Bottom row:</strong> adjusted values.
</div>

Partial correlations correctly identify that $$U$$ is correlated to both $$X$$ and $$Y$$, and in turn that those are not correlated once we account for the effect of $$U$$. Note that while the true correlation between $$X$$ and $$U$$ is $$\rho_{X, U}$$, $$\rho_{X, U \mid Y} \neq \rho_{X, U}$$. This is because $$Y$$ contains an additional noise term that makes the adjustment imperfect.

# Estimating the precision matrix

Let's touch some code now. I computed the correlations in the toy example above by implementing the literal definition: fitting a linear model for each pair of variables and computing the correlation between the residuals. However, we can also use our knowledge of the precision matrix to compute the same result more efficiently:

```python
import numpy as np
import pandas as pd

np.random.seed(42)

N = 200

df = pd.DataFrame(
    {
        "U": np.random.normal(0, 1, N),
        "X": np.random.normal(0, 0.5, N),
        "Y": np.random.normal(0, 0.5, N),
    }
)
df["X"] = df["U"] + df["X"]
df["Y"] = df["U"] + df["Y"]

# compute the covariance matrix
df = df - df.mean()
covariance = np.dot(df.T, df) / (N - 1)

# compute the precision matrix
precision = np.linalg.inv(covariance)

# factorize it to get the partial correlations
norm_matrix = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
partial_corrs = precision / norm_matrix

print(np.round(partial_corrs, 2))
```

```
[[ 1.   -0.75  0.13]
 [-0.75  1.   -0.68]
 [ 0.13 -0.68  1.  ]]
```

Two disclaimers are needed here. One is that this code will not work when $$\Sigma$$ is [non-invertible](https://en.wikipedia.org/wiki/Singular_matrix). In those cases, [the **pseudoinverse** matrix](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) could take its place, although we would be moving away from the theory and hence we should be careful interpreting the results. The other is that the matrix inversion will be inaccurate when $$\Sigma$$ is [ill-conditioned](https://en.wikipedia.org/wiki/Condition_number). In those cases, [regularization](https://scikit-learn.org/stable/modules/covariance.html) can help.

<!-- TODO regularization: https://scikit-learn.org/stable/modules/covariance.html -->
