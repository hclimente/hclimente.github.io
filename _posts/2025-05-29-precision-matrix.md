---
layout: post
title: Covariance and precision
date: 2025-07-16 11:59:00
description: Learning the hidden structure of data
tags: linear_algebra graphs statistics
giscus_comments: true
related_posts: false
---

Imagine we have a set of 3 variables ($$U$$, $$X$$, and $$Y$$), with one of them being upstream of the other two ($$X \leftarrow U \rightarrow Y$$):

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

We want to discover this structure from observational data. Since both $$X$$ and $$Y$$ are caused by $$U$$, a correlation is not very enlightening and will just return the fully connected graph:

{% include figure.liquid loading="eager" path="assets/python/2025-05-29-precision-matrix/img/correlations.webp" class="img-fluid" %}

<div class="caption">
    Scatter plots of each pair of variables on 200 observations, each with the correlation between the variables and the associated P-value indicated above.
</div>

A sensible way of going about it is to study the correlation between each pair of variables after adjusting for the remaining variable. If we assume all relationships are linear, these are called **partial correlations**. Partial correlations are designed to reveal direct relationships by removing the influence of confounding variables. Here is a naive implementation:

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

    residuals = np.empty((p, p), dtype=object)

    for i in range(p):
        for j in range(i + 1, p):
            covariates_indices = [k for k in range(p) if k != i and k != j]
            X_covars = X[:, covariates_indices]

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

Partial correlations correctly identify that $$U$$ is correlated with both $$X$$ and $$Y$$, and in turn that those are not correlated once we account for the effect of $$U$$:

{% include figure.liquid loading="eager" path="assets/python/2025-05-29-precision-matrix/img/partial_correlations.webp" class="img-fluid" %}

<div class="caption">
    Scatter plots of the residuals of each pair of variables on 200 observations, each with the <em>partial correlation</em> between the variables and its associated P-value indicated above.
</div>

Note that while $$\hat{\rho}_{X, U} \approx \rho_{X, U} = 0.8944$$, $$\hat{\rho}_{X, U \mid Y} \neq \rho_{X, U}$$. This is because $$Y$$ contains an additional noise term that makes the adjustment imperfect. Also note that we have identified the **structure** of the data ($$X - U - Y$$), but not its **causal** structure ($$X \rightarrow U \rightarrow Y$$, $$X \leftarrow U \leftarrow Y$$ or $$X \rightarrow U \rightarrow Y$$).

A downside of this approach is its computational complexity. For an $$n \times p$$ input matrix:

- Memory complexity: $$\mathcal{O}(np^2)$$, dominated by storing $${p \choose 2} = \mathcal{O}(p^2)$$ residuals, each of length $$n$$.
- Time complexity: $$\mathcal{O}(np^4)$$, dominated by computing $${p \choose 2} = \mathcal{O}(p^2)$$ least squares problems, each of complexity $$\mathcal{O}(np^2)$$.

This is quite computational intensive, which will become a problem in real-world problems. Can we do better? Enter the **precision matrix**, a nice mathematical object to do this at scale.

# The precision matrix

{% details Need to dust off the basics? Variance, covariance and correlation %}

The **variance** of a random variable $$X$$ is defined as

$$
\sigma_X^2 = \mathbf{E}(X - \mathbf{E}(X))^2
$$

The variance takes values in $$[0, \infty)$$, and measures how disperse the outcomes of the RV are from its mean. Notably, the **(scalar) precision** is defined as $$\frac 1 \sigma_X^2$$, so high variance equals low precision and vice versa.

The **covariance** between two random variables, $$X_1$$ and $$X_2$$, is defined as:

$$
\text{Cov}(X_1, X_2) = \mathbf{E}((X_1 - \mathbf{E}(X_1))(X_2 - \mathbf{E}(X_2))).
$$

Observe that if $$X_1 = X_2$$, $$\sigma_{X_1}^2 = \sigma_{X_2}^2 = \text{Cov}(X_1, X_2)$$.

The covariance takes values in $$(-\sigma_{X_1} \sigma_{X_2}, \sigma_{X_1} \sigma_{X_2})$$, and measures the degree to which two random variables are linearly related. The **correlation** $$\rho$$ normalizes the covariance, rescaling it to the $$[-1, 1]$$ range:

$$
\rho_{X_1, X_2} = \frac {\text{Cov}(X_1, X_2)} {\sigma_{X_1} \sigma_{X_2}}
$$

{% enddetails %}

The **covariance matrix** of a set of random variables ties the variance and the covariance together. If $$\mathbf{X}$$ is a column vector such that

$$
\mathbf{X} = \begin{pmatrix}
  X_1 \\
  X_2 \\
  \vdots \\
  X_n
\end{pmatrix}
$$

then its associated covariance matrix $$\mathbf{\Sigma}$$ is

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

Finally, the **precision matrix** $$\mathbf{\Sigma}^{-1}$$ is the inverse of the covariance matrix, i.e., $$\mathbf{\Sigma} \mathbf{\Sigma}^{-1} = \mathbf{I}$$. $$\mathbf{\Sigma}$$ is not guaranteed to be invertible, and hence $$\mathbf{\Sigma}^{-1}$$ may not exist. Let's ignore this case for now, and jump to where things start getting interesting. $$\mathbf{\Sigma}^{-1}$$ can be decomposed as follows:

$$
\mathbf{\Sigma}^{-1} =
D
\begin{pmatrix}
    1                                           & -\rho_{X_1, X_2 \mid X_3, \dots, X_n}          & \cdots & -\rho_{X_1, X_n \mid X_2, \cdots, X_{n-1}}      \\
    -\rho_{X_2, X_1 \mid X_3, \cdots, X_n}      & 1                                              & \cdots & -\rho_{X_2, X_n \mid X_1, X_3, \cdots, X_{n-1}} \\
    \vdots                                      & \vdots                                         & \ddots & \vdots                                          \\
    -\rho_{X_n, X_1 \mid X_2, \cdots, X_{n-1}}  & -\rho_{X_n, X_2 \mid X_1, X_3 \cdots, X_{n-1}} & \cdots & 1
\end{pmatrix}
D
$$

where $$D$$ is a normalization matrix:

$$
D =
\begin{pmatrix}
    \frac 1 {\sigma_{X_1 \mid X_2, \cdots, X_n}} &                                                   &        & 0 \\
                                                 & \frac 1 {\sigma_{X_2 \mid X_1, X_3, \cdots, X_n}} &        &   \\
                                                 &                                                   & \ddots &   \\
    0                                            &                                                   &        & \frac 1 {\sigma_{X_n \mid X_1, \cdots, X_{n-1}}}
\end{pmatrix}
$$

The entries $$\rho_{X_., X_. \mid \dots}$$ in the middle matrix are our precious **partial correlations**.

# Estimating the precision matrix

Let's revisit our motivating example equipped with our newfound knowledge: instead of fitting $$\mathcal{O}(p^2)$$ linear models, let's reach the same result using linear algebra. First, we will estimate the covariance matrix using the maximum likelihood approach. Then, we will invert it to obtain the precision matrix.

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

    # showing how the sausage is made
    # but could be replaced by covariance = np.cov(X, rowvar=False)
    centered_X = X - X.mean(axis=0)
    covariance = np.dot(centered_X.T, centered_X) / (n - 1)

    precision = np.linalg.inv(covariance)

    normalization_factors = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial_correlations = - precision / normalization_factors

    return partial_correlations
```

This implementation is not only more compact, but has a more favorable computational complexity:

- Memory complexity: $$\mathcal{O}(p^2)$$, dominated by the intermediate matrices.
- Time complexity: $$\mathcal{O}(np^2 + p^3)$$, dominated by the computation of the covariance matrix and by the matrix inversion.

Furthermore, this implementation is [vectorized]({% post_url 2024-02-04-python-vectors %}) which further boosts performance. As a quick benchmark, on a random $$1000 \times 100$$ matrix, the original `pcorr_residuals` took 40.85 seconds; the updated `pcorr_linalg` took only 0.0007 seconds.

As with many elegant results in linear algebra, things start breaking down when our covariance matrix is [ill-conditioned](https://en.wikipedia.org/wiki/Condition_number) or outright [non-invertible](https://en.wikipedia.org/wiki/Singular_matrix). In [high-dimensional problems](https://en.wikipedia.org/wiki/High-dimensional_statistics), $$\Sigma$$ is non-invertible (and hard to estimate in the first place). In such cases, we could use [the pseudoinverse matrix](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) instead of the inverse. But that's just a patch: we will get results, but we are outside of the theory and interpreting the results is not as straightforward. However, when the matrix is ill-conditioned, there is a potential path to salvation: [regularization](https://scikit-learn.org/stable/modules/covariance.html).

# Regularized estimation

Adding a regularization step to the covariance matrix estimation will result in a better conditioned matrix. A common approach is _shrinking_ our empirical covariance towards another matrix, the _target_:

$$
\hat{\mathbf{\Sigma}} = (1 - \alpha) \hat{\mathbf{\Sigma}}_\text{MLE} + \alpha T
$$

where $$\alpha \in [0, 1]$$ is a parameter and $$T$$ is the target matrix, a highly structured matrix that encodes our assumption about what a _true_ covariance matrix should look like. A possible and aggressive target matrix is a diagonal matrix, which encodes the assumption of zero covariance between variables. By upweighting the diagonal elements and downweighting the off-diagonal elements, this matrix will have a better condition than $$\Sigma_\text{MLE}$$.

The problem becomes then tuning $$\alpha$$. A common way to compute the $$\alpha$$ is the [Ledoit-Wolf shrinkage method](https://web.archive.org/web/20141205061842/http://www.econ.uzh.ch/faculty/ledoit/publications/honey.pdf), which finds the $$\alpha$$ that minimizes the mean squared error between the real and the estimated matrix. Its [scikit-learn implementation](https://github.com/scikit-learn/scikit-learn/blob/68483539614102ba8e083277ed7123e6a9fece53/sklearn/covariance/_shrunk_covariance.py#L25) assumes that the target matrix is $$T = \mu I$$, where $$\mu$$ is the average variance.

Alternatively, we can use graphical lasso to estimate a sparse precision matrix. Conceptually, this is a bit easier to swallow: in many situations, most variables being conditionally uncorrelated is a valid assumption. The [graphical lasso](https://en.wikipedia.org/wiki/Graphical_lasso) does just that; it is a penalized estimator of the precision matrix.

$$
\hat{\mathbf{\Sigma}}^{-1} = \operatorname{argmin}_{\mathbf{\Sigma}^{-1} \succ 0} \left(\operatorname{tr}(\mathbf{\Sigma} \mathbf{\Sigma}^{-1}) - \log \det \mathbf{\Sigma}^{-1} - \lambda \|\mathbf{\Sigma}^{-1}\|_1 \right).
$$

The $$- \lambda \|\mathbf{\Sigma}^{-1}\|_1$$ term will favor sparse matrices, with a strength proportional to the magnitude of $$\lambda$$. While tuning $$\lambda$$ is in itself a challenge, a common approach is using [cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLassoCV.html).

Let's bring this point home by looking at a high-dimensional example (20 samples, 20 features).

{% include figure.liquid loading="eager" path="assets/python/2025-05-29-precision-matrix/img/high_dimensional_experiments.webp" class="img-fluid" %}

<div class="caption">
    Ground truth and estimated covariance matrix, precision matrix and structure of a high-dimensional example. The data generation process involved 20 samples, with 20 features each, sampled from a 0-mean multivariate Normal distribution. The estimated structure using the Ledoit-Wolf used a soft threshold (abs(correlation) > 0.1); otherwise, the fully connected graph would be shown.
</div>

As we can see, **maximum likelihood estimation** absolutely fails. Due to the extremely ill-conditioned covariance matrix, the precision matrix is completely off scale, with values ranging from -1.8e+15 to 1.0e+15. **Ledoit-Wolf** succeeds at computing a sensible-looking precision matrix. But recovering a structure, e.g., by thresholding it, is quite a hard task. Last, **graphical lasso** is able to find a relatively sparse structure. While it is still far from the ground truth, it prunes away most of the spurious correlations and keeps most of the true links. [As expected](https://scikit-learn.org/stable/modules/covariance.html#sparse-inverse-covariance), most of the true links are larger in absolute value, and further pruning it would return something close to the true structure.

More than anything, this little exercise shows how hard this endeavour is, and serves as a good caution to high-dimensional statistics. Beware!

## Conclusions

Under certain assumptions, the precision matrix helps us discover the internal structure of the data. When should we use what to estimate it?

1. **Empirical inverse (MLE):** fast and exact, but blows up if $$p$$ approaches $$n$$ or $$\hat Σ$$ is singular. Use it when $$n \gg p$$ and $$\hat Σ$$ is well‑conditioned.
1. **Shrinkage (Ledoit-Wolf):** automatically picks $$\alpha$$ to stabilize $$\hat Σ$$, yielding a dense but well‑behaved precision. Use it when $$\frac p n$$ is moderate.
1. **Graphical Lasso (cross‑validated $$\lambda$$):** trades off likelihood vs. sparsity to prune weak edges and reveal a parsimonious conditional‑independence network. Use it in high‑dimensional settings.
