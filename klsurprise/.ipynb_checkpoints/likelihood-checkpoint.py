import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import solve_triangular


@jit
def logL2_jitted(data_2_model, chol_cov_2, D):
    """
    JIT-compatible log-likelihood computation.

    Args:
    - data_2_model: Model data (already computed from theta).
    - chol_cov_2: Cholesky factor of the covariance matrix.
    - D: Data space vector.

    Returns:
    - The log-likelihood.
    """
    n = D.shape[0]
    diff = data_2_model - D
    solve = solve_triangular(chol_cov_2, diff, lower=True)
    log_det_cov = 2 * jnp.sum(jnp.log(jnp.diagonal(chol_cov_2)))
    log_likelihood = -0.5 * (
        n * jnp.log(2 * jnp.pi) + log_det_cov + jnp.dot(solve, solve)
    )

    return log_likelihood
