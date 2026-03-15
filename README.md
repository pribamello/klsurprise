# klsurprise

A Python package for computing the **non-Gaussian Surprise statistic** between two datasets using nested sampling. The Surprise quantifies tension between datasets by comparing the Kullback-Leibler Divergence (KLD) of their posterior distributions against the expected KLD from the Posterior Predictive Distribution (PPD).

Built on [JAX](https://github.com/google/jax) for automatic differentiation and GPU/TPU acceleration, and [dynesty](https://dynesty.readthedocs.io/) for dynamic nested sampling.

If you use this code, please cite:

> Riba Mello et al., *Open Journal of Astrophysics*, vol. 8, 2025.
> [https://doi.org/10.33232/001c.138626](https://doi.org/10.33232/001c.138626)

---

## Installation

**Requirements:** Python >= 3.9

> **Note (Debian/Ubuntu with Python 3.12+):** System Python is protected by
> [PEP 668](https://peps.python.org/pep-0668/) and will refuse bare `pip install`
> commands. You **must** use a virtual environment as shown below.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 2. Install from source (recommended for development)

```bash
git clone https://github.com/<your-username>/klsurprise.git
cd klsurprise
pip install -e .
```

Or, for a non-editable install:

```bash
pip install .
```

### 3. To run the examples (adds `getdist`, `matplotlib`, `pandas`, and pinned `setuptools`)

```bash
pip install -e ".[examples]"
```

Or use the one-step install script:

```bash
bash install_examples.sh
```

This creates a virtual environment, installs the package in editable mode with all
example dependencies, and pins `setuptools<82` (see compatibility note below).

---

> **Compatibility note (15 Mar 2026 -- jax-cosmo 0.1.0):**
> `jax-cosmo` 0.1.0 imports `pkg_resources` from `setuptools` at runtime to
> read its own package version. `setuptools` >= 82.0.0 (released 2025)
> [removed `pkg_resources`](https://setuptools.pypa.io/en/latest/pkg_resources.html),
> which causes `jax-cosmo` to fail on import. Until `jax-cosmo` releases a
> fix, `setuptools` must be pinned below 82. The `[examples]` optional
> dependency group already includes `"setuptools>=64,<82"`, and the
> `install_examples.sh` script handles this automatically.

---

## Quick start

```python
import numpy as np
import jax
import klsurprise as kls

# Define the parameter space domain (flat prior bounds)
domain = np.array([
    [0.3, 1.0],    # h
    [0.05, 1.0],   # Omega_m
    [-1.0, 1.0],   # Omega_k
    [-3.0, -0.4],  # w
])

# Define the log-likelihood for dataset 1
@jax.jit
def logL_1(theta):
    ...  # your log-likelihood function

# Define the data model for dataset 2 (maps parameters -> data space)
def data_2_model(theta):
    ...  # returns a data vector

# Covariance matrix and observed data vector for dataset 2
cov_matrix_2 = ...   # (n x n) covariance matrix
data_vector_2 = ...  # (n,) observed data

# Initialize the Surprise statistics
sup = kls.surprise_statistics(
    logL_1,
    data_2_model,
    covariance_matrix_2=cov_matrix_2,
    domain=domain,
    data_2=data_vector_2,
    data_1_name="dataset1_NS.pkl",   # cache nested sampling results
    data_2_name="dataset2_NS.pkl",
)

# Compute the Surprise
results = sup.surprise_function_call(
    Nkld=100,                           # number of PPD samples for the KLD distribution
    result_path="surprise_results.hdf5", # save results to HDF5
    n_jobs=-1,                           # use all CPU cores
)

print(f"S = {results['S']:.2f} nats")
print(f"p-value = {results['p_value']:.4f}")
print(f"Discordance = {results['sigma_discordance']:.2f} sigma")
```

---

## How it works

The Surprise statistic `S` measures how unexpected a second dataset `D2` is given the posterior from a first dataset `D1`:

```
S = KLD(p(theta|D2) || p(theta|D1)) - <KLD>_PPD
```

where `<KLD>_PPD` is the expected KLD computed from the Posterior Predictive Distribution of `D2` given `D1`.

The pipeline:

1. **Nested Sampling** -- Compute posterior samples for `D1` (and optionally `D2`) using `dynesty`.
2. **PPD Generation** -- Draw mock data vectors from the Posterior Predictive Distribution `PPD(D2|D1)`.
3. **KLD Distribution** -- For each PPD sample, run nested sampling and compute `KLD(p_mock || p1)`. This builds the expected KLD distribution.
4. **Surprise** -- Compare the observed `KLD(p2 || p1)` against the expected distribution to get `S`, a p-value, and the sigma-level discordance.

---

## Output

`surprise_function_call()` returns a dictionary with:

| Key | Description |
|-----|-------------|
| `S` | Surprise statistic (nats). Only if `data_2` was provided. |
| `S_dist` | Distribution of Surprise values from the PPD. |
| `kld21` | `KLD(p2 \|\| p1)` between the two posteriors. Only if `data_2` was provided. |
| `kld_exp` | Expected KLD (mean of the KLD distribution). |
| `kld_dist` | Full KLD distribution array. |
| `p_value` | p-value of the observed Surprise. Only if `data_2` was provided. |
| `sigma_discordance` | Equivalent Gaussian sigma level. Only if `data_2` was provided. |
| `domain` | The parameter domain used. |

Results can be automatically saved to HDF5 via the `result_path` argument.

---

## Example: DESI BAO vs Pantheon+ SNIa (owCDM)

A complete working example is included in `examples/snia_bao/`, comparing DESI BAO+BBN data against Pantheon+SH0ES Type Ia supernovae under an open wCDM cosmological model.

```bash
cd examples/snia_bao
python run_owcdm.py
```

This example uses:
- `BAO_likelihood_DESI.py` -- DESI BAO likelihood with BBN prior, marginalized over baryon density.
- `SNIa_likelihood_pantheon.py` -- Pantheon+SH0ES likelihood for distance moduli.
- `data/` -- Pantheon+SH0ES data files.

---

## Package structure

```
klsurprise/
    __init__.py       # Public API
    core.py           # surprise_statistics class (main orchestrator)
    likelihood.py     # JIT-compiled Gaussian log-likelihood
    sampling.py       # Nested sampling (dynesty) and batch processing
    kld.py            # KLD computation and parallel distribution evaluation
    ppd.py            # Posterior Predictive Distribution generation
    io.py             # HDF5 output
    utils.py          # Priors, p-values, sigma discordance
```

---

## Notes and limitations

- **JAX compatibility required.** All user-provided log-likelihood and model functions must be JAX-compatible (use `jax.numpy` instead of `numpy` for array operations inside likelihoods).
- **Gaussian likelihood for D2.** The code assumes the second dataset's likelihood is Gaussian, parameterized by a covariance matrix and a model function.
- **Flat priors only.** Currently, only flat (top-hat) priors defined by `domain` are supported for nested sampling.
- **Nested sampling results are cached.** Provide `data_1_name` / `data_2_name` to save and reload expensive nested sampling runs between sessions.

---

## Dependencies

- [JAX](https://github.com/google/jax) >= 0.4.20
- [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) >= 0.1.0
- [dynesty](https://dynesty.readthedocs.io/) >= 2.1
- [numpy](https://numpy.org/) >= 1.26
- [scipy](https://scipy.org/) >= 1.11
- [h5py](https://www.h5py.org/) >= 3.10
- [joblib](https://joblib.readthedocs.io/) >= 1.3
- [tqdm](https://tqdm.github.io/) >= 4.66

All dependencies are installed automatically via `pip install`.

---

## License

MIT
