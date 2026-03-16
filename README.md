# klsurprise

Compute the **non-Gaussian Surprise statistic** between two datasets using nested sampling.

The Surprise quantifies tension between datasets by comparing the
Kullback-Leibler Divergence (KLD) of their posteriors against the expected
KLD from the Posterior Predictive Distribution (PPD).

Built on [JAX](https://github.com/google/jax) and
[dynesty](https://dynesty.readthedocs.io/).

If you use this code, please cite:

> Riba Mello et al., *Open Journal of Astrophysics*, vol. 8, 2025.
> [https://doi.org/10.33232/001c.138626](https://doi.org/10.33232/001c.138626)

---

## Installation

**Requires:** Python >= 3.9

> **Debian/Ubuntu with Python 3.12+:** system Python is protected by
> [PEP 668](https://peps.python.org/pep-0668/). Use a virtual environment.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install
git clone https://github.com/pribamello/klsurprise.git
cd klsurprise
pip install -e .                 # editable install

# 3. (Optional) Install example dependencies
pip install -e ".[examples]"     # adds getdist, matplotlib, pandas
```

Or use the one-step script:

```bash
bash install_examples.sh
```

> **Compatibility note (15 Mar 2026 -- jax-cosmo 0.1.0):**
> `jax-cosmo` 0.1.0 uses `pkg_resources`, which was removed in
> `setuptools` >= 82. The `[examples]` extra already pins
> `setuptools>=64,<82`. The install script handles this automatically.

---

## Quick start

```python
import numpy as np
import jax
import klsurprise as kls

domain = np.array([
    [0.3, 1.0],    # h
    [0.05, 1.0],   # Omega_m
    [-1.0, 1.0],   # Omega_k
    [-3.0, -0.4],  # w
])

@jax.jit
def logL_1(theta):
    ...  # log-likelihood for dataset 1

def data_2_model(theta):
    ...  # maps parameters -> data vector for dataset 2

sup = kls.surprise_statistics(
    logL_1,
    data_2_model,
    covariance_matrix_2=cov_matrix_2,
    domain=domain,
    data_2=data_vector_2,
    data_1_name="d1_NS.pkl",
    data_2_name="d2_NS.pkl",
)

results = sup.surprise_function_call(Nkld=100, n_jobs=-1)

print(f"S = {results['S']:.2f} nats")
print(f"p-value = {results['p_value']:.4f}")
print(f"Discordance = {results['sigma_discordance']:.2f} sigma")
```

---

## How it works

```
S = KLD(p(theta|D2) || p(theta|D1))  -  <KLD>_PPD
```

| Step | What happens |
|------|--------------|
| 1. Nested Sampling | Compute posteriors for D1 and D2 with `dynesty`. |
| 2. PPD Generation | Draw mock data vectors from PPD(D2\|D1). |
| 3. KLD Distribution | For each mock, run NS and compute KLD(p_mock \|\| p1). |
| 4. Surprise | Compare the observed KLD against the expected distribution. |

---

## API Reference

### Main pipeline

#### `surprise_statistics`

```python
kls.surprise_statistics(
    logL1,                # callable(theta) -> float
    data_2_model_fun,     # callable(theta) -> array
    covariance_matrix_2,  # (n, n) array
    domain,               # (ndim, 2) array — flat prior bounds
    data_2=None,          # (n,) observed data vector (optional)
    data_1_name=None,     # str — cache file for D1 NS results
    data_2_name=None,     # str — cache file for D2 NS results
    init_NS=False,        # run nested sampling on construction
)
```

The central class. Orchestrates nested sampling, PPD generation, KLD
computation, and Surprise evaluation.

**Main method:**

```python
results = sup.surprise_function_call(
    Nkld,                 # int — number of PPD samples for KLD distribution
    result_path=None,     # str — save results to HDF5
    n_effective=15000,    # int — target effective samples for NS
    n_jobs=-1,            # int — parallel workers (-1 = all cores)
)
```

Returns a dictionary:

| Key | Description |
|-----|-------------|
| `S` | Surprise value (nats) |
| `S_dist` | Surprise distribution from PPD |
| `kld21` | KLD(p2 \|\| p1) |
| `kld_exp` | Expected KLD (mean of KLD distribution) |
| `kld_dist` | Full KLD distribution array |
| `p_value` | p-value of the observed Surprise |
| `sigma_discordance` | Equivalent Gaussian sigma |
| `domain` | Parameter domain used |

Keys `S`, `kld21`, `p_value`, and `sigma_discordance` are only present when
`data_2` was provided.

---

### Gaussian Surprise (analytical)

```python
from klsurprise import SurpriseGauss

sg = SurpriseGauss(chain_2, chain_1)   # each is (N, ndim) samples array
```

Fast closed-form Surprise assuming both posteriors are multivariate
Gaussians. No nested sampling needed.

| Method | Returns |
|--------|---------|
| `sg.calculate_kld(Nsamples=10000)` | `(kld_value, kld_expected, kld_distribution)` |
| `sg.calculate_surprise(Nsamples=10000)` | `(S_value, S_distribution)` |

Static methods for direct computation:

| Method | Description |
|--------|-------------|
| `SurpriseGauss.kld(chain_2, chain_1)` | Analytical KLD between two Gaussians |
| `SurpriseGauss.expected_kld(cov2, cov1)` | Expected KLD (Eq. A38, Seehars et al. 2014) |
| `SurpriseGauss.surprise(chain_2, chain_1)` | Analytical Surprise value |
| `SurpriseGauss.S_dist(cov2, cov1, N)` | Sample the Surprise distribution |

---

### Standalone functions

#### Nested sampling

```python
from klsurprise import run_nested_sampling, load_create_NS_file

# Run dynesty directly
res = run_nested_sampling(loglikelihood, ndim, domain)

# Load from cache or run and save
res = load_create_NS_file("cache.pkl", loglikelihood, ndim, domain)
```

`run_nested_sampling` accepts keyword arguments for fine-tuning:
`nlive`, `nlive_batch`, `n_effective`, `dlogz`, `maxiter`, `maxbatch`,
`static_NS`, `dynamic_NS`, `print_progress`.

#### KLD computation

```python
from klsurprise import KLD_numerical

kld = KLD_numerical(
    res_p, logP,      # dynesty result + log-prob for distribution p
    res_q, logQ,      # dynesty result + log-prob for distribution q
    domain=domain,
)
```

Computes KLD(p || q) numerically using equally-weighted posterior samples
and evidence-normalized log-probabilities.

#### PPD generation

```python
from klsurprise import create_ppd_chain

ppd = create_ppd_chain(
    th1_samples,       # (N, ndim) posterior samples from D1
    data_model_fun,    # callable(theta) -> data vector
    cov_matrix,        # data-space covariance
    sample_size=10,    # Gaussian draws per posterior sample
    n_jobs=4,
)
```

#### Utilities

```python
from klsurprise import find_pval, sigma_discordance

p = find_pval(S_dist, S_observed)     # fraction of S_dist > S
sigma = sigma_discordance(p)          # sqrt(2) * erfinv(1 - p)
```

#### I/O

```python
from klsurprise import save_dict_to_hdf5

save_dict_to_hdf5("results.hdf5", results_dict)
```

---

## Package structure

```
klsurprise/
    __init__.py            Public API exports
    core.py                surprise_statistics class
    surprise_gaussian.py   Analytical Gaussian Surprise
    likelihood.py          JIT-compiled Gaussian log-likelihood
    sampling.py            Nested sampling + caching
    kld.py                 KLD computation + parallel distribution
    ppd.py                 Posterior Predictive Distribution
    io.py                  HDF5 output
    utils.py               Priors, p-values, sigma conversion
```

---

## Example: DESI BAO vs Pantheon+ SNIa

A complete working example lives in `examples/snia_bao/`:

```bash
cd examples/snia_bao
python run_owcdm.py
```

Compares **DESI BAO+BBN** (D1) against **Pantheon+SH0ES** (D2) under an
open wCDM model. Supports four cosmological models: owCDM, wCDM, oLCDM, FLCDM.

Files:
- `run_owcdm.py` -- entry-point script
- `BAO_likelihood_DESI.py` -- BAO likelihood (marginalized over baryon density)
- `SNIa_likelihood_pantheon.py` -- Pantheon+SH0ES likelihood
- `plot_tools.py` -- triangle plots and histograms
- `data/` -- Pantheon+SH0ES data files

---

## Notes

- All likelihoods must be **JAX-compatible** (`jax.numpy`, not `numpy`).
- Dataset 2 likelihood is assumed **Gaussian** (covariance + model function).
- Only **flat (uniform) priors** are supported.
- NS results are **cached** to `.pkl` files for reuse across sessions.
- Two paths to Surprise: full non-Gaussian pipeline (`surprise_statistics`,
  expensive) or analytical Gaussian approximation (`SurpriseGauss`, fast).

---

## Dependencies

| Package | Version |
|---------|---------|
| [JAX](https://github.com/google/jax) | >= 0.4.20 |
| [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) | >= 0.1.0 |
| [dynesty](https://dynesty.readthedocs.io/) | >= 2.1 |
| [numpy](https://numpy.org/) | >= 1.26 |
| [scipy](https://scipy.org/) | >= 1.11 |
| [h5py](https://www.h5py.org/) | >= 3.10 |
| [joblib](https://joblib.readthedocs.io/) | >= 1.3 |
| [tqdm](https://tqdm.github.io/) | >= 4.66 |

All installed automatically via `pip install`.

---

## License

MIT
