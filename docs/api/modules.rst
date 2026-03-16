klsurprise public API
=====================

This page documents the public interface of the ``klsurprise`` package.

Main class
----------

.. autoclass:: klsurprise.surprise_statistics
   :members:
   :undoc-members:
   :show-inheritance:

Gaussian Surprise
-----------------

.. autoclass:: klsurprise.SurpriseGauss
   :members:
   :undoc-members:
   :show-inheritance:

Likelihood
----------

.. autofunction:: klsurprise.logL2_jitted

Nested Sampling
---------------

.. autofunction:: klsurprise.run_nested_sampling

.. autofunction:: klsurprise.load_create_NS_file

.. autofunction:: klsurprise.process_batch

KL Divergence
-------------

.. autofunction:: klsurprise.KLD_numerical

.. autofunction:: klsurprise.kld_worker

.. autofunction:: klsurprise.compute_kld_distribution

Posterior Predictive Distribution
---------------------------------

.. autofunction:: klsurprise.generate_samples

.. autofunction:: klsurprise.create_ppd_chain

I/O
---

.. autofunction:: klsurprise.save_dict_to_hdf5

.. autofunction:: klsurprise.load_dict_from_hdf5

Utilities
---------

.. autofunction:: klsurprise.sampler

.. autofunction:: klsurprise.calculate_flat_prior_volume

.. autofunction:: klsurprise.find_pval

.. autofunction:: klsurprise.sigma_discordance
