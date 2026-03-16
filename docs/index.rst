klsurprise documentation
========================

.. image:: https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg
   :target: https://arxiv.org/abs/2501.xxxxx

Compute the **non-Gaussian Surprise** statistic between two datasets using
nested sampling.

``klsurprise`` quantifies dataset concordance beyond the Gaussian
approximation.  The core quantity is the *Surprise*:

.. math::

   S = D_{\mathrm{KL}}\!\bigl(p\,|\,D_2\;\|\;p\,|\,D_1\bigr)
     - \bigl\langle D_{\mathrm{KL}} \bigr\rangle_{\mathrm{PPD}}

where the expected KLD is estimated from the Posterior Predictive Distribution
(PPD) of dataset 2.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/example_snia_bao

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
