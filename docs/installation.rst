Installation
============

Requirements
------------

* Python >= 3.9
* A working JAX installation (CPU or GPU)

Install from source
-------------------

Clone the repository and install in a virtual environment:

.. code-block:: bash

   git clone https://github.com/pribamello/klsurprise.git
   cd klsurprise

   python -m venv .venv
   source .venv/bin/activate

   pip install -e .

To run the example notebooks you will also need extra dependencies:

.. code-block:: bash

   pip install -e ".[examples]"

.. note::

   ``jax-cosmo`` requires ``setuptools < 82``.  The ``[examples]`` extra
   already pins this, so there is nothing extra to do if you install with
   the command above.

Dependencies
------------

The following packages are installed automatically:

.. list-table::
   :header-rows: 1

   * - Package
     - Minimum version
   * - dynesty
     - 2.1
   * - h5py
     - 3.10
   * - jax
     - 0.4.20
   * - jaxlib
     - 0.4.20
   * - joblib
     - 1.3
   * - numpy
     - 1.26
   * - scipy
     - 1.11
   * - tqdm
     - 4.66
