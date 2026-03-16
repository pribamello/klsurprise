# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the project root so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = "klsurprise"
copyright = "2025, Riba Mello et al."
author = "Riba Mello et al."
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "nbsphinx_link",
]

# Napoleon settings (for NumPy / Google style docstrings)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True

# nbsphinx settings – never execute notebooks during the build
nbsphinx_execute = "never"

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx mapping for cross-referencing external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "dynesty": ("https://dynesty.readthedocs.io/en/stable/", None),
}

# Files / directories to ignore
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output --------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# Short title for the sidebar
html_short_title = "klsurprise"
