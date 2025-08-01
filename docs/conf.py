# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
# sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
sys.path.insert(0, str(Path("../src").resolve()))
sys.path.insert(0, str(Path("..").resolve()))
# sys.path.insert(0, str(Path("../..").resolve()))
print(sys.path)
print(sys.executable)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "awkward-zipper"
copyright = "2025, Peter Fackeldey, Maxym Naumchyk"
author = "Peter Fackeldey, Maxym Naumchyk"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
