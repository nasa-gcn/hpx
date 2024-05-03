# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from importlib.metadata import metadata
from sphinx_astropy.conf.v2 import *  # noqa: F403

meta = metadata("hpx")
project = meta["Name"]
# copyright = "2024, Leo Singer"
author = meta["Author-Email"]
release = meta["Version"]

autosummary_generate = True
