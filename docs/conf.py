# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from importlib.metadata import metadata
from sphinx_astropy.conf.v2 import *  # noqa: F403
from sphinx_astropy.conf.v2 import html_theme_options

meta = metadata("hpx")
project = meta["Name"]
# copyright = "2024, Leo Singer"
author = meta["Author-Email"]
release = meta["Version"]

html_theme_options.update(
    {
        "github_url": "https://github.com/astropy/astropy",
        "use_edit_page_button": True,
    }
)

html_context = {
    "github_user": "astropy",
    "github_repo": "astropy",
    "github_version": "main",
    "doc_path": "docs",
}

autosummary_generate = True
