"""Sphinx configuration for bolojax docs."""

project = "bolojax"
author = "Brodi Elwood"

extensions = [
    "autoapi.extension",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_math_dollar",
    "myst_parser",
]

# sphinx-autoapi
autoapi_dirs = ["../src/bolojax"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

autodoc_typehints = "description"

# Render $...$ as inline math
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# autoapi generates ambiguous cross-refs for the 'type' discriminator field
# shared across OpticalElement subclasses
suppress_warnings = ["ref.python"]

html_theme = "furo"
html_title = "bolojax"
