import os
import sys

sys.path.append(".")

project = "AnoMed Deanonymizer"
copyright = "2025, Yannik Potdevin"
author = "Yannik Potdevin"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",  # to parse NumPy docstrings
    "myst_parser",  # to parse .md files too
    "autodoc2",  # this is more compatible with MyST than Sphinx' autodoc
]

templates_path = ["_templates"]
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
html_theme = "sphinx_rtd_theme"

myst_enable_extensions = [
    "fieldlist",
]

autodoc2_packages = [
    "../src/anomed_deanonymizer",
]
autodoc2_docstring_parser_regexes = [
    # this will render all docstrings as Markdown
    (r".*", "docstrings_parser"),
]
autodoc2_module_all_regexes = [
    r"anomed_deanonymizer\..*",
]
autodoc2_hidden_objects = ["private", "inherited"]
