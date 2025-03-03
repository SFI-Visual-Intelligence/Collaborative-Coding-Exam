project = "Collaborative Coding Exam"
copyright = "2025, SFI Visual Intelligence"
author = "SFI Visual Intelligence"
release = "1.1.0"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "myst_parser",  # in order to use markdown
    "autoapi.extension",  # in order to generate API documentation
    "sphinx.ext.mathjax",  # in order to render math equations
]

# search this directory for Python files
autoapi_dirs = ["../CollaborativeCoding"]

myst_enable_extensions = [
    "colon_fence",  # ::: can be used instead of ``` for better rendering
    "dollarmath",  # $...$ can be used for math equations
]

html_theme = "sphinx_rtd_theme"

html_css_files = ["custom.css"]
