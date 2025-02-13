project = "Collaborative Coding Exam"
copyright = "2025, SFI Visual Intelligence"
author = "SFI Visual Intelligence"
release = "0.0.1"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "myst_parser",  # in order to use markdown
    "autoapi.extension",  # in order to generate API documentation
]

# search this directory for Python files
autoapi_dirs = ["../CollaborativeCoding"]

myst_enable_extensions = [
    "colon_fence",  # ::: can be used instead of ``` for better rendering
]

html_theme = "sphinx_rtd_theme"
