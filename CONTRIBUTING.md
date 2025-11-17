# Contributing to WoodTapper

You are more than welcome to propose and contribute to new developments. You can contribute by:
- raising problems you encounter by opening an GitHub issue
- resolving issues already detected and marked open
- developing new features in the roadmap or that would fit well in the library
- providing additional examples of use
- fixing typos, improving code quality, comments, etc...
- developing additional tests to improve the coverage

We encourage you to open an [issue](https://github.com/artefactory/woodtapper/issues) so that we can discuss your different ideas.

# Workflow

The easiest workflow to integrate contributions is the following:
1. Fork the repository
2. Clone your fork
3. Commit and push changes to your fork
4. Open a Pull Request from your fork to the *main* branch of the original repository

# Local Setup

Clone the repository and install the right environment for development using:
```bash
git clone https://github.com/artefactory/woodtapper.git
cd woodtapper
pip install -e .[dev,docs]
```

# Documentation
New features should be documented with:
- NumPy style Docstring
- An example in a notebook
- MkDoc based on the notebook (It is pretty much automatized)

# Tests & CI

Current CI should pass for any change to validated. It currently ensures style based verifications as well as runs some tests. When a Pull Request is opened, the CI is automatically run.
