# earth-text
Adding language to Clay

## Installation

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html) and then run the following commands to create the earth-text-env environment:

```bash
conda env create -f environment.yml

conda activate earth-text-env
```

Next, install the package:

```bash
pip install -e .
```

or if you want development dependencies as well:

```bash
pip install -e .[dev]
```

### Optional, but highly recommended

Install [pre-commit](https://pre-commit.com/) by running the following command to automatically run code formatting and linting before each commit:

```bash
pre-commit install
```

If using pre-commit, each time you commit, your code will be formatted, linted, checked for imports, merge conflicts, and more. If any of these checks fail, the commit will be aborted.

## Adding a new package

To add a new package to the environment, open pyproject.toml file and add the package name to "dependencies" list. Then, run the following command to install the new package:

```bash
pip install -e . # or .[dev]
```
