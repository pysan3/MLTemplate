# MLTemplate

My Machine Learning Template.


## Setup

### Requirements

- `poetry >= 1.2`
    - [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)[How to install `poetry`.](How to install `poetry`.)
    - `curl -sSL https://install.python-poetry.org | python3 -`
- `python v3.11.1`
    - `pyenv` will be useful.
    - [https://github.com/pyenv/pyenv#installation](https://github.com/pyenv/pyenv#installation)[How to install `pyenv`.](How to install `pyenv`.)
    - `curl https://pyenv.run | bash`


### Clone

```bash
git clone git@github.com:pysan3/MLTemplate.git && cd MLTemplate
```


### Create Environment

- Do this before opening VSCode. (Will setup linter and formatter)
- Same commands on Windows as well.

```zsh
# Setup local python
pyenv install 3.11.1
python -V # ==> Check python version is `3.11.1`

# Start poetry setup
poetry install
poetry shell # Activates poetry environment.
```

-  Do you see `(mltemplate-py3.11)` before your `$PS1`?


## Run

### Help

```zsh
# Activate poetry environment
poetry shell

mlt --help
```


### Training

**WIP**
