# Developer Corner
## Pre-commit
We use pre-commit hooks to standardize some coding standards and guidelines across the codebase. After cloning this
repo, follow the below steps to install pre-commit hooks in your developer environment.

```
# Using pip
pip install pre-commit
# Or, using conda
# conda install -c conda-forge pre-commit
# run pre-commit install to set up the git hook scripts
pre-commit install
```

## Pull Requests
Create feature branches by using the following naming convention and then create pull requests.
```
git branch -b $USER/my-awesome-feature
```
