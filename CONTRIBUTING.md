# Contributing to Volume Segmantics
First of all, thank you for your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

We use github to host code, to track issues and feature requests, as well as accept pull requests.

We have a [code of conduct](CODE_OF_CONDUCT.md) that describes how to participate in our community.

## Contributions

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests.

### Setting up your own local development environment

This project was written using [Poetry](https://python-poetry.org/) for packaging and dependency management. As such, if you want to make your own changes to the code, is suggested that you:

1. Install Poetry according to [these instructions](https://python-poetry.org/docs/master/#installing-with-the-official-installer).
2. Clone the repository to your machine
3. `cd` to the repository directory and run `poetry install` to create a virtual-env with dependencies installed.
4. To activate the virtual-env you can use the command `poetry shell` to access to installed dependencies. Alternatively, you can use `poetry run <command>` to execute the command within the environment.
5. To run the tests from outside the environment the command to use is `poetry run pytest`. If the environment is activated, run `pytest tests/`. By default, these commands run the CPU and GPU tests. To exclude the GPU tests, append the flag `-m "not gpu"`. **Please add tests if you've added code that can be tested. If you have the hardware, please run all tests, including GPU tests, before submitting changes**.
6. If you've added methods to the public API, make sure that docstrings have been added/updated.
7. Upload your changes to a new branch and issue a pull request :sparkles:. 

### License - Any contributions you make will be under the Apache Software License v2.0
In short, when you submit code changes, your submissions are understood to be under the same [Apache v2.0 License](LICENSE.md) that covers the project. Feel free to contact the maintainers if that's a concern.

## Reporting bugs

### Report bugs using Github's [issues](https://github.com/DiamondLightSource/volume-segmantics/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/DiamondLightSource/volume-segmantics/issues/new/choose).

### Write bug reports with detail, background, and sample code

**Good Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)


## References
This document was adapted from the open-source contribution guidelines for [Transcriptase](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62) which were in turn adapted from [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)
