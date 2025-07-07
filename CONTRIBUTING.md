# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Developer Installation

If something goes wrong at any point during installing the library please see how
[our CI/CD on GitHub Actions](.github/workflows/build-main.yml) installs and builds the
project as it will always be the most up-to-date.

## Test Resources and Git LFS

 This project uses Git Large File Storage (LFS) to store large test resources under /tests/resources/.
 Before cloning, make sure you have Git LFS installed:

 ### macOS
 ```
 brew install git-lfs
 git lfs install
 ```

 ### Debian/Ubuntu
 ```
 sudo apt-get install git-lfs
 git lfs install
 ```

 ### Or install manually: https://git-lfs.github.com/

 ⚠️ If you skip git lfs install, large files will appear as plain-text pointers and your tests may fail.
 
## Get Started!

Ready to contribute? Here's how to set up `bioio-ome-zarr` for local development.

1. Fork the `bioio-ome-zarr` repo on GitHub.

2. Clone your fork locally:

    ```bash
    git clone git@github.com:{your_name_here}/bioio-ome-zarr.git
    ```

3. Install the project in editable mode. (It is also recommended to work in a virtualenv or anaconda environment):

    ```bash
    cd bioio-ome-zarr/
    just setup-dev
    ```

4. Create a branch for local development:

    ```bash
    git checkout -b {your_development_type}/short-description
    ```

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
    Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linting and
   tests with [just](https://github.com/casey/just):

    ```bash
    just build
    ```

6. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin {your_development_type}/short-description
    ```

7. Submit a pull request through the GitHub website.

## Just Commands

For development commands we use [just](https://github.com/casey/just).

```bash
just
```
```
Available recipes:
    build                    # run lint and then run tests
    clean                    # clean all build, python, and lint files
    default                  # list all available commands
    install                  # install with all deps
    lint                     # lint, format, and check all files
    release                  # release a new version
    tag-for-release version  # tag a new version
    test                     # run tests
    update-from-cookiecutter # update this repo using latest cookiecutter-bioio-reader
```

## Deploying

A reminder for the maintainers on how to deploy.
Make sure the main branch is checked out and all desired changes
are merged. Then run:

```bash
just tag-for-release "vX.Y.Z"
just release
```

The presence of a tag starting with "v" will trigger the `publish` step in the
main github workflow, which will build the package and upload it to PyPI. The
version will be injected into the package metadata by
[`setuptools-scm`](https://github.com/pypa/setuptools_scm)
