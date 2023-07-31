# Setting Up the Repository

1. Turn your project into a GitHub repository:
    - Make an account on [github.com](https://github.com)
    - Go to [make a new repository](https://github.com/new)
    - _Recommendations:_
        - _It is strongly recommended to make the repository name the same as the Python
            package name_
        - _A lot of the following optional steps are *free* if the repository is Public,
            plus open source is cool_
    - After a GitHub repo has been created, run the commands listed under:
        "...or push an existing repository from the command line"
2. Ensure that you have set GitHub pages to build the `gh-pages` branch by selecting the
   `gh-pages` branch in the dropdown in the "GitHub Pages" section of the
   repository settings.
   ([Repo Settings](https://github.com/BrianWhitneyAI/bioio-ome-zarr/settings))
3. Register your project with PyPI:
    - Make an account on [pypi.org](https://pypi.org)
    - Go to your GitHub repository's settings and under the
      [Secrets tab](https://github.com/BrianWhitneyAI/bioio-ome-zarr/settings/secrets/actions),
      add a secret called `PYPI_TOKEN` with your password for your PyPI account.
      Don't worry, no one will see this password because it will be encrypted.
    - Next time you push to the branch `main` after using
      `just tag-for-release {version}` and `just release`, GitHub
      actions will build and deploy your Python package to PyPI.

You can delete this file once complete!