# list all available commands
default:
  just --list

# clean all build, python, and lint files
clean:
	rm -fr build
	rm -fr dist
	rm -fr .eggs
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .coverage
	rm -fr coverage.xml
	rm -fr htmlcov
	rm -fr .pytest_cache
	rm -fr .mypy_cache

# install with all deps
install:
	pip install -e .[lint,test]
	pre-commit install

setup-dev:
	just install
	python scripts/download_test_resources.py

# lint, format, and check all files
lint:
	pre-commit run --all-files

# run tests
test:
	pytest --cov-report xml --cov-report html --cov=bioio_ome_zarr bioio_ome_zarr/tests

# run performance tests
benchmark:
	python scripts/benchmark.py

# run lint and then run tests
build:
	just lint
	just test

# tag a new version
tag-for-release version:
	git tag -a "{{version}}" -m "{{version}}"
	echo "Tagged: $(git tag --sort=-version:refname| head -n 1)"

# release a new version
release:
	git push --follow-tags

# update this repo using latest cookiecutter-bioio-reader
update-from-cookiecutter:
	pip install cookiecutter
	cookiecutter gh:bioio-devs/cookiecutter-bioio-reader --config-file .cookiecutter.yaml --no-input --overwrite-if-exists --output-dir ..