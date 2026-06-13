"""quack, a knowledge layer over your local work that helps LLMs navigate it."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quackspace")
except PackageNotFoundError:
    __version__ = "0.0.0"
