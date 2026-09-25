import importlib.metadata

try:
    __version__ = importlib.metadata.version("transformer_lab")
except importlib.metadata.PackageNotFoundError:
    pass
