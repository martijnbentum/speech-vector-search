from pathlib import Path


DEFAULT_STORAGE_DIR = Path("data")
DEFAULT_TOKEN_NAME = "tokens"
DEFAULT_PROTOTYPE_NAME = "prototypes"


def default_storage_dir():
    '''return the default storage directory.
    no parameters            returns default storage directory
    '''
    return DEFAULT_STORAGE_DIR


def default_name():
    '''return the default base name for saved files.
    no parameters            returns default base name
    '''
    return default_token_name()


def default_token_name():
    '''return the default base name for token files.
    no parameters            returns default token base name
    '''
    return DEFAULT_TOKEN_NAME


def default_prototype_name():
    '''return the default prototype artifact directory name.
    no parameters            returns default prototype name
    '''
    return DEFAULT_PROTOTYPE_NAME


def token_embeddings_path(directory=None, name=None):
    '''build the token embedding path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    directory = resolve_directory(directory)
    name = resolve_token_name(name)
    return directory / (name + ".embeddings.npz")


def token_metadata_path(directory=None, name=None):
    '''build the token metadata path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    directory = resolve_directory(directory)
    name = resolve_token_name(name)
    return directory / (name + ".metadata.jsonl")


def prototype_vectors_path(directory=None, name=None):
    '''build the prototype vector path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    return prototype_directory(directory, name) / "prototypes.npy"


def prototype_metadata_path(directory=None, name=None):
    '''build the prototype metadata path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    return prototype_directory(directory, name) / "metadata.jsonl"


def config_path(directory=None, name=None):
    '''build the config path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    return prototype_directory(directory, name) / "config.json"


def prototype_directory(directory=None, name=None):
    '''build the prototype artifact directory.
    directory                optional storage directory
    name                     optional artifact directory name
    '''
    directory = resolve_directory(directory)
    name = resolve_prototype_name(name)
    return directory / name


def resolve_directory(directory):
    '''resolve the storage directory.
    directory                optional storage directory
    '''
    if directory is None: return default_storage_dir()
    return Path(directory)


def resolve_name(name):
    '''resolve the base name for saved files.
    name                     optional base name without extension
    '''
    if name is None: return default_name()
    return name


def resolve_token_name(name):
    '''resolve the base name for token files.
    name                     optional base name without extension
    '''
    if name is None:
        return default_token_name()
    return name


def resolve_prototype_name(name):
    '''resolve the name for prototype artifacts.
    name                     optional artifact directory name
    '''
    if name is None:
        return default_prototype_name()
    return name
