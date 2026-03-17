import os


DEFAULT_STORAGE_DIR = "data"
DEFAULT_NAME = "tokens"


def default_storage_dir():
    '''return the default storage directory.
    no parameters            returns default storage directory
    '''
    return DEFAULT_STORAGE_DIR


def default_name():
    '''return the default base name for saved files.
    no parameters            returns default base name
    '''
    return DEFAULT_NAME


def token_embeddings_path(directory=None, name=None):
    '''build the token embedding path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    directory = resolve_directory(directory)
    name = resolve_name(name)
    return os.path.join(directory, name + ".npz")


def token_metadata_path(directory=None, name=None):
    '''build the token metadata path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    directory = resolve_directory(directory)
    name = resolve_name(name)
    return os.path.join(directory, name + ".jsonl")


def prototype_vectors_path(directory=None, name=None):
    '''build the prototype vector path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    directory = resolve_directory(directory)
    name = resolve_name(name)
    return os.path.join(directory, name + ".npy")


def prototype_metadata_path(directory=None, name=None):
    '''build the prototype metadata path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    directory = resolve_directory(directory)
    name = resolve_name(name)
    return os.path.join(directory, name + ".jsonl")


def config_path(directory=None, name=None):
    '''build the config path.
    directory                optional storage directory
    name                     optional base name without extension
    '''
    directory = resolve_directory(directory)
    name = resolve_name(name)
    return os.path.join(directory, name + ".json")


def resolve_directory(directory):
    '''resolve the storage directory.
    directory                optional storage directory
    '''
    if directory is None: return default_storage_dir()
    return directory


def resolve_name(name):
    '''resolve the base name for saved files.
    name                     optional base name without extension
    '''
    if name is None: return default_name()
    return name
