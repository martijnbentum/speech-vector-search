from pathlib import Path

DEFAULT_STORAGE_DIR = Path("data")
DEFAULT_PROTOTYPE_NAME = "prototypes"


def make_path(directory, name, suffix, overwrite=False, load=False):
    if directory is None:
        directory = DEFAULT_STORAGE_DIR
    else:
        directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = DEFAULT_PROTOTYPE_NAME
    path = directory / f'{name}_{suffix}'
    path = _handle_suffix(path, suffix)

    if not overwrite and not load:
        if path.exists():
            raise FileExistsError(f"file exists: {path}")
    if load:
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
    return path


def _handle_suffix(path, suffix):
    if suffix == 'vectors':
        return path.with_suffix('.npy')
    if suffix == 'metadata':
        return path.with_suffix('.jsonl')
    if suffix == 'config':
        return path.with_suffix('.json')
    m = f"invalid suffix: {suffix}, must be one of 'vectors', "
    m += f"'metadata', or 'config'"
    raise ValueError(m)
