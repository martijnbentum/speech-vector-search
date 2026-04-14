import json
import os


def ensure_directory(path):
    '''create directory if needed.
    path                    directory path
    '''
    os.makedirs(path, exist_ok=True)


def read_json(path):
    '''load json from disk.
    path                    json file path
    '''
    with open(path) as handle:
        return json.load(handle)


def write_json(data, path):
    '''write json to disk.
    data                    object to serialize
    path                    json file path
    '''
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def save_jsonl(rows, path):
    '''save metadata rows to jsonl.
    rows                     metadata records
    path                     jsonl file path
    '''
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row_to_dict(row)) + "\n")


def load_jsonl(path):
    '''load rows from jsonl.
    path                    jsonl file path
    '''
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            rows.append(row)
    return rows


def row_to_dict(row):
    '''convert a metadata row-like object to a dict.'''
    if isinstance(row, dict):
        return row
    if hasattr(row, 'to_dict'):
        return row.to_dict()
    raise ValueError('row must be a dict or expose to_dict')


def label_value(row):
    '''extract the comparable label from a row-like object.'''
    if isinstance(row, dict):
        for key in ('label', 'word', 'text'):
            if key in row and row[key]:
                return row[key]
    else:
        for key in ('label', 'word', 'text'):
            if hasattr(row, key):
                value = getattr(row, key)
                if value:
                    return value
    raise ValueError('row must contain a label')


def infer_label_key(row):
    '''infer which field stores the label on a row-like object.'''
    if isinstance(row, dict):
        for key in ('label', 'word', 'text'):
            if key in row and row[key]:
                return key
    else:
        for key in ('label', 'word', 'text'):
            if hasattr(row, key) and getattr(row, key):
                return key
    raise ValueError('row must contain a label field')
