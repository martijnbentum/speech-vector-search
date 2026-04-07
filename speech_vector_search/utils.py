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


def infer_label_key(row):
    '''find label key in metadata row.
    row                     metadata record
    '''
    if hasattr(row, "label"):
        return "label"
    if "label" in row:
        return "label"
    raise ValueError("metadata row must contain 'label'")


def label_value(row):
    '''return the label value from a metadata row.
    row                     metadata record
    '''
    return row[infer_label_key(row)]

def save_jsonl(rows, path):
    '''save metadata rows to jsonl.
    rows                     metadata records
    path                     jsonl file path
    '''
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

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
