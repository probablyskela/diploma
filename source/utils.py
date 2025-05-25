import json
import hashlib


def hash_dict(d: dict) -> int:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()
