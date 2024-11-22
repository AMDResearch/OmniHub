def flatten_dict(d, parent_key=""):
    """
    Flatten a nested dictionary by prepending parent keys to each key, and
    converting lists to comma-separated values.

    Args:
    - d (dict): The dictionary to flatten.
    - parent_key (str): Current prefix for the key (used for recursion).

    Returns:
    - dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}#{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))

    items = {k: ",".join(v) if isinstance(v, list) else v for k, v in items}
    return items
