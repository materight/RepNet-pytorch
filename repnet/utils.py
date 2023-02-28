"""Utility functions."""
import os
import requests
from pytube import YouTube



def flatten_dict(dictionary: dict, parent_key: str = '', sep: str = '.', keep_last: bool = False, skip_none: bool = False):
    """Flatten a nested dictionary into a single dictionary with keys separated by `sep`."""
    items = {}
    for k, v in dictionary.items():
        key_prefix = parent_key if parent_key else ''
        key_suffix = k if not skip_none or k is not None else ''
        key_sep = sep if key_prefix and key_suffix else ''
        new_key = key_prefix + key_sep + key_suffix
        if isinstance(v, dict) and (not keep_last or isinstance(next(iter(v.values())), dict)):
            items.update(flatten_dict(v, new_key, sep=sep, keep_last=keep_last, skip_none=skip_none))
        else:
            items[new_key] = v
    return items



def download_file(url: str, dst: str):
    """Download a file from a given url."""
    if 'www.youtube.com' in url:
        yt_object = YouTube(url)
        output_dir, output_file = os.path.split(dst)
        yt_object.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first().download(output_dir, output_file)
    else:
        response = requests.get(url)
        with open(dst, 'wb') as file:
            file.write(response.content)
