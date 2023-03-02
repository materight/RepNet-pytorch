"""Utility functions."""
import os
import shutil
import requests
import yt_dlp



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



YOUTUB_DL_DOMAINS = ['youtube.com', 'imgur.com', 'reddit.com']
def download_file(url: str, dst: str):
    """Download a file from a given url."""
    if any(domain in url for domain in YOUTUB_DL_DOMAINS):
        # Download video from YouTube
        with yt_dlp.YoutubeDL(dict(format='bestvideo[ext=mp4]/mp4', outtmpl=dst, quiet=True)) as ydl:
            ydl.download([url])
    elif url.startswith('http://') or url.startswith('https://'):
        # Download file from HTTP
        response = requests.get(url, timeout=10)
        with open(dst, 'wb') as file:
            file.write(response.content)
    elif os.path.exists(url) and os.path.isfile(url):
        # Copy file from local path
        shutil.copyfile(url, dst)
    else:
        raise ValueError(f'Invalid url: {url}')
