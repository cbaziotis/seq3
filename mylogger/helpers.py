import json
import os
from collections import defaultdict
from pathlib import Path

from glob2 import glob


def files_to_dict(dirs, safe=True):
    file_data = defaultdict(dict)

    for dir in dirs:
        for file in glob(os.path.join(dir + "/*.py")):
            _dir = os.path.split(dir)[1]
            filename = os.path.basename(file)
            if safe:
                filename = filename.replace('.', '[dot]')
            file_data[_dir][filename] = Path(file).read_text()

    return file_data


def dict_to_html(config):
    indent = 2
    msg = json.dumps(config, indent=indent)
    msg = "\n".join([line[2:].rstrip() for line in msg.split("\n")
                     if len(line.strip()) > 3])
    # format with html
    msg = msg.replace('{', '')
    msg = msg.replace('}', '')
    # msg = msg.replace('\n', '<br />')
    return msg
