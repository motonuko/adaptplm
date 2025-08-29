import json
from pathlib import Path

from adaptplm.viz.exp_result import ExpResult


def parse(file_path: Path) -> ExpResult:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return ExpResult.from_dict(data)
