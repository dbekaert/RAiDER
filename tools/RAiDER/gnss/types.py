import argparse
from pathlib import Path
from typing import Optional


class RAiDERCombineArgs(argparse.Namespace):
    raider_file: Path
    raider_folder: Path
    gnss_folder: Path
    gnss_file: Optional[Path]
    raider_column_name: str
    column_name: str
    out_name: Path
    local_time: Optional[str]
