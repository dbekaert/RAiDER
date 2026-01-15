import argparse
from pathlib import Path
from typing import List, Optional

class RAiDERCombineArgs(argparse.Namespace):
    raider_file: Path
    raider_folder: List[Path]
    gnss_folder: List[Path]
    gnss_file: Optional[Path]
    raider_column_name: str
    column_name: str
    out_name: Path
    local_time: Optional[str]
    obs_errlimit: float
    min_pct_days: float
    timeinterval: Optional[str]
    allow_nan_for_negative: bool
    verbose: bool
