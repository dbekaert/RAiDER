"""
Script to assist in regenerating each Python version's lockfiles after making
changes to environment.yml.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import yaml
from tqdm import tqdm

from RAiDER.logger import logger


# Matches the "dependencies" entry for python.
# First group: like " - python"
# Second group: like ">=3.9"
PATTERN_PYTHON_DEP = re.compile(r'^(\s*-\s*python)([<>=~]?=?.+)$', re.MULTILINE)


def generate_lock(out_path: Path, version: str, template: str) -> None:
    """
    Use conda-lock to generate a lockfile for this version of the
    environment.yml file, and place it at the specified output path.
    """
    logger.info(f'Generating lockfile for Python {version}...')
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        env_path = Path(tmp_dir_str) / 'environment.yml'

        # Hardcode a copy of the environment.yml file to this Python version
        with env_path.open('w', encoding='utf-8') as f_tmp_env:
            f_tmp_env.write(re.sub(PATTERN_PYTHON_DEP, f'\\1={version}', template))

        # Platforms explicitly listed in order to exclude win-64, since isce3
        # and wand (and therefore RAiDER) are not compatible with Windows.
        cmd = f'conda-lock -f {env_path} --lockfile {out_path} -p linux-64 -p osx-64 -p osx-arm64'
        logger.debug(f'>>> {cmd}')
        result = subprocess.run(cmd.split(' '), stdout=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise Exception(f'Command exited with non-zero status {result.returncode}: "{cmd}"')


def main() -> None:
    """(Re)generates a lockfile for each Python version in .circleci/config.yml."""
    with Path('environment.yml').open(encoding='utf-8') as f_env:
        yml_content = f_env.read()

    # Read RAiDER's supported Python versions from CircleCI config.
    # The last entry in the list will be placed in the project root.
    with Path('.circleci/config.yml').open(encoding='utf-8') as f_ci_config:
        # fmt: off
        versions: list[str] = yaml.safe_load(f_ci_config) \
            ['workflows']['all-tests']['jobs'][0] \
            ['build']['matrix']['parameters']['python-version']
        # fmt: on

    for i, version in tqdm(enumerate(versions), total=len(versions), unit='lockfiles written'):
        if i < len(versions) - 1:
            generate_lock(Path(f'.circleci/conda-lock-{version}.yml'), version, yml_content)
        else:
            generate_lock(Path('conda-lock.yml'), version, yml_content)


if __name__ == '__main__':
    main()
