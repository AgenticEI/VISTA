"""Inspect configuration and Harness contracts without launching an experiment."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meta_harness.config import EvolutionConfig
from rpent.harness import load_harness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = EvolutionConfig.load(args.config)
    config.require_ready()
    package = load_harness(config.initial_harness)
    print('Configuration and Harness contracts: OK')
    print('Robot:', config.robot)
    print('H0:', config.initial_harness.relative_to(ROOT))
    print('Harness files:', len(package.files))
    for suite in config.suites:
        print(f'{suite.id}: {len(suite.tasks)} tasks, '
              f'{len(config.matrix(suite.id, "parallel"))} cells per full taskwise round')
    print('No simulator, GPU service, or model request was started.')


if __name__ == '__main__':
    main()
