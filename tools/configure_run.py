"""Create a local schema-2 config with explicitly selected model endpoints."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meta_harness.config import EvolutionConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['spatial', 'object', 'goal', '10'],
                        help='Select one LIBERO suite while retaining its full task/scene matrix.')
    parser.add_argument('--planner-model', required=True,
                        help='Full planner identifier, e.g. openai-responses:<model-id>.')
    parser.add_argument('--meta-model', required=True,
                        help='Model identifier accepted by your Codex-compatible provider.')
    parser.add_argument('--api-base-url', required=True,
                        help='Base URL shared by the planner and Meta provider.')
    args = parser.parse_args()
    if args.output.resolve() == args.template.resolve():
        parser.error('Write a separate local config; keep the template unchanged.')
    if args.output.exists():
        parser.error('Output already exists; select a new path.')
    raw = json.loads(args.template.read_text())
    if args.suite:
        suites = [suite for suite in raw['suites'] if suite['id'] == args.suite]
        if len(suites) != 1:
            parser.error(f'Template does not contain suite {args.suite!r}.')
        raw['suites'] = suites
        raw['mode'] = 'probe'  # A suite subset, with unchanged tasks and budgets.
    command = raw['episode']['command']
    command[command.index('--model') + 1] = args.planner_model
    if '--base-url' in command:
        command[command.index('--base-url') + 1] = args.api_base_url
    else:
        command.extend(['--base-url', args.api_base_url])
    raw['meta']['model'] = args.meta_model
    raw['meta']['provider'] = {
        'name': 'configured', 'base_url': args.api_base_url,
        'env_key': 'OPENAI_API_KEY', 'wire_api': 'responses',
    }
    config = EvolutionConfig.parse(raw, root=ROOT)
    config.require_ready()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, indent=2) + '\n')
    print(f'Wrote {args.output}; no services or model requests were started.')


if __name__ == '__main__':
    main()
