"""python -m mecheval <command> [...]: check-data, run, validate, score, report, reproduce, annotate."""
import importlib
import sys

COMMANDS = {'check-data': 'data', 'run': 'run', 'validate': 'validate', 'score': 'score', 'report': 'report',
            'reproduce': 'reproduce', 'annotate': 'annotate'}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__.strip())
        print('\n'.join(f'  {c:<11} python -m mecheval {c} --help' for c in COMMANDS))
        return 2
    command, args = sys.argv[1], sys.argv[2:]
    sys.argv[0] = f'python -m mecheval {command}'     # shown in --help
    module = importlib.import_module(f'.{COMMANDS[command]}', __package__)
    return module.main(args)


sys.exit(main())
