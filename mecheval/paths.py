"""Where MechEval finds the hooked game programs, the reference environment files and its data.

MECHEVAL_GAMES may point to another directory with the same layout (gamesofts/ and environment_files/); the hooked
programs and the reference files are checked against their SHA-256 digests before use.
"""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
GAMES = Path(os.environ.get('MECHEVAL_GAMES', PACKAGE/'games')).resolve()
GAME_SOURCES = GAMES                                   # on sys.path, so that `gamesofts` is importable
ENVIRONMENT_FILES = GAMES/'environment_files'
DATA = PACKAGE/'data'
