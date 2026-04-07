from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from rust_mjai_bot import RustMjaiBot


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = [
    ROOT / "tools" / "fixtures" / "competition_call_choice.jsonl",
    ROOT / "tools" / "fixtures" / "competition_riichi_discard.jsonl",
]


def build_python_oracle(fixture: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "mjai_oracle.py"), "--fixture", str(fixture)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class RustMjaiBotStateTest(unittest.TestCase):
    def test_wrapper_matches_python_oracle(self) -> None:
        for fixture in FIXTURES:
            oracle = build_python_oracle(fixture)
            bot = RustMjaiBot(player_id=int(oracle["player_id"]))
            for step in oracle["steps"]:
                events = step["events"]
                reaction = bot.react(json.dumps(events, ensure_ascii=True, separators=(",", ":")))
                self.assertEqual(reaction, step["snapshot"]["state"]["reaction"])
                self.assertEqual(bot.snapshot(), step["snapshot"])


if __name__ == "__main__":
    unittest.main()