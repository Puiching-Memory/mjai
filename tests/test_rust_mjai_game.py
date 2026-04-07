from __future__ import annotations

import unittest

from rust_mjai_bot import kyoku_to_zero_indexed_kyoku, to_rank


class RustMjaiGameTest(unittest.TestCase):
    def test_to_rank_matches_expected_ordering(self) -> None:
        self.assertEqual(to_rank([25000, 25000, 25000, 25000]), [1, 2, 3, 4])
        self.assertEqual(to_rank([2500, 60000, 5000, -900]), [3, 1, 2, 4])

    def test_kyoku_conversion(self) -> None:
        self.assertEqual(kyoku_to_zero_indexed_kyoku("E", 1), 0)
        self.assertEqual(kyoku_to_zero_indexed_kyoku("S", 1), 4)
        self.assertEqual(kyoku_to_zero_indexed_kyoku("W", 2), 9)


if __name__ == "__main__":
    unittest.main()