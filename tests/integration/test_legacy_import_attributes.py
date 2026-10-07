"""Characterize five incidental 0.1.x import attributes without advertising them.

These consumer imports survived issue #188's lint repair. Removal requires the
deliberate namespace audit in docs/CI_POLICY.md; they are not new public API.
The standard-library suite also runs against an isolated normal wheel.
"""

from __future__ import annotations

import unittest


class LegacyImportAttributeTests(unittest.TestCase):
    def test_validation_problem_id(self) -> None:
        from finite_difference_options.integrations._compiled_pde_contracts import EXPECTED_PROBLEM_ID as canonical
        from finite_difference_options.integrations._compiled_pde_validation import EXPECTED_PROBLEM_ID as legacy

        self.assertIs(legacy, canonical)

    def test_grid_metrics_solver(self) -> None:
        from finite_difference_options.integrations.compiled_pde_black_scholes_route import (
            _solve_compiled_black_scholes_grid as canonical,
        )
        from finite_difference_options.validation.fd_evidence.grid_metrics import (
            _solve_compiled_black_scholes_grid as legacy,
        )

        self.assertIs(legacy, canonical)

    def test_perturbations_solver(self) -> None:
        from finite_difference_options.integrations.compiled_pde_black_scholes_route import (
            _solve_compiled_black_scholes_grid as canonical,
        )
        from finite_difference_options.validation.fd_evidence.perturbations import (
            _solve_compiled_black_scholes_grid as legacy,
        )

        self.assertIs(legacy, canonical)

    def test_verification_solver(self) -> None:
        from finite_difference_options.integrations.compiled_pde_black_scholes_route import (
            _solve_compiled_black_scholes_grid as canonical,
        )
        from finite_difference_options.validation.fd_verification import _solve_compiled_black_scholes_grid as legacy

        self.assertIs(legacy, canonical)

    def test_verification_upper_boundary(self) -> None:
        from finite_difference_options.integrations.compiled_pde_black_scholes_route import (
            _upper_call_boundary as canonical,
        )
        from finite_difference_options.validation.fd_verification import _upper_call_boundary as legacy

        self.assertIs(legacy, canonical)


if __name__ == "__main__":
    unittest.main()
