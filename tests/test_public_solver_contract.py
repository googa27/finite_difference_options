"""Public solver-contract smoke aliases for issue #141 exact gate."""

from __future__ import annotations

from finite_difference_options.integrations import released_fd_solver_contract
from finite_difference_options.integrations.compiled_pde_adapter import packaged_compiled_black_scholes_fixture


def test_public_solver_contract_keeps_compiled_pde_on_separate_adapter() -> None:
    contract = released_fd_solver_contract()
    fixture = packaged_compiled_black_scholes_fixture()

    assert "public-synthetic.compiled-pde.black-scholes-call.v0" not in contract.supported_problem_ids
    assert (
        "finite_difference_options.integrations.compiled_pde_adapter:solve_compiled_pde_payload"
        in contract.entry_points
    )
    assert "googa27/finite_difference_options#141" in contract.issue_refs
    assert fixture["schema_version"] == "finite-difference-options.compiled-pde-adapter-fixture/v0"
    assert fixture["compiled_operator_result"]["accepted"] is True
