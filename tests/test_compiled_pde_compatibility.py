"""Public module and serialization identities captured from the c85b403 wheel."""

from __future__ import annotations

import base64
import inspect
import json
import pickle
from pathlib import Path

from finite_difference_options import integrations
from finite_difference_options.integrations import compiled_pde_adapter as adapter


def test_compiled_adapter_legacy_public_contract() -> None:
    reference = json.loads((Path(__file__).parent / "fixtures/compiled_pde_adapter_v0_contract.json").read_text())
    assert adapter.__all__ == reference["exports"]
    for name, expected in reference["metadata"].items():
        member = getattr(adapter, name)
        assert member.__module__ == expected["module"]
        assert str(inspect.signature(member)) == expected["signature"]
        if name in integrations.__all__:
            assert getattr(integrations, name) is member
    screen = adapter.screen_compiled_pde_payload(adapter.packaged_compiled_black_scholes_fixture())
    assert json.loads(json.dumps(screen.as_dict())) == reference["screen"]
    # These bytes are trusted, checked-in synthetic DTOs from the pre-extraction wheel.
    for encoded, expected in zip(
        reference["pickles"],
        [adapter.CompiledPDEDiagnostic("unsupported", "synthetic", "payload"), screen],
        strict=True,
    ):
        restored = pickle.loads(base64.b64decode(encoded))
        assert type(restored) is type(expected)
        assert restored == expected
        assert pickle.loads(pickle.dumps(restored, protocol=4)) == expected


def test_fd_verification_legacy_public_contract() -> None:
    from finite_difference_options.validation import fd_verification

    reference = json.loads((Path(__file__).parent / "fixtures/compiled_pde_adapter_v0_contract.json").read_text())[
        "verification"
    ]
    assert fd_verification.__all__ == reference["exports"]
    for name, expected in reference["metadata"].items():
        member = getattr(fd_verification, name)
        assert member.__module__ == expected["module"]
        assert str(inspect.signature(member)) == expected["signature"]
    for name, expected in reference["constants"].items():
        assert getattr(fd_verification, name) == expected
