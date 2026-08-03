from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "scan_private_identifiers.py"


def _load_scanner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scan_private_identifiers", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scan_private_identifiers = _load_scanner()


def _openai_prefixed_token(prefix: str = "proj") -> str:
    return "sk-" + prefix + "-" + ("A" * 24)


def test_scan_detects_dotenv_files_without_printing_secret(capsys, tmp_path: Path) -> None:
    token = _openai_prefixed_token()
    dotenv = tmp_path / ".env.local"
    dotenv.write_text("OPENAI_API_KEY=" + token + "\n", encoding="utf-8")

    assert scan_private_identifiers.main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "openai_style_token: .env.local" in output
    assert token not in output


def test_scan_detects_openai_svcacct_prefixed_keys_in_decodable_text(tmp_path: Path) -> None:
    token = _openai_prefixed_token("svcacct")
    secret_file = tmp_path / "config"
    secret_file.write_text("token = " + token + "\n", encoding="utf-8")

    hits = scan_private_identifiers.scan_root(tmp_path)

    assert scan_private_identifiers.ScanHit("openai_style_token", "config") in hits


def test_scan_detects_legacy_openai_keys_and_github_tokens(tmp_path: Path) -> None:
    legacy_openai = "sk-" + ("B" * 24)
    github_token = "ghp_" + ("C" * 24)
    text_file = tmp_path / "credentials.txt"
    text_file.write_text("\n".join([legacy_openai, github_token]), encoding="utf-8")

    hits = scan_private_identifiers.scan_root(tmp_path)

    assert scan_private_identifiers.ScanHit("openai_style_token", "credentials.txt") in hits
    assert scan_private_identifiers.ScanHit("github_token", "credentials.txt") in hits


def test_scan_skips_binary_dotenv_files(tmp_path: Path) -> None:
    binary_dotenv = tmp_path / ".env.production"
    binary_dotenv.write_bytes(b"\x00OPENAI_API_KEY=" + _openai_prefixed_token().encode("ascii"))

    assert scan_private_identifiers.scan_root(tmp_path) == []


def test_dotenv_templates_are_explicit_and_still_scanned_for_real_secrets(tmp_path: Path) -> None:
    assert scan_private_identifiers.SAFE_DOTENV_TEMPLATES == {
        ".env.example",
        ".env.sample",
        ".env.template",
    }
    placeholder_template = tmp_path / ".env.sample"
    placeholder_template.write_text(
        "OPENAI_API_KEY=" + "sk-" + "proj-" + "REPLACE_WITH_OPENAI_KEY" + "\n",
        encoding="utf-8",
    )
    real_secret_template = tmp_path / ".env.example"
    real_secret_template.write_text("OPENAI_API_KEY=" + _openai_prefixed_token() + "\n", encoding="utf-8")

    hits = scan_private_identifiers.scan_root(tmp_path)

    assert scan_private_identifiers.ScanHit("openai_style_token", ".env.example") in hits
    assert scan_private_identifiers.ScanHit("openai_style_token", ".env.sample") not in hits
