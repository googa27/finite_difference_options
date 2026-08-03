from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTS = {".md", ".py", ".toml", ".yml", ".yaml", ".json", ".txt"}
IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
PATTERNS = {
    "rut_like_identifier": re.compile(
        r"\b\d{1,2}\.\d{3}\.\d{3}-[0-9Kk]\b|\b\d{7,8}-[0-9Kk]\b"
    ),
    "github_token": re.compile(r"ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    "openai_style_token": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "private_data_path": re.compile(r"data/private/"),
}
ALLOWED_PRIVATE_PATH_REFS = {
    ".gitignore",
    "AGENTS.md",
    "scripts/scan_private_identifiers.py",
}


def main() -> int:
    hits: list[tuple[str, str]] = []
    for path in ROOT.rglob("*"):
        if any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if not path.is_file() or path.suffix not in TEXT_EXTS:
            continue
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            if name == "private_data_path" and rel in ALLOWED_PRIVATE_PATH_REFS:
                continue
            if pattern.search(text):
                hits.append((name, rel))
    if hits:
        print(
            "Potential private identifier/secret references found; values intentionally not printed:"
        )
        for name, rel in sorted(set(hits)):
            print(f"- {name}: {rel}")
        return 1
    print("private identifier scan OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
