from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
    "rut_like_identifier": re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-[0-9Kk]\b|\b\d{7,8}-[0-9Kk]\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "openai_style_token": re.compile(r"\b(?:sk-(?:proj|svcacct)-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,})\b"),
    "private_data_path": re.compile(r"data/private/"),
}
ALLOWED_PRIVATE_PATH_REFS = {
    ".gitignore",
    "AGENTS.md",
    "scripts/scan_private_identifiers.py",
}
SAFE_DOTENV_TEMPLATES = {
    ".env.example",
    ".env.sample",
    ".env.template",
}
SAFE_TEMPLATE_VALUE_PATTERNS = {
    "openai_style_token": re.compile(
        r"sk-(?:proj|svcacct)-(?:REPLACE_WITH_[A-Z0-9_]+|YOUR_[A-Z0-9_]+|EXAMPLE_[A-Z0-9_]+)"
    ),
}


@dataclass(frozen=True, order=True)
class ScanHit:
    pattern_name: str
    relative_path: str


def is_dotenv_path(path: Path) -> bool:
    """Return whether a path uses a dotenv name that commonly carries secrets."""

    name = path.name
    return name == ".env" or name.startswith(".env.")


def is_ignored(path: Path, root: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.relative_to(root).parts)


def read_decodable_text(path: Path) -> str | None:
    """Read UTF-8 text when decodable, skipping binary-looking files."""

    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def candidate_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or is_ignored(path, root):
            continue
        if is_dotenv_path(path) or path.suffix in TEXT_EXTS:
            yield path
            continue
        if read_decodable_text(path) is not None:
            yield path


def scan_root(root: Path) -> list[ScanHit]:
    hits: list[ScanHit] = []
    root = root.resolve()
    for path in candidate_text_files(root):
        rel = path.relative_to(root).as_posix()
        text = read_decodable_text(path)
        if text is None:
            continue
        for name, pattern in PATTERNS.items():
            if name == "private_data_path" and rel in ALLOWED_PRIVATE_PATH_REFS:
                continue
            for match in pattern.finditer(text):
                if is_safe_template_match(rel, name, match.group(0)):
                    continue
                hits.append(ScanHit(name, rel))
                break
    return sorted(set(hits))


def is_safe_template_match(relative_path: str, pattern_name: str, matched_value: str) -> bool:
    """Allow only explicit placeholder tokens in exact dotenv template names."""

    if Path(relative_path).name not in SAFE_DOTENV_TEMPLATES:
        return False
    safe_pattern = SAFE_TEMPLATE_VALUE_PATTERNS.get(pattern_name)
    return safe_pattern is not None and safe_pattern.fullmatch(matched_value) is not None


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        print("usage: scan_private_identifiers.py [root]", file=sys.stderr)
        return 2
    root = Path(args[0]).resolve() if args else ROOT
    hits = scan_root(root)
    if hits:
        print("Potential private identifier/secret references found; values intentionally not printed:")
        for hit in hits:
            print(f"- {hit.pattern_name}: {hit.relative_path}")
        return 1
    print("private identifier scan OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
