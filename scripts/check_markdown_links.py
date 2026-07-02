#!/usr/bin/env python3
"""Check repository-local Markdown links.

The checker is intentionally small and dependency-free so documentation health can
stay in blocking CI without adding a docs toolchain. It validates relative links
in Markdown files and ignores external URLs, anchors-only links, e-mail links,
images, and fenced code blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import unquote, urlparse

EXTERNAL_SCHEMES = {"http", "https", "mailto", "tel", "ftp", "data"}
DEFAULT_EXCLUDES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
}


def _iter_markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.md"):
        if any(part in DEFAULT_EXCLUDES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _strip_fenced_code(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            output.append("\n")
            continue
        output.append("\n" if in_fence else line)
    return "".join(output)


def _find_label_end(line: str, start: int) -> int | None:
    depth = 0
    escaped = False
    for cursor in range(start + 1, len(line)):
        char = line[cursor]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "[":
            depth += 1
        elif char == "]":
            if depth == 0:
                return cursor
            depth -= 1
    return None


def _iter_link_targets(line: str):
    index = 0
    while index < len(line):
        start = line.find("[", index)
        if start == -1:
            return
        if start > 0 and line[start - 1] == "!":
            index = start + 1
            continue
        label_end = _find_label_end(line, start)
        if label_end is None:
            return
        if label_end + 1 >= len(line) or line[label_end + 1] != "(":
            index = label_end + 1
            continue

        cursor = label_end + 2
        depth = 0
        in_angle = False
        quote: str | None = None
        escaped = False
        target: list[str] = []
        while cursor < len(line):
            char = line[cursor]
            if escaped:
                target.append(char)
                escaped = False
            elif char == "\\":
                target.append(char)
                escaped = True
            elif quote is not None:
                target.append(char)
                if char == quote:
                    quote = None
            elif char in {"'", '"'}:
                target.append(char)
                quote = char
            elif in_angle:
                target.append(char)
                if char == ">":
                    in_angle = False
            elif char == "<":
                target.append(char)
                in_angle = True
            elif char == "(":
                target.append(char)
                depth += 1
            elif char == ")":
                if depth == 0:
                    yield "".join(target)
                    index = cursor + 1
                    break
                target.append(char)
                depth -= 1
            else:
                target.append(char)
            cursor += 1
        else:
            return


def _link_destination(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<"):
        closing = target.find(">")
        if closing != -1:
            return target[1:closing].strip()
    return target.split(maxsplit=1)[0]


def _normalise_target(raw_target: str) -> str | None:
    target = _link_destination(raw_target)
    if not target:
        return None
    parsed = urlparse(target)
    if parsed.scheme.lower() in EXTERNAL_SCHEMES or parsed.netloc:
        return None
    if target.startswith("#"):
        return None
    path_part = target.split("#", 1)[0].split("?", 1)[0]
    if not path_part:
        return None
    return unquote(path_part)


def validate_links(repo_root: Path) -> list[str]:
    failures: list[str] = []
    for markdown_file in _iter_markdown_files(repo_root):
        rel_file = markdown_file.relative_to(repo_root)
        text = _strip_fenced_code(markdown_file.read_text(encoding="utf-8"))
        for line_number, line in enumerate(text.splitlines(), start=1):
            for raw_target in _iter_link_targets(line):
                target = _normalise_target(raw_target)
                if target is None:
                    continue
                if target.startswith("/"):
                    candidate = repo_root / target.lstrip("/")
                else:
                    candidate = markdown_file.parent / target
                candidate = candidate.resolve()
                try:
                    candidate.relative_to(repo_root.resolve())
                except ValueError:
                    failures.append(
                        f"{rel_file}:{line_number}: link escapes repository: {target}"
                    )
                    continue
                if not candidate.exists():
                    failures.append(
                        f"{rel_file}:{line_number}: missing relative link target: {target}"
                    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    args = parser.parse_args()
    repo_root = args.root.resolve()
    failures = validate_links(repo_root)
    if failures:
        print("Markdown link check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Markdown link check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
