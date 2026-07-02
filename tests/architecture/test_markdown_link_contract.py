import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_markdown_links.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_markdown_links", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_markdown_link_checker_accepts_existing_relative_targets_with_titles(
    tmp_path,
) -> None:
    checker = _load_checker()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "target.md").write_text("# Target\n", encoding="utf-8")
    (tmp_path / "docs" / "target(with-parens).md").write_text(
        "# Target with parens\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "index.md").write_text(
        "[plain](target.md)\n"
        '[titled](target.md "Target doc")\n'
        '[angle](<target.md> "Target doc")\n'
        "[parens](target(with-parens).md)\n"
        '[angle-parens](<target(with-parens).md> "Target doc")\n',
        encoding="utf-8",
    )

    assert checker.validate_links(tmp_path) == []


def test_markdown_link_checker_reports_missing_relative_targets(tmp_path) -> None:
    checker = _load_checker()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.md").write_text(
        "[missing](absent.md)\n", encoding="utf-8"
    )

    assert checker.validate_links(tmp_path) == [
        "docs/index.md:1: missing relative link target: absent.md"
    ]


def test_markdown_link_checker_validates_outer_target_for_linked_images(
    tmp_path,
) -> None:
    checker = _load_checker()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "img.png").write_bytes(b"synthetic")
    (tmp_path / "docs" / "index.md").write_text(
        "[![diagram](img.png)](missing-diagram-page.md)\n",
        encoding="utf-8",
    )

    assert checker.validate_links(tmp_path) == [
        "docs/index.md:1: missing relative link target: missing-diagram-page.md"
    ]


def test_markdown_link_checker_resolves_root_relative_repo_links(tmp_path) -> None:
    checker = _load_checker()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CI_POLICY.md").write_text("# CI Policy\n", encoding="utf-8")
    (tmp_path / "docs" / "index.md").write_text(
        "[policy](/docs/CI_POLICY.md)\n", encoding="utf-8"
    )

    assert checker.validate_links(tmp_path) == []


def test_markdown_link_checker_ignores_fenced_code_external_and_anchor_links(
    tmp_path,
) -> None:
    checker = _load_checker()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.md").write_text(
        "[external](https://example.com/x.md)\n"
        "[protocol-relative](//example.com/file.md)\n"
        "[anchor](#local)\n"
        "```bash\n"
        'grep -r "\\[.*\\](.*\\.md)" docs/\n'
        "```\n",
        encoding="utf-8",
    )

    assert checker.validate_links(tmp_path) == []
