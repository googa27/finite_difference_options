"""Strict unified-hunk ranges for bounded advisory context, with no provider."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pr_review_context", Path(__file__).resolve().parents[1] / "scripts/classify_pr_review.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize(
    "patch,additions,deletions",
    [
        ("@@ -0,0 +0,1 @@\n+new", 1, 0),
        ("@@ -0,1 +1,1 @@\n-old\n+new", 1, 1),
        ("@@ -1,0 +1,1 @@\n+one\n@@ -2,0 +1,1 @@\n+two", 2, 0),
        ("@@ -1,1 +1,0 @@\n-one\n@@ -1,1 +2,0 @@\n-two", 0, 2),
        ("@@ -9,0 +9,1 @@\n+one\n@@ -2,0 +2,1 @@\n+two", 2, 0),
        ("\\ No newline at end of file\n@@ -0,0 +1,1 @@\n+new", 1, 0),
        ("@@ -0,0 +1,1 @@\n\\ No newline at end of file\n+new", 1, 0),
        ("@@ -0,0 +1,1 @@\n+new\n\\ No newline at end of file\n\\ No newline at end of file", 1, 0),
        ("@@ -0,0 +0,0 @@", 0, 0),
    ],
)
def test_zero_nonempty_ranges_overlap_descent_and_orphan_markers_refuse(patch, additions, deletions):
    with pytest.raises(_MODULE.ReviewRefusal):
        _MODULE._added_lines(patch, additions, deletions)


@pytest.mark.parametrize(
    "patch,additions,deletions,want",
    [
        ("@@ -0,0 +1,1 @@\n+new\n\\ No newline at end of file", 1, 0, [1]),
        ("@@ -1,1 +0,0 @@\n-old\n\\ No newline at end of file", 0, 1, []),
        ("@@ -1,1 +1,1 @@\n-old\n\\ No newline at end of file\n+new", 1, 1, [1]),
        ("@@ -1,1 +1,1 @@\n-old\n+new\n@@ -9,1 +9,1 @@\n-old\n+new", 2, 2, [1, 9]),
    ],
)
def test_native_zero_count_file_boundaries_markers_and_ordered_hunks_work(patch, additions, deletions, want):
    assert _MODULE._added_lines(patch, additions, deletions) == want


@pytest.mark.parametrize(
    "patch",
    [
        "@@ -1,1 +1,1 @@\n-old\n+new\n@@ -5,1 +6,1 @@\n-old\n+new",
        "@@ -1,1 +1,1 @@\n-old\n+new\n@@ -6,1 +5,1 @@\n-old\n+new",
        "@@ -2,1 +3,1 @@\n-old\n+new",
    ],
)
def test_impossible_unchanged_prefix_or_inter_hunk_gaps_refuse(patch):
    with pytest.raises(_MODULE.ReviewRefusal):
        _MODULE._added_lines(patch, patch.count("+new"), patch.count("-old"))


@pytest.mark.parametrize(
    "patch,additions,deletions,want",
    [
        ("@@ -1,1 +1,1 @@\n-old\n+new\n@@ -1,0 +2,1 @@\n+extra", 2, 1, [1, 2]),
        ("@@ -1,1 +1,1 @@\n-old\n+new\n@@ -2,1 +1,0 @@\n-delete", 1, 2, [1]),
        ("@@ -1,0 +2,1 @@\n+extra\n@@ -5,1 +6,1 @@\n-old\n+new", 2, 1, [2, 6]),
        ("@@ -2,1 +1,0 @@\n-delete\n@@ -5,1 +4,1 @@\n-old\n+new", 1, 2, [4]),
    ],
)
def test_zero_count_boundary_offsets_and_corresponding_gaps_work(patch, additions, deletions, want):
    assert _MODULE._added_lines(patch, additions, deletions) == want
