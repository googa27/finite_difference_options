# Post-Refactor Cleanup Plan

## Plan Creation Date
2025-08-24

## Task Description
This plan outlines the steps to clean up the codebase after the recent major architectural refactoring. The goal is to remove redundant code, update documentation, and improve CI/CD processes to create a clean, consistent, and maintainable codebase.

## Detailed Requirements
- Remove all redundant directories and files.
- Ensure the `src` directory has a consistent, clean structure.
- Consolidate and clean up test files.
- Update all documentation to reflect the refactored codebase.
- Create an Architectural Decision Record (ADR) for the refactoring.
- Enhance the CI pipeline with linting and ensure it tests the correct code paths.

## Implementation Approach

### Phase 1: Critical Redundancy Removal
1.  **Delete `backup_src/` directory:** This is a backup and is no longer needed.
2.  **Delete `react-frontend/` directory:** This is an unused frontend application.
3.  **Commit changes:** Create a git commit for the deletions.
4.  **Run tests:** Ensure that the project still works after these deletions.

### Phase 2: Code and Test Consolidation
1.  **Analyze `src` directory:** Identify and remove obsolete Python files that have been replaced by the new subdirectory structure.
2.  **Analyze `tests` directory:** Identify, merge, and/or remove obsolete and temporary test files (e.g., `test_*_refactor.py`).
3.  **Run tests:** Continuously run the test suite to ensure no functionality is broken during cleanup.
4.  **Commit changes:** Create a git commit for the code and test consolidation.

### Phase 3: Documentation Update
1.  **Review and Update `README.md`:** Update all instructions, code examples, and descriptions to match the refactored code.
2.  **Review and Update `docs/`:** Go through the documents in the `docs` folder and update them as necessary.
3.  **Create ADR:** Write a new ADR in `docs/adr/` explaining the large-scale refactoring.
4.  **Commit changes:** Create a git commit for the documentation updates.

### Phase 4: CI/CD Enhancement
1.  **Analyze `.github/workflows/ci.yml`:** Review the CI pipeline to ensure it's using the correct paths for testing and other jobs.
2.  **Add Linting Step:** Add a new step to the CI pipeline to run `ruff check .` and `black --check .` on the codebase.
3.  **Commit changes:** Create a git commit for the CI/CD enhancements.

## Files to Modify
- `.github/workflows/ci.yml`: To add linting and verify paths.
- `README.md`: To update documentation.
- `docs/adr/`: To add a new ADR file.
- The plan involves deleting multiple files and directories.

## Testing Strategy
- After each major change (especially deletions and file moves), the full test suite will be run using `pytest -n auto`.
- The CI pipeline will be monitored after the changes to ensure it passes.

## Acceptance Criteria
- The `backup_src/` and `react-frontend/` directories are gone.
- The `src/` directory contains only the new, consistent subdirectory structure.
- The `tests/` directory is clean of temporary files.
- The documentation accurately reflects the state of the code.
- The CI pipeline includes a passing linting step.
- All tests pass.

## Dependencies
- None.

## Estimated Complexity
- MEDIUM: The work is straightforward but involves touching many parts of the project, requiring careful verification at each step.

## Potential Issues
- Deleting files from `src` might break imports that haven't been updated. Careful testing is required.
- The CI pipeline might have subtle dependencies on the old structure.
