# Documentation Update Triggers

## File Change Triggers

### Source Code Changes
```yaml
src/processes/*.py:
  triggers:
    - update: "reference/model-catalog.md"
    - update: "reference/api-reference.md" 
    - validate: "guides/advanced-usage.md examples"
  frequency: immediate

src/pricing/*.py:
  triggers:
    - update: "reference/api-reference.md"
    - update: "guides/quick-start.md examples"
    - validate: "examples/real-world-cases.md"
  frequency: immediate

src/solvers/*.py:
  triggers:
    - update: "reference/api-reference.md"
    - update: "architecture/system-diagram.md"
  frequency: immediate

tests/test_*.py:
  triggers:
    - update: "compliance/validation-report.md"
    - update: "reference/performance-benchmarks.md"
  frequency: daily
```

### Configuration Changes
```yaml
pyproject.toml:
  triggers:
    - update: "guides/quick-start.md dependencies"
    - update: "PROJECT_CONTEXT.md dependencies"
    - validate: "all installation examples"
  frequency: immediate

requirements*.txt:
  triggers:
    - update: "guides/quick-start.md"
    - validate: "installation documentation"
  frequency: immediate

README.md:
  triggers:
    - sync: "PROJECT_CONTEXT.md overview"
    - validate: "consistency with main docs"
  frequency: immediate
```

## Scheduled Triggers

### Daily (Automated)
- Check `.gemini_project/project_tasks.sqlite` for new tasks
- Update `planning/roadmap.md` with task status changes
- Validate all internal documentation links
- Check for broken external references

### Weekly (AI-Driven)
- Execute all code examples in `guides/`
- Update performance benchmarks if new test results available
- Review and consolidate entries in `planning/ideas.md`
- Validate API documentation against current code signatures

### Monthly (Comprehensive)
- Regenerate complete `compliance/validation-report.md`
- Audit all external links for validity and relevance
- Update `architecture/dependency-map.md` with current dependencies
- Review template variables for accuracy

### Release Triggers
- Update all template variables in `TEMPLATE_CONFIG.md`
- Generate comprehensive release notes in `development/release-notes.md`
- Validate complete documentation set for consistency
- Create new ADR if significant architectural changes occurred

## Validation Rules

### Code Example Validation
```bash
# Extract and test all Python code blocks
grep -A 20 "```python" docs/**/*.md | python -m doctest

# Validate import statements work
python -c "from src.processes import *; from src.pricing import *"
```

### Link Validation
```bash
# Check internal links
find docs/ -name "*.md" -exec grep -l "\[.*\](.*\.md)" {} \; | xargs validate_internal_links.py

# Check external links (monthly)
find docs/ -name "*.md" -exec grep -l "http" {} \; | xargs validate_external_links.py
```

### Template Consistency
```bash
# Ensure no TEMPLATE_ variables remain unfilled
grep -r "TEMPLATE_" docs/ && echo "ERROR: Unfilled template variables found"

# Validate YAML in TEMPLATE_CONFIG.md
python -c "import yaml; yaml.safe_load(open('docs/TEMPLATE_CONFIG.md').read().split('```yaml')[1].split('```')[0])"
```

## Error Handling

### Trigger Failures
- **Code example fails**: Update example to match current API
- **Link validation fails**: Fix broken links or mark as deprecated
- **Template validation fails**: Fill missing variables or fix syntax
- **API sync fails**: Update documentation to match code changes

### Escalation Matrix
- **Critical (API mismatch)**: Immediate fix required, block releases
- **High (broken examples)**: Fix within 24 hours
- **Medium (outdated benchmarks)**: Update within 1 week  
- **Low (formatting issues)**: Fix during next scheduled update

## Integration Commands

### Manual Trigger Execution
```bash
# Force update of specific documentation
python scripts/update_docs.py --trigger src/processes/

# Validate specific documentation section
python scripts/validate_docs.py --section reference/

# Generate validation report
python scripts/generate_validation_report.py
```

### CI/CD Integration
```yaml
# .github/workflows/docs.yml
on:
  push:
    paths: ['src/**', 'docs/**', 'pyproject.toml']
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  validate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Documentation
        run: python scripts/validate_all_docs.py
```
