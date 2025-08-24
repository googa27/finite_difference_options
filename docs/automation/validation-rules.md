# Documentation Validation Rules

## Code Example Validation

### Python Code Blocks
```bash
# Extract and validate all Python code examples
find docs/ -name "*.md" -exec grep -l "```python" {} \; | while read file; do
    echo "Validating $file"
    python -m doctest "$file" || echo "FAILED: $file"
done

# Test import statements
python -c "
try:
    from src.processes import *
    from src.pricing import *
    from src.solvers import *
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
"
```

### API Consistency
```python
# Validate API documentation matches actual code
import inspect
from src.processes import StochasticProcess
from src.pricing import UnifiedPricingEngine

def validate_api_docs():
    # Check method signatures match documentation
    for cls in [StochasticProcess, UnifiedPricingEngine]:
        methods = inspect.getmembers(cls, predicate=inspect.ismethod)
        # Compare with documented signatures in reference/api-reference.md
```

## Link Validation

### Internal Links
```bash
# Check all internal markdown links resolve
find docs/ -name "*.md" -print0 | xargs -0 grep -n "\[.*\](.*\.md)" | while IFS=: read file line content; do
    link=$(echo "$content" | sed -n 's/.*(\([^)]*\.md\)).*/\1/p')
    if [[ ! -f "docs/$link" ]]; then
        echo "❌ Broken link in $file:$line -> $link"
    fi
done
```

### External Links
```bash
# Validate external HTTP links (run monthly)
find docs/ -name "*.md" -print0 | xargs -0 grep -n "http" | while IFS=: read file line content; do
    url=$(echo "$content" | grep -o 'https\?://[^)]*')
    if ! curl -s --head "$url" | head -n 1 | grep -q "200 OK"; then
        echo "❌ Broken external link in $file:$line -> $url"
    fi
done
```

## Template Validation

### Variable Completion
```bash
# Ensure no unfilled template variables
if grep -r "TEMPLATE_" docs/ --exclude="TEMPLATE_CONFIG.md"; then
    echo "❌ Unfilled template variables found"
    exit 1
else
    echo "✅ All template variables filled"
fi
```

### YAML Syntax
```python
# Validate YAML configuration
import yaml

def validate_template_config():
    with open('docs/TEMPLATE_CONFIG.md', 'r') as f:
        content = f.read()
    
    yaml_blocks = content.split('```yaml')[1:]
    for i, block in enumerate(yaml_blocks):
        yaml_content = block.split('```')[0]
        try:
            yaml.safe_load(yaml_content)
            print(f"✅ YAML block {i+1} valid")
        except yaml.YAMLError as e:
            print(f"❌ YAML block {i+1} invalid: {e}")
```

## Content Quality Rules

### Documentation Standards
- **Headings**: Must follow hierarchical structure (H1 → H2 → H3)
- **Code Blocks**: Must specify language for syntax highlighting
- **Links**: Must use descriptive text, not raw URLs
- **Examples**: Must be executable and current
- **ADRs**: Must follow standard ADR format

### Consistency Checks
```bash
# Check heading hierarchy
find docs/ -name "*.md" -exec grep -n "^#" {} \; | awk -F: '{
    file=$1; line=$2; heading=$3
    level = length(heading) - length(ltrim(heading, "#"))
    print file":"line" Level "level": "heading
}'

# Validate code block languages
find docs/ -name "*.md" -exec grep -n "^```" {} \; | grep -v "```[a-z]" | while IFS=: read file line content; do
    if [[ "$content" == '```' ]]; then
        echo "⚠️  Code block without language in $file:$line"
    fi
done
```

## Performance Validation

### Benchmark Accuracy
```python
# Validate performance claims in documentation
def validate_benchmarks():
    import time
    from src.pricing import UnifiedPricingEngine
    from src.processes import create_gbm_process
    
    # Time a standard pricing operation
    process = create_gbm_process(r=0.05, sigma=0.2)
    engine = UnifiedPricingEngine(process)
    
    start_time = time.time()
    # Run benchmark operation
    elapsed = time.time() - start_time
    
    # Compare with documented performance claims
    print(f"Actual performance: {elapsed:.4f}s")
```

### Memory Usage
```python
# Monitor memory usage of examples
import psutil
import os

def validate_memory_usage():
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run memory-intensive example
    # ... code execution ...
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory
    
    print(f"Memory usage: {memory_used:.2f} MB")
```

## Automated Quality Gates

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml addition
repos:
  - repo: local
    hooks:
      - id: validate-docs
        name: Validate Documentation
        entry: python scripts/validate_docs.py
        language: python
        files: ^docs/.*\.md$
        
      - id: check-links
        name: Check Internal Links
        entry: bash scripts/check_internal_links.sh
        language: bash
        files: ^docs/.*\.md$
```

### CI/CD Validation
```yaml
# GitHub Actions workflow
name: Documentation Quality
on:
  pull_request:
    paths: ['docs/**']
  
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Validate documentation
        run: |
          python scripts/validate_all_docs.py
          bash scripts/check_all_links.sh
          python scripts/validate_examples.py
```

## Error Reporting

### Validation Report Format
```markdown
# Documentation Validation Report
Generated: {{ timestamp }}

## Summary
- ✅ Code Examples: {{ passed }}/{{ total }}
- ✅ Internal Links: {{ passed }}/{{ total }}
- ⚠️  External Links: {{ passed }}/{{ total }}
- ✅ Template Variables: All filled
- ✅ YAML Syntax: Valid

## Issues Found
### Critical
- [ ] API documentation mismatch in reference/api-reference.md:42

### High Priority  
- [ ] Broken example in guides/quick-start.md:15

### Medium Priority
- [ ] Outdated benchmark in reference/performance-guide.md:28

## Recommendations
1. Update API documentation to match current code
2. Fix broken code example
3. Re-run performance benchmarks
```

### Escalation Thresholds
- **Block Release**: Any critical issues (API mismatches)
- **Require Fix**: >5 high priority issues
- **Warning Only**: Medium/low priority issues
- **Auto-Fix**: Formatting and minor consistency issues
