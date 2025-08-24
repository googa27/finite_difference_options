# AI Agent Automation Rules

## Primary Responsibilities
1. **Maintain PROJECT_CONTEXT.md** - Keep as single source of truth for AI agents
2. **Trigger-Based Updates** - Automatically update docs based on code changes
3. **Quality Assurance** - Validate documentation consistency and accuracy
4. **Template Management** - Ensure reusability across similar projects

## Trigger-Based Updates

### Code Changes → Documentation Updates
```yaml
triggers:
  - path: "src/processes/*.py"
    action: "Update reference/model-catalog.md"
    frequency: "immediate"
    
  - path: "src/pricing/*.py" 
    action: "Update reference/api-reference.md"
    frequency: "immediate"
    
  - path: "tests/test_*.py"
    action: "Update compliance/validation-report.md"
    frequency: "daily"
    
  - path: "pyproject.toml"
    action: "Update guides/quick-start.md dependencies"
    frequency: "immediate"
    
  - path: "README.md"
    action: "Sync with PROJECT_CONTEXT.md overview"
    frequency: "immediate"
```

### Scheduled Updates
- **Daily**: 
  - Check `.gemini_project/project_tasks.sqlite` → Update `planning/roadmap.md`
  - Validate internal links in all documentation
  
- **Weekly**: 
  - Execute all code examples in guides/
  - Update performance benchmarks if tests ran
  - Review and consolidate `planning/ideas.md`
  
- **Monthly**: 
  - Regenerate `compliance/validation-report.md`
  - Audit all external links for validity
  - Update `architecture/dependency-map.md`
  
- **On Release**: 
  - Update all template variables in TEMPLATE_CONFIG.md
  - Generate comprehensive release notes
  - Validate complete documentation set

## Quality Assurance Rules

### Documentation Standards
- **Executable Examples**: All code examples must run successfully
- **Link Validation**: All internal/external links must resolve
- **Template Consistency**: All template variables must be populated
- **ADR Format**: All architectural decisions follow ADR template
- **API Sync**: API documentation matches actual code signatures

### Validation Commands
```bash
# Validate all code examples
find docs/ -name "*.md" -exec python -m doctest {} \;

# Check internal links
grep -r "\[.*\](.*\.md)" docs/ | validate_links.py

# Verify template variables
grep -r "TEMPLATE_" docs/ | check_populated.py

# Test API documentation sync
python -m scripts.validate_api_docs
```

## Template Replication Protocol

### For New Projects
1. **Copy Structure**: `cp -r docs/ /path/to/new-project/docs/`
2. **Update Config**: Edit `TEMPLATE_CONFIG.md` with new project variables
3. **Initialize**: Run AI setup prompt with project-specific context
4. **Customize**: Adapt domain-specific content in guides/ and compliance/
5. **Validate**: Execute validation commands to ensure consistency

### Template Variables to Update
```yaml
required_updates:
  - PROJECT_NAME: "New Project Name"
  - MAIN_DOMAIN: "Project Domain"
  - PRIMARY_MODELS: ["Model1", "Model2"]
  - TARGET_USERS: ["User Type 1", "User Type 2"]
  - DEPLOYMENT_TYPES: ["Type1", "Type2"]
```

## AI Agent Prompts

### Standard Maintenance Prompt
```
Reference @docs/PROJECT_CONTEXT.md for complete project understanding.
Check recent changes in src/ and update corresponding documentation.
Validate all examples still work with current codebase.
Update roadmap from latest tasks.sqlite entries.
```

### Release Preparation Prompt
```
Reference @docs/PROJECT_CONTEXT.md for project context.
Generate comprehensive validation report.
Update all performance benchmarks.
Ensure all documentation reflects current release state.
Create release notes from recent changes.
```

### Template Setup Prompt
```
Initialize new project documentation from template.
Reference TEMPLATE_CONFIG.md for project-specific variables.
Customize all template placeholders.
Validate documentation structure for new domain.
```

## Error Handling

### Common Issues & Solutions
- **Broken Examples**: Update code to match current API
- **Missing Links**: Add proper cross-references
- **Outdated Benchmarks**: Re-run performance tests
- **Template Variables**: Populate all TEMPLATE_ placeholders
- **API Mismatches**: Sync documentation with actual code

### Escalation Rules
- **Critical**: API documentation doesn't match code → Immediate fix required
- **High**: Examples don't execute → Fix within 24 hours  
- **Medium**: Performance benchmarks outdated → Update within 1 week
- **Low**: Minor formatting issues → Fix during next scheduled update

## Integration Points

### With .gemini_project/
- Read `project_tasks.sqlite` for roadmap updates
- Sync project status with documentation
- Track completion of documentation tasks

### With Source Code
- Monitor git commits for trigger-based updates
- Extract API signatures for reference documentation
- Validate examples against current codebase

### With CI/CD
- Run documentation validation in pipeline
- Generate reports on documentation coverage
- Block releases if critical documentation issues exist
