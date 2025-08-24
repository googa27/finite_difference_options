# Template Configuration

## Project Configuration
```yaml
project:
  name: "Finite Difference Options Pricing"
  description: "Unified multi-dimensional PDE pricing framework for financial derivatives"
  domain: "Quantitative Finance"
  main_package: "src"
  version: "1.0.0"
  
architecture:
  type: "Domain-Driven Design"
  packages: 
    - "processes"    # Stochastic process implementations
    - "pricing"      # Financial instruments and engines
    - "solvers"      # PDE solvers (1D, multi-D)
    - "utils"        # Shared utilities and validation
  
  interfaces:
    - "FastAPI REST API"
    - "Streamlit Web App" 
    - "CLI (Typer)"
    - "Python Package"

models:
  affine_1d: ["GBM", "OU", "CIR"]
  nonaffine_1d: ["CEV"]
  affine_multid: ["Heston"]
  nonaffine_multid: ["SABR"]
  
ai_settings:
  auto_update: true
  validation_frequency: "weekly"
  benchmark_tracking: true
  primary_reference: "docs/PROJECT_CONTEXT.md"
  
compliance:
  regulatory: ["Basel III", "FRTB", "CCAR"]
  validation_required: true
  documentation_level: "enterprise"
  risk_frameworks: ["VaR", "Expected Shortfall", "Greeks"]
```

## Template Usage Instructions

### 1. Fork Structure
```bash
# Copy entire documentation structure
cp -r docs/ /path/to/new-project/docs/

# Copy AI maintenance files
cp AGENTS.md /path/to/new-project/
cp .gemini_project/ /path/to/new-project/ -r
```

### 2. Configure Project Variables
Edit the YAML configuration above with your project specifics:

**Required Changes:**
- `project.name`: Your project name
- `project.description`: Brief project description
- `project.domain`: Your domain (e.g., "Fixed Income", "Equity Derivatives")
- `architecture.packages`: Your main code packages
- `models.*`: Your available models by category

**Optional Changes:**
- `interfaces`: Deployment interfaces you support
- `compliance.regulatory`: Relevant regulations
- `ai_settings.*`: AI maintenance preferences

### 3. Initialize AI Context
Run this prompt with your AI agent:
```
Initialize documentation from template using TEMPLATE_CONFIG.md.
Update all TEMPLATE_ placeholders with project-specific values.
Reference the configuration YAML for project details.
Validate documentation structure for the new domain.
```

### 4. Customize Domain Content

**Financial Domain Adaptations:**
- `guides/quick-start.md`: Update with domain-specific examples
- `reference/model-catalog.md`: List your available models
- `compliance/regulatory-checklist.md`: Add relevant regulations
- `architecture/design-principles.md`: Adapt to your domain patterns

**Code Integration:**
- Update `reference/api-reference.md` with your API
- Modify `guides/advanced-usage.md` for your use cases
- Adapt `compliance/validation.md` for your validation approach

### 5. Validation Checklist
- [ ] All TEMPLATE_ variables replaced
- [ ] Code examples use your project's API
- [ ] Model catalog reflects your implementations
- [ ] Compliance docs match your regulatory requirements
- [ ] AI_MAINTENANCE.md triggers updated for your structure
- [ ] PROJECT_CONTEXT.md reflects your architecture

## Template Variables Reference

### Global Replacements
```yaml
TEMPLATE_PROJECT_NAME: "{{ project.name }}"
TEMPLATE_DOMAIN: "{{ project.domain }}"
TEMPLATE_MAIN_PACKAGE: "{{ project.main_package }}"
TEMPLATE_VERSION: "{{ project.version }}"
```

### Architecture Variables
```yaml
TEMPLATE_PACKAGES: "{{ architecture.packages | join(', ') }}"
TEMPLATE_INTERFACES: "{{ architecture.interfaces | join(', ') }}"
TEMPLATE_TYPE: "{{ architecture.type }}"
```

### Model Variables
```yaml
TEMPLATE_AFFINE_1D: "{{ models.affine_1d | join(', ') }}"
TEMPLATE_NONAFFINE_1D: "{{ models.nonaffine_1d | join(', ') }}"
TEMPLATE_AFFINE_MULTID: "{{ models.affine_multid | join(', ') }}"
TEMPLATE_NONAFFINE_MULTID: "{{ models.nonaffine_multid | join(', ') }}"
```

### Compliance Variables
```yaml
TEMPLATE_REGULATORY: "{{ compliance.regulatory | join(', ') }}"
TEMPLATE_RISK_FRAMEWORKS: "{{ compliance.risk_frameworks | join(', ') }}"
```

## Domain-Specific Templates

### Quantitative Finance (Current)
- Focus: PDE solving, stochastic processes, derivatives pricing
- Models: Black-Scholes family, stochastic volatility, interest rate
- Compliance: Basel, FRTB, model validation

### Fixed Income (Example)
```yaml
project:
  domain: "Fixed Income"
models:
  yield_curve: ["Nelson-Siegel", "Svensson", "Spline"]
  credit: ["CIR++", "Hull-White", "Vasicek"]
compliance:
  regulatory: ["Basel III", "CCAR", "CECL"]
```

### Equity Derivatives (Example)
```yaml
project:
  domain: "Equity Derivatives"
models:
  equity: ["Black-Scholes", "Heston", "Bates"]
  volatility: ["SABR", "SVI", "Local Vol"]
compliance:
  regulatory: ["FRTB", "CVA", "XVA"]
```

## Maintenance Schedule

### After Template Setup
- Week 1: Validate all examples work
- Week 2: Customize compliance documentation
- Week 3: Add domain-specific models
- Week 4: Complete API documentation

### Ongoing Maintenance
- **Monthly**: Review template improvements
- **Quarterly**: Update regulatory requirements
- **Annually**: Refresh model catalog and benchmarks
