# AI Agent Workspace Protocol

## Overview
This directory contains tools and documentation to help future AI agents understand the current state of the project and continue development effectively.

## Key Files

### CURRENT_STATUS.md
Contains a comprehensive overview of the current project state, including:
- Completed work
- Ongoing tasks
- Known issues
- Testing status
- How to continue development

### ai_protocol.py
A Python script that provides helper functions for:
- Creating development plans
- Recording issues encountered
- Updating project status

## How to Use

### 1. Understand Current State
Start by reading `CURRENT_STATUS.md` to understand:
- What's already been completed
- What needs to be done next
- Known issues and limitations
- How to test and verify functionality

### 2. Plan New Work
Before starting new work:
1. Create a plan using the protocol script:
   ```bash
   python .agent_workspace/ai_protocol.py plan "performance_optimization" "Optimize ADI solver for better performance"
   ```
2. This creates a plan file in `.agent_workspace/plans/`

### 3. Record Issues
When encountering problems:
1. Record the issue:
   ```bash
   python .agent_workspace/ai_protocol.py issue "test_framework_problem" "Some tests failing due to initialization issues"
   ```
2. This creates an issue file in `.agent_workspace/issues/`

### 4. Update Status
After completing significant work:
```bash
python .agent_workspace/ai_protocol.py update "Completed performance optimization for ADI solver"
```

## Directory Structure
```
.agent_workspace/
├── CURRENT_STATUS.md          # Current project status
├── ai_protocol.py             # Helper script
├── plans/                     # Development plans
│   ├── PLAN_TEMPLATE.md       # Template for new plans
│   └── [task_name].md         # Individual task plans
└── issues/                    # Encountered issues
    ├── ISSUE_TEMPLATE.md      # Template for new issues
    └── [issue_name].md        # Individual issue reports
```

## Best Practices for Future AI Agents

### 1. Always Test Incrementally
- Test after every small change
- Don't write large amounts of code without testing
- Verify core functionality works before moving on

### 2. Document Everything
- Record all plans before implementation
- Document any issues encountered
- Update status after significant milestones

### 3. Follow the Plan
- Create a detailed plan before coding
- Stick to the plan but be flexible for discoveries
- Update the plan if requirements change

### 4. Use Version Control
- Make frequent, small commits with descriptive messages
- Branch for major features
- Tag releases appropriately

## Getting Started
1. Read `CURRENT_STATUS.md`
2. Choose a task from the "Ongoing Work" section
3. Create a plan using the protocol script
4. Implement incrementally with frequent testing
5. Document any issues encountered
6. Update the current status when complete

## Contact
For questions about this protocol, review the implementation in `ai_protocol.py` or check the git commit history.