#!/usr/bin/env python3
"""
AI Agent Development Protocol Script

This script documents the protocol that future AI agents should follow
when working on this project.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

class AIAgentProtocol:
    """Protocol for AI agents working on this project."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.workspace = self.project_root / ".agent_workspace"
        self.plans_dir = self.workspace / "plans"
        self.issues_dir = self.workspace / "issues"
        
    def initialize_workspace(self):
        """Ensure workspace directories exist."""
        self.workspace.mkdir(exist_ok=True)
        self.plans_dir.mkdir(exist_ok=True)
        self.issues_dir.mkdir(exist_ok=True)
        
        # Create if they don't exist
        current_status = self.workspace / "CURRENT_STATUS.md"
        if not current_status.exists():
            current_status.write_text("# Current Project Status\n\n[Auto-generated]\n")
            
        plan_template = self.plans_dir / "PLAN_TEMPLATE.md"
        if not plan_template.exists():
            plan_template.write_text(self._get_plan_template())
            
        issue_template = self.issues_dir / "ISSUE_TEMPLATE.md"
        if not issue_template.exists():
            issue_template.write_text(self._get_issue_template())
    
    def _get_plan_template(self) -> str:
        """Get the plan template."""
        return """# DEVELOPMENT PLAN

## Plan Creation Date
{}

## Task Description
[Brief description of what needs to be implemented]

## Detailed Requirements
[List specific requirements for the implementation]

## Implementation Approach
[Step-by-step approach to implementation]

## Files to Modify
- [File 1]: [Brief description of changes]
- [File 2]: [Brief description of changes]

## Testing Strategy
[How to test the implementation]

## Acceptance Criteria
[List of criteria that must be met for completion]

## Dependencies
[List any dependencies that need to be addressed first]

## Estimated Complexity
[LOW/MEDIUM/HIGH - Justification]

## Potential Issues
[List potential issues that might be encountered]
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _get_issue_template(self) -> str:
        """Get the issue template."""
        return """# ISSUE REPORT

## Issue Discovery Date
{}

## Issue Description
[Brief description of the issue encountered]

## Error Message
[Exact error message if applicable]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Files Involved
- [File 1]: [Role in the issue]
- [File 2]: [Role in the issue]

## Root Cause Analysis
[Analysis of what's causing the issue]

## Attempted Solutions
1. [Solution 1]: [Result]
2. [Solution 2]: [Result]

## Working Solution
[What fixed the issue, if resolved]

## Related Issues
[Links to related issues or PRs]

## Prevention Strategies
[How to prevent similar issues in the future]
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def create_plan(self, task_name: str, description: str = "") -> Path:
        """Create a new plan file."""
        plan_file = self.plans_dir / f"{task_name.replace(' ', '_').lower()}.md"
        
        if plan_file.exists():
            print(f"Warning: Plan '{task_name}' already exists at {plan_file}")
            return plan_file
            
        template = self._get_plan_template()
        if description:
            template = template.replace("[Brief description of what needs to be implemented]", description)
            
        plan_file.write_text(template)
        print(f"Created plan: {plan_file}")
        return plan_file
    
    def record_issue(self, issue_name: str, description: str = "") -> Path:
        """Record a new issue."""
        issue_file = self.issues_dir / f"{issue_name.replace(' ', '_').lower()}.md"
        
        if issue_file.exists():
            print(f"Warning: Issue '{issue_name}' already exists at {issue_file}")
            return issue_file
            
        template = self._get_issue_template()
        if description:
            template = template.replace("[Brief description of the issue encountered]", description)
            
        issue_file.write_text(template)
        print(f"Recorded issue: {issue_file}")
        return issue_file
    
    def get_current_status(self) -> str:
        """Get current project status."""
        status_file = self.workspace / "CURRENT_STATUS.md"
        if status_file.exists():
            return status_file.read_text()
        return "No current status file found."
    
    def update_current_status(self, update_text: str):
        """Update current project status."""
        status_file = self.workspace / "CURRENT_STATUS.md"
        
        if status_file.exists():
            current_content = status_file.read_text()
            updated_content = f"{current_content}\n\n## Update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{update_text}"
        else:
            updated_content = f"# Current Project Status\n\n## Last Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{update_text}"
            
        status_file.write_text(updated_content)
        print(f"Updated status: {status_file}")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python ai_protocol.py [command]")
        print("Commands:")
        print("  init - Initialize workspace")
        print("  plan <name> [description] - Create new plan")
        print("  issue <name> [description] - Record new issue")
        print("  status - Show current status")
        return
    
    protocol = AIAgentProtocol(".")
    
    command = sys.argv[1]
    
    if command == "init":
        protocol.initialize_workspace()
        print("Workspace initialized.")
        
    elif command == "plan":
        if len(sys.argv) < 3:
            print("Usage: python ai_protocol.py plan <name> [description]")
            return
        name = sys.argv[2]
        description = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        protocol.create_plan(name, description)
        
    elif command == "issue":
        if len(sys.argv) < 3:
            print("Usage: python ai_protocol.py issue <name> [description]")
            return
        name = sys.argv[2]
        description = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        protocol.record_issue(name, description)
        
    elif command == "status":
        print(protocol.get_current_status())
        
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()