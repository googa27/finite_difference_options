#!/usr/bin/env python3
"""
Demo script showing how to use the AI Agent Protocol
"""

import sys
from pathlib import Path

def demo_protocol():
    """Demonstrate the AI agent protocol."""
    print("=== AI Agent Protocol Demo ===\n")
    
    # Initialize protocol
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from .agent_workspace.ai_protocol import AIAgentProtocol
    protocol = AIAgentProtocol(str(project_root))
    
    print("1. Creating a plan for performance optimization...")
    plan_file = protocol.create_plan(
        "performance_optimization", 
        "Optimize the ADI solver for better computational performance"
    )
    print(f"   Created plan: {plan_file}\n")
    
    print("2. Recording an issue with test framework...")
    issue_file = protocol.record_issue(
        "test_framework_problem",
        "Some tests failing due to parameter validation framework issues"
    )
    print(f"   Recorded issue: {issue_file}\n")
    
    print("3. Updating current status...")
    protocol.update_current_status(
        "Demoed the AI agent protocol. Created performance optimization plan and recorded test framework issue."
    )
    print("   Status updated.\n")
    
    print("4. Showing current status...")
    print("-" * 40)
    print(protocol.get_current_status())
    print("-" * 40)
    
    print("\n=== Demo Complete ===")
    print("Check the .agent_workspace/ directory for created files.")

if __name__ == "__main__":
    demo_protocol()