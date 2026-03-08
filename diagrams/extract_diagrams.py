#!/usr/bin/env python3
"""
Extract all Mermaid diagrams from the AI Design Patterns article.
"""

import re
import os
from pathlib import Path

# File paths
ARTICLE_PATH = Path("d:/projects/CXOne/ai-design-patterns-article/ai-design-patterns-article.md")
DIAGRAMS_DIR = Path("d:/projects/CXOne/ai-design-patterns-article/diagrams")

# Diagram metadata: (filename, title, description)
DIAGRAM_METADATA = [
    ("diagram-01-architecture.mmd", "LLM Application Stack", "Architecture layers for LLM applications"),
    ("diagram-02-chunking.mmd", "Chunking Strategy", "Comparison of document chunking approaches"),
    ("diagram-03-hybrid-search.mmd", "Hybrid Search", "Dense and sparse retrieval combination"),
    ("diagram-04-query-rewriting.mmd", "Query Rewriting", "Multi-query expansion with context"),
    ("diagram-05-metadata-filtering.mmd", "Metadata Filtering", "Hierarchical metadata filtering strategy"),
    ("diagram-06-semantic-caching.mmd", "Semantic Caching", "RAG response caching with embeddings"),
    ("diagram-07-react-agent.mmd", "ReAct Agent Loop", "Thought-action-observation cycle"),
    ("diagram-08-specialist-agents.mmd", "Specialist Agents", "Multi-agent coordination pattern"),
    ("diagram-09-planning-agent.mmd", "Planning & Reflection", "Agent planning with reflection loops"),
    ("diagram-10-memory-system.mmd", "Memory System", "Multi-tier memory architecture"),
    ("diagram-11-circuit-breaker.mmd", "Circuit Breaker", "Loop detection and intervention"),
    ("diagram-12-prompt-injection.mmd", "Prompt Injection Defense", "Multi-layer security validation"),
]

def extract_mermaid_diagrams(article_content):
    """Extract all Mermaid diagrams from the article."""
    # Pattern to match ```mermaid ... ```
    pattern = r'```mermaid\n(.*?)```'
    matches = re.findall(pattern, article_content, re.DOTALL)
    return matches

def main():
    print(f"Reading article from: {ARTICLE_PATH}")

    with open(ARTICLE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    diagrams = extract_mermaid_diagrams(content)
    print(f"Found {len(diagrams)} Mermaid diagrams")

    # Ensure diagrams directory exists
    DIAGRAMS_DIR.mkdir(exist_ok=True, parents=True)

    # Write each diagram to a file
    for i, (diagram_content, (filename, title, description)) in enumerate(zip(diagrams, DIAGRAM_METADATA)):
        output_path = DIAGRAMS_DIR / filename

        # Clean up the diagram content
        diagram_content = diagram_content.strip()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(diagram_content)

        print(f"  [{i+1:2d}] Saved: {filename} ({title})")

    # Create DIAGRAM_INDEX.md
    create_diagram_index(diagrams)

    print(f"\nAll diagrams saved to: {DIAGRAMS_DIR}")

def create_diagram_index(diagrams):
    """Create an index file listing all diagrams."""
    index_path = DIAGRAMS_DIR / "DIAGRAM_INDEX.md"

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# Diagram Index\n\n")
        f.write("This directory contains all Mermaid diagrams extracted from the AI Design Patterns article.\n\n")
        f.write("## Diagrams\n\n")

        for i, (filename, title, description) in enumerate(DIAGRAM_METADATA, 1):
            f.write(f"### {i}. {title}\n\n")
            f.write(f"- **File**: `{filename}`\n")
            f.write(f"- **PNG**: `{filename.replace('.mmd', '.png')}`\n")
            f.write(f"- **Description**: {description}\n")
            f.write(f"- **Pattern**: {get_pattern_from_index(i)}\n\n")

    print(f"Created diagram index: {index_path}")

def get_pattern_from_index(index):
    """Map diagram index to pattern name."""
    patterns = {
        1: "Foundation / Architecture",
        2: "RAG Optimization - Chunking",
        3: "RAG Optimization - Hybrid Search",
        4: "RAG Optimization - Query Rewriting",
        5: "RAG Optimization - Metadata Filtering",
        6: "RAG Optimization - Semantic Caching",
        7: "Agent Patterns - ReAct",
        8: "Agent Patterns - Specialist Agents",
        9: "Agent Patterns - Planning & Reflection",
        10: "Agent Patterns - Memory System",
        11: "Production Patterns - Circuit Breaker",
        12: "Production Patterns - Prompt Injection Defense",
    }
    return patterns.get(index, "Unknown")

if __name__ == "__main__":
    main()
