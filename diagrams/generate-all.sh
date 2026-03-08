#!/bin/bash
# Generate all Mermaid diagrams as PNG images
# Prerequisites: npm install -g @mermaid-js/mermaid-cli

set -e  # Exit on error

DIAGRAMS_DIR="d:/projects/CXOne/ai-design-patterns-article/diagrams"
cd "$DIAGRAMS_DIR"

echo "==================================="
echo "Mermaid Diagram Generator"
echo "==================================="
echo ""

# Check if mmdc is installed
if ! command -v mmdc &> /dev/null; then
    echo "ERROR: Mermaid CLI (mmdc) not found!"
    echo ""
    echo "Please install it first:"
    echo "  npm install -g @mermaid-js/mermaid-cli"
    echo ""
    echo "Or install locally in this directory:"
    echo "  npm install @mermaid-js/mermaid-cli"
    echo "  Then use: npx mmdc instead of mmdc"
    exit 1
fi

echo "Found mmdc: $(which mmdc)"
echo ""

# Configuration
WIDTH=1200
BACKGROUND="white"
THEME="default"

# Counter for success/failure
SUCCESS=0
FAILED=0

# Array of diagram files
DIAGRAMS=(
    "diagram-01-architecture.mmd"
    "diagram-02-chunking.mmd"
    "diagram-03-hybrid-search.mmd"
    "diagram-04-query-rewriting.mmd"
    "diagram-05-metadata-filtering.mmd"
    "diagram-06-semantic-caching.mmd"
    "diagram-07-react-agent.mmd"
    "diagram-08-specialist-agents.mmd"
    "diagram-09-planning-agent.mmd"
    "diagram-10-memory-system.mmd"
    "diagram-11-circuit-breaker.mmd"
    "diagram-12-prompt-injection.mmd"
)

echo "Generating PNG images..."
echo "Settings: ${WIDTH}px width, ${BACKGROUND} background, ${THEME} theme"
echo ""

# Generate each diagram
for diagram in "${DIAGRAMS[@]}"; do
    base_name="${diagram%.mmd}"
    output_file="${base_name}.png"

    echo -n "  Processing: $diagram ... "

    if [ ! -f "$diagram" ]; then
        echo "FAILED (file not found)"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Generate PNG using mmdc
    if mmdc -i "$diagram" -o "$output_file" -w "$WIDTH" -b "$BACKGROUND" -t "$THEME" > /dev/null 2>&1; then
        echo "SUCCESS"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "==================================="
echo "Generation Summary"
echo "==================================="
echo "  Success: $SUCCESS"
echo "  Failed:  $FAILED"
echo "  Total:   ${#DIAGRAMS[@]}"
echo ""

if [ $SUCCESS -gt 0 ]; then
    echo "PNG files saved to: $DIAGRAMS_DIR"
fi

exit 0
