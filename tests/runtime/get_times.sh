#!/bin/bash

# Usage: ./extract_elapsed_times.sh [directory] [pattern] [output_file]
# Defaults: current directory, *.log files, elapsed_times.txt

DIR="${1:-.}"
PATTERN="${2:-*.log}"
OUTPUT_FILE="${3:-elapsed_times.txt}"

> "$OUTPUT_FILE"  # Clear previous output

# Loop over files sorted naturally
for file in $(ls -v "$DIR"/$PATTERN 2>/dev/null); do
    grep "Elapsed time:" "$file" | awk -v fname="$file" -F': ' '{printf "%s: %s\n", fname, $2}' >> "$OUTPUT_FILE"
done

echo "Results saved to $OUTPUT_FILE"