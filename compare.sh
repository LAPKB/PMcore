#!/bin/bash

# Check if enough arguments are supplied
if [ "$#" -ne 4 ]; then
    echo "Usage: ./time_compare.sh <example_name> <N> <commit-hash-1> <commit-hash-2>"
    exit 1
fi

# Input arguments
EXAMPLE_NAME=$1
N=$2
COMMIT1=$3
COMMIT2=$4

# Check if bc is installed
if ! command -v bc &> /dev/null
then
    echo "'bc' is required but it's not installed. Please install it using your package manager."
    exit 1
fi

# Output file based on example name
OUTPUT_FILE="${EXAMPLE_NAME}_time_comparison_results.txt"
echo "Time comparison results" > $OUTPUT_FILE
echo "Example: $EXAMPLE_NAME" >> $OUTPUT_FILE
echo "===================================" >> $OUTPUT_FILE

# Function to run the example N times and log the average time
function time_commit() {
    COMMIT=$1
    EXAMPLE_NAME=$2
    N=$3
    TOTAL_TIME=0

    echo "Checking out commit $COMMIT"
    git checkout $COMMIT > /dev/null 2>&1

    # Running the example N times
    echo "Running $EXAMPLE_NAME in release mode for $N times"
    for i in $(seq 1 $N); do
        echo "Run $i: executing 'cargo run' for commit $COMMIT..."
        
        # Capture both stdout and stderr to handle errors from cargo run
        RUN_TIME=$( { /usr/bin/time -f "%e" cargo run --release --example $EXAMPLE_NAME 2>&1 | grep -Eo '^[0-9]+\.[0-9]+$'; } 2>&1 )

        # Validate that RUN_TIME is a valid float
        if [[ $RUN_TIME =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "Run $i: ${RUN_TIME}s"
            TOTAL_TIME=$(echo "$TOTAL_TIME + $RUN_TIME" | bc)
        else
            echo "Error: Unable to capture timing for run $i. Got: $RUN_TIME"
            exit 1
        fi
    done

    # Calculate average time
    AVERAGE_TIME=$(echo "scale=3; $TOTAL_TIME / $N" | bc)
    echo "Average time for commit $COMMIT: ${AVERAGE_TIME}s" >> $OUTPUT_FILE
    echo "-----------------------------------" >> $OUTPUT_FILE
}

# Time the first commit
time_commit $COMMIT1 $EXAMPLE_NAME $N

# Time the second commit
time_commit $COMMIT2 $EXAMPLE_NAME $N

# Switch back to the current branch
git checkout - > /dev/null 2>&1

# Output completion message
echo "Timing complete. Results saved to $OUTPUT_FILE."
