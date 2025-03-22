#!/bin/bash

# Array of LLM models to evaluate
#"gemini-1.5-flash" "gemini-2.0-flash" "gemini-1.5-pro" "gpt-3.5-turbo-0125" "gpt-4-turbo-2024-04-09" "gpt-4o-2024-11-20" "claude-3-7-sonnet-20250219" "claude-3-5-haiku-20241022"
MODELS=(
"gemini-1.5-flash" "gemini-2.0-flash" "gemini-1.5-pro" "gpt-3.5-turbo-0125" "gpt-4-turbo-2024-04-09" "gpt-4o-2024-11-20"
)


NUM_RUN=10
NUM_BLOCKS=15
SEED=10


FILE_NAME="falling_tower"

echo "Evaluating model: $MODEL"
# Loop through falling heights
for falling_height in 3 5 7 9 11 13
do
  echo "Running with falling height: $falling_height"
  python main.py --file_name ${FILE_NAME} --task falling_tower --model gemini-1.5-flash gemini-2.0-flash gemini-1.5-pro gpt-3.5-turbo-0125 gpt-4-turbo-2024-04-09 gpt-4o-2024-11-20 --num_blocks ${NUM_BLOCKS} --falling_height $falling_height --num_runs $NUM_RUN --starting_seed $SEED
  echo "----------------------------------------"
done
