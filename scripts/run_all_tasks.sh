#!/bin/bash


# Define variations in input arguments
# ("information_gathering" "measuring" "cognitive_effort" "generate_configurations" "evaluate_configuration" "pick_configuration" "execution" "plan_and_execute" "full")
TASKS=("measuring" "cognitive_effort" "generate_configurations" "evaluate_configuration" "pick_configuration" "execution" "full")

#MODELS=("gemini-1.5-flash" "gemini-2.0-flash" "gemini-1.5-pro" "gpt-3.5-turbo-0125" "gpt-4-turbo-2024-04-09" "gpt-4o-2024-11-20" "o1-mini-2024-09-12" "claude-3-7-sonnet-20250219" "claude-3-5-haiku-20241022")

NUM_RUNS=10
SEED=10


# Loop through variations and execute the command
for TASK in "${TASKS[@]}"; do
  # Construct the full command with current arguments
  full_command="python3 main.py --models gemini-2.0-flash gemini-1.5-pro gemini-1.5-flash --task ${TASK} --num_blocks 3 4 5 --result_folder drive_results/2025-04-04-noise-0.3-expected_regret --num_runs ${NUM_RUNS} --starting_seed ${SEED} --noise 0.3 --perturb_prob 0.2 --distraction_prob 0.2" 

  # Execute the command
  echo "Running: ${full_command}"
  ${full_command}
done
