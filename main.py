from agents import langchain_agent

from blocksworld_environment.blocksworld_environment import BlocksWorld
from tasks.information_gathering import InformationGatheringTask, MeasuringCapability
from tasks.cognitive_effort import CognitiveEffortTask, GenerateConfigurationsCapability, EvaluateConfigurationCapability, PickConfigurationCapability
from tasks.full_task import FullTask, PlanAndExecuteTask, ExecuteTask
from tasks.falling_tower_task import FallingTowerTask, BuildTowerWithAllBlocksCapability
from anagram_environment.arrange_letters import ArrangeLettersTask, ConstructWordCapability
from anagram_environment.anagram import AnagramTask, GeneratePermutationsCapability, CheckIfWordCapability

import argparse
import csv
import datetime
import sys
import os
import datetime
import threading
import queue


tasks = {
    'information_gathering':   InformationGatheringTask,
    'measuring':               MeasuringCapability,
    "cognitive_effort":        CognitiveEffortTask,
    "generate_configurations": GenerateConfigurationsCapability,
    "evaluate_configuration":  EvaluateConfigurationCapability,
    "pick_configuration":      PickConfigurationCapability,
    "execution":               ExecuteTask,
    "plan_and_execute":        PlanAndExecuteTask,
    "full":                    FullTask,
    "falling_tower":           FallingTowerTask,
    "build_tower_with_all_blocks": BuildTowerWithAllBlocksCapability,
    "arrange_letters":         ArrangeLettersTask,
    "construct_words":         ConstructWordCapability,
    "anagram":                 AnagramTask,
    "permutation":             GeneratePermutationsCapability,
    "isword":                  CheckIfWordCapability,
}

models = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
]


def output_csv(result_queue, filename):
    # First result we pop separately, to initialise the csv writer
    result = result_queue.get()

    # Try to open the file in read mode to check existing columns
    full_path = os.path.join('results', filename + ".csv")
    if os.path.exists(full_path):
        with open(full_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_fieldnames = reader.fieldnames
            # Compare existing fieldnames with the given fieldnames
            if existing_fieldnames != list(result.keys()):
                # if they don't match, we create a new, time-stamped filename instead
                now = datetime.datetime.now().isoformat()
                print(f"Warning: fieldnames don't match in {filename}, storing csv outputs in {filename + now}.csv instead.")
                filename += now

    # Now we're ready to open the writer
    csv_file = open(os.path.join('results', filename + ".csv"), 'a', newline = "")
    csv_writer = csv.DictWriter(csv_file, result.keys())
    if csv_file.tell() == 0:
        csv_writer.writeheader()
    csv_writer.writerow(result)
    csv_file.flush()
    result_queue.task_done()

    # Now the writer is opened, and we keep popping the queue until we encounter an empty one, indicating that the task is done
    while True:
        result = result_queue.get()
        if result is None:  # signal to end
            break
        csv_writer.writerow(result)
        csv_file.flush()
        result_queue.task_done()
    csv_file.close()


def run_task_sequence(task_sequence, env, llm, result_queues):
    """Executes a sequence of tasks within a thread."""
            #output_files = []
    for i, task in enumerate(task_sequence):
        print(f"Run {i} for model {model} on {task} with {env.number_of_blocks} blocks and seed {env.seed}")
        task_instance = tasks[task](seed=seed, output_file=output_file, number_of_blocks=number_of_blocks, **vars(args))
        result = task_instance.run(llm)
        result_queues[task].put(result) # put the result into the queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', type=str, default=["gemini-2.0-flash"], help=", ".join(models))
    parser.add_argument("--tasks", nargs='+', type=str, default=["information_gathering"], help=", ".join(list(tasks.keys())))
    parser.add_argument("--num_blocks", nargs='+', type=int, default=[3])
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_steps_per_run", type=int, default=None)
    parser.add_argument("--distraction_prob", type=float, default=0.2)
    parser.add_argument("--perturb_prob", type=float, default=0.2)
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--starting_seed", type=int, default=None)
    parser.add_argument("--falling_height", type=int, default=None)
    parser.add_argument("--extra_prompt", type=str, default="")

    args = parser.parse_args()

    # Task
    for task in args.tasks:
        if task not in tasks:
            raise ValueError(f"No such task: {args.tasks}, choose one of {list(tasks.keys())}.")

    # Create results dir
    if args.file_name:
        os.makedirs("results", exist_ok=True)

    ## Actual run ##
    threads = []
    result_queues = {task: queue.Queue() for task in args.tasks}
    if args.models != ["all"]:
        print(args.models, "not equal to all")
        models = args.models
    for model in models:
        for number_of_blocks in args.num_blocks:
            for i in range(args.num_runs):
                seed = args.starting_seed + i if args.starting_seed is not None else None
                if args.file_name:
                    output_file = open(os.path.join('results', f"{model}_{number_of_blocks}_{i}.txt"), "w")
                    print(f"Run {i} for model {model} on {args.tasks} with {number_of_blocks} blocks and seed {seed}", file=output_file)
                else:
                    output_file = None

                env = BlocksWorld(**vars(args))
                llm = langchain_agent.LangchainAgent(model, extra_prompt=args.extra_prompt, output_file=output_file)
                thread = threading.Thread(target=run_task_sequence, args=(args.tasks, env, llm, result_queues))
                threads.append(thread)
                thread.start()

    if args.file_name:
        writer_threads = {}
        for task in args.tasks:
            writer_threads[task] = threading.Thread(target=output_csv, args=(result_queues[task], task))
            writer_threads[task].start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    if args.file_name:
        for task in args.tasks:
            result_queues[task].put(None)
            writer_threads[task].join()
