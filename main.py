from agents import langchain_agent

from blocksworld_environment.blocksworld_environment import BlocksWorld
from tasks.information_gathering import InformationGatheringTask, MeasuringCapability, BuildTwoBlockTowerCapability
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
import traceback


TASK_CLASS = {
    'information_gathering':   InformationGatheringTask,
    'measuring':               MeasuringCapability,
    'two_block_tower':         BuildTwoBlockTowerCapability,
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


def output_csv(result_queue, filename, folder):
    # First result we pop separately, to initialise the csv writer
    result = result_queue.get()

    # Try to open the file in read mode to check existing columns
    full_path = os.path.join(folder, filename + ".csv")
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
    csv_file = open(os.path.join(folder, filename + ".csv"), 'a', newline = "")
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


def run_task_sequence(task_sequence, env, llm, result_queues, output_file):
    """Executes a sequence of tasks within a thread."""
    results = {task: [] for task in task_sequence}  #  for sharing with other tasks
    for i, task in enumerate(task_sequence):
        try:
            print(f"Running model {model} on {task},{i} with {env.number_of_blocks} blocks and seed {env.seed}")
            task_instance = TASK_CLASS[task](env, output_file=output_file,
                                             preceding_tasks = task_sequence[:i],
                                             task = task,
                                             preceding_results = results,
                                             result_queues = result_queues,
                                             **vars(args))
            task_instance.run(llm)
        except Exception as e:
            print(f"Task {task} failed for model {llm}\n{traceback.format_exc()}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', type=str, default=["gemini-2.0-flash"])
    parser.add_argument("--tasks", nargs='+', type=str, default=["information_gathering"], help=", ".join(list(TASK_CLASS.keys())))
    parser.add_argument("--num_blocks", nargs='+', type=int, default=[3])
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_steps_per_run", type=int, default=None)
    parser.add_argument("--distraction_prob", type=float, default=0.2)
    parser.add_argument("--perturb_prob", type=float, default=0.2)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--starting_seed", type=int, default=None)
    parser.add_argument("--falling_height", type=int, default=None)
    parser.add_argument("--extra_prompt", type=str, default="")

    args = parser.parse_args()

    # Task
    for task in args.tasks:
        if task not in TASK_CLASS:
            raise ValueError(f"No such task: {args.tasks}, choose one of {list(TASK_CLASS.keys())}.")

    # Create results dir
    if args.result_folder:
        os.makedirs(args.result_folder, exist_ok=True)

    ## Actual run ##
    threads = []
    result_queues = {task: queue.Queue() for task in args.tasks}  # for writing to file
    output_files = []
    for model in args.models:
        for number_of_blocks in args.num_blocks:
            for i in range(args.num_runs):
                seed = args.starting_seed + i if args.starting_seed is not None else None
                if args.result_folder:
                    output_file = open(os.path.join(args.result_folder, f"{model}_{number_of_blocks}_{seed}.txt"), "w")
                    output_files.append(output_file)
                    print(f"Run {i} for model {model} on {args.tasks} with {number_of_blocks} blocks and seed {seed}", file=output_file)
                else:
                    output_file = None

                env = BlocksWorld(number_of_blocks=number_of_blocks, seed=seed, **vars(args))
                llm = langchain_agent.LangchainAgent(model, extra_prompt=args.extra_prompt, output_file=output_file)
                thread = threading.Thread(target=run_task_sequence, args=(args.tasks, env, llm, result_queues, output_file))
                threads.append(thread)
                thread.start()

    if args.result_folder:
        writer_threads = {}
        for task in args.tasks:
            writer_threads[task] = threading.Thread(target=output_csv, args=(result_queues[task], task, args.result_folder))
            writer_threads[task].start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    if args.result_folder:
        for task in args.tasks:
            result_queues[task].put(None)
        for task in args.tasks:
            writer_threads[task].join()
        for output_file in output_files:
            output_file.close()
