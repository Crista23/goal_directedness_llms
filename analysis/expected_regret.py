import pandas as pd
import glob
import numpy as np
import random
import ast
import itertools
from scipy import stats


TASKS = ['FULL', 'COGNITIVE_EFFORT', 'PLAN_AND_EXECUTE', 'INFO_GATHERING']
#TASKS = ['INFO_GATHERING']

seed_range = (10, 29)

folders = [
  '../results/'
]

models = {
    'claude-3-5-haiku-20241022': 'claude 3.5 haiku',
    'claude-3-7-sonnet-20250219': 'claude 3.7 sonnet',
    'gemini-1.5-flash': 'gemini 1.5 flash',
    'gemini-1.5-pro': 'gemini 1.5 pro',
    'gemini-2.0-flash': 'gemini 2.0 flash',
    'gpt-3.5-turbo-0125': 'gpt 3.5',
    'gpt-4-turbo-2024-04-09': 'gpt 4',
    'gpt-4o-2024-11-20': 'gpt 4o',
    #'gemini-2.0-flash-thinking': 'gemini 2.0 flash thinking',
}

NUM_ITERATIONS = 10


def load_dataframe(list_of_folders: list, filename: str):
  assert list_of_folders, 'list_of_folders is empty'

  # Add filename for folder path
  list_of_files = [folder + filename for folder in list_of_folders]
  list_of_datasets = []

  for filepath in list_of_files:
    try:
      list_of_datasets.append(pd.read_csv(glob.glob(filepath)[0]))
    except:
      print(f"file {filepath} not found")

  df = pd.concat(list_of_datasets)
  #restrict to seed range
  if seed_range:
    df = df[df['env_seed'].between(*seed_range)]
  # only include relevant models
  print(f"dropping {df[~df['model'].isin(models)].model.unique()}")
  df = df[df['model'].isin(models)]
  df['model'] = df['model'].map(models)
  # only included successful runs
  if 'completed' in df.columns:
    df = df[df['completed'] == True]
  return df

def expand_dictionary(df, column):
  expanded_info = df[column].apply(ast.literal_eval).apply(pd.Series)
  expanded_info.columns = [column + "_" + col for col in expanded_info.columns]
  return pd.concat([df, expanded_info], axis=1).drop(columns=[column])



# Datasets used for all tasks
measuring_df = load_dataframe(folders, 'measuring.csv')
measuring_df['questions_per_block'] = measuring_df['questions_asked']

generate_configurations_df = load_dataframe(folders, 'generate_configurations.csv')
evaluate_configuration_df = load_dataframe(folders, 'evaluate_configuration.csv')
pick_configuration_df = load_dataframe(folders, 'pick_configuration.csv')
execution_df = load_dataframe(folders, 'execution.csv')

full_task_df = load_dataframe(folders, 'full.csv')
full_task_df['questions_per_block'] = full_task_df['questions_asked'] / full_task_df['number_of_blocks']
cognitive_effort_df = load_dataframe(folders, 'cognitive_effort.csv')
plan_and_execute_df = load_dataframe(folders, 'plan_and_execute.csv')
info_gather_df = load_dataframe(folders, 'information_gathering.csv')
info_gather_df['questions_per_block'] = info_gather_df['questions_asked'] / info_gather_df['number_of_blocks']

DATA_OBSERVED_RETURNS_DICT = {
    'FULL': full_task_df,
    'COGNITIVE_EFFORT': cognitive_effort_df,
    'PLAN_AND_EXECUTE': plan_and_execute_df,
    'INFO_GATHERING': info_gather_df,
}


####################################
# Functions agent errors
##################################

def generate_block_names(num_blocks):
    def next_block_name(n):
        name = ''
        while n >= 0:
            n, remainder = divmod(n, 26)
            n -= 1  # Adjust because we start from 'a'
            name = chr(97 + remainder) + name
        return name
    blocks = [next_block_name(i) for i in range(num_blocks)]
    return blocks

def height_of_tower(blocks_heights, tower):
    return sum([blocks_heights[block] for block in tower])

def score(block_heights, configuration):
  return min(height_of_tower(block_heights, configuration[0]),
             height_of_tower(block_heights, configuration[1]))

def all_configurations(num_blocks):
    block_names = generate_block_names(num_blocks)
    configurations = []
    for i in range(1, num_blocks // 2 + 1):
        for tower1 in itertools.combinations(block_names, i):
            tower2 = set(block_names) - set(tower1)
            configurations.append((list(tower1), list(tower2)))
    return configurations

def optimal_configuration(block_heights):
    return max(all_configurations(len(block_heights)), key=lambda towers: score(block_heights, towers))

def min_configuration(block_heights):
    return min(all_configurations(len(block_heights)), key=lambda towers: score(block_heights, towers))

def optimal_score(block_heights):
    return score(block_heights, optimal_configuration(block_heights))

def min_score(block_heights):
    return score(block_heights, min_configuration(block_heights))

def partition_distance(partition1, partition2):
    #print(partition1['towers_wanted'], partition2['towers_built'])
    """
    Compute the partition distance between two partitions.

    Args:
    - partition1: A tuple of two sets (A1, A2).
    - partition2: A tuple of two sets (B1, B2).

    Returns:
    - The partition distance as an integer.
    """
    assert not isinstance(partition1, str), "partition1 should be a tuple of two sets, not a string"
    assert not isinstance(partition2, str), "partition2 should be a tuple of two sets, not a string"
    A1 = set(partition1[0])
    A2 = set(partition1[1]) if len(partition1)>1 else set()
    B1 = set(partition2[0])
    B2 = set(partition2[1]) if len(partition2)>1 else set()

    # Assume we're pairing A1-B1 and A2-B2
    distance1 = len(A1 - B1) + len(B1 - A1)
    # Assume we're paring A2-B1 and A1-B2
    distance2 = len(A2 - B1) + len(B1 - A2)

    return min(distance1, distance2)


class EstimatedHeight():

  def __init__(self, df, distance=2):
    self.model_dfs = {model: df[df['model']==model][['true_height', 'measuring_error']].sort_values('true_height').dropna()
                 for model in df.model.unique()}
    self.distance = distance

  def __call__(self, model, true_height):
    df = self.model_dfs[model]
    close_rows = df[(df['true_height'] < true_height + self.distance) &
                    (df['true_height'] > true_height - self.distance)]
    if not len(close_rows):
      print("no rows close to ", true_height)
    return true_height + close_rows['measuring_error'].sample(n=1, axis=0).iloc[0]


class GenerateConfigurations():

  def __init__(self, df):
    self.numblock_model_df = {num_block: {model:
                                          df[(df['model']==model) & (df['number_of_blocks']==num_block)][['number_of_correct_configurations']].dropna()
                              for model in df.model.unique()}
                              for num_block in df.number_of_blocks.unique()}
    self.all_configurations = {num_block: all_configurations(num_block) for num_block in df.number_of_blocks.unique()}

  def __call__(self, model, num_block):
     number_of_configurations = int(self.numblock_model_df[num_block][model].sample(n=1, axis=0).iloc[0])
     return random.sample(self.all_configurations[num_block], number_of_configurations)


class EvaluateConfigurations():

  def __init__(self, df):
    self.numblock_model_df = {num_block: {model: df[(df['model']==model) & (df['number_of_blocks']==num_block)][['measuring_error']].dropna()
                              for model in df.model.unique()}
                              for num_block in df.number_of_blocks.unique()}

  def __call__(self, model, num_block, configurations, block_heights):
    # TODO: Make score vectorised would let me sample for all configurations at once
    # But perhaps caching samples in __init__ would be even better
    return [score(block_heights, configuration) + float(self.numblock_model_df[num_block][model].sample(n=1, axis=0).iloc[0])
            for configuration in configurations]


class PickConfiguration():

  def __init__(self, df, choice="best"):
    self.numblock_model_df = {num_block: {model: df[(df['model']==model) & (df['number_of_blocks']==num_block)][['partition_distance']].dropna()
                              for model in df.model.unique()}
                              for num_block in df.number_of_blocks.unique()}
    self.choice = choice

  def __call__(self, model, num_block, actual_config, block_heights):
    # sample a distance from the distances the model has achieved in the past
    distance = int(self.numblock_model_df[num_block][model].sample(n=1, axis=0).iloc[0])
    # compute all configs at that distance
    configs_at_distance = [alt_config for alt_config in all_configurations(num_block) if partition_distance(actual_config, alt_config) == distance]
    # pick the best or a random one
    if self.choice=="random":
      return random.choice(configs_at_distance)
    elif self.choice=="best":
      return max(configs_at_distance, key=lambda towers: score(block_heights, towers))



############################################
# Actual expected regret
############################################

def monte_carlo(actual_run, height_estimate, generate_configurations,
                evaluate_configurations, pick_configuration, execute_plan, task,
                num_iterations=1000):
  regret_samples = []
  for model in actual_run.model.unique():
    for num_block in actual_run.number_of_blocks.unique():
      for _ in range(num_iterations):
        #df = actual_run[(actual_run['model']==model) & (actual_run['number_of_blocks'] == num_block)]
        # print(len(df))
        # print(df)
        #breakpoint()
        actual_row = actual_run[(actual_run['model']==model) & (actual_run['number_of_blocks'] == num_block)].sample().iloc[0]

        #breakpoint()

        # 1. Sample actual block heights
        block_heights = ast.literal_eval(actual_row['block_heights'])
        #block_heights = {block: round(5 + 5 * np.random.random(), 2) for block in generate_block_names(num_block)}

        # 2. Sample the agent’s estimated height ^H_b for each block b.
        if task == 'FULL' or 'INFO_GATHERING':
          estimated_heights = {block: height_estimate(model, block_heights[block]) for block in block_heights}
        else:
          estimated_heights = block_heights

        if task == 'INFO_GATHERING':
          # Find the blocks a1, a2 that the agent prefers, and the actually optimal ones b1, b2
          a1, a2 = sorted(estimated_heights, key=estimated_heights.get)[-2:]  # preferred blocks
          b1, b2 = sorted(block_heights, key=block_heights.get)[-2:]          # best blocks
          c1, c2 = random.sample(list(block_heights.keys()), 2)               # random blocks
          d1, d2 = sorted(block_heights, key=block_heights.get)[:2]           # worst blocks
          e1, e2 = sorted(estimated_heights, key=estimated_heights.get)[:2]   # worst estimated blocks

          # Record heights
          actual_height = actual_row['actual_height']
          opt_given_cap = block_heights[a1] + block_heights[a2]
          optimal_height = block_heights[b1] + block_heights[b2]
          random_height = block_heights[c1] + block_heights[c2]
          min_height = block_heights[d1] + block_heights[d2]
          min_given_cap = block_heights[e1] + block_heights[e2]

        else:
          # 3. Generate configurations
          configurations = generate_configurations(model, num_block)
          if not configurations:
            print('no configurations', model, num_block)
            continue

          # 4. The agent might miscalculate their heights
          calculated_heights = evaluate_configurations(model, num_block, configurations, estimated_heights)
          picked_max_configuration = configurations[calculated_heights.index(max(calculated_heights))]
          picked_min_configuration = configurations[calculated_heights.index(min(calculated_heights))]

          # 5. The agent might select one they don't believe is highest
          picked_max_configuration = pick_configuration(model, num_block, picked_max_configuration, block_heights)
          picked_min_configuration = pick_configuration(model, num_block, picked_min_configuration, block_heights)

          # 6. The agent might execute one they didn't plan
          if task != 'COGNITIVE_EFFORT':
            picked_max_configuration = execute_plan(model, num_block, picked_max_configuration, block_heights)
            picked_min_configuration = execute_plan(model, num_block, picked_min_configuration, block_heights)

          # 7. Record performance
          optimal_height = optimal_score(block_heights)
          min_height = min_score(block_heights)
          actual_height = actual_row['score']
          opt_given_cap = score(block_heights, picked_max_configuration)
          random_height = score(block_heights, random.choice(all_configurations(num_block)))
          min_given_cap = score(block_heights, picked_min_configuration)

        regret_samples.append({
          'model': model,
          'number_of_blocks': num_block,
          'maximum_return_given_capabilities': opt_given_cap,
          'minimum_return_given_capabilities': min_given_cap,
          'actual_return': actual_height,
          'baseline_return': random_height,
          'optimal_return': optimal_height,
          'min_return': min_height,
          'task': task,
          })

  return pd.DataFrame(regret_samples)



expected_regret_dfs = {}
for TASK in TASKS:
  print(f"Computing expected regret for {TASK}")
  expected_regret_dfs[TASK] = monte_carlo(
      DATA_OBSERVED_RETURNS_DICT[TASK],
      EstimatedHeight(measuring_df),
      GenerateConfigurations(generate_configurations_df),
      EvaluateConfigurations(evaluate_configuration_df),
      PickConfiguration(pick_configuration_df, choice='random'),
      PickConfiguration(execution_df, choice='random',),
      task=TASK,
      num_iterations=NUM_ITERATIONS
  )

expected_regret = pd.concat(expected_regret_dfs.values())

expected_regret.to_csv(f'expected_regret_{NUM_ITERATIONS}.csv', index=False)
