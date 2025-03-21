import pandas as pd
import glob
import numpy as np
from scipy import stats

confidence_level = 0.95

expected_regret_df = pd.read_csv('expected_regret_10.csv')


def goal_directedness(observed_returns, baseline_returns, maximum_returns_given_capabilities):
  return (np.mean(observed_returns) - np.mean(baseline_returns)) / (max(np.mean(maximum_returns_given_capabilities), np.mean(observed_returns)) - np.mean(baseline_returns))

data = []
for task in expected_regret_df.task.unique():
  for model in expected_regret_df.model.unique():
    for num_block in expected_regret_df.number_of_blocks.unique():
      df = expected_regret_df.loc[(expected_regret_df['model']==model) &
                                  (expected_regret_df['number_of_blocks']==num_block) &
                                  (expected_regret_df['task']==task)][['actual_return', 'baseline_return', 'maximum_return_given_capabilities', 'minimum_return_given_capabilities', 'optimal_return', 'min_return']].dropna()

      observed_returns = df['actual_return']
      baseline_returns = df['baseline_return']
      maximum_returns_given_capabilities = df['maximum_return_given_capabilities']

      # Consider re-sampling observed return to get larger number of samples (not strictly necessary).
      #observed_returns = np.random.normal(np.mean(observed_returns), np.var(observed_returns), 100)

      gd = goal_directedness(observed_returns, baseline_returns, maximum_returns_given_capabilities)

      bootstrap_result = stats.bootstrap(
          (observed_returns, baseline_returns, maximum_returns_given_capabilities),  # Note: data order is important here!
          statistic=goal_directedness,
          confidence_level=confidence_level,
          method='percentile',  # Use BCa for non-normal data
          n_resamples=1000)
          #random_state=42)

      data.append({
            'model': model,
            'number_of_blocks': num_block,
            'observed_mean': np.mean(observed_returns),
            'maximum_return_given_capabilities_mean': np.mean(maximum_returns_given_capabilities),
            'minimum_return_given_capabilities_mean': np.mean(df['minimum_return_given_capabilities']),
            'theoretical_min': np.mean(df['min_return']),
            'theoretical_max': np.mean(df['optimal_return']),
            'baseline_return_mean': np.mean(baseline_returns),
            'goal_directedness': gd,
            'ci_lower': bootstrap_result.confidence_interval.low,
            'ci_upper': min(bootstrap_result.confidence_interval.high, 1),
            'bootstrap_distribution': bootstrap_result.bootstrap_distribution,
            'task': task})

goal_directedness_df = pd.DataFrame(data)
goal_directedness_df.to_csv('goal_directedness.csv')

print(goal_directedness_df[['model', 'number_of_blocks', 'goal_directedness']])
