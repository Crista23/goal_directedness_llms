This codebase accompanies the paper "Evaluating the Goal-Directedness of Large Language Models".

We provide below instructions on running the code.

## Setting up the environment

Create the pre-defined conda environment (llm_goals.yml) by running

```
conda env create -f llm_goals_env.yml 
```

To update the conda environment with additional packages, first add new packages in `llm_goals.yml` file, then run:

```
conda env update --file llm_goals.yml --prune --name llm_goals
```

To activate the environment run:

```
conda activate llm_goals
```

To deactivate the environment run:

```
conda deactivate
```

## Setting up API keys for querying LLM models


