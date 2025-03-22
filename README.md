This codebase accompanies the paper **Evaluating the Goal-Directedness of Large Language Models**.

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

Add the API keys for querying LLM models to the `.env` file:

```
GOOGLE_API_KEY = "your-api-key-string-here"
OPENAI_API_KEY = "your-api-key-string-here"
ANTHROPIC_API_KEY = "your-api-key-string-here"
```

## TEST ENVIRONMENTS

We make available a custom implementation of the BlocksWorld environment, as well as an additional Anagram environment.
Please see `blocksworld_environment` and `anagram environment` folders.

## TASKS

We evaluate LLM goal directedness on four tasks: 
1. Information gathering
2. Cognitive effort
3. Plan and execute
4. Combined task

Please see `tasks` folder for their implementation.

## Running the code

Please see the `scripts` folder. 

The command below will launch the evals for all tasks.

```
./scripts/run_all_tasks.sh
```

## Analysis of the results

Please see `analysis` folder for detailed instructions.



To cite this paper, please use the the bib entry below (TO ADD):


