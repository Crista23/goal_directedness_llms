from langchain_agent import LangchainAgent
import agency_evals
import pytest
import itertools
import os
if not os.path.exists("test_logs"):
    os.makedirs("test_logs")


@pytest.mark.parametrize("task, model", itertools.product(agency_evals.tasks, agency_evals.models))
def test_task_model(task, model):
    output_file = open(os.path.join("test_logs", f"{task}_{model}.txt"), 'w')
    task = agency_evals.tasks[task](number_of_blocks=3, falling_height=2, distraction_prob=0, perturb_prob=0, output_file=output_file)
    llm = LangchainAgent(model, output_file=output_file)
    task.run(llm)
    results = task.evaluate()
    print(task.evaluate(), file=output_file)
    assert not results['agent_error']
    assert not results['environment_error']
    if 'completed' in results:
        assert results['completed']
    elif 'passed' in results:
        assert results['passed']
    else:
        assert results['regret'] == 0
