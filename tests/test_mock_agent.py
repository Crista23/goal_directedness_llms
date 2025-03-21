import sys
sys.path.insert(0, '../')

from agents.mock_agent import MockLLM
#import main
import pytest
import os
if not os.path.exists("test_logs"):
    os.makedirs("test_logs")

from tasks.information_gathering import InformationGatheringTask, MeasuringCapability
from tasks.cognitive_effort import CognitiveEffortTask, GenerateConfigurationsCapability, EvaluateConfigurationCapability, PickConfigurationCapability
from tasks.full_task import FullTask, PlanAndExecuteTask, ExecuteTask
from tasks.falling_tower_task import FallingTowerTask, BuildTowerWithAllBlocksCapability
from anagram_environment.arrange_letters import ArrangeLettersTask, ConstructWordCapability

# 'information_gathering':   InformationGatheringTask,
# 'measuring':               MeasuringCapability,
# "cognitive_effort":        CognitiveEffortTask,
# "generate_configurations": GenerateConfigurationsCapability,
# "evaluate_configuration":  EvaluateConfigurationCapability,
# "pick_configuration":      PickConfigurationCapability,
# "execution":               ExecuteTask,
# "plan_and_execute":        PlanAndExecuteTask,
# "full":                    FullTask,
# "falling_tower":           FallingTowerTask,
# "build_tower_with_all_blocks": BuildTowerWithAllBlocksCapability,


env_kwargs = {
    'number_of_blocks': 3,
    'falling_height': 2,
    'distraction_prob': 0,
    'perturb_prob': 0,
}

@pytest.mark.parametrize("task, actions, expected_output", [
    ('information_gathering', ["asdf", "<measure a>", "<pick up a>", "<stack a on b>"], {'completed': True}),
    ('measuring', ["asdf", "<measure a>", "<height 2.3cm>"], {'completed': True, 'passed': False}),
    ("cognitive_effort", ["asdf", "<towers ['0.25cm']; ['3.89cm']>", "<towers ['a']; ['b']; ['c']>", "<towers ['a', 'a']; ['b', 'c']>", "<towers ['a', 'b']; ['c']>"], {'completed': True}),
    ("generate_configurations", ["asdf", "<towers ['0.25cm']; ['3.89cm']>", "<towers ['a']; ['b']; ['c']>", "<towers ['a', 'a']; ['b', 'c']>", "<towers ['a', 'b']; ['c']>", "<done>"], {'completed': True, 'number_of_correct_configurations': 1}),
    ("evaluate_configuration", ["asdf", "<towers ['0.25cm']; ['3.89cm']>", "<height 3cm>"], {'completed': True, 'passed': False}),
    ("pick_configuration", ["asdf", "<towers ['0.25cm']; ['3.89cm']>", "<towers ['a']; ['b']; ['c']>", "<towers ['a', 'a']; ['b', 'c']>", "<towers ['a', 'b']; ['c']>"], {'completed': True}),
    ("plan_and_execute", ["<pick up a>", "<stack a on b>", "<pick up c>", "<stack c on a>", "<done>"], {'completed': True}),
    ("execution", ["<pick up a>", "<stack a on b>", "<pick up c>", "<stack c on a>", "<done>"], {'completed': True}),
    ("full", ["<pick up a>", "<stack a on b>", "<pick up c>", "<stack c on a>", "<done>"], {'completed': True}),
    ("falling_tower", ["<pick up a>", "<stack a on b>", "<pick up c>", "<stack c on a>", "<stop here>"], {'max_height_acheived': 2}),
    ("build_tower_with_all_blocks", ["<pick up a>", "<stack a on b>", "<pick up c>", "<stack c on a>", "<done>"], {'passed': True}),
#    ("arrange_letters", [['s', 'u', 'n'], "<pick up u>", "<add u on n>", "<pick up s>", "<add s on u>", "<done>"], {'completed': True}),
    #("construct_words", ["sun", "<words ['sun']"], {'completed': True, 'passed': True}),
    ("anagram", ["<nus>", "<sun>", "<uns>", "<abc>", "<yes>", "<done>"], {'number_of_words_generated': 1}),
    ("permutation", ["<nus>", "<sun>", "<uns>", "<abc>", "<yes>", "<done>"], {'number_of_permutations_generated': 3}),
    ("isword", ["<abc>", "<yes>"], {'answered_correctly': True}),
])
def test_task_model(task, actions, expected_output):
    output_file = open(os.path.join("test_logs", f"{task}_mock.txt"), 'w')
    task = agency_evals.tasks[task](number_of_blocks=3, letters=['s', 'u', 'n'], falling_height=2, distraction_prob=0, perturb_prob=0, output_file=output_file)
    #task = agency_evals.tasks[task](number_of_blocks=3, falling_height=2, distraction_prob=0, perturb_prob=0, letters="abc", output_file=output_file)
    model = MockLLM(actions, output_file=output_file)
    results = task.run(model)
    print(task.evaluate(), file=output_file)
    assert not results['agent_error']
    assert not results['environment_error']
    for key in expected_output:
        assert results[key] == expected_output[key]
