from blocksworld_environment.blocksworld_environment import BlocksWorld, Action, PickUp, PutDown, Stack, Unstack, Measure, Help, list_to_natural_language
from tasks.task import Task
from agents import langchain_agent

import math
import time


class InformationGatheringTask(Task):
    """
    This task tests the agent's information gathering abilities by asking it to build
    a maximally high two-block tower. The agent doesn't initially know the height of
    each tower, and has to use noisy measurements to figure these out.
    """


    def set_up(self, max_steps_per_run = None, **kwargs):
        self.env.goal_description = "The goal is to build a maximally high tower using just two of the available blocks."
        self.env.set_actions([PickUp, PutDown, Stack, Unstack, Measure, Help])
        if kwargs.get('noise', None):
            self.max_steps = max_steps_per_run or 20 * self.env.number_of_blocks
        else:
            self.max_steps = max_steps_per_run or self.env.number_of_blocks + 10

    def stop_condition(self):
        return self.env.step_count > min(self.max_steps, self.env.total_questions_asked + 20) or self.env.most_blocks_in_a_tower >= 2

    def evaluate(self):
        optimal_height = sum(sorted([block.height for block in self.env.blocks.values()])[-2:])
        actual_height = self.env.height_of_highest_tower
        completed = self.env.most_blocks_in_a_tower >= 2
        return {
            'completed': completed,
            'regret': optimal_height - actual_height if completed else None,
            'actual_height': actual_height,
            'max_height_tower': optimal_height,
        }


#########################################################
#
# Build two-blocks capability
#
#########################################################


class BuildTwoBlockTowerCapability(InformationGatheringTask):
    """
    This task assumes that the agent has already obtained estimates of all block heights,
    and tests whether the agent is able to build a maximally high two-block tower based
    on those estimates.
    """

    def set_up(self, **kwargs):
        super().set_up(**kwargs)
        del self.env.actions[str(Measure)]

    def evaluate(self):
        results = super().evaluate()
        measuring_results = self.preceding_results['measuring']
        estimated_block_heights = {measuring_results[i]['block_to_measure']: measuring_results[i]['estimate'] for i in range(self.env.number_of_blocks)}
        assert set(estimated_block_heights.keys()) == set(self.env.block_names), "seems like not all measuring tasks were run prior to this subtask"
        estimated_optimal_height = sum(sorted([height for height in estimated_block_heights.values()])[-2:])
        if self.env.most_blocks_in_a_tower >= 2:
            tower = [tower for tower in self.env.get_state()['towers'] if len(tower)==2][0]
            estimated_height = estimated_block_heights[tower[0].name] + estimated_block_heights[tower[1].name]
        else:
            estimated_regret = None
            estimated_height = None
        results.update({
            'local_regret': estimated_optimal_height - estimated_height,
            'believed_height': estimated_height
        })
        return results


#########################################################
#
# Measuring capability
#
#########################################################

class SubmitHeightAction(Action):
    def __init__(self, env):
        self.estimate = None
        super().__init__(env)

    def string_specifies_action(self, action_str):
        return (action_str.startswith('height') and len(action_str.split(' ')) == 2)

    def execute_str(self, action_str):
        return self.execute(float(action_str.split(' ')[1].replace("cm", "")))

    def execute(self, estimate):
        self.times_executed += 1
        self.estimate = estimate
        self.env.done = True
        return f"Submitted height {self.estimate}cm."

    def possible_applications(self):
        return ["<height Xcm>"]

    def describe(self):
        return "When you are done measuring, submit your answer with <height Xcm>, where X is the height of the block. For example <height 1.32cm>."


class MeasuringCapability(Task):

    def set_up(self, max_steps_per_run = None, randomize_block=False, **kwargs):
        self.env.set_actions([Measure, SubmitHeightAction])
        seed = self.env.seed or int(time.time())
        if randomize_block:
            self.block = self.env.block_names[seed % len(self.env.block_names)]
        else:
            self.block = self.env.block_names[len(self.preceding_tasks) % len(self.env.block_names)]
        self.true_height = self.env.block_heights[self.block]

        self.env.goal_description = f"Your goal is to figure out the true height of block '{self.block}'."
        if kwargs.get('noise', None):
            self.max_steps = max_steps_per_run or 20 * self.env.number_of_blocks + 10
        else:
            self.max_steps = max_steps_per_run or self.env.number_of_blocks + 10

    def initial_instructions(self):
        if self.preceding_tasks:
            return super().initial_instructions()
        else:
            return f"There are blocks {list_to_natural_language(self.env.block_names)}. {self.env.goal_description} {self.env.describe_interface()}"

    def evaluate(self):
        estimate = self.env.actions[str(SubmitHeightAction)].estimate
        result = {
            'completed': self.env.done,
            'passed': self.env.done and math.isclose(estimate, self.true_height, abs_tol=0.01),
            'estimate': estimate,
            'true_height': self.true_height,
            'block_to_measure': self.block,
            'measuring_error': estimate - self.true_height if estimate else None,
            '|measuring_error|': abs(estimate - self.true_height) if estimate else None,
            'regret': abs(estimate - self.true_height) if estimate else None,
        }
        return result




if __name__ == "__main__":
    from mock_agent import MockLLM
    task = MeasuringCapability(number_of_blocks=3)
    llm = MockLLM(["<height 3.2cm>"])
    task.run(llm)
    print(task.evaluate())
