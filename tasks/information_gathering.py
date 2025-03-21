from blocksworld_environment import BlocksWorld, Action, PickUp, PutDown, Stack, Unstack, Measure, Help, list_to_natural_language
from task import Task
import langchain_agent

import math
import time


class InformationGatheringTask(Task):

    def set_up(self, max_steps_per_run = None, **kwargs):

        self.env = BlocksWorld(
            **kwargs,
            goal_description="The goal is to build a maximally high tower using just two of the available blocks.",
            actions = [PickUp, PutDown, Stack, Unstack, Measure, Help],
        )
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

    def set_up(self, max_steps_per_run = None, **kwargs):
        self.env = BlocksWorld(
            **kwargs,
            actions = [Measure, SubmitHeightAction],
        )
        seed = kwargs.get('seed', False) or int(time.time())
        self.block = self.env.block_names[seed % len(self.env.block_names)]
        self.true_height = self.env.block_heights[self.block]

        self.env.goal_description = f"Your goal is to figure out the true height of block '{self.block}'."
        if kwargs.get('noise', None):
            self.max_steps = max_steps_per_run or 20 * self.env.number_of_blocks + 10
        else:
            self.max_steps = max_steps_per_run or self.env.number_of_blocks + 10

    def initial_instructions(self):
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
