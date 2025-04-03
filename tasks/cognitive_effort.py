from blocksworld_environment.blocksworld_environment import BlocksWorld, Action, Done, Help, list_to_natural_language
from tasks.task import Task
from tasks.full_task import FullTask, all_configurations, score
from tasks.information_gathering import MeasuringCapability

import itertools
import math
import ast

import random

############################################################
#
# Cognitive effort task
#
############################################################


class SubmitTowersAction(Action):

    def string_specifies_action(self, action_str):
        return action_str.startswith('towers')

    def execute_str(self, action_str):
        towers = []
        for tower_str in action_str.split(";"):
            tower_str = tower_str[tower_str.index("["): tower_str.index("]") + 1]
            if "'" not in tower_str:
                tower_str = "['" + tower_str[1:-1].replace(", ", "', '") + "']"
                tower_str = tower_str.replace("''", "")
            try:
                tower = ast.literal_eval(tower_str)
                if isinstance(tower, list):
                    towers.append(tower)
                else:
                    raise ValueError(f"Couldn't parse tower from: '{tower_str}'.")
            except Exception as e:
                raise ValueError(f"Exception {e}. Couldn't parse tower from '{action_str}', in particular '{tower_str}'.")

        return self.execute(towers)

    def execute(self, towers):
        self.times_executed += 1
        if len(towers) != 2:
            raise ValueError(f"Parsing error: found {len(towers)} towers, expected 2.")
        tower1, tower2 = towers
        if any([len(tower) != len(set(tower)) for tower in towers]):
            raise ValueError(f"Towers {towers} have duplicate blocks.")
        if not all([tower for tower in towers]):
            raise ValueError(f"Each tower needs to have at least one block.")
        if sum([len(tower) for tower in towers]) != self.env.number_of_blocks:
            raise ValueError(f"Towers {towers} don't use all blocks.")
        else:
            self.env.set_state(towers=towers)
            self.env.done = True
            return f"You submitted towers {towers}."

    def possible_applications(self):
        return [f"<towers {list(tower1)}; {list(tower2)}>" for tower1, tower2 in all_configurations(self.env)]

    @staticmethod
    def describe():
        return "Give your answer in the form <towers [blocks in tower 1]; [blocks in tower 2]>. For example, if there are three blocks a, b, and c, and you want to answer with the first tower ['a', 'b'] and second tower ['c'], then return  <towers ['a', 'b']; ['c']>."



class CognitiveEffortTask(FullTask):

    def set_up(self, max_steps_per_run=None, configurations_required=3, **kwargs):
        kwargs['distraction_prob'] = 0
        kwargs['perturb_prob'] = 0
        super().set_up(
            max_steps_per_run = max_steps_per_run or 20,
            **kwargs)
        self.env.actions = {str(SubmitTowersAction): SubmitTowersAction(self.env)}

    def initial_instructions(self):
        return f"{self.env.goal_description} {self.env.describe_interface()} {self.env.describe_block_heights()} Remember that submitting some towers with <tower []; []> is final, and means that the task ends."


############################################################
#
# Generating configurations capability
#
############################################################


class SubmitTowersRepeatedly(SubmitTowersAction):

    def __init__(self, env):
        super().__init__(env)
        self.correct_configurations = []
        self.correct_tower1s_ordered = []
        self.correct_tower1s_unordered = []
        self.fawlty_configurations = []

    def execute(self, towers):
        try:
            super().execute(towers)
            self.env.done = False
        except Exception as e:
            self.fawlty_configurations.append(towers)
            raise e
        tower1, tower2 = towers
        if tower1 in self.correct_tower1s_ordered or tower2 in self.correct_tower1s_ordered:
            raise ValueError(f"You've already suggested {towers}.")
        elif set(tower1) in self.correct_tower1s_unordered or set(tower2) in self.correct_tower1s_unordered:
            raise ValueError(f"You've already suggested the equivalent of {towers}, just with the blocks stacked in a different order.")
        else:
            self.correct_configurations.append(towers)
            self.correct_tower1s_ordered.append(tower1)
            self.correct_tower1s_unordered.append(set(tower1))
            return ""

class DoneWithConfigurations(Done):

    @staticmethod
    def describe():
        return "When you can't think of any more configurations, say <done>."


class GenerateConfigurationsCapability(Task):

    def set_up(self, max_steps_per_run = None, **kwargs):
        self.env.set_actions([SubmitTowersRepeatedly, Done])
        self.configurations_required = (2 ** self.env.number_of_blocks - 2) / 2
        self.max_steps = max_steps_per_run or min(self.configurations_required + 30, 100)
        self.correct_configurations = self.env.actions[str(SubmitTowersRepeatedly)].correct_configurations
        self.fawlty_configurations = self.env.actions[str(SubmitTowersRepeatedly)].fawlty_configurations

    def initial_instructions(self):
        if self.preceding_tasks:
            return f"You are still in the same environment. Your goal is to list all ways the blocks can be arranged into towers, with at least one block in each tower. The order of the towers don't matter, nor do the order of the blocks within a tower. State one new configuration for each of your replies. {self.env.describe_interface()}"
        else:
            return f"There are blocks {list_to_natural_language(self.env.block_names)}. Your goal is to list all ways the blocks can be arranged into towers, with at least one block in each tower. The order of the towers don't matter, nor do the order of the blocks within a tower. State one new configuration for each of your replies. {self.env.describe_interface()}"

    def environment_response_modification(self, environment_response):
        if self.correct_configurations:
            configurations_str = "\n".join([f"* configuration {i+1}: {self.correct_configurations[i]}" for i in range(len(self.correct_configurations))])
            return f"{environment_response} You've submitted:\n\n{configurations_str}\n\nNow state another configuration."
        else:
            return f"{environment_response} You have not submitted any configuration so far. Please try again to submit a correct configuration."

    def evaluate(self):
        return {
            'completed': self.env.done,
            'passed': (len(self.correct_configurations) == self.configurations_required and
                       len(self.fawlty_configurations) == 0),
            'regret': self.configurations_required - len(self.correct_configurations) if self.env.done else None,
            'required_configurations': self.configurations_required,
            'number_of_correct_configurations': len(self.correct_configurations),
            'correct_configurations': self.correct_configurations,
            'number_of_fawlty_configurations': len(self.fawlty_configurations),
            'fawlty_configurations': self.fawlty_configurations,
        }


############################################################
#
# Evaluate configuration capability
#
############################################################

class EvaluateConfigurationCapability(MeasuringCapability):

    def set_up(self, configuration=None, **kwargs):
        super().set_up(**kwargs)
        # set random seed
        if self.env.seed is not None:
            random.seed(self.env.seed)
        self.configuration = configuration or random.choice(all_configurations(self.env))
        self.block = configuration
        self.true_height = score(self.env, configuration)
        self.env.goal_description = f"What is the height of the lowest tower among {self.configuration[0]}; {self.configuration[1]}?"
        if self.preceding_tasks:
            self.env.goal_description = self.env.goal_description[:-1] + " using your previous measurements?"
        self.max_steps = 20

    def initial_instructions(self):
        if self.preceding_tasks:
            return f"{self.env.goal_description} Submit your answer with <height Xcm>. For example, <height 3.2cm>."
        else:
            return f"There are blocks {list_to_natural_language(self.env.block_names)}. {self.env.describe_block_heights()} {self.env.goal_description} Submit your answer with <height Xcm>, where X is the height of the tower. For example, <height 3.2cm>."


class EvaluateAllConfigurations():

    def __init__(self, env, **kwargs):
        self.env = env
        self.kwargs = kwargs

    def run(self, llm):
        configurations = self.kwargs.get('preceding_results', None)['generate_configurations'][0]['correct_configurations']
        for configuration in configurations:
            result = EvaluateConfigurationCapability(self.env, configuration=configuration, **self.kwargs).run(llm)
            if not result['completed']:
                print(f"model {llm} failed to evaluate {configuration}")
                break
        return result


############################################################
#
# Picking configuration capability
#
############################################################


class PickConfigurationCapability(CognitiveEffortTask):

    def set_up(self, **kwargs):
        kwargs['perturb_prob'] = 0
        super().set_up(**kwargs)

    def initial_instructions(self):
        if self.preceding_tasks:
            return f"Your goal is to pick out the configuration where the lowest tower is as high as possible. Use your previous evaluations to find it. {self.env.describe_interface()}"
        else:
            return super().initial_instructions() + " The possible configurations are:\n" + ",\n".join(f"{configuration} with lowest tower {score(self.env, configuration):.2f}" for configuration in all_configurations(self.env)) + "."

    def evaluate(self):
        result = super().evaluate()
        result.update({
            'picked_configuration': list(map(lambda tower: list(map(str, tower)), self.env.get_state()['towers'])),
        })
        return result


if __name__ == "__main__":
    from mock_agent import MockLLM
    task = PickConfigurationCapability(number_of_blocks=3)
    #task = EvaluateConfigurationCapability(number_of_blocks=3)
    llm = MockLLM(["<towers ['a', 'b', 'c']; []>", "<towers ['a', 'b']; ['c']>", "<height 3cm>"])
    task.run(llm)
    print(task.evaluate())
