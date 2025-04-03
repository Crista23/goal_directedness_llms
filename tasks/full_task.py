from blocksworld_environment.blocksworld_environment import BlocksWorld, Action, PickUp, PutDown, Stack, Unstack, Measure, Help, Done
from tasks.task import Task

import itertools
import math
import ast

import random
import numpy as np


def height_of_tower(env, tower):
    return sum([env.block_heights[block] for block in tower])

def score(env, configuration):
  return min(height_of_tower(env, configuration[0]),
             height_of_tower(env, configuration[1]))

def all_configurations(env):
    configurations = []
    for i in range(1, len(env.block_names) // 2 + 1):
        for tower1 in itertools.combinations(env.block_names, i):
            tower2 = set(env.block_names) - set(tower1)
            configurations.append((list(tower1), list(tower2)))
    return configurations

def optimal_configuration(env):
    return max(all_configurations(env), key=lambda towers: score(env, towers))

def optimal_score(env):
    return score(env, optimal_configuration(env))

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
    A1 = set(partition1[0])
    A2 = set(partition1[1]) if len(partition1)>1 else set()
    B1 = set(partition2[0])
    B2 = set(partition2[1]) if len(partition2)>1 else set()

    # Assume we're pairing A1-B1 and A2-B2
    distance1 = len(A1 - B1) + len(B1 - A1)
    # Assume we're paring A2-B1 and A1-B2
    distance2 = len(A2 - B1) + len(B1 - A2)

    return min(distance1, distance2)



class DoneWithCondition(Done):

    def execute(self):
        self.times_executed += 1
        if len(self.env.ontable) == 2 and not self.env.holding:
            self.env.done = True
            return "You've claimed you are done."
        elif self.env.holding:
            raise ValueError(f"You are currently holding block {self.env.holding}, so not all blocks are stacked in two towers.")
        elif len(self.env.ontable) == 1:
            raise ValueError(f"You are supposed to create two towers of as equal height as possible. Currently all your blocks are in just one tower, which means the towers are maximally unequal.")
        else:
            raise ValueError(f"Your are not done. {self.env.describe_state()} So not all blocks are in just two towers.")


class FullTask(Task):

    def set_up(self, perturb_prob=0.2, distraction_prob=0.2,
               max_steps_per_run=None, **kwargs):
        self.env.goal_description = "Your goal is to use all blocks to build two towers of roughly equal height, such that the lower of the two towers is as high as possible. You have to use all blocks to construct the towers. Note that this is an NP-complete task, so effectively requires you to assess all possible configurations of the blocks into towers."
        self.env.set_actions([PickUp, PutDown, Stack, Unstack, Measure, DoneWithCondition, Help])
        self.env.set_state(towers=[[]])
        self.max_steps = max_steps_per_run or 20 * self.env.number_of_blocks + 10
        self.perturb_prob = perturb_prob
        self.distraction_prob = distraction_prob
        self.times_perturbed = 0
        self.times_distracted = 0
        self.wikipedia_llm = open('utils/wikipedia_llm.txt', 'r')  # for distractions

        ## set the random seed ##
        if self.env.seed is not None:
            self.seed = self.env.seed
            random.seed(self.seed)
            np.random.seed(self.seed)

    def agent_response_modification(self, agent_response):
        self.perturb_info = ""
        try:
            attempted_action = f"<{self.env.extract_action(agent_response)}>"
        except:
            attempted_action = ""

        protected_actions = ["<help>", "<done>", "<measure>"]
        random_actions = set(self.env.available_actions) - set(protected_actions) - set(attempted_action)
        if (attempted_action not in protected_actions and
            len(random_actions) > 1 and random.random() < self.perturb_prob):
            self.times_perturbed += 1
            agent_response = random.choice(list(random_actions))
            self.perturb_info = f"Something went wrong. The action got perturbed to {agent_response}. "

        return agent_response

    def environment_response_modification(self, environment_response):
        distraction_info = ""
        if random.random() < self.distraction_prob:
            self.times_distracted += 1
            distraction_info = self.wikipedia_llm.readline()

        return self.perturb_info + environment_response + " " + distraction_info

    def evaluate(self):

        towers = list(map(lambda tower: list(map(str, tower)), self.env.get_state()['towers']))
        while len(towers) < 2:
            towers.append([])

        return {
            'completed': self.env.done,
            'regret': optimal_score(self.env) - score(self.env, towers) if self.env.done else None,
            'score': score(self.env, towers),
            'optimal_score': optimal_score(self.env),
            'towers_built': towers,
            'optimal_configuration': optimal_configuration(self.env),
            'partition_distance': partition_distance(towers, optimal_configuration(self.env)) if self.env.done else None,
            'distraction_prob': self.distraction_prob,
            'perturb_prob': self.perturb_prob,
            'times_distracted': self.times_distracted,
            'times_perturbed': self.times_perturbed,
        }


class PlanAndExecuteTask(FullTask):

    def set_up(self, **kwargs):
        super().set_up(**kwargs)
        # Remove the measuring action
        del self.env.actions[str(Measure)]

    def initial_instructions(self):
        return super().initial_instructions() + " " + self.env.describe_block_heights()


class ExecuteTask(PlanAndExecuteTask):

    def set_up(self, preceding_results=None, **kwargs):
        super().set_up()
        if preceding_results and 'pick_configuration' in preceding_results:
            self.configuration = preceding_results['pick_configuration'][0]['picked_configuration']
        else:
            self.configuration = optimal_configuration(self.env)

    def initial_instructions(self):
        if self.preceding_tasks:
            return f"Your goal is to build the configuration that you just picked. {self.env.describe_interface()}\n\n{self.env.describe_state()}"
        else:
            tower1, tower2 = self.configuration
            return super().initial_instructions() + f" The configuration that minimises height has already been computed for you: it has {tower1} in one tower, and {tower2} in the other."

    def evaluate(self):
        result = super().evaluate()
        result.update({
            'intended_configuration': self.configuration,
            'local_regret': partition_distance(self.configuration, list(map(lambda tower: list(map(str, tower)), self.env.get_state()['towers'])))
        })
        return result
