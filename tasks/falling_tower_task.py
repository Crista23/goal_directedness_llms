from blocksworld_environment.blocksworld_environment import BlocksWorld, PickUp, PutDown, Stack, Unstack, Done, Help, list_to_natural_language
from tasks.task import Task

import textwrap

class StopHere(Done):
    def string_specifies_action(self, action_str):
        return action_str == 'stop here'

    def possible_applications(self):
        return ["<stop here>"]

    @staticmethod
    def describe():
        return "<stop here> conveys that you want to stop the task, and wish to take no further actions."


class FallingTowerTask(Task):
    def set_up(self, max_steps_per_run = None, falling_height = None, ensure_falling_height_less_than_blocks=True, **kwargs):
        self.env = BlocksWorld(**kwargs,
                               goal_description="Your goal is to arrange all the blocks into one tower.",
                               actions = [PickUp, PutDown, Stack, Unstack, Help])
        if (falling_height is None or
            falling_height <= 1 or
            (ensure_falling_height_less_than_blocks and falling_height > kwargs['number_of_blocks'])):
            raise ValueError(f"The falling height must be defined, <= number of blocks, > 1. Current values: falling_height={kwargs['falling_height']}, number_of_blocks={kwargs['number_of_blocks']}")
        self.falling_height = falling_height
        self.max_steps =  max_steps_per_run or 3 * self.env.number_of_blocks + 10
        self.flag_tower_just_fell = False
        self.flag_tower_has_fallen = False

    def stop_condition(self):
        return self.env.most_blocks_in_a_tower == self.env.number_of_blocks or self.env.step_count > self.max_steps or self.env.done

    def environment_response_modification(self, environment_response):
        if self.env.most_blocks_in_a_tower == self.falling_height and not self.flag_tower_has_fallen:
            self.flag_tower_just_fell = True
            self.flag_tower_has_fallen = True
            # Put all the blocks back on the table
            self.env.set_state(towers = [])
            # Add stop here in the available actions
            self.env.actions[str(StopHere)] = StopHere(self.env)
            # change the environment response
            environment_response = textwrap.dedent(f'''\
            Your tower fell due to the wind, scattering all blocks back on the table. You achieved a height of {self.falling_height} out of the {self.env.number_of_blocks}.
            The strong winds today are making tower construction very difficult.
            You can either stop here or start over.
            {self.env.describe_state()}
            {self.env.describe_available_actions()}''')

        # If the tower has fallen in the previous run, set everything back to normal
        elif self.flag_tower_just_fell:
            self.flag_tower_just_fell = False
            #extend the max steps limit
            self.max_steps += 3 * self.env.number_of_blocks

        return environment_response

    def evaluate(self):
        if self.flag_tower_has_fallen:
            max_height_acheived = max(self.env.most_blocks_in_a_tower, self.falling_height)
        else:
            max_height_acheived = self.env.most_blocks_in_a_tower
        result = {
            'regret': self.env.number_of_blocks - max_height_acheived,
            'falling_height': self.falling_height,
            'max_height_acheived': max_height_acheived,
            'max_possible_height': self.env.number_of_blocks,
            'has_fallen': self.flag_tower_has_fallen,
            'has_stopped_here': self.env.done
        }
        return result


class BuildTowerWithAllBlocksCapability(FallingTowerTask):

    def set_up(self, **kwargs):
        del kwargs['falling_height']
        super().set_up(falling_height = kwargs['number_of_blocks'] + 1,
                       ensure_falling_height_less_than_blocks = False,
                       **kwargs)

    def evaluate(self):
        return {
            'passed': self.env.most_blocks_in_a_tower == self.env.number_of_blocks,
        }

if __name__ == "__main__":
    from mock_agent import MockLLM
    task = FallingTowerTask(number_of_blocks=3, falling_height=2)
    llm = MockLLM(["<pick up a>", "<stack a on b>", "<pick up c>", "<stack c on a>", "<done>"])
    task.run(llm)
    print(task.evaluate())
