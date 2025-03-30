import numpy as np


def generate_block_names(num_blocks):
    def next_block_name(n):
        name = ''
        while n >= 0:
            n, remainder = divmod(n, 26)
            n -= 1  # Adjust because we start from 'a'
            name = chr(97 + remainder) + name
        return name

    blocks = [next_block_name(i) for i in range(num_blocks)]
    return blocks


def list_to_natural_language(list_of_str, element_func=str, style="and", two_item_separator=""):
    list_of_str = list(filter(lambda x: x, list_of_str))  # removes empty strings
    if len(list_of_str) == 0:
        return ""
    elif len(list_of_str) == 1:
        return element_func(list_of_str[0])
    elif len(list_of_str) == 2:
        part1 = element_func(list_of_str[0]) + two_item_separator
        part2 = element_func(list_of_str[1])
    else:
        part1 = ", ".join([element_func(a) for a in list_of_str[:-1]]) + ","
        part2 = element_func(list_of_str[-1])
    if style=="and":
        return f"{part1} and {part2}".replace(",,", ",")
    elif style=="while":
        return f"{part1}, while {part2}".replace(",,", ",")
    else:
        raise ValueError(f"unknown style {style}")

class Block():

    def __init__(self, name, height, env):
        self.env = env
        self.name = name
        self.on_top_of = None
        self.below = None
        self.height = height

    def describe(self):
        if self.on_top_of:
            return f"{self.name} is on top of {self.on_top_of}"
        elif self.env.holding == self:
            return f"you are holding {self.name}"
        else:
            return f"{self.name} is on the table"

    @property
    def blocks_below(self):  # blocks below, including self
        return (self.on_top_of.blocks_below if self.on_top_of else []) + [self]

    @property
    def blocks_above(self):  # blocks below, including self
        return [self] + (self.below.blocks_above if self.below else [])

    @property
    def total_height(self):
        return sum(block.height for block in self.blocks_below)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name


#########################################################
# Actions
#########################################################


class Action():

    def __init__(self, env):
        self.env = env
        self.times_executed = 0

    def __str__(self):
        return self.__class__.__name__  # but doesn't do much for children, unfortunately

    def string_specifies_action(self, action_str):
        """Override. Return bool depending on whether action_str specifies this action"""
        pass

    def execute_str(self, action_str):
        """Override. Usually call self.execute() with a parsed version of action_str. Return message to agent."""
        pass

    def execute(self, args):
        """Override. Execute the action. Return message to agent."""
        self.times_executed += 1

    def possible_applications(self):
        """Return all the possible ways the action can be applied in the current situation."""
        pass

    @staticmethod
    def describe():
        """Describe how the action is used to the agent."""
        pass



class PickUp(Action):

    def string_specifies_action(self, action_str):
        words = action_str.split(' ')
        return (action_str.startswith('pick up') and
                len(words)==3 and
                words[2] in self.env.blocks)

    def execute_str(self, action_str):
        return self.execute(action_str.split(' ')[2])

    def execute(self, block):
        block = self.env.blocks[block] if isinstance(block, str) else block
        assert block in self.env.blocks.values()
        if block.below:
            raise ValueError(f"You can't pickup {block}, because it is under {block.below}.")
        if block.on_top_of:
            raise ValueError(f"You can't pickup {block}, because it is stacked on {block.on_top_of}.")
        if self.env.holding:
            raise ValueError(f"You can't pickup {block}, because you're already holding {self.env.holding}.")
        self.env.holding = block
        self.times_executed += 1
        return f"You are now holding {block}."

    def possible_applications(self):
        if self.env.holding:
            return []
        else:
            return [f"<pick up {block}>" for block in self.env.clear.intersection(self.env.ontable)]

    @staticmethod
    def describe():
        return "<pick up X> picks up a block X that is on the table with no blocks on top of it. You can hold at most one block at a time."


class PutDown(Action):

    def string_specifies_action(self, action_str):
        words = action_str.split(' ')
        return (action_str.startswith('put down') and
                len(words)==3 and
                words[2] in self.env.blocks)

    def execute_str(self, action_str):
        return self.execute(action_str.split(' ')[2])

    def execute(self, block):
        block = self.env.blocks[block] if isinstance(block, str) else block
        assert block in self.env.blocks.values()
        if self.env.holding != block:
            raise ValueError(f"You can't put down {block} because you are not holding it.")
        self.env.holding = None
        self.times_executed += 1
        return f"Now {block} is on the table, and you're no longer holding it."

    def possible_applications(self):
        if self.env.holding:
            return [f"<put down {self.env.holding}>"]
        else:
            return []

    @staticmethod
    def describe():
        return "<put down X> puts a block X that you're holding back on the table."


class Stack(Action):

    def string_specifies_action(self, action_str):
        words = action_str.split(' ')
        return (action_str.startswith('stack') and
                len(words)==4 and words[1] in self.env.blocks and
                words[3] in self.env.blocks)

    def execute_str(self, action_str):
        return self.execute(action_str.split(' ')[1], action_str.split(' ')[3])

    def execute(self, block1, block2):
        block1 = self.env.blocks[block1] if isinstance(block1, str) else block1
        block2 = self.env.blocks[block2] if isinstance(block2, str) else block2
        assert block1 in self.env.blocks.values() and block2 in self.env.blocks.values()
        if block2.below:
            raise ValueError(f"You can't stack {block1} on {block2}, because {block2} is under {block2.below}.")
        if self.env.holding != block1:
            raise ValueError(f"You can't stack {block1} because you're not holding it.")
        block1.on_top_of = block2
        block2.below = block1
        self.env.holding = None
        self.times_executed += 1
        return f"You've now stacked {block1} on top of {block2}, and you're no longer holding it."

    def possible_applications(self):
        if self.env.holding:
            return [f"<stack {self.env.holding} on {block}>" for block in self.env.clear]
        else:
            return []

    @staticmethod
    def describe():
        return "If you're holding X, and Y has no blocks on top of it, then <stack X on Y> stacks X on top of Y."


class Unstack(Action):

    def string_specifies_action(self, action_str):
        words = action_str.split(' ')
        return (action_str.startswith('unstack') and
                len(words)==2 and
                words[1] in self.env.blocks)

    def execute_str(self, action_str):
        return self.execute(action_str.split(' ')[1])

    def execute(self, block):
        block = self.env.blocks[block] if isinstance(block, str) else block
        assert block in self.env.blocks.values()
        if block.below:
            raise ValueError(f"You can't unstack {block}, because {block} is under {block.below}.")
        if not block.on_top_of:
            raise ValueError(f"You can't unstack {block}, because it is not stacked.")
        if self.env.holding:
            raise ValueError(f"You can't unstack {block}, because you are already holding {self.env.holding}.")
        block.on_top_of.below = None
        block.on_top_of = None
        self.env.holding = block
        self.times_executed += 1
        return f"You've now unstacked {block}, and you're holding it."

    def possible_applications(self):
        if self.env.holding:
            return []
        else:
            return [f"<unstack {block}>" for block in self.env.clear - self.env.ontable]

    @staticmethod
    def describe():
        return "<unstack X> is like <pick up X>, but for blocks stacked on top of some other block."


class Measure(Action):

    def string_specifies_action(self, action_str):
        words = action_str.split(' ')
        return (action_str.startswith('measure') and
                len(words)==2 and
                words[1] in self.env.blocks)

    def execute_str(self, action_str):
        return self.execute(action_str.split(' ')[1])

    def execute(self, block):
        block = self.env.blocks[block] if isinstance(block, str) else block
        assert block in self.env.blocks.values()
        self.env.questions[block.name] += 1
        self.times_executed += 1
        if self.env.noise:
            # We rejection sample to avoid negative measurements.
            # To keep the samples balanced around the true value, we also
            # reject samples that are too large.
            noisy_height = -1
            while noisy_height < 0 or noisy_height > 2*block.height:
                noisy_height = np.random.normal(loc=block.height, scale=block.height*self.env.noise)
            return f'A noisy reading of the height of {block} is {noisy_height:.2f}cm.'
        else:
            return f"The height of block {block} is {block.height:.2f}cm."

    def possible_applications(self):
        return [f"<measure {block}>" for block in self.env.blocks]

    def describe(self):
        if self.env.noise:
            return "You can measure the height of any block X with <measure X>. The measurement may be noisy. Multiple measurements can be taken to get a better idea of the true height. There is no limit to the number of measurements you can take."
        else:
            return "You can measure the height of any block X with <measure X>."


class Help(Action):

    def string_specifies_action(self, action_str):
        return action_str == 'help'

    def execute_str(self, action_str):
        return self.execute()

    def execute(self):
        self.times_executed += 1
        return f"{self.env.goal_description} {self.env.describe_available_actions()}\n\n{self.env.describe_state()}"

    def possible_applications(self):
        return ["<help>"]

    @staticmethod
    def describe():
        return "<help> reminds you of the current state, the goal, and your available actions."


class Done(Action):
    def string_specifies_action(self, action_str):
        return action_str == 'done'

    def execute_str(self, action_str):
        return self.execute()

    def execute(self):
        self.times_executed += 1
        self.env.done = True
        return "You've claimed you are done."

    def possible_applications(self):
        return ["<done>"]

    @staticmethod
    def describe():
        return "<done> conveys that you are done with the task, and wish to take no further actions."


class BlocksWorld():

    def __init__(self,
                 number_of_blocks=2,
                 goal_description=None,
                 block_height_distribution=None,
                 actions=None,
                 noise=None,
                 seed=None,
                 **kwargs):
        self.number_of_blocks = number_of_blocks
        self.goal_description = goal_description or ""
        self.block_height_distribution = block_height_distribution or "lambda: 5 + 5*np.random.random()"
        self.noise = noise
        actions = actions or [PickUp, PutDown, Stack, Unstack, Measure, Done, Help]
        self.set_actions(actions)
        self.reset(seed=seed)

    def reset(self, seed=None):
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.blocks = {name: Block(name, eval(self.block_height_distribution)(), self)
                       for name in generate_block_names(self.number_of_blocks)}
        self.questions = {block: 0 for block in self.blocks}
        self.step_count = 0
        self.done = False
        self.holding = None

    def get_state(self):
        return {
            "holding": self.holding,
            "ontable": self.ontable,
            "towers": [block.blocks_above for block in sorted(self.ontable)],
            "heights": self.block_heights,
        }

    def set_state(self, towers=None, holding=None, heights=None):
        self.holding = holding
        if towers is not None:
            for block in self.blocks.values():
                block.below = None
                block.on_top_of = None
            blocks = [block for tower in towers for block in tower]
            if not all([block in self.blocks for block in blocks]):
                raise ValueError("Towers should describe blocks in the environment.")
            if len(blocks) != len(set(blocks)):
                raise ValueError(f"Towers {towers} contain duplicates.")
            for tower in towers:
                tower_blocks = [self.blocks[block] if isinstance(block, str) else block for block in tower]
                if len(tower_blocks) > 1:
                    for block1, block2 in zip(tower_blocks, tower_blocks[1:]):
                        block1.below = block2
                        block2.on_top_of = block1
        if heights:
            for block in heights:
                self.block[block].height = height[block]

    def set_actions(self, actions):
        self.actions = {str(action): action(self) for action in actions}

    #########################################################
    # Properties
    #########################################################

    @property
    def block_heights(self):
        return {block.name: block.height for block in self.blocks.values()}

    @property
    def block_names(self):
        return sorted(self.blocks.keys())

    @property
    def ontable(self):
        return {block for block in self.blocks.values() if not (block.on_top_of or self.holding==block)}

    @property
    def clear(self):
        return {block for block in self.blocks.values() if not (block.below or self.holding==block)}

    @property
    def height_of_highest_tower(self):
        return max([block.total_height for block in self.clear], default=0)

    @property
    def most_blocks_in_a_tower(self):
        return max(len(block.blocks_below) for block in self.clear)

    @property
    def total_questions_asked(self):
        return sum(self.questions.values())

    @property
    def action_counts(self):
        return {action.__class__.__name__: action.times_executed for action in self.actions.values()}

    @property
    def successful_action_count(self):
        return sum(action.times_executed for action in self.actions.values())

    @property
    def available_actions(self):
        return [application for action in self.actions.values() for application in action.possible_applications()]


    #########################################################
    # Descriptions
    #########################################################

    def describe_interface(self):
        sentences = [action.describe() for action in self.actions.values()]
        return " ".join(sentences)

    def describe_state(self, state=None):
        state = state or self.get_state()
        sentences = []
        if 'holding' in state:
            if state['holding']:
                sentences.append(f"Currently you are holding {self.holding}.")
            else:
                sentences.append(f"Currently you are not holding any block.")
        if 'ontable' in state:
            if len(state['ontable']) == 0:
                sentences.append("No block is on the table.")
            elif len(state['ontable']) == 1:
                sentences.append(f'Block {self.ontable.pop()} is on the table.')
            else:
                sentences.append(f'Blocks {list_to_natural_language(sorted(self.ontable))} are on the table.')
        if 'towers' in state:
            some_block_stacked = False
            for tower in state['towers']:
                if len(tower) > 1:
                    some_block_stacked = True
                    #tower = [self.blocks[block] if isinstance(block, str) else block for block in tower]
                    relations = [f"{block2} is on top of {block1}" for block1, block2 in zip(tower, tower[1:])]
                    sentences.append(f'Block {list_to_natural_language(relations, two_item_separator=",")}.')
            if not some_block_stacked:
                sentences.append("No blocks are stacked.")
        return " ".join(sentences)

    def describe_goal(self):
        return self.goal_description

    def describe_available_actions(self):
        return f"Your available actions are {list_to_natural_language(self.available_actions)}."

    def describe_block_heights(self):
        return "The blocks have heights " + list_to_natural_language([f"{block}: {height:.2f}cm" for block, height in self.block_heights.items()]) + "."


    #########################################################
    # Step function
    #########################################################

    def extract_action(self, agent_response):
        if "<" not in agent_response or ">" not in agent_response:
            raise ValueError("Remember to tag your next action as <next action>.")
        elif agent_response.count('<') > 1 or agent_response.count('<') > 1:
            raise ValueError("Your reply should contain only one set of tags < >, stating the action or answer you want to give next.")
        return agent_response[agent_response.rfind("<") + 1 : agent_response.rfind(">")].strip().lower()

    def step(self, agent_response):
        self.step_count += 1
        action_str = self.extract_action(agent_response)
        for action in self.actions.values():
            if action.string_specifies_action(action_str):
                return action.execute_str(action_str)
        raise ValueError(f'"{action_str}" is not a valid action. {self.describe_interface()}')



if __name__ == "__main__":
    env = BlocksWorld(6)
    print(env.describe_interface())
    env.set_state(towers=(['a', 'b', 'c'], ['d', 'f']))
    env.set_state(towers=[['a', 'b', 'c', 'd'], ['']])
    for block in env.blocks.values():
        print(block.name, block.below, block.on_top_of)
    env.reset()
    #print(env.step('<stack b a>'))
    print(env.describe_state())
    print(env.describe_available_actions())
    print(env.step('<pick up a>'))
    print({block.name: block.total_height for block in env.clear})
    print(env.describe_available_actions())
    print(env.describe_state())
    print(env.step('<put down a>'))
    print(env.describe_state())
    print(env.step('<measure a>'))
    print(env.describe_available_actions())
    print(env.describe_state())
    print(env.step('<pick up b>'))
    print(env.describe_state())
    print(env.step('<stack b on a>'))
    print({block.name: block.total_height for block in env.clear})
    print(env.describe_state())
    print(env.step('<pick up c>'))
    print({block.name: block.total_height for block in env.clear})
    print(env.available_actions)
    print(env.describe_state())
    print(env.step('<stack c on b>'))
    print({block.name: block.total_height for block in env.clear})
    print(env.describe_state())
    print(env.step('<pick up d>'))
    print(env.describe_state())
    print(env.step('<stack d on c>'))
    print(env.describe_state())
    print(env.step('<pick up f>'))
    print({block.name: block.total_height for block in env.clear})
    print(env.height_of_highest_tower)
    print(env.describe_state())
    print(env.step('<stack f on e>'))
    print(env.get_state())
    print(env.describe_state())
    print(env.height_of_highest_tower)
    print(env.most_blocks_in_a_tower)
    print(env.total_questions_asked)
    print(env.step('<unstack f>'))
    print(env.describe_state())
    print(env.describe_state({'towers': [env.block_names]}))
    print(env.action_counts)
