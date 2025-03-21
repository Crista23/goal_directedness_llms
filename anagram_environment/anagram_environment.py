import numpy as np
from blocksworld_environment.blocksworld_environment import list_to_natural_language, Action

def generate_letters(num_letters):
    def next_letter(n):
        name = ''
        while n >= 0:
            n, remainder = divmod(n, 26)
            n -= 1  # Adjust because we start from 'a'
            name = chr(97 + remainder) + name
        return name

    letters = [next_letter(i) for i in range(num_letters)]
    return letters

class Letter():

    def __init__(self, name, env):
        self.env = env
        self.name = name
        self.on_top_of = None
        self.before = None

    def describe(self):
        if self.on_top_of:
            return f"{self.name} is after {self.on_top_of}"
        elif self.env.holding == self:
            return f"you are holding {self.name}"
        else:
            return f"{self.name} is available"

    @property
    def letters_before(self):  
        return (self.on_top_of.letters_before if self.on_top_of else []) + [self]

    @property
    def letters_after(self):  
        return [self] + (self.before.letters_after if self.before else [])

    @property
    def total_letters(self):
        return sum(1 for letter in self.letters_before)

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
                words[2] in self.env.letters)

    def execute_str(self, action_str):
        return self.execute(action_str.split(' ')[2])

    def execute(self, letter):
        #letter = self.env.letters[letter] if isinstance(letter, str) else letter
        assert letter in self.env.letters
        letter = Letter(letter, self)
        if letter.before:
            raise ValueError(f"You can't pickup {letter}, because it is under {letter.before}.")
        if letter.on_top_of:
            raise ValueError(f"You can't pickup {letter}, because it is stacked on {letter.on_top_of}.")
        if self.env.holding:
            raise ValueError(f"You can't pickup {letter}, because you're already holding {self.env.holding}.")
        self.env.holding = letter
        self.times_executed += 1
        return f"You are now holding {letter}."

    def possible_applications(self):
        if self.env.holding:
            return []
        else:
            return [f"<pick up {letter}>" for letter in self.env.clear.intersection(self.env.ontable)]

    @staticmethod
    def describe():
        return "<pick up X> picks up a letter X that is on the table with no letters on top of it. You can hold at most one letter at a time."


class PutDown(Action):

    def string_specifies_action(self, action_str):
        words = action_str.split(' ')
        return (action_str.startswith('put down') and
                len(words)==3 and
                words[2] in self.env.letters)

    def execute_str(self, action_str):
        return self.execute(action_str.split(' ')[2])

    def execute(self, letter):
        letter = self.env.letters[letter] if isinstance(letter, str) else letter
        assert letter in self.env.letters.values()
        if self.env.holding != letter:
            raise ValueError(f"You can't put down {letter} because you are not holding it.")
        self.env.holding = None
        self.times_executed += 1
        return f"Now {letter} is on the table, and you're no longer holding it."

    def possible_applications(self):
        if self.env.holding:
            return [f"<put down {self.env.holding}>"]
        else:
            return []

    @staticmethod
    def describe():
        return "<put down X> puts a letter X that you're holding back on the table."


class Add(Action):

    def string_specifies_action(self, action_str):
        words = action_str.split(' ')
        return (action_str.startswith('add') and
                len(words)==4 and words[1] in self.env.letters and
                words[3] in self.env.letters)

    def execute_str(self, action_str):
        return self.execute(action_str.split(' ')[1], action_str.split(' ')[3])

    def execute(self, letter1, letter2):
        
        assert letter1 in self.env.letters and letter2 in self.env.letters
        letter1 = Letter(letter1, self)
        letter2 = Letter(letter2, self)
        
        if letter2.before:
            raise ValueError(f"You can't add {letter1} on {letter2}, because {letter2} is under {letter2.before}.")
        if self.env.holding.name != letter1.name:
            raise ValueError(f"You can't add {letter1} because you're not holding it.")
        letter1.on_top_of = letter2
        letter2.before = letter1
        self.env.holding = None
        self.times_executed += 1
        return f"You've now added {letter1} on top of {letter2}, and you're no longer holding it."

    def possible_applications(self):
        if self.env.holding:
            return [f"<add {self.env.holding} on {letter}>" for letter in self.env.clear]
        else:
            return []

    @staticmethod
    def describe():
        return "If you're holding X, and Y has no letters after it, then <add X to Y> adds X at the end of Y."


class Remove(Action):

    def string_specifies_action(self, action_str):
        words = action_str.split(' ')
        return (action_str.startswith('remove') and
                len(words)==2 and
                words[1] in self.env.letters)

    def execute_str(self, action_str):
        return self.execute(action_str.split(' ')[1])

    def execute(self, letter):
        #letter = self.env.letters[letter] if isinstance(letter, str) else letter
        assert letter in self.env.letters
        letter = Letter(letter, self)
        if letter.before:
            raise ValueError(f"You can't remove {letter}, because {letter} before {letter.before}.")
        if not letter.on_top_of:
            raise ValueError(f"You can't remove {letter}, because it is not added.")
        if self.env.holding:
            raise ValueError(f"You can't remove {letter}, because you are already holding {self.env.holding}.")
        letter.on_top_of.before = None
        letter.on_top_of = None
        self.env.holding = letter
        self.times_executed += 1
        return f"You've now removed {letter}, and you're holding it."

    def possible_applications(self):
        if self.env.holding:
            return []
        else:
            return [f"<remove {letter}>" for letter in self.env.clear - self.env.ontable]

    @staticmethod
    def describe():
        return "<remove X> is like <pick up X>, but for letters added on top of some other letter."


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


class AnagramWorld():

    def __init__(self,
                 letters=[],
                 goal_description=None,
                 actions=None,
                 noise=None,
                 seed=None,
                 **kwargs):
        self.letters = letters
        self.number_of_letters = len(letters)
        self.goal_description = goal_description or ""
        self.noise = noise

        actions = actions or [PickUp, PutDown, Add, Remove, Done, Help]
        self.actions = {str(action): action(self) for action in actions}
        self.submitted_word = ""
        self.reset(seed=seed)

    def reset(self, seed=None):
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.step_count = 0
        self.done = False
        self.holding = None

    def get_state(self):
        return {
            "holding": self.holding,
            "ontable": self.ontable,
            "words": [letter.letters_after for letter in sorted(self.ontable)],
        }

    def set_state(self, state):
        for letter in self.letters:
            letter.below = None
            letter.on_top_of = None
        if 'words' in state:
            words = [letter for word in state['words'] for letter in word]
            if len(words) != len(set(words)):
                raise ValueError(f"Words {state['words']} contain duplicates.")
            for word in state['words']:
                word_letters = [self.words[letter] if isinstance(letter, str) else letter for letter in word]
                
                if len(word_letters) > 1:
                    for letter1, letter2 in zip(word_letters, word_letters[1:]):
                        letter1.below = letter2
                        letter2.on_top_of = letter1
                
        if 'holding' in state:
            self.holding = state['holding']


    #########################################################
    # Properties
    #########################################################

    @property
    def letter_names(self):
        return sorted(self.letters)

    @property
    def ontable(self):
        letters = set()
        for letter in self.letters:
            letter = Letter(letter, self)
            if not (letter.on_top_of or self.holding==letter):
                letters.add(letter)
        return letters

        #return {letter for letter in self.letters if not (letter.on_top_of or self.holding==letter)}

    @property
    def clear(self):
        letters = set()
        for letter in self.letters:
            letter = Letter(letter, self)
            if not (letter.before or self.holding==letter):
                letters.add(letter)
        return letters

        #return {letter for letter in self.letters if not (letter.before or self.holding==letter)}

    @property
    def action_counts(self):
        return {str(action): Action(action).times_executed for action in self.actions}

    @property
    def successful_action_count(self):
        #print("actions", self.actions)
        total_times_executed = 0
        for action in self.actions:
            action = Action(action)
            times_executed = action.times_executed
            total_times_executed += times_executed
        return total_times_executed

        #    print("action", action, type(action))
        #    print("times_executed", action.times_executed)
        #return sum(action.times_executed for action in self.actions)

    @property
    def available_actions(self):
        applications = []
        for action in self.actions:
            print("action", action)
            action = Action(action)
            print("possible_applications", action.possible_applications())
            for application in action.possible_applications():
                applications.append(application)
        return applications

        #return [application for action in self.actions for application in action.possible_applications()]


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
                sentences.append(f"Currently you are not holding any letter.")
        if 'ontable' in state:
            if len(state['ontable']) == 0:
                sentences.append("No letter is on the table.")
            elif len(state['ontable']) == 1:
                sentences.append(f'Letter {self.ontable.pop()} is on the table.')
            else:
                sentences.append(f'Letter {list_to_natural_language(sorted(self.ontable))} are on the table.')
        if 'words' in state:
            some_letter_stacked = False
            for word in state['words']:
                if len(word) > 1:
                    some_letter_stacked = True
                    relations = [f"{letter2} is on top of {letter1}" for letter1, letter2 in zip(word, word[1:])]
                    sentences.append(f'Letter {list_to_natural_language(relations, two_item_separator=",")}.')
            if not some_letter_stacked:
                sentences.append("No letters are stacked.")
        return " ".join(sentences)

    def describe_goal(self):
        return self.goal_description

    def describe_available_actions(self):
        return f"Your available actions are {list_to_natural_language(self.available_actions)}."

    #########################################################
    # Step function
    #########################################################

    def extract_action(self, agent_response):
        if "<" not in agent_response or ">" not in agent_response:
            raise ValueError("Remember to tag your next action as <next action>.")
        elif agent_response.count('<') > 1 or agent_response.count('<') > 1:
            raise ValueError("Your reply should contain only one set of tags < >.")
        return agent_response[agent_response.rfind("<") + 1 : agent_response.rfind(">")].strip().lower()

    def step(self, agent_response):
        self.step_count += 1
        action_str = self.extract_action(agent_response)
        for action in self.actions.values():
            if action.string_specifies_action(action_str):
                return action.execute_str(action_str)
        raise ValueError(f'"{action_str}" is not a valid action.')



if __name__ == "__main__":
    env = AnagramWorld(6)
    print(env.describe_interface())
    env.set_state((['a', 'b', 'c', 'd', 'e', 'f']))
    for letter in env.letters.values():
        print(letter.name, letter.below, letter.on_top_of)
    env.reset()
    #print(env.step('<add b a>'))
    print(env.describe_state())
    print(env.describe_available_actions())
    print(env.step('<pick up a>'))
    print({letter.name for letter in env.clear})
    print(env.describe_available_actions())
    print(env.describe_state())
    print(env.step('<put down a>'))
    print(env.describe_state())
    print(env.describe_available_actions())
    print(env.describe_state())
    print(env.step('<pick up b>'))
    print(env.describe_state())
    print(env.step('<add b on a>'))
    print({letter.name for letter in env.clear})
    print(env.describe_state())
    print(env.step('<pick up c>'))
    print({letter.name for letter in env.clear})
    print(env.available_actions)
    print(env.describe_state())
    print(env.step('<add c on b>'))
    print({letter.name for letter in env.clear})
    print(env.describe_state())
    print(env.step('<pick up d>'))
    print(env.describe_state())
    print(env.step('<add d on c>'))
    print(env.describe_state())
    print(env.step('<pick up f>'))
    print({letter.name for letter in env.clear})
    print(env.describe_state())
    print(env.step('<add f on e>'))
    print(env.get_state())
    print(env.describe_state())
    print(env.step('<remove f>'))
    print(env.describe_state())
    print(env.describe_state({'words': [env.letter_names]}))
    
