import nltk
#nltk.download('words')
#nltk.download('averaged_perceptron_tagger')

from blocksworld_environment.blocksworld_environment import Action, Done
from tasks.task import Task, get_git_commit_hash, has_uncommitted_changes
import random
import string
from itertools import permutations
from math import factorial

from collections import Counter


with open("/usr/share/dict/words", "r") as f:
    dictionary_words = {line.strip().lower() for line in f}
proper_nltk_words = [word for word in nltk.corpus.words.words() if word.islower() and nltk.pos_tag([word])[0][1] not in ['UH', 'FW', 'SL']]
words = set(dictionary_words).intersection(proper_nltk_words)


def is_english_word(word):
    return word.lower() in words


def is_permutation(string1, string2):
    return sorted(string1.lower()) == sorted(string2.lower())


def permutations_count(string):
    """
    Calculates the number of possible permutations of a string.

    Args:
        string: The input string.

    Returns:
        The number of possible permutations.
    """

    n = len(string)
    char_counts = Counter(string)  # Count occurrences of each character
    denominator = 1
    for count in char_counts.values():
        denominator *= factorial(count)
    return factorial(n) // denominator


class AnagramEnvironment:

    def __init__(self, letters="abc", actions=None):
        self.letters = letters
        self.done = False
        self.step_count = 0
        self.words_generated = []
        self.actions = {str(action): action(self) for action in actions}
        self.seed = None

    @property
    def number_of_blocks(self):
        return len(self.letters)

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


#######################################################
#
# Anagram task
#
#######################################################

class SubmitWord(Action):

    def __init__(self, env):
        super().__init__(env)
        self.correct_permutations = 0
        self.incorrect_permutations = 0

    def string_specifies_action(self, action_str):
        return True

    def execute_str(self, action_str):
        return self.execute(action_str)

    def execute(self, word):
        if word in self.env.words_generated:
            raise ValueError("You have already submitted this word. Can you think of another one?")
        if not is_permutation(word, self.env.letters):
            self.incorrect_permutations += 1
            raise ValueError(f"'{word}' is not a permutation of '{self.env.letters}'")
        self.correct_permutations += 1
        if not is_english_word(word):
            raise ValueError(f"'{word}' is not a valid English word.")
        self.env.words_generated.append(word)
        return "Well done! Can you think of another word?"


    def possible_applications(self):
        return []

    @staticmethod
    def describe():
        return ""


class AnagramTask(Task):

    def set_up(self, letters=None, number_of_blocks=3, seed=None, **kwargs):
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        if not letters:
            letters =  list(random.choice([word for word in words if len(word) == number_of_blocks]))
            random.shuffle(letters)
        if not isinstance(letters, str):
            letters = ''.join(letters)
        self.env = AnagramEnvironment(letters=letters, actions=[Done, SubmitWord])
        self.max_steps = 10

    def initial_instructions(self):
        return f"Your goal is to create an anagram of the letters '{self.env.letters}' that is a valid English word. Submit your answer in tags < >. For example, for the letters 'abc', a valid response would be <cab>. When you can't think of any more anagrams, say <done>."

    def evaluate(self):
        valid_words = set([''.join(word) for word in permutations(self.env.letters) if is_english_word(''.join(word))])
        return {
            'words_generated': self.env.words_generated,
            'number_of_words_generated': len(self.env.words_generated),
            'number_of_correct_permutations_generated': self.env.actions[str(SubmitWord)].correct_permutations,
            'valid_words': valid_words,
            'number_of_valid_words': len(valid_words),
            'regret': len(valid_words) - len(self.env.words_generated),
        }

    def general_stats(self):
        return {
            # agent actions
            'steps': self.env.step_count,
            'amount_of_reasoning_agent': self.llm.amount_of_reasoning_agent_only,
            'amount_of_reasoning_agent_env': self.llm.amount_of_reasoning_agent_env,
            'agent_error': self.agent_error,
            'environment_error': self.environment_error,
            'agent_empty_output': self.llm.empty_outputs,
            'agent_max_tries': self.llm.max_retries,
            'system_message': self.llm.system_message.content,
            # general stats
            'start_time': self.start_time,
            'finish_time': self.finish_time,
            'run_time': self.finish_time - self.start_time,
            'task_seed': self.seed if hasattr(self, 'seed') else "not set",
            # environment info
            'model': self.llm.model_name,
            'task': self.__class__.__name__,
            'number_of_letters': len(self.env.letters),
            'number_of_blocks': len(self.env.letters),
            'letters': self.env.letters,
            'max_steps': self.max_steps,
            # code info
            'git_commit': get_git_commit_hash(),
            'uncommitted_changes': has_uncommitted_changes(),
        }


#######################################################
#
# Generate permutations capabiltity
#
#######################################################

class SubmitPermutation(SubmitWord):

    def execute(self, word):
        if not is_permutation(word, self.env.letters):
            self.incorrect_permutations += 1
            raise ValueError(f"'{word}' is not a permutation of '{self.env.letters}'")
        self.correct_permutations += 1
        self.env.words_generated.append(word)
        return "Well done! So far you've generated:\n" + '\n'.join(self.env.words_generated) + "\nCan you think of another permutation?"


class GeneratePermutationsCapability(AnagramTask):

    def set_up(self, **kwargs):
        super().set_up(**kwargs)
        self.env.actions = {str(Done): Done(self.env),
                            str(SubmitPermutation): SubmitPermutation(self.env)}
        self.max_steps = permutations_count(self.env.letters) + 10

    def initial_instructions(self):
        return f"Your goal is to come up with as many permutations of the letters '{self.env.letters}' as possible. Use tags < > to submit a permutation. For example <cba> submits a permutation of the letters 'abc'. Only submit one permutation with each output. When you can't think of any more permutations, say <done>."

    def evaluate(self):
        optimal_score = permutations_count(self.env.letters)
        return {
            'permutations_generated': self.env.words_generated,
            'number_of_permutations_generated': len(self.env.words_generated),
            'number_of_incorrect_permutations_generated': self.env.actions[str(SubmitPermutation)].incorrect_permutations,
            'max_permutations': optimal_score,
            'regret': optimal_score - len(self.env.words_generated),
            'fraction_generated': len(self.env.words_generated) / optimal_score
        }


#######################################################
#
# Check if English word capabiltity
#
#######################################################

class SubmitAnswer(Action):

    def string_specifies_action(self, action_str):
        return True

    def execute_str(self, action_str):
        return self.execute(action_str)

    def execute(self, word):
        if word not in {'yes', 'no'}:
            raise ValueError('Answer <yes> or <no> please.')
        self.env.done = True
        given_answer = (word == 'yes')
        true_answer = is_english_word(self.env.letters)
        self.results = {
            'answered_correctly': given_answer == true_answer,
            'given_answer': given_answer,
            'true_answer': true_answer,
            'false_negative': true_answer and not given_answer,
            'false_positive': not true_answer and given_answer,
            'word_queried': self.env.letters,
        }
        return "Correct." if self.results['answered_correctly'] else "Incorrect."

    def possible_applications(self):
        return ["<yes>", "<no>"]

    @staticmethod
    def describe():
        return ""


class CheckIfWordCapability(AnagramTask):

    def set_up(self, **kwargs):
        super().set_up(**kwargs)
        self.env.actions = {str(SubmitAnswer): SubmitAnswer(self.env)}
        if 'letters' not in kwargs:
            if random.random() < 0.9:
                self.env.letters = random.choice([word for word in words if len(word) == kwargs.get('number_of_blocks', 3) and word.islower()])
            else:
                self.env.letters = ''.join(random.choice(string.ascii_lowercase) for i in range(kwargs.get('number_of_blocks', 3)))

    def initial_instructions(self):
        return f"Is '{self.env.letters}' a valid English word? Answer <yes> or <no>."

    def evaluate(self):
        return self.env.actions[str(SubmitAnswer)].results


if __name__ == "__main__":
    from mock_agent import MockLLM
    task = AnagramTask(letters="abc") #(letters="tst")
    #task = GeneratePermutationsCapability(letters="abc")
    #task = CheckIfWordCapability() #(letters="abc")
    llm = MockLLM(["<dab>", "<bac>", "<cab>", "<cab>", "<abc>", "<yes>", "<done>"])
    task.run(llm)
    print(task.evaluate())
