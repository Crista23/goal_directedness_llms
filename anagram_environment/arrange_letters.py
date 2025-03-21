from anagram_environment.anagram_environment import AnagramWorld, PickUp, PutDown, Add, Remove, Done, Help
from blocksworld_environment.blocksworld_environment import list_to_natural_language, Action
from tasks.task import Task, get_git_commit_hash, has_uncommitted_changes

import nltk
import time
import itertools

#nltk.download('words') 
from nltk.corpus import words

# Check if a word is valid English
def is_english_word(word):
    return word.lower() in words.words()

def all_combinations(letters):
    return ["".join(perm) for perm in itertools.permutations(letters) if is_english_word("".join(perm))]

class ArrangeLettersTask(Task):

    def set_up(self, max_steps_per_run = None, **kwargs):
        self.env = AnagramWorld(
            **kwargs,
            goal_description="The goal is to build an English word using all the available letters.",
            actions = [PickUp, PutDown, Add, Remove, Done, Help],
        )
        
        self.max_steps = max_steps_per_run or self.env.number_of_letters + 10

    def stop_condition(self):
        built_word =  "".join([letter[0].name for letter in self.env.get_state()["words"]])
        return self.env.step_count > self.max_steps or is_english_word(built_word)

    def evaluate(self):
        built_word =  "".join([letter[0].name for letter in self.env.get_state()["words"]])
        completed = is_english_word(built_word)
        return {
            'completed': completed,
        }

    def general_stats(self):

        return {
            # agent actions
            'steps': self.env.step_count,
            'successful_actions': self.env.successful_action_count,
            'failed_actions': self.env.step_count - self.env.successful_action_count,
            'actions': self.env.action_counts,
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
            'env_seed': self.env.seed,
            # environment info
            'model': self.llm.model_name,
            'task': self.__class__.__name__,
            'number_of_letters': self.env.number_of_letters,
            'letters': self.env.letter_names,
            'max_steps': self.max_steps,
            # code info
            'git_commit': get_git_commit_hash(),
            'uncommitted_changes': has_uncommitted_changes(),
        }


class SubmitWordAction(Action):

    def string_specifies_action(self, action_str):
        return action_str.startswith('words')

    def execute_str(self, action_str):
        words = []

        action_str = action_str[action_str.index("[") + 1 : action_str.index("]")]
        for word in action_str.split(";"):
            words.append(word)

        return self.execute(words)

    def execute(self, words):
        self.times_executed += 1
        if len(words) == 0:
            raise ValueError(f"Parsing error: found no words.")

        for word in words:
            if len(set(word)) != self.env.number_of_letters:
                raise ValueError(f"Word {word} does not use all letters.")
            if not (set(word).issubset(self.env.letters.keys())):
                raise ValueError(f"Word {word} uses letters different from the ones given.")
        
        self.env.set_state(words=words)
        self.env.submitted_word = words[0]
        self.env.done = True
        return f"You submitted words {words}."

    def possible_applications(self):
        words = all_combinations("".join(self.env.letters.keys()))
        return [f"<words {words}>"]

    @staticmethod
    def describe():
        return "Give your final answer in the form <words [word1; word2; word3]>. For example, if there are three letters s, u, and n, you want to return  <words ['sun']>."


class ConstructWordCapability(Task):

    def set_up(self, max_steps_per_run = None, **kwargs):
        self.env = AnagramWorld(
            **kwargs,
            )

        self.env.actions = {str(SubmitWordAction): SubmitWordAction(self.env)}

        seed = kwargs.get('seed', False) or int(time.time())
        self.env.goal_description = f"Your goal is to construct an English word out of the given letters '{self.env.letter_names}'."
        self.max_steps = max_steps_per_run or self.env.number_of_letters + 10

    def initial_instructions(self):
        return f"There are letters {list_to_natural_language(self.env.letter_names)}. {self.env.goal_description} {self.env.describe_interface()}"

    def evaluate(self):
        valid_word = is_english_word(self.env.submitted_word)
        result = {
            'completed': self.env.done,
            'passed': self.env.done and valid_word,
        }
        return result

    def general_stats(self):

        return {
            # agent actions
            'steps': self.env.step_count,
            'successful_actions': self.env.successful_action_count,
            'failed_actions': self.env.step_count - self.env.successful_action_count,
            'actions': self.env.action_counts,
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
            'env_seed': self.env.seed,
            # environment info
            'model': self.llm.model_name,
            'task': self.__class__.__name__,
            'number_of_letters': self.env.number_of_letters,
            'letters': self.env.letter_names,
            'max_steps': self.max_steps,
            # code info
            'git_commit': get_git_commit_hash(),
            'uncommitted_changes': has_uncommitted_changes(),
        }


if __name__ == "__main__":
    from mock_agent import MockLLM
    task = ConstructWordCapability(number_of_letters=3)
    llm = MockLLM("<words ['sun']>")
    task.run(llm)
    print(task.evaluate())
