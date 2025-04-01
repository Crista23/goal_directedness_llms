from blocksworld_environment.blocksworld_environment import BlocksWorld
import datetime
import subprocess
import traceback


def get_git_commit_hash():
  """Returns the current Git commit hash."""
  try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    return commit_hash
  except subprocess.CalledProcessError:
    return "Not a git repository"


def has_uncommitted_changes():
  """Checks if there are any uncommitted changes in the repository."""
  try:
    output = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8')
    return bool(output.strip())  # True if there are changes, False otherwise
  except subprocess.CalledProcessError:
    return False  # Not a git repository


class Task():
    """
    Abstract task class to inherit, based on the philosophy that a task consists of:
    * An environment set up, managed by set_up()
    * A stopping condition, managed by stop_condition()
    * An evaluation, managed by evaluate()

    It also possible to specify step-specific functions by overriding
    agent_response_modification() and/or environment_response_modification().
    """

    def __init__(self, env, **kwargs):
        """Initial setup. Usually no need to override."""
        self.env = env
        self.env.done = False
        self.max_steps = kwargs.get('max_steps_per_run', 20)  # often overriden by set_up()
        self.output_file = kwargs.get('output_file', None)
        self.agent_error = False
        self.environment_error = False
        self.preceding_tasks = kwargs.get('preceding_tasks', [])
        self.task = kwargs.get('task', None)
        self.preceding_results = kwargs.get('preceding_results', {})
        self.result_queues = kwargs.get('result_queues', None)
        self.set_up(**kwargs)

    def set_up(self, **kwargs):
        """
        Override this for task specific environment setup, etc.
        Don't forget to seed for any randomness sources you use, typically:
        ## set the random seed ##
        if env.seed is not None:
            self.seed = env.seed
            random.seed(self.seed)
            np.random.seed(self.seed)
        """
        pass

    def initial_instructions(self):
        """Initial instructions given to the agent. Override if needed."""
        same_env = "You are still in the same environment. " if self.preceding_tasks else ""
        interface = "" if self.preceding_tasks and self.preceding_tasks[-1]==self.task else self.env.describe_interface()
        return f"{same_env}{self.env.goal_description} {interface}\n\n{self.env.describe_state()}"

    def stop_condition(self):
        """Override this to set a task specific stopping condition"""
        if self.env.step_count > self.max_steps:
          print(f"Warning: {self.llm.model_name} exceeded max_steps {self.max_steps} on task {self.__class__.__name__}")
        return self.env.step_count > self.max_steps or self.env.done

    def agent_response_modification(self, agent_response):
        """Override to apply a function to the agent's response at every
        environment step."""
        return agent_response

    def environment_response_modification(self, environment_response):
        """Override to apply a function to the environment's response at every step."""
        return environment_response

    def run(self, llm):
        """Manages agent-environment interaction. Usually no need to override."""
        self.llm = llm
        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        self.preceding_steps = self.env.step_count
        self.preceding_questions = self.env.total_questions_asked
        self.preceding_reasoning_agent = self.llm.amount_of_reasoning_agent_only
        self.preceding_reasoning_agent_env = self.llm.amount_of_reasoning_agent_env
        environment_response = self.initial_instructions()
        while not self.stop_condition():
            try:
                agent_response = llm(environment_response)
            except Exception:
                self.agent_error = True
                print(f"\nAGENT ERROR, BREAKING EXECUTION. EXCEPTION TRACE:\n{traceback.format_exc()}", file=self.output_file)
                break

            agent_response = self.agent_response_modification(agent_response)

            try:
                environment_response = self.env.step(agent_response)
            except ValueError as e:
                environment_response = str(e)
            except Exception as e:
                self.environment_error = True
                print(f"\nENVIRONMENT ERROR, BREAKING EXECUTION. EXCEPTION TRACE:\n{traceback.format_exc()}", file=self.output_file)

            environment_response = self.environment_response_modification(environment_response)
        self.finish_time = datetime.datetime.now(datetime.timezone.utc)
        result = self.evaluate()
        result.update(self.general_stats())
        self.preceding_results[self.task].append(result)
        print("=================================== Results ====================================", file=self.output_file)
        for key, value in result.items():
            print(f"{key}: {value}", file=self.output_file)
        if self.output_file:
            self.output_file.flush()
        if self.result_queues:
            print(f"Finished {self.llm.model_name} on {self.env.number_of_blocks} blocks and seed {self.env.seed}.")
            self.result_queues[self.task].put(result)
        else:
            return result

    def evaluate(self):
        """
        Override and return a dictionary with task specific stats, typically
        {'completed': bool, 'regret': float} and any others that seem relevant.
        This function is executed at the end of run().
        """
        pass

    def general_stats(self):
        """Usually no need to override, use evaluate to compute task specific stats"""
        return {
            # agent actions
            'steps': self.env.step_count - self.preceding_steps,
            'preceding_steps': self.preceding_steps,
            'total_steps': self.env.step_count,
            'successful_actions': self.env.successful_action_count,
            'failed_actions': self.env.step_count - self.env.successful_action_count,
            'questions_asked': self.env.total_questions_asked - self.preceding_questions,
            'questions_asked_total': self.env.total_questions_asked,
            'question_blocks': self.env.questions,
            'actions': self.env.action_counts,
            'amount_of_reasoning_agent': self.llm.amount_of_reasoning_agent_only - self.preceding_reasoning_agent,
            'amount_of_reasoning_agent_env': self.llm.amount_of_reasoning_agent_env - self.preceding_reasoning_agent_env,
            'total_reasoning_start': self.preceding_reasoning_agent_env,
            'total_reasoning_end': self.llm.amount_of_reasoning_agent_env,
            'agent_error': self.agent_error,
            'environment_error': self.environment_error,
            'agent_empty_output': self.llm.empty_outputs,
            'agent_max_tries': self.llm.max_retries,
            'system_message': self.llm.system_message.content,
            # environment state
            'most_blocks_in_a_tower': self.env.most_blocks_in_a_tower,
            'height_of_highest_tower': self.env.height_of_highest_tower,
            'tower_heights' : [block.total_height for block in self.env.clear],
            'number_of_towers': len(self.env.ontable),
            # general stats
            'preceding_tasks': len(self.preceding_tasks),
            'start_time': self.start_time,
            'finish_time': self.finish_time,
            'run_time': self.finish_time - self.start_time,
            'task_seed': self.seed if hasattr(self, 'seed') else "not set",
            'env_seed': self.env.seed,
            # environment info
            'block_heights': self.env.block_heights,
            'model': self.llm.model_name,
            'task': self.__class__.__name__,
            'number_of_blocks': self.env.number_of_blocks,
            'max_steps': self.max_steps,
            'measuring_noise': self.env.noise,
            'height_dist': self.env.block_height_distribution,
            # code info
            'git_commit': get_git_commit_hash(),
            'uncommitted_changes': has_uncommitted_changes(),
        }
