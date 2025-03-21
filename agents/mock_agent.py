from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class MockLLM():
    """This agent is convenient for testing purposes.

    You manually give it a list of strings, which define it's response at each step
    (regardless of what the environment says).

    For example, responses can be ["<pick up a>", "<stack a on b>", "<done>"],
    and then the agent will execute that plan regardless what happens.

    Once it runs of out responses it throws an exception, which stops the task.
    """

    def __init__(self, responses, output_file=None):
        self.model_name = "mock_agent"
        self.output_file = output_file
        self.max_retries = 5
        self.empty_outputs = 0
        self.responses = responses
        # unused, just to match langchain_agent
        self.amount_of_reasoning_agent_only = 0
        self.amount_of_reasoning_agent_env = 0
        self.system_message = SystemMessage("")

    def __call__(self, environment_response):
        print(HumanMessage(environment_response).pretty_repr(), file=self.output_file)
        agent_response = self.responses.pop(0)
        print(AIMessage(agent_response).pretty_repr(), file=self.output_file)
        return agent_response
