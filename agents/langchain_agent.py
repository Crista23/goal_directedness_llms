import dotenv
import time
import random

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


dotenv.load_dotenv()


class LangchainAgent():

   def __init__(self, model_name, extra_prompt="", output_file=None, max_retries=20):
      self.model_name = model_name
      self.output_file = output_file
      self.max_retries = max_retries
      self.empty_outputs = 0
      self.history = ChatMessageHistory()
      self.config = {"configurable": {"session_id": 0}}
      self.system_message = SystemMessage("You are an agent inhabiting an interactive environment, trying to solve the task you're given. You can only specify one action per output. The action should be identified by tags < >. You can reason step-by-step before specifying your action." + (('\n\n' + extra_prompt) if extra_prompt else ""))
      self.output_file.flush() if self.output_file else None
      self.prompt = ChatPromptTemplate.from_messages(
        [
                self.system_message,
                MessagesPlaceholder(variable_name="messages"),
        ])
      if "gemini" in model_name:  # Google Gemini
         self.model=ChatGoogleGenerativeAI(model=self.model_name)
      elif "claude" in model_name:
         self.model = ChatAnthropic(model=model_name)
      elif "o1-mini" in model_name:
         # o1-mini doesn't allow a system message
         self.system_message.content = None
         self.prompt = ChatPromptTemplate.from_messages(
             [
                MessagesPlaceholder(variable_name="messages"),
             ])
         # o1-mini requires temperature=1
         self.model=ChatOpenAI(model=self.model_name, temperature=1)
      else:  # OpenAI Model
         self.model=ChatOpenAI(model=self.model_name)

      self.with_message_history = RunnableWithMessageHistory(self.prompt | self.model,
                                                             lambda x: self.history)

      # statistics
      self.amount_of_reasoning_agent_only = 0
      self.amount_of_reasoning_agent_env = 0

      print(self.system_message.pretty_repr(), file=self.output_file)


   def __call__(self, environment_response):
      print(HumanMessage(environment_response).pretty_repr(), file=self.output_file)
      self.output_file.flush() if self.output_file else None
      agent_response = None #self.with_message_history.invoke(environment_response, self.config)
      number_of_tries = 0
      while number_of_tries < self.max_retries:
         try:
            agent_response = self.with_message_history.invoke(environment_response, self.config)
            if agent_response.content:
               break  # API call worked, let's proceed
            else:
               print(f"Warning, received empty reply from agent API for {self.model_name}.", end='')
         except Exception as e:
            print(f"Warning, caught api exception {str(e)} for model {self.model_name}.", end='')
         number_of_tries += 1
         delay = 20 * number_of_tries + random.randint(0, 120)
         print(f" Wating {delay} seconds before retrying.")
         time.sleep(delay)
      else:
         print(f"Agent {self.model_name} exceeded API max retries")
         raise ValueError("API kept giving empty responses.")
      self.empty_outputs += number_of_tries
      print(agent_response.pretty_repr(), file=self.output_file)
      self.output_file.flush() if self.output_file else None
      self.amount_of_reasoning_agent_only += len(agent_response.content)
      self.amount_of_reasoning_agent_env += len(environment_response) + len(agent_response.content)
      return agent_response.content


if __name__ == '__main__':
   agent = LangchainAgent("gpt-3.5-turbo")
   agent('hello')
   print(agent.prompt)
   print(agent.history)
