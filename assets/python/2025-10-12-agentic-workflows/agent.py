from langchain_ollama import ChatOllama

# dataclass next step
from typing import Optional
from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    description: str
    function: callable


@dataclass
class Step:
    input: str
    output: Optional[str] = None
    tool: Tool = None


class Agent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1", temperature=0.1)
        self.tools = []
        self.memory = [Step(input="Initialize agent")]
        self.prompt = "You are a helpful assistant. Use the tools provided to answer the user's questions."

    def add_tool(self, tool: Tool):
        self.tools.append(tool)

    def run(self):
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the agent.")
                break

            self.memory.append(user_input)
            response = self._generate_response(user_input)
            print(f"Agent: {response}")
            next_action = self._decide_next_action()
            self.memory.append(next_action)

    def _decide_next_action(self):
        if self.memory:
            return "continue"
        return "exit"
