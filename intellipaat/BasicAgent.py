import google.genai as genai
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import os
from dotenv import load_dotenv

load_dotenv()

# print(os.getenv("GOOGLE_API_KEY"))

gemini_api_key = os.getenv("GOOGLE_API_KEY")


client = genai.Client(api_key=gemini_api_key)

# init LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-lite-latest",
    google_api_key=gemini_api_key,
    temperature=0,
    max_output_tokens=512
)


def calculate(expression):
    return eval(expression)


def average_pill_weight(name):
    pill_weights = {
        "Paracetamol": "500 mg",
        "Aspirin": "250 mg",
        "Ibuprofen": "400 mg"
    }
    return f"The average weight of {name} is {pill_weights.get(name, 'Unknown')}"

known_actions = {
    "calculate": calculate,
    "average_pill_weight": average_pill_weight
}


class Agent:
    messages = []

    def __int__(self):
        self.messages = []  # message history - maintain

    def __call__(self, message):
        self.messages.append(HumanMessage(content=message))
        response = self.execute()
        self.messages.append(AIMessage(content=response))
        return response

    def execute(self):
        response = llm.invoke(self.messages)
        return response.content


action_re = re.compile(r'Action: \s*(\w+):\s*(.*?)(?:\n|$)', re.MULTILINE)


def run_query(prompt, max_turns=10):
    agent = Agent()
    next_prompt = prompt

    for turn in range(max_turns):
        print(f"Turn {turn + 1}: ")
        response = agent(next_prompt)
        print(f"Assistant: \n {response} \n")

        actions = action_re.findall(response[0]["text"])
        print(actions)
        if actions:
            action, action_input = actions[0]
            action_input = action_input.strip()
            print(f"Action: {action}")
            print(f"Action Input: {action_input}")
            print(f"Known Actions: {known_actions}")
            if action not in known_actions:
                raise ValueError(f" Unknown action: {action}")

            observation = known_actions[action](action_input)

            print(f"Observation: {observation} \n")
            next_prompt = f"Observation: {observation}"

        else:
            if "Answer: " in response:
                print(f"Final Answer provided")
                break
            else:
                print(f"No action found and no final answer. Stopping.")
                break


instruction_and_question = """
    You are a helpful assistant following a ReAct reasoning pattern.
    For each step:
    1. Use "Thought": to think step by step about the problem
    2. If you need more information, use 'Action: <action_name>:<input>' on a new line
    3. After each action. I will provide an 'Observation:' with the result
    4. Continue thinking and acting until you can provide a final 'Answer:'

    Available Actions:
        - calculate: perform a python calculation (e.g., calculate: 5 + 3)
        - average_pill_weight: get the average_pill_weight of a specific pill (e.g., average_pill_weight: Aspirin)

        Important : Only perform ONE action at a time, then wait for the observation.

        Question : I have two pills, Aspirin and Paracetamol, what is their combined weight.
    """

run_query(instruction_and_question)
