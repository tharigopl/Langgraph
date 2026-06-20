from typing import List, Union, Annotated, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, END, START

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_content: str


@tool
def update(content: str) -> str:
    """Updates the document with the provided content. Call this when the user wants to write or modify the document."""
    #return f'Document updated with the content successfully! The current content is:\n {content}'
    return content  # return the new content — custom tool node reads this to update state


@tool
def save(
    filename: str,
    state: Annotated[AgentState, InjectedState]  # InjectedState reads document_content from state
) -> str:
    """Saves the current document to disk as a text file. Call this when the user wants to save or finish.
       Args: filename for the text file
    """
    document_content = state["document_content"]  # read from state, not global

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    except Exception as e:
        return f'Error saving the document: {str(e)}'


tools = [update, save]
tools_by_name = {t.name: t for t in tools}

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, use the 'save' tool.
    - Always show the current document state after modifications.

    The current document content is: {state["document_content"]}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f'\n🤖 AI: {response.content}')
    if response.tool_calls:
        print(f'USING TOOLS: {[tc["name"] for tc in response.tool_calls]}')

    return {"messages": [user_message, response]}


def custom_tool_node(state: AgentState) -> AgentState:
    """Custom tool node that runs tools AND updates document_content in state."""
    last_message = state["messages"][-1]
    tool_results = []
    new_document_content = state["document_content"]  # start with current content

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "update":
            # run the tool
            result = tools_by_name[tool_name].invoke(tool_args)
            new_document_content = result  # update state with new content
            display_result = f"Document updated successfully! Current content:\n{new_document_content}"

        elif tool_name == "save":
            # InjectedState is handled automatically when invoked via tool node
            # but since we're custom, pass state manually
            result = tools_by_name[tool_name].invoke({**tool_args, "state": state})
            display_result = result

        else:
            result = tools_by_name[tool_name].invoke(tool_args)
            display_result = result

        tool_results.append(ToolMessage(
            content=display_result,
            tool_call_id=tool_call["id"]
        ))

    return {
        "messages": tool_results,
        "document_content": new_document_content  # write updated content back to state
    }


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]
    if not messages:
        return "continue"

    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
                "saved" in message.content.lower() and
                "document" in message.content.lower()):
            return "end"

    return "continue"


def agent_should_continue(state: AgentState) -> str:
    """Route agent to tools if tool calls exist, otherwise end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    # if messages only has the hardcoded first exchange, keep going
    if len(state["messages"]) <= 2:  # only HumanMessage + first AIMessage
        return "agent"  # loop back so user can type
    return "end"


def print_messages(messages):
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f'\nTool Result: {message.content}')


def run_document_agent():
    print("\n============== Drafter ==============")
    state = {"messages": [], "document_content": ""}  # initialise document_content

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n============== Drafter Finished ==============")


if __name__ == '__main__':
    print('Drafter')

    graph = StateGraph(AgentState)

    graph.add_node("agent", our_agent)
    graph.add_node("tools", custom_tool_node)  # custom node instead of ToolNode

    graph.set_entry_point("agent")

    # conditional from agent — only go to tools if there are tool calls
    graph.add_conditional_edges(
        "agent",
        agent_should_continue,
        {
            "tools": "tools",
            "end": END,
            "agent": "agent"
        }
    )
    #graph.add_edge("agent", "tools")

    graph.add_conditional_edges(
        "tools",
        should_continue,
        {
            "continue": "agent",
            "end": END,
        },
    )

    app = graph.compile()

    run_document_agent()
