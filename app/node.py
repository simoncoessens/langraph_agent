from pydantic import BaseModel
from typing import Literal
import os
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from decouple import config



class Q1_SubquestionAnswers(BaseModel):
    q1: Literal[True, False, None] = Field(
        None, description="Is the system a machine-based system?"
    )
    q2: Literal[True, False, None] = Field(
        None, description="May the system exhibit adaptiveness after deployment?"
    )
    q3: Literal[True, False, None] = Field(
        None, description="Is it capable of inferring how to generate outputs?"
    )
    q4: Literal[True, False, None] = Field(
        None, description="Can the system outputs influence environments?"
    )
    q5: Literal[True, False, None] = Field(
        None, description="Is the system designed with varying autonomy levels?"
    )

    def all(self):
        return all([self.q1, self.q2, self.q3, self.q4, self.q5])

    def all_not(self):
        return all([not self.q1, not self.q2, not self.q3, not self.q4, not self.q5])

    def __iter__(self):
        for attr in ["q1", "q2", "q3", "q4", "q5"]:
            yield getattr(self, attr) 


prompt = """
Analyze the user's response to determine if it contains a clear, direct answer to each of the following questions. Only respond with:
- "True" if the response explicitly and directly confirms the answer to the question without needing assumptions.
- "False" if the response explicitly denies the answer.
- "None" if there is no clear and explicit information in the response to answer the question.

Questions:
1. Is the system a machine-based system?
2. May the system exhibit adaptiveness after deployment?
3. Is it capable, for explicit or implicit objectives, of inferring, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions?
4. Can the system outputs influence physical or virtual environments?
5. Is the system designed to operate with varying levels of autonomy?

Be cautious to only answer "True" or "False" where there is explicit, unambiguous information in the user's response, and use "None" if the answer is not directly clear.

Question: What type of system are you developing?
User's response: {response}
""".strip()

prompt_template = PromptTemplate(input_variables=["response"], template=prompt)
api_key = config("OPENAI_API_KEY")
gpt = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
llm_q1 = gpt.with_structured_output(Q1_SubquestionAnswers)
q1_chain = prompt_template | llm_q1


class GraphState(MessagesState):
    subquestion_answers: Q1_SubquestionAnswers = Q1_SubquestionAnswers()


# Define the function to process the user's response
def process_response(state: GraphState) -> GraphState:
    messages = state["messages"]
    # user response is every even number message
    user_response = ""
    for i in range(0, len(messages), 2):
        user_response += messages[i].content + "\n"
    sub_answers_new = q1_chain.invoke({"response": user_response})
    if "subquestion_answers" in state:
        sub_answers_existing = state["subquestion_answers"]
    else:
        sub_answers_existing = Q1_SubquestionAnswers()
    for attr in ["q1", "q2", "q3", "q4", "q5"]:
        if not getattr(sub_answers_existing, attr):
            setattr(sub_answers_existing, attr, getattr(sub_answers_new, attr))
    return {"subquestion_answers": sub_answers_existing}


# Define the function to check if all subquestions are answered
def check_completion(state: GraphState) -> bool:
    answers = state["subquestion_answers"]
    all_not_none = all([answer is not None for answer in answers])
    return all_not_none


# Define the function to inform the user about vague parts
def inform_user(state: GraphState) -> str:
    vague_parts = [
        f"{desc}"
        for desc, answer in zip(
            [
                "Is the system a machine-based system?",
                "May the system exhibit adaptiveness after deployment?",
                "Is it capable of inferring how to generate outputs?",
                "Can the system outputs influence environments?",
                "Is the system designed with varying autonomy levels?",
            ],
            state["subquestion_answers"],
        )
        if answer is None
    ]
    vague_str = "\n".join(vague_parts)
    suggestion = (
        "The following parts of your description are vague or unclear:\n"
        f"{vague_str}\n\n"
        "Please provide more details to clarify these aspects."
    )
    return {"messages": AIMessage(content=suggestion)}


# Initialize the StateGraph
graph_builder = StateGraph(GraphState)

# Add nodes to the graph
graph_builder.add_node("process_response", process_response)
graph_builder.add_node("inform_user", inform_user)

# Define the edges with conditions
graph_builder.add_edge(START, "process_response")
graph_builder.add_conditional_edges(
    "process_response", check_completion, {True: END, False: "inform_user"}
)
graph_builder.add_edge("inform_user", END)
graph = graph_builder.compile()
