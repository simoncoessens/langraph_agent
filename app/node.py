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
from pinecone import Pinecone
import os
#from dotenv import load_dotenv

#load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("aiact")

def fetch_context(article_ids):
    all_fetched_texts = []

    for article_id in article_ids:
        # Try to fetch the article normally
        result = index.fetch(ids=[article_id])

        if article_id in result['vectors']:
            # Article found
            all_fetched_texts.append(f"{article_id}:\n{result['vectors'][article_id]['metadata']['text']}")
        else:
            # Article not found, try to fetch subsections
            subsection_ids = [f"{article_id}.{i:03d}" for i in range(1, 100)]
            result = index.fetch(ids=subsection_ids)

            # Keep track of whether any subsections were found
            subsections_found = False

            # Process fetched subsections
            for subsection_id in subsection_ids:
                if subsection_id in result['vectors']:
                    subsections_found = True
                    all_fetched_texts.append(f"{subsection_id}:\n{result['vectors'][subsection_id]['metadata']['text']}")

            if not subsections_found:
                # No subsections found, print a message to the terminal
                print(f"No article found for ID: {article_id}")

    return "\n\n".join(all_fetched_texts)


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
gpt = ChatOpenAI(model="gpt-4o", temperature=0)
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


# Define the function to process the user's response
# def generate_explanations(state: GraphState) -> GraphState:
#     messages = state["messages"]
#     # user response is every even number message
#     user_response = ""
#     for i in range(0, len(messages), 2):
#         user_response += messages[i].content + "\n"
#     return {
#         "explanations": generate_agent_explanations(user_response, explanation_chain)
#     }


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
    user_anwser = state["messages"][-1].content
    general_template = """
You are Anna, the AI assistant here to guide users in answering the **Main Question**: {question}. Your role is to provide clear, supportive assistance while keeping things conversational and concise.

In the last message the user said: {user_answer}

Now if the user is asking for clarification on a topic, read the AI act articles and explain based on that. Keep your response short, friendly, and focus on the user's question or the next steps they should take to complete the questionnaire accurately.
If you reference AI articles, then reference them and put the article name in bold.

If the user is explaining something about his system, confirm that you understand and ask him to give the missing details if needed.

**Summary of Missing Details**: The following parts of the user's description are unclear:
{suggestion}

Please encourage the user to clarify these aspects so they can progress smoothly with the questionnaire. 

**Relevant AI Act Articles**: 
{articles}
""".strip()

    general_prompt = PromptTemplate(
        template=general_template,
        partial_variables={
            "articles": fetch_context(
                article_ids=[
                    "article_003",
                    "recital_rct_12",
                    "recital_rct_97",
                    "recital_rct_100",
                ]
            )
        },
    )
    chain = general_prompt | gpt
    response = chain.invoke({"question": "Describe the AI system that will be target of this compliance analysis", "suggestion": suggestion, 'user_answer': user_anwser})
    return {"messages": response}


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
graph = graph_builder.compile(checkpointer=MemorySaver())
