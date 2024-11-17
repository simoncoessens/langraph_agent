from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
# Import your custom classes and functions
from app.node import Q1_SubquestionAnswers, process_response, check_completion, inform_user, GraphState, graph_builder  
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage 
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Compile the graph
graph = graph_builder.compile(checkpointer=MemorySaver())



config = {"configurable": {"thread_id": 2}}

# Initialize FastAPI app
app = FastAPI()

# Define a data model for incoming requests
class UserInput(BaseModel):
    response: str

# Create an endpoint that processes user input
@app.post("/process")
async def process_input(data: UserInput):
    # Create an initial graph state with the user's input
    
    # Run the graph logic
    state = graph.invoke({"messages": [HumanMessage(data.response)]}, config=config)
    print(state["messages"])


    # Return the response as JSON
    return {"response": state["messages"]}

# If needed, define more routes or logic here
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)