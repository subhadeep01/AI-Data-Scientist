# # =============================
# # Multi-Agent Data Scientist System
# # Using LangGraph + LangChain
# # =============================

import pandas as pd
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from graph import build_graph

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# =============================
# STATE
# =============================

class DSState(TypedDict):
    data: pd.DataFrame
    target_column: str
    processed_data: pd.DataFrame
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    best_model: Any
    model_name: str
    metrics: Dict
    report: str

# =============================
# LLM
# =============================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=api_key
)

# =============================
# LOAD YOUR OWN DATASET HERE
# =============================

df = pd.read_csv("heart.csv")   
target_column = "HeartDisease"            

# =============================
# RUN GRAPH
# =============================

graph = build_graph(llm, DSState)

initial_state = {
    "data": df,
    "target_column": target_column,
    "processed_data": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "best_model": None,
    "model_name": "",
    "metrics": {},
    "report": ""
}

result = graph.invoke(initial_state)

print("Best Model:", result["model_name"])
print("Metrics:", result["metrics"])
print("\nReport:\n", result["report"])