from codify import Codify
from config import groq_client, openai_client
import os
import openai
from langchain_openai import ChatOpenAI
from agent import Agent
from pydantic import BaseModel
from typing import List

# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def extract_k_key_points(query: str, k:int):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                      temperature=0,
                        max_tokens=300,
                        timeout=None,
                        max_retries=2)
    
    system_prompt = f"""
    You are a medical expert that can extract key points from a clinical query.
    You will be given clinical notes written by different people for the same patient.
    You must extract the key diagonsis from the notes based on ICD-9 paradigm and give your supporting evidence.
    There might be typos and format errors in clinical notes.
    The response should be a list of diagnosis and a list of evidence. Length of both lists should be equal to {k}.
    Example:
    ["diagnosis1", "diagnosis2", "diagnosis3", "diagnosis4", "diagnosis5"...]
    ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5"...]
    Do not return any other information.

    query: {query}
    """
    # context = f"""
    # query: {query}
    # """
    
    # Make the LangChain API call
    response = llm.predict(system_prompt)
    decisions = response.strip('/n').split('\n')
    diagonosis = decisions[0].strip('[').strip(']').split(',')
    evidence = decisions[1].strip('[').strip(']').split(',')
    return diagonosis,evidence


