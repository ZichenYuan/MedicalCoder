# defines extract agents
from codify import Codify
from config import groq_client, openai_client
import os
import openai
from langchain_openai import ChatOpenAI
from utils import truncate_text
import time
from typing import List
from pydantic import BaseModel


# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def gpt3_extract_k_diagnosis(query: str, k: int, max_retries: int = 3):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     temperature=0,
                     max_tokens=500,
                     timeout=None,
                     max_retries=3) 
    
    system_prompt = f"""
    You are a medical expert that can extract key diagnosis from a clinical query.
    You will be given clinical notes written by different people for the same patient.
    You must extract the diagonsis descriptions from the notes based on ICD-9 paradigm and give your supporting evidence.
    There might be typos and format errors in clinical notes.
    The response should be a list of diagnosis and a list of evidence. Length of both lists should be equal to {k}.
    Example:
    ["diagnosis1", "diagnosis2", "diagnosis3", "diagnosis4", "diagnosis5"...]
    ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5"...]
    Do not return any other information.

    query: {query}
    """

    response = llm.predict(system_prompt)
    decisions = response.strip('/n').split('\n')
    diagonosis = decisions[0].strip('[').strip(']').split(',')
    evidence = decisions[1].strip('[').strip(']').split(',')
    return diagonosis, evidence


def gpt3_extract_k_sentences(query: str, k: int, max_retries: int = 3):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     temperature=0,
                     max_tokens=500,
                     timeout=None,
                     max_retries=3)
    
    system_prompt = f"""
    You are a medical expert that can extract diagnosis descriptions from a clinical query.
    You will be given clinical notes written by different people for the same patient.
    You must extract the diagonsis descriptions from the notes based on ICD-9 paradigm and give your supporting evidence.
    There might be typos and format errors in clinical notes.
    The response should be a list of sentences and a list of evidence. Length of both lists should be equal to {k}.
    Example:
    ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"...]
    ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5"...]
    Do not return any other information.

    query: {query}
    """

    response = llm.predict(system_prompt)
    decisions = response.strip('/n').split('\n')
    diagonosis = decisions[0].strip('[').strip(']').split(',')
    evidence = decisions[1].strip('[').strip(']').split(',')
    return diagonosis, evidence
    


def gpt4_extract_k_diagnosis(llm, query: str, k: int, max_retries: int = 3):
    # llm = ChatOpenAI(model_name="gpt-4",
    #                  temperature=0,
    #                  max_tokens=500,
    #                  timeout=None,
    #                  max_retries=3)
    
    # Truncate the query if it's too long
    truncated_query = truncate_text(query)
    
    system_prompt = f"""
    You are a medical expert that can extract key diagnosis from clinical notes written by different people for the same patient.
    You must extract {k} diagonsis with as much details as possible from the notes based on ICD-9 paradigm and give your supporting evidence.
    There might be typos and format errors in clinical notes.
    The response should be a list of diagnosis in one sentence and a list of evidence. Length of both lists should be equal to {k}.
    Example:
    ["Tuberculosis of bronchus, bacteriological or histological examination not done", "Herpes zoster with other nervous system complications", "diagnosis3", "diagnosis4", "diagnosis5"...]
    ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5"...]
    Do not return any other information.

    query: {truncated_query}
    """
    
    for attempt in range(max_retries):
        try:
            response = llm.predict(system_prompt)
            decisions = response.strip('/n').split('\n')
            diagonosis = decisions[0].strip('[').strip(']').split(',')
            evidence = decisions[1].strip('[').strip(']').split(',')
            return diagonosis, evidence
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"Rate limit exceeded after {max_retries} attempts. Please wait a minute before trying again.")
                return [], []
            wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
            print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

def gpt4_summarize_key_points(query: str, max_retries: int = 3):
    llm = ChatOpenAI(model_name="gpt-4",
                     temperature=0,
                     max_tokens=500,
                     timeout=None,
                     max_retries=3)
    
    # Truncate the query if it's too long
    truncated_query = truncate_text(query)
    
    system_prompt = f"""
    You are a medical expert that can extract key points from a clinical query.
    You will be given clinical notes written by different people for the same patient.
    You must summarize the key points from the notes based on ICD-9 paradigm in one short paragraph.
    Fix the typos, abbreviations, and format errors in clinical notes.
    Do not return any other information.

    query: {truncated_query}
    """

    for attempt in range(max_retries):
        try:
            response = llm.predict(system_prompt)
            return response
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"Rate limit exceeded after {max_retries} attempts. Please wait a minute before trying again.")
                return ""
            wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
            print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

# def summarize_k_sentences(query: str, k: int, max_retries: int = 3):
#     llm = ChatOpenAI(model_name="gpt-4",
#                      temperature=0,
#                      max_tokens=500,
#                      timeout=None,
#                      max_retries=3)
    
#     # Truncate the query if it's too long
#     truncated_query = truncate_text(query)
    
#     system_prompt = f"""
#     You are a medical expert that can summarize key points from a clinical query.
#     You will be given clinical notes written by different people for the same patient.
#     You must summarize the key points from the notes based on ICD-9 paradigm in {k} sentences.
#     Fix the typos, abbreviations, and format errors in clinical notes.
#     Do not return any other information.
#     The response should be a list of sentences separated by spaces. Length of the list should be equal to {k}.
#     Example:
#     ["sentence1" "sentence2" "sentence3"  "sentence4"  "sentence5"...]

#     query: {truncated_query}
#     """

#     for attempt in range(max_retries):
#         try:
#             response = llm.predict(system_prompt)
#             print(f'response:{response}')
            
#             # Parse the response into our Pydantic model
#             response_text = response.strip()
#             # Remove the outer brackets and split by quotes
#             sentences = [s.strip().strip('"') for s in response_text.strip('[]').split('" "')]
            
#             # Create and validate the response model
#             summary_response = SummaryResponse(sentences=sentences)
            
#             # Ensure we have exactly k sentences
#             if len(summary_response.sentences) != k:
#                 print(f"Warning: Expected {k} sentences but got {len(summary_response.sentences)}")
            
#             return summary_response.sentences
            
#         except openai.RateLimitError as e:
#             if attempt == max_retries - 1:
#                 print(f"Rate limit exceeded after {max_retries} attempts. Please wait a minute before trying again.")
#                 return []
#             wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
#             print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
#             time.sleep(wait_time)
#         except Exception as e:
#             print(f"Error processing response: {e}")
#             return []

    
# #testing
# if __name__ == "__main__":
#     import pickle
#     sample_file = "random_samples.pkl"
#     with open(sample_file, 'rb') as f:
#             descriptions, codes_list, document_metadatas, ids = pickle.load(f)
    
#     test_description = descriptions[0]
#     # summary = summarize_key_points(test_description)
#     # print(summary)
#     codify = Codify()
#     # result = codify.get_ranked_top_k_icd_codes(10, summary)
#     result = summarize_k_sentences(test_description, 8)
#     print(result)
#     for i in result:
#         code = codify.get_ranked_top_k_icd_codes(3, i)
#         print(code)
#     print(codes_list)
