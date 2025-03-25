# old version of baseline_gpt.py
# fail beacuse of exceeding maximum token

from codify import Codify
from langchain_openai import ChatOpenAI
from agent import Agent, ExtractModel
from pydantic import BaseModel


def extract_key_points(query: str):
    llm = Agent(ai_provider="openai_client", model="gpt-3.5-turbo", max_token=300, response_model=ExtractModel)
    
    system_prompt = f"""
    You are a medical expert that can extract key points from a clinical query.
    You will be given clinical notes written by different people for the same patient.
    You must extract the key diagonsis from the notes based on ICD-9 paradigm and give your supporting evidence.
    There might be typos and format errors in clinical notes.
    The response should be a list of diagnosis and a list of evidence. Length of both lists should be equal and at least 5.
    Example:
    ["code1", "code2", "code3", "code4", "code5"...]
    ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5"...]
    Do not return any other information.
    """
    context = f"""
    query: {query}
    """
    response = llm.inference(context, system_prompt)
    return response

def baseline_k_predict(query: str, k:int):
    llm = Agent(ai_provider="openai_client", model="gpt-3.5-turbo", max_token=300, response_model=ExtractModel)
    
    system_prompt = f"""
    You are a medical expert that can extract precise diagnosis from clinical notes.
    You will be given clinical notes written by different people for the same patient.
    You must autocode the ICD-9 code from the notes and give your supporting evidence.
    There might be typos and format errors in clinical notes.
    The response should be a list of diagnosis and a list of evidence. Length of both lists should be equal to {k}.
    Example:
    ["code1", "code2", "code3", "code4", "code5"...]
    ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5"...]
    Do not return any other information.
    query: {query}
    """
    context = f"""
    query: {query}
    """

    response = llm.inference(context, system_prompt)
    return response


# codes, evidence = baseline_k_predict(clinical_notes, 5)

# codify = Codify()
# for i in range(len(evidence)):
#     result = codify.get_ranked_top_k_icd_codes(1, codes[i])
#     print(f'baseline result:{result}')

# output
# descriptions = ['"Left temporal intraparenchymal hemorrhage"', ' "Developmental venous anomaly"', ' "Cavernous malformation"']
# evidence = ['Left temporal bleed identified on CT scan and MRI', 'Patient presented with confusion and nonsensical speech', 'Normal MRA and MRV results indicating no aneurysm or vascular malformation']
# descriptions = ['"Left temporal intraparenchymal hemorrhage"', ' "Expressive aphasia"', ' "Developmental venous anomaly"', ' "Cavernous malformation"', ' "Aneurysm"']
# evidence = ['"CT scan head showed left temporal bleed"', ' "MRI head showed left temporal hemorrhage with mild surrounding edema"', ' "Angiogram showed suggestion of a small venous angioma in the left temporal region"', ' "MRI brain showed acute left temporal intraparenchymal hemorrhage with mild mass effect"', ' "CTA head showed acute left temporal intraparenchymal hemorrhage and curvilinear region of contrast enhancement posterior to the hemorrhage"']
 