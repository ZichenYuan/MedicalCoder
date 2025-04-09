# Using GPT-3.5 to extract key diagnosis from clinical notes 
from codify import Codify
from agent import Agent, ExtractModel

def extract_key_points(query: str):
    llm = Agent(ai_provider="openai_client", model="gpt-3.5-turbo", max_token=300, response_model=ExtractModel)
    
    system_prompt = f"""
    You are a medical expert that can extract precise diagnosis from clinical notes.
    You will be given clinical notes written by different people for the same patient.
    You must extract the key diagonsis from the notes based on ICD-9 paradigm and give your supporting evidence.
    There might be typos and format errors in clinical notes.
    The response should be a list of diagnosis and a list of evidence. Length of both lists should be equal and at least 5.
    Example:
    ["diagnosis1", "diagnosis2", "diagnosis3", "diagnosis4", "diagnosis5"]
    ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5"]wo
    Do not return any other information.
    """
    context = f"""
    query: {query}
    """
    
    response = llm.inference(context, system_prompt)
    return response


def extract_k_key_points(query: str, k:int):
    llm = Agent(ai_provider="openai_client", model="gpt-3.5-turbo", max_token=300, response_model=ExtractModel)
    
    system_prompt = f"""
    You are a medical expert that can extract precise diagnosis from a clinical query.
    You will be given clinical notes written by different people for the same patient.
    You must extract the key diagonsis from the notes based on ICD-9 paradigm and give your supporting evidence.
    There might be typos and format errors in clinical notes.
    The response should be a list of diagnosis and a list of evidence. Length of both lists should be equal to {k}.
    Example:
    ["diagnosis1", "diagnosis2", "diagnosis3", "diagnosis4", "diagnosis5"...]
    ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5"...]
    Do not return any other information.
    """
    context = f"""
    query: {query}
    """
    
    response = llm.inference(context, system_prompt)
    return response
# clinical_text = """
# Patient has a history of Type 2 Diabetes Mellitus, diagnosed 5 years ago. Currently taking Metformin 500mg BID.
# Reports episodes of hypertension and high blood pressure, recently prescribed Lisinopril 10mg daily.
# Previous hospital visit due to pneumonia last year. No known allergies. Lab results show increased HbA1c levels.
# """
# extracted = extract_k_key_points(clinical_text, 3)
# print(extracted.diagnosis)
# descriptions = ['"Left temporal intraparenchymal hemorrhage"', ' "Expressive aphasia"', ' "Developmental venous anomaly"', ' "Cavernous malformation"', ' "Aneurysm"']
# evidence = ['"CT scan head showed left temporal bleed"', ' "MRI head showed left temporal hemorrhage with mild surrounding edema"', ' "Angiogram showed suggestion of a small venous angioma in the left temporal region"', ' "MRI brain showed acute left temporal intraparenchymal hemorrhage with mild mass effect"', ' "CTA head showed acute left temporal intraparenchymal hemorrhage and curvilinear region of contrast enhancement posterior to the hemorrhage"']
 
# codify = Codify()
# for i in range(len(evidence)):
#     result = codify.get_ranked_top_k_icd_codes_with_evidence(3, descriptions[i],evidence[i])
#     print(f'rag result:{result}')

