# Using GPT-3.5 and GPT-4 as baseline to autocode ICD9 codes

# note: use langchain_openai to avoid exceeding maximum allowed tokens
from codify import Codify
from config import groq_client, openai_client
import os
import openai
from langchain_openai import ChatOpenAI
import time
from utils import truncate_text


# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def baseline_gpt4_k_predict(llm, query: str, k: int, max_retries: int = 3):
    # llm = ChatOpenAI(model_name="gpt-4",
    #                  temperature=0,
    #                  max_tokens=500,
    #                  timeout=None,
    #                  max_retries=3)
    
    # Truncate the query if it's too long
    truncated_query = truncate_text(query)
    
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
    query: {truncated_query}
    """
    
    for attempt in range(max_retries):
        try:
            response = llm.predict(system_prompt)
            decisions = response.strip('/n').split('\n')
            codes = decisions[0].strip('[').strip(']').split(',')
            evidence = decisions[1].strip('[').strip(']').split(',')
            
            # get rid of quotation marks and dots
            cleaned_codes = [code.strip().replace('"', '').replace('.', '') for code in codes]
            return cleaned_codes, evidence
            
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"Rate limit exceeded after {max_retries} attempts. Please wait a minute before trying again.")
                return [], []
            wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
            print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

def baseline_gpt3_k_predict(query: str, k:int):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                      temperature=0,
                        max_tokens=300,
                        timeout=None,
                        max_retries=2)
    
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
    # context = f"""
    # query: {query}
    # """
    
    # Make the LangChain API call
    response = llm.predict(system_prompt)
    decisions = response.strip('/n').split('\n')
    codes = decisions[0].strip('[').strip(']').split(',')
    evidence = decisions[1].strip('[').strip(']').split(',')
    # print(codes)
    # print(evidence)

    # get rid of quotation marks and dots
    cleaned_codes = [code.strip().replace('"', '').replace('.', '') for code in codes]

    return cleaned_codes,evidence


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
 