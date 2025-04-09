# This is a multi-agent system that uses a series of agents to assign ICD-9-CM codes to a clinical note.

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils import get_random_sample
import openai
import os
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from utils import truncate_text

# Set OpenAI API key
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


class CleanerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["raw_notes"],
            template="""
            You are a medical expert that summarize clinical notes.
            You will be given clinical notes written by different people for the same patient.
            
            Summarize the medically relevant information.  Omit formats, repetitive information, and  irrelevant narrative details.

            Raw Clinical Notes: {raw_notes}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def clean(self, cleaned_text):
        cleaned_text = truncate_text(cleaned_text)
        return self.chain.run(cleaned_text)

class ClassificationAgent:
    def __init__(self, llm, k):
        self.llm = llm
        self.k = k
        self.prompt = PromptTemplate(
            input_variables=["extracted_entities"],
            template="""
            You are a medical expert that can extract and group the medical information into chapters based on ICD-9-CM paradigm.
            The response should be a list of descriptions with the chapter, subcategory(if any), and specific traits(if any), with length {k}.
            The response should be in the following format:
            {{
                1. Chapter: Diseases of the respiratory system
                Subcategory: Pneumonia
                Specific traits: MRSA pneumonia, multifocal pneumonia with opacities in the right upper lobe, left lower lobe, and right lower lobe, moderate left pleural effusion, trace right pleural effusion, intubation and mechanical ventilation, discordant breathing.

                2. Chapter: Diseases of the circulatory system
                Subcategory: Other forms of heart disease
                Specific traits: Atrial/ventricular ectopy, severely depressed left ventricular systolic function, frequent PVCs, stable blood pressure.

                ...
            }}

            Summarized Medical Information: {extracted_entities}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def classify(self, extracted_entities):
        formatted_input = {
            "extracted_entities": extracted_entities,
            "k": self.k
        }
        return self.chain.run(formatted_input)

class SummaryAgent:
    def __init__(self, llm, k):
        self.llm = llm
        self.k = k
        self.prompt = PromptTemplate(
            input_variables=["classified_entities"],
            template="""
            You are a medical expert that can diagnose the patient based on given information. Extract diagnosis from the given information based on ICD-9-CM coding paradigm. 
            The response should be a list of {k} diagnosis. Do not return any other information.

            The response should be in the following format:
            {{
                ["Actinic reticuloid and actinic granuloma", "Anal sphincter tear complicating delivery, not associated with third-degree perineal laceration, unspecified as to episode of care or not applicable", ...]
            }}

            Classified Entities: {classified_entities}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def summarize(self, classified_entities):
        formatted_input = {
            "classified_entities": classified_entities,
            "k": self.k
        }
        return self.chain.run(formatted_input)

# class SpecificityAgent:
#     def __init__(self, llm):
#         self.llm = llm
#         self.prompt = PromptTemplate(
#             input_variables=["classified_entities"],
#             template="""
#             Determine the most specific ICD-9-CM codes in each entity. Output all the relevant codes in a list.
#             Classified Entities: {classified_entities}
#             """
#         )
#         self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

#     def specify(self, classified_entities):
#         return self.chain.run(classified_entities)

# class RerankingAgent:
#     def __init__(self, llm, k=5):
#         self.llm = llm
#         self.k = k
#         self.prompt = PromptTemplate(
#             input_variables=["codes"],
#             template="""
#             ICD-9-CM Codes: {codes}
#             You are a medical coding expert that can rerank ICD-9-CM diagnosis codes based on their relevance to a query and a list of ICD-9-CM references.
#             You must return only the top {k} ICD-9-CM codes.
#             You must return the passage that lead to the most relevant ICD-9-CM codes from the query.
#             Do not include your resoning in your response, just return the ICD-9-CM code and the content of the ICD-9-CM reference.
#             The response should be in the following format:
#             {{
#                 "codes": ["41001", "5589", ...],
#                 "descriptions": ["Acute myocardial infarction", "Infectious gastroenteritis and colitis, unspecified", ...],
#                 "keywords": ["passage1", "passage2", "passage3"]
#             }}
#             Do not return any other information.
#             """
#         )
#         self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

#     def rerank(self, codes):
#         # Format the input with both codes and k
#         formatted_input = {
#             "codes": codes,
#             "k": self.k
#         }
#         return self.chain.run(formatted_input)

# class ValidationAgent:
#     def __init__(self, llm):
#         self.llm = llm
#         self.prompt = PromptTemplate(
#             input_variables=["final_codes"],
#             template="""
#             Final ICD-9-CM Codes: {final_codes}

#             Validate these codes based on ICD-9-CM coding standards. Provide a rationale for each code.
#             """
#         )
#         self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

#     def validate(self, final_codes):
#         return self.chain.run(final_codes)

class MultiAgentICD9:
    def __init__(self, llm, k=5):
        self.llm = llm
        self.k = k
        self.cleaner = CleanerAgent(llm)
        self.classifier = ClassificationAgent(llm, k)
        self.summary = SummaryAgent(llm, k) 
        # self.specificity = SpecificityAgent(llm)
        # self.reranker = RerankingAgent(llm, k)
        

        self.coordinator_prompt = PromptTemplate(
            input_variables=["task"],
            template="""
            Task: {task}

            """
        )
        self.coordinator_chain = LLMChain(llm=self.llm, prompt=self.coordinator_prompt)

    def execute_task(self, raw_notes):
        task_description = "Assign ICD-9-CM codes to this unstructured patient record."
        plan = self.coordinator_chain.run(task_description)

        cleaned_text = self.cleaner.clean(raw_notes)
        classified_entities = self.classifier.classify(cleaned_text)
        queries = self.summary.summarize(classified_entities)
        # specific_codes = self.specificity.specify(classified_entities)
        # ranked_results = self.reranker.rerank(specific_codes)

        print(f'queries:{type(queries)}')




        return {
            "plan": plan.strip(),
            "cleaned_text": cleaned_text.strip(),
            "classified_entities": classified_entities.strip(),
            "summary_sentence": queries.strip(),
            # "specific_codes": specific_codes.strip(),
            # "ranked_results": ranked_results.strip()
        }
        

def main():
    import pickle

    csv_file_path = "mimic3_full.csv"
    # Try to load existing samples from file
    sample_file = "random_samples.pkl"
    try:
        with open(sample_file, 'rb') as f:
            descriptions, codes_list, document_metadatas, ids = pickle.load(f)
            print("Loaded existing random samples")
    except:
        # If file doesn't exist or can't be read, generate new samples
        print("Generating new random samples...")
        descriptions, codes_list, document_metadatas, ids = get_random_sample(csv_file_path, 5)
        
        # Save the samples
        with open(sample_file, 'wb') as f:
            pickle.dump((descriptions, codes_list, document_metadatas, ids), f)
            print("Saved random samples to file")

    # Example Usage
    llm = ChatOpenAI(model_name="gpt-4",
                     temperature=0,
                     max_tokens=500,
                     timeout=None) 
    multi_agent_icd9 = MultiAgentICD9(llm, 9)

    clinical_notes = descriptions[0]

    result = multi_agent_icd9.execute_task(clinical_notes)
    print("\n=== Multi-Agent ICD-9 Coding Results ===")
    print("\nPlanning:")
    print(result["plan"])
    print("\nCleaned Clinical Text:")
    print(result["cleaned_text"]) 
    print("\nClassified Medical Entities:")
    print(result["classified_entities"])
    print("\nSummary Sentence:")
    print(result["summary_sentence"])
    # print("\nSpecific ICD-9 Codes:")
    # print(result["specific_codes"])
    # print("\nRanked Results:")
    # print(result["ranked_results"])
    print("\n=====================================")
    
    print(f'Answer code:{codes_list[0]}')

if __name__ == "__main__":
    main()
