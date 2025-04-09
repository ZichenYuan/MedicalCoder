# This is a multi-agent system that uses a series of agents to extract key diagnosis from clinical notes.
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
            You are a medical expert that can diagnose the patient based on given information. Extract specificdiagnosis from the given information. 
            The response should be a list of {k} diagnosis. Do not return any other information.

            The response should be in the following format:
            ["Actinic reticuloid and actinic granuloma", "Anal sphincter tear complicating delivery, not associated with third-degree perineal laceration, unspecified as to episode of care or not applicable", ...]

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
        k = self.k
        task_description = f"""You are a medical expert that can extract key diagnosis from clinical notes written by different people for the same patient.
        You should follow these steps:
        1. Clean the clinical notes
        2. Classify the medical entities
        3. Summarize the medical entities and extract the key diagnosis

        The response should be a list of diagnosis in one sentence. Length of the listshould be equal to {k}."""
        
        plan = self.coordinator_chain.run(task_description)
        cleaned_text = self.cleaner.clean(raw_notes)
        classified_entities = self.classifier.classify(cleaned_text)
        queries = self.summary.summarize(classified_entities)
        
        try:
            formated_queries = queries.strip('/n').strip('[').strip(']').split(',')
            cleaned_queries = [q.strip().replace('"', '').replace('.', '') for q in formated_queries]
            print(f'formated_queries:{cleaned_queries}')
        except:
            # queries = self.summary.summarize(classified_entities)
            print(f'queries:{queries}')
        # remove the first and last character of queries

        # specific_codes = self.specificity.specify(classified_entities)
        # ranked_results = self.reranker.rerank(specific_codes)

        # return formated_queries


        return {
            "plan": plan.strip(),
            "cleaned_text": cleaned_text.strip(),
            "classified_entities": classified_entities.strip(),
            "summary_sentence": cleaned_queries,
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
    print("\n=====================================")
    
    print(f'Answer code:{codes_list[0]}')

if __name__ == "__main__":
    main()
