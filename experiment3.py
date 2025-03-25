import csv
import random
from codify import Codify
from datetime import datetime, timedelta
import time
from langfuse.decorators import observe
from itertools import islice
from extract_gpt import extract_k_key_points
from baseline_gpt import baseline_k_predict
from utils import get_codes, calculate_hit_rate

def normalize_code(code):
    return code.replace('.', '').upper()

def code_match(true_code, predicted_code):
    true_main = normalize_code(true_code).split()[0]
    predicted_main = normalize_code(predicted_code).split()[0]
    return true_main == predicted_main

@observe()
def run_experiment(csv_file_path):
    codify = Codify()
    results = []
    skipped_samples = 0

    descriptions = []
    codes_list = []
    document_metadatas = []
    ids =[]
    # Read the CSV file

        
    descriptions = []
    codes_list = []
    document_metadatas = []
    ids = []

    previous_id = None
    previous_codes = None
    concatenated_description = ""

    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in islice(reader, 100):
            description = row['text'].strip()
            codes = row['dia_code'].strip()
            id = row['subject_id'].strip()
            
            if description and codes:
                if id == previous_id and codes == previous_codes:
                    # Concatenate description for the same id and codes
                    concatenated_description += " " + description
                else:
                    # Store the previous entry before resetting
                    if previous_id is not None and previous_codes is not None:
                        descriptions.append(concatenated_description)
                        codes_list.append(previous_codes)
                        document_metadatas.append({"type": "ICD-9", "code": previous_codes})
                        ids.append(previous_id)
                    
                    # Reset for the new entry
                    concatenated_description = description
                    previous_id = id
                    previous_codes = codes
        
    # Add the last entry after the loop
    if previous_id is not None and previous_codes is not None:
        descriptions.append(concatenated_description)
        codes_list.append(previous_codes)
        document_metadatas.append({"type": "ICD-9", "code": previous_codes})
        ids.append(previous_id)

    length = len(descriptions)
    
    for i in range(length):
        description = descriptions[i]

        pipeline_pred_code_ls = []
        # extracted_description = extract_key_points(description)
        answer_code_str = codes_list[i]
        answer_code_lst = answer_code_str.split(',')

        baseline_response = baseline_k_predict(description, len(answer_code_lst))
        pipeline_response = extract_k_key_points(description, len(answer_code_lst))

        baseline_pred_code_ls, _ = baseline_response
        dia, _ = pipeline_response #diagnosis from extract agent
        print(f'query:{dia}')
        for j in range(len(dia)):
            result1 = codify.get_ranked_top_k_icd_codes(1, dia[j])
            # print(f'rag result:{result1}')
            pipeline_code_pred = get_codes(result1)
            # result2 = codify.get_ranked_top_k_icd_codes_with_evidence(1, dia[j], evi[j])
            # print(f'code_pred:{pipeline_code_pred}')
            # print(f'ragwe result:{result2}')
            pipeline_pred_code_ls.append(pipeline_code_pred[0])
        
        baseline_hit_rate = calculate_hit_rate(baseline_pred_code_ls, answer_code_lst)
        pipeline_hit_rate = calculate_hit_rate(pipeline_pred_code_ls, answer_code_lst)
        print(f'baseline:{baseline_pred_code_ls}, hit rate:{baseline_hit_rate}')
        print(f'pipeline:{pipeline_pred_code_ls}, hit rate:{pipeline_hit_rate}')
        print(f'Answer code:{answer_code_str}')
        

if __name__ == "__main__":
    csv_file_path = "mimic3_full.csv"
    run_experiment(csv_file_path)