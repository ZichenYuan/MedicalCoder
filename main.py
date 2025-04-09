import csv
import random
from codify import Codify
from langfuse.decorators import observe
from extract_gpt import gpt4_extract_k_diagnosis
from baseline_gpt import baseline_gpt4_k_predict
from utils import get_codes, calculate_hit_rate, get_random_sample
from multi_agent_workflow import MultiAgentICD9
from langchain_openai import ChatOpenAI

def normalize_code(code):
    return code.replace('.', '').upper()

def code_match(true_code, predicted_code):
    true_main = normalize_code(true_code).split()[0]
    predicted_main = normalize_code(predicted_code).split()[0]
    return true_main == predicted_main


# extract k diagnosis descriptions from clinical notes and extract k codes
def get_pipeline1_result(llm, description, k, codify):
    dia, _ = gpt4_extract_k_diagnosis(llm, description, k)
    print(f'queries:{dia}')
    codes = []
    for j in range(len(dia)):
        result1 = codify.get_ranked_top_k_icd_codes(1, dia[j])
        code = get_codes(result1)
        # print(f'result1:{code}')
        if code:
            codes.append(code[0])
    return codes

# summarize in k sentences and extract k codes
# def get_pipeline2_result(description, k, codify):
#     sentences = summarize_k_sentences(description, k)
#     print(f'sentences:{sentences}')
#     codes = []
#     for j in range(len(sentences)):
#         result1 = codify.get_ranked_top_k_icd_codes(1, sentences[j])
#         code = get_codes(result1)
#         # print(f'result2:{code}')
#         if code:
#             codes.append(code[0])
#     return codes


@observe()
def run_experiment(descriptions, codes_list):

    codify = Codify()
    num_samples = len(descriptions)
    llm = ChatOpenAI(model_name="gpt-4",
                     temperature=0,
                     max_tokens=500,
                     timeout=None,
                     max_retries=3)
    
    for i in range(num_samples):
        description = descriptions[i]

        # extracted_description = extract_key_points(description)
        answer_code_str = codes_list[i]
        answer_code_lst = answer_code_str.split(',')

        # gpt baseline
        baseline_pred_code_ls, _ = baseline_gpt4_k_predict(llm, description, len(answer_code_lst))

        # pipeline 1
        pipeline1_pred_code_ls = get_pipeline1_result(llm,description, len(answer_code_lst), codify)

        #pipeline 2
        multi_agent_icd9 = MultiAgentICD9(llm, 9)
        unformatted_queries = multi_agent_icd9.execute_task(description)
        queries = unformatted_queries.strip('[').strip(']').split(',')
        pipeline2_pred_code_ls = []
        for query in queries:
            result1 = codify.get_ranked_top_k_icd_codes(1, query)
            code = get_codes(result1)
            if code:
                pipeline2_pred_code_ls.append(code[0])


        baseline_hit_rate = calculate_hit_rate(baseline_pred_code_ls, answer_code_lst)
        pipeline1_hit_rate = calculate_hit_rate(pipeline1_pred_code_ls, answer_code_lst)
        pipeline2_hit_rate = calculate_hit_rate(pipeline2_pred_code_ls, answer_code_lst)
        print(f'baseline:{baseline_pred_code_ls}, hit rate:{baseline_hit_rate}')
        print(f'pipeline1:{pipeline1_pred_code_ls}, hit rate:{pipeline1_hit_rate}')
        print(f'pipeline2:{pipeline2_pred_code_ls}, hit rate:{pipeline2_hit_rate}')
        print(f'Answer code:{answer_code_str}')
        

if __name__ == "__main__":
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

    # print(f'descriptions:{len(descriptions[1])}')
    # Clean up descriptions to remove newlines and extra whitespace
    descriptions = [desc.replace('\n', ' ').replace('\r', ' ').strip() for desc in descriptions]
    # Remove multiple spaces
    descriptions = [' '.join(desc.split()) for desc in descriptions]
    run_experiment(descriptions, codes_list)