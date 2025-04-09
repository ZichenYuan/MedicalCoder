import json
import re
import time
import tiktoken
import random
import csv  

def fix_malformed_json(json_str):
    # Fix list with key-value pair → convert to dict
    json_str = re.sub(r'\[\s*"(\w+)"\s*:\s*(\d+)\s*\]', r'{"\1": \2}', json_str)

    # You can add more rules here to fix common issues
    return json_str

data1 = {
    'id': 'chatcmpl-7a5a871e-e125-418c-9b05-e3d17eeacc0a',
    'choices': [{
        'finish_reason': 'stop',
        'index': 0,
        'logprobs': None,
        'message': {
            'content': '{\n  "codes": ["09322", "3963", "3962"],\n  "descriptions": ["Syphilitic endocarditis of aortic valve", "Mitral valve insufficiency and aortic valve insufficiency", "Mitral valve insufficiency and aortic valve stenosis"],\n  "keywords": ["974", "4612", "4611"]\n}',
            'refusal': None,
            'role': 'assistant',
            'audio': None,
            'function_call': None,
            'tool_calls': None
        }
    }],
    'created': 1742508614,
    'model': 'llama-3.1-8b-instant',
    'object': 'chat.completion',
    'service_tier': None,
    'system_fingerprint': 'fp_8d1c73bd4c',
    'usage': {
        'completion_tokens': 82,
        'prompt_tokens': 1575,
        'total_tokens': 1657,
        'completion_tokens_details': None,
        'prompt_tokens_details': None
    },
    'x_groq': {
        'id': 'req_01jptsps0tfd6vjb0qfytjfhtz'
    }
}

data2 = {
    'id': 'chatcmpl-f28022db-3baa-41f8-a12b-4ebcb9c8544d',
    'choices': [
        {
            'finish_reason': 'stop',
            'index': 0,
            'logprobs': None,
            'message': {
                'content': '{\n  "codes": ["36236"],\n  "descriptions": ["Venous tributary (branch) occlusion"],\n  "keywords": ["passage_id": 3700]\n}',
                'refusal': None,
                'role': 'assistant',
                'audio': None,
                'function_call': None,
                'tool_calls': None
            }
        }
    ],
    'created': 1742759359,
    'model': 'llama-3.1-8b-instant',
    'object': 'chat.completion',
    'service_tier': None,
    'system_fingerprint': 'fp_a4265e44d5',
    'usage': {
        'completion_tokens': 41,
        'prompt_tokens': 1612,
        'total_tokens': 1653,
        'completion_tokens_details': None,
        'prompt_tokens_details': None
    },
    'x_groq': {
        'id': 'req_01jq28tx1feydr7sw2tm0h9h8f'
    }
}

# def get_codes(data):
#     # Extract the string from the nested 'content' field and parse it as JSON
#     content_str = data['choices'][0]['message']['content']
#     # content_json = json.loads(content_str)

#     try:
#         content_json = json.loads(content_str)
#     except json.JSONDecodeError as e:
#         print(f"JSONDecodeError: {e}")

#     # Return the list after "codes"
#     codes = content_json['codes']
#     return codes


def get_codes(data):
    try:
        # Get the string from the 'content' field
        content_str = data['choices'][0]['message']['content']
        fixed_str = fix_malformed_json(content_str)
        # print(fixed_str)
        # Convert the string into a JSON object
        content_json = json.loads(fixed_str)
        # Return the 'codes' list
        return content_json.get('codes', [])
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing data: {e}")
        return []


def calculate_hit_rate(code_pred_ls:list, answer_code_ls: list):
    hit_rate = 0
    for code in code_pred_ls:
        if code in answer_code_ls:
            hit_rate += 1
    return hit_rate/len(answer_code_ls)


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def truncate_text(text: str, max_tokens: int = 6000) -> str:
    """Truncate text to fit within token limit."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])




def get_random_sample(csv_file_path, num_samples=10):
    descriptions = []
    codes_list = []
    document_metadatas = []
    ids = []

    while len(descriptions) < num_samples:
        # Reset for new sample
        concatenated_description = ""
        current_id = None
        current_codes = None
        
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Get random starting point
            rows = list(reader)
            start_idx = random.randint(0, len(rows)-1)
            current_row = rows[start_idx]
            
            # Process first row of sample
            description = current_row['text'].strip()
            codes = current_row['dia_code'].strip()
            current_id = current_row['subject_id'].strip()
            
            if description and codes:
                concatenated_description = description
                current_codes = codes
                
                # Look at subsequent rows
                for row in rows[start_idx+1:]:
                    next_id = row['subject_id'].strip()
                    next_description = row['text'].strip()
                    
                    # If same ID, concatenate description
                    if next_id == current_id:
                        concatenated_description += " " + next_description
                    else:
                        break
                        
                # Add completed sample
                descriptions.append(concatenated_description)
                # print(f'description:{descriptions[-1]}')
                codes_list.append(current_codes)
                document_metadatas.append({"type": "ICD-9", "code": current_codes})
                ids.append(current_id)

    return descriptions, codes_list, document_metadatas, ids