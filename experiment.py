import csv
import random
from codify import Codify
from datetime import datetime, timedelta
import time
from langfuse.decorators import observe
def normalize_code(code):
    return code.replace('.', '').upper()

def code_match(true_code, predicted_code):
    true_main = normalize_code(true_code).split()[0]
    predicted_main = normalize_code(predicted_code).split()[0]
    return true_main == predicted_main

@observe()
def run_experiment(csv_file_path, sample_size=100):
    codify = Codify()
    results = []
    skipped_samples = 0

    with open(csv_file_path, 'r') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        csv_reader = csv.reader(csvfile, dialect)
        all_rows = list(csv_reader)

    print(f"Total rows in CSV: {len(all_rows)}")

    number_of_traitement = 0
    page = 1
    number_of_correct_codes = 0
    total_samples = len(all_rows)
    for row in all_rows[sample_size * (page - 1):sample_size]:
    # for row in all_rows:
        row = random.choice(all_rows)
        # row = ["Infectious gastroenteritis and colitis, unspecified | A09"]
        if len(row) == 1:  # Now we expect only one column
            parts = row[0].split('|')
            if len(parts) == 2:
                description = parts[0].strip()
                true_code = parts[1].strip()
            else:
                print(f"Skipping row: {row} (incorrect number of columns)")
                skipped_samples += 1
        if len(row) == 2:  # Now we expect two columns
                description = row[0].strip()
                true_code = row[1].strip()

        print(f"\nProcessing: {description}")
        number_of_traitement += 1
        # Regular prediction
        result = codify.get_ranked_icd_codes(description)
        predicted_code = normalize_code(result['top_one']['code'])
        predicted_description = result['top_one']['description']

        # Control group prediction
        # control_result = codify.get_control_group_output(description)
        # control_predicted_code = normalize_code(control_result['top_one']['code'])
        # control_predicted_description = control_result['top_one']['description']

        top_one_match = code_match(true_code, predicted_code)
        # control_top_one_match = code_match(true_code, control_predicted_code)
        if top_one_match:
            number_of_correct_codes += 1
        print(f"Diagnosis Description: {description}")
        print(f"Reference Code: {true_code}")
        print(f"Retrieve-Rank: {predicted_code}")
        # print(f"Vanilla GPT-3.5-turbo: {control_predicted_code}")
        print(f"Retrieve-Rank Match: {'Yes' if top_one_match else 'No'}")
        # print(f"Vanilla GPT-3.5 turbo match: {'Yes' if control_top_one_match else 'No'}")

        results.append({
            'description': description,
            'true_code': normalize_code(true_code),
            'predicted_code': predicted_code,
            'predicted_description': predicted_description,
            # 'control_predicted_code': control_predicted_code,
            # 'control_predicted_description': control_predicted_description,
            'top_one_match': top_one_match,
            # 'control_top_one_match': control_top_one_match,
        })
        print(f"Rest to process: {total_samples - number_of_traitement}")

    # Calculate overall statistics
    total_samples = len(results)
    top_one_accuracy = number_of_correct_codes / total_samples * 100
    # control_top_one_accuracy = sum(r['control_top_one_match'] for r in results) / total_samples * 100

    print(f"\nTotal processed samples: {total_samples}")
    print(f"Number of correct prediction: {number_of_correct_codes}")
    print(f"Skipped samples: {skipped_samples}")
    print(f"Top One Accuracy: {top_one_accuracy:.2f}%")
    # print(f"Control Top One Accuracy: {control_top_one_accuracy:.2f}%")

    # Log results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiment_results_{timestamp}.csv"
    negative_results = f"negative_results{timestamp}.csv"
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Diagnosis Description', 'Reference Code', 'Retrieve-Rank', 'Vanilla GPT-3.5-turbo', 'Retrieve-Rank Match', 'Vanilla GPT-3.5 turbo match']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                'Diagnosis Description': result['description'],
                'Reference Code': result['true_code'],
                'Retrieve-Rank': result['predicted_code'],
                # 'Vanilla GPT-3.5-turbo': result['control_predicted_code'],
                'Retrieve-Rank Match': 'Yes' if result['top_one_match'] else 'No',
                # 'Vanilla GPT-3.5 turbo match': 'Yes' if result['control_top_one_match'] else 'No'
            })
    with open(negative_results, 'w', newline='') as csvfile:
        fieldnames = ['Diagnosis Description', 'Reference Code', 'Retrieve-Rank', 'Vanilla GPT-3.5-turbo', 'Retrieve-Rank Match', 'Vanilla GPT-3.5 turbo match']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if not result['top_one_match']:
                writer.writerow({
                    'Diagnosis Description': result['description'],
                    'Reference Code': result['true_code'],
                    'Retrieve-Rank': result['predicted_code'],
                    # 'Vanilla GPT-3.5-turbo': result['control_predicted_code'],
                    'Retrieve-Rank Match': 'Yes' if result['top_one_match'] else 'No',
                    # 'Vanilla GPT-3.5 turbo match': 'Yes' if result['control_top_one_match'] else 'No'
                })

    print(f"Results logged to {output_file}")

if __name__ == "__main__":
    start = time.time()
    csv_file_path = "ICD-10_formatted.csv"
    run_experiment(csv_file_path, sample_size=1000)
    end = time.time()
    print(f"Total time taken: {timedelta(seconds=end - start)}")