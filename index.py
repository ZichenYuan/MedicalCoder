from ragatouille import RAGPretrainedModel
import csv
import os

# Load the RAG model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


icd10_documents = []
document_ids = []
document_metadatas = []

# with open('ICD-10.csv', 'r') as csvfile:
#     csv_reader = csv.reader(csvfile, delimiter='\t')
#     for i, row in enumerate(csv_reader):
#         if len(row) < 2:
#             print(f"Skipping row {i} due to incorrect format: {row}")
#             continue
#         code = row[0]
#         description = ' '.join(row[1:])  # Join all remaining fields for the description
#         icd10_documents.append(description)
#         document_ids.append(code)
#         document_metadatas.append({"type": "ICD-10", "code": code})



# with open('icd10pcs_codes_2020.txt', 'r', encoding='utf-8') as txtfile:
#     csv_reader = csv.reader(txtfile)  # Default delimiter is comma, which works for your file

#     for i, row in enumerate(csv_reader):
#         if len(row) < 2:
#             print(f"Skipping row {i} due to incorrect format: {row}")
#             continue

#         code = row[0].strip()
#         description = row[1].strip()  # The description is the second column

#         # Append to respective lists
#         icd10_documents.append(description)
#         document_ids.append(code)
#         document_metadatas.append({"type": "ICD-10", "code": code})
# print(f"Processed {len(icd10_documents)} valid rows")


# Directory containing text files
data_dir = 'ICD-9-CM-v32-master-descriptions'


# Lists to store extracted data
icd_documents = []
document_ids = []
document_metadatas = []

# Iterate through all files in the directory
for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)

    # Ensure it's a text file of long descriptions of ICD9 diagnosis codes
    if filename.endswith("LONG_DX.txt"):
        try:
            # Open with encoding handling
            with open(file_path, 'r', encoding='utf-8', errors='replace') as txtfile:
                for i, line in enumerate(txtfile):
                    parts = line.strip().split(" ", 1)  # Split only at the first space
                    if len(parts) < 2:
                        print(f"Skipping line {i} in {filename} due to incorrect format: {line}")
                        continue

                    code = parts[0].strip()
                    description = parts[1].strip()

                    # Append to respective lists
                    icd_documents.append(description)
                    document_ids.append(code)
                    document_metadatas.append({"type": "ICD-9", "code": code})

            print(f"Processed {len(icd_documents)} valid rows from {filename}")

        # Uncomment if there is a UnicodeDecodeError
        # except UnicodeDecodeError:
        #     print(f"Unicode error reading {filename}, trying alternative encoding...")
        #     try:
        #         # Try an alternative encoding
        #         with open(file_path, 'r', encoding='ISO-8859-1') as txtfile:
        #             for i, line in enumerate(txtfile):
        #                 parts = line.strip().split(" ", 1)  # Split only at the first space
        #                 if len(parts) < 2:
        #                     print(f"Skipping line {i} in {filename} due to incorrect format: {line}")
        #                     continue

        #                 code = parts[0].strip()
        #                 description = parts[1].strip()

        #                 # Append to respective lists
        #                 icd_documents.append(description)
        #                 document_ids.append(code)
        #                 document_metadatas.append({"type": "ICD-9", "code": code})

        #         print(f"Successfully processed {filename} using ISO-8859-1 encoding.")

        except Exception as e:
            print(f"Failed to read {filename} even with alternative encoding: {e}")

# Summary
print(f"Total documents processed: {len(icd_documents)}")


# Create the index
index_path = RAG.index(
    index_name="icd10_index",
    collection=icd10_documents,
    document_ids=document_ids,
    document_metadatas=document_metadatas,
)

print(f"Index created and saved at: {index_path}")

