import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
import os
import json
import pickle
from utils import get_random_sample

class ICDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertICDPredictor(nn.Module):
    def __init__(self, n_classes, bert_model_name='bert-base-uncased'):
        super(BertICDPredictor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.fc(output)

def load_all_icd_codes(cms_file_path='ICD-9-CM-v32-master-descriptions/CMS32_DESC_LONG_DX.txt'):
    """Load all potential ICD codes from the CMS file."""
    all_codes = []
    with open(cms_file_path, 'r') as f:
        for line in f:
            # Split by whitespace and take the first part as the code
            parts = line.strip().split()
            if parts:
                code = parts[0]
                all_codes.append(code)
    return all_codes

def create_code_mapping(codes_list, mapping_file='icd_code_mapping.json', cms_file_path='ICD-9-CM-v32-master-descriptions/CMS32_DESC_LONG_DX.txt'):
    """Create a mapping from ICD codes to indices and save it to a file."""
    # Load all potential codes from the CMS file
    all_potential_codes = load_all_icd_codes(cms_file_path)
    
    # Extract unique codes from the dataset
    dataset_codes = set()
    for codes in codes_list:
        for code in codes:
            dataset_codes.add(code)
    
    # Find codes in the dataset that are not in the CMS file
    missing_codes = dataset_codes - set(all_potential_codes)
    if missing_codes:
        print(f"Warning: {len(missing_codes)} codes in the dataset are not in the CMS file.")
        print(f"First few missing codes: {list(missing_codes)[:5]}")
    
    # Create mapping using all potential codes
    code_to_idx = {code: idx for idx, code in enumerate(all_potential_codes)}
    
    # Save mapping to file
    with open(mapping_file, 'w') as f:
        json.dump(code_to_idx, f)
    
    return code_to_idx

def load_code_mapping(codes_list, mapping_file='icd_code_mapping.json', cms_file_path='ICD-9-CM-v32-master-descriptions/CMS32_DESC_LONG_DX.txt'):
    """Load the code-to-index mapping from a file."""
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            print(f"Loading code mapping from {mapping_file}")
            return json.load(f)
    else:
        print(f"Creating code mapping in {mapping_file}")
        return create_code_mapping(codes_list, mapping_file, cms_file_path)
        
def load_and_preprocess_data(pickle_file, mapping_file='icd_code_mapping.json', cms_file_path='ICD-9-CM-v32-master-descriptions/CMS32_DESC_LONG_DX.txt'):
    """Load and preprocess the data from pickle file."""
    try:
        with open(pickle_file, 'rb') as f:
            descriptions, codes_list, document_metadatas, ids = pickle.load(f)
            print("Loaded existing random samples")
    except:
        # If file doesn't exist or can't be read, generate new samples
        print("Generating new random samples from mimic3_full.csv...")
        descriptions, codes_list, document_metadatas, ids = get_random_sample("mimic3_full.csv", 10000)
        
        # Save the samples
        with open(pickle_file, 'wb') as f:
            pickle.dump((descriptions, codes_list, document_metadatas, ids), f)
            print("Saved random samples to file")
    
    # Load or create code mapping
    # if create_mapping:
    #     code_to_idx = create_code_mapping(codes_list, mapping_file, cms_file_path)
    # else:
    #     code_to_idx = load_code_mapping(mapping_file)
    #     if code_to_idx is None:
    #         raise ValueError(f"Mapping file {mapping_file} not found. Set create_mapping=True to create it.")

    code_to_idx = load_code_mapping(codes_list, mapping_file, cms_file_path)
    
    # Get the number of all unique ICM-9 codes
    n_codes = len(code_to_idx)
    
    # Create multi-label encoding for each sample
    labels = []
    for codes in codes_list:
        label = np.zeros(n_codes)
        for code in codes:
            if code in code_to_idx:  # Only include codes that are in our mapping
                label[code_to_idx[code]] = 1
        labels.append(label)
    
    return descriptions, np.array(labels), list(code_to_idx.keys()), code_to_idx

def train_model(model, train_loader, val_loader, device, num_epochs=5):
    """Train the BERT model."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_bert_icd_model.pth')

def evaluate_model(model, test_loader, device, unique_codes):
    """Evaluate the model on test data."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs) > 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics for each ICD code
    for i, code in enumerate(unique_codes):
        print(f"\nMetrics for ICD code: {code}")
        print(classification_report(all_labels[:, i], all_preds[:, i]))

def predict_icd_codes(model, tokenizer, text, code_to_idx, device, threshold=0.5):
    """Predict ICD codes for a new text."""
    model.eval()
    
    # Tokenize the text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).cpu().numpy()[0]
    
    # Convert predictions to ICD codes
    predicted_codes = []
    for code, idx in code_to_idx.items():
        if preds[idx] == 1:
            predicted_codes.append(code)
    
    return predicted_codes

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    texts, labels, unique_codes, code_to_idx = load_and_preprocess_data(
        'large_random_samples.pkl', 
        create_mapping=True,
        cms_file_path='ICD-9-CM-v32-master-descriptions/CMS32_DESC_LONG_DX.txt'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = ICDDataset(X_train, y_train, tokenizer)
    val_dataset = ICDDataset(X_val, y_val, tokenizer)
    test_dataset = ICDDataset(X_test, y_test, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Initialize model
    model = BertICDPredictor(n_classes=len(unique_codes))
    model = model.to(device)
    
    # Train model
    print("Training model...")
    train_model(model, train_loader, val_loader, device)
    
    # Load best model and evaluate
    print("Evaluating model...")
    model.load_state_dict(torch.load('best_bert_icd_model.pth'))
    evaluate_model(model, test_loader, device, unique_codes)
    
    # Example of predicting ICD codes for a new text
    print("\nExample prediction:")
    sample_text = "Patient presents with type 2 diabetes mellitus and hypertension."
    predicted_codes = predict_icd_codes(model, tokenizer, sample_text, code_to_idx, device)
    print(f"Text: {sample_text}")
    print(f"Predicted ICD codes: {predicted_codes}")

if __name__ == "__main__":
    main() 