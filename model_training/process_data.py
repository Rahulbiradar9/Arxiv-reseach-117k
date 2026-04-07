import csv
import json
import re
import sys
import os

def process_pipeline(input_csv, output_json):
    keywords_dict = {
        "AI": ["neural", "deep learning", "transformer", "cnn"],
        "Networks": ["network", "protocol", "routing"],
        "Security": ["security", "encryption", "attack"],
        "Systems": ["system", "operating system", "distributed"]
    }
    
    label_to_index = {"AI": 0, "Networks": 1, "Security": 2, "Systems": 3}
    
    # Increase CSV field size limit just in case
    csv.field_size_limit(10**7)
    
    results = []
    print(f"Starting processing of {input_csv}...")
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                title = row.get('title', '')
                abstract = row.get('abstract', '')
                
                raw_text = f"{title} {abstract}"
                
                # Clean text: lowercase, remove special characters, keep alphanumeric
                text = raw_text.lower()
                text = re.sub(r'[^a-z0-9\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if not text:
                    continue
                    
                explanation = {}
                labels = []
                vector = [0, 0, 0, 0]
                
                for label, kws in keywords_dict.items():
                    found_kws = []
                    for kw in kws:
                        # use regex to ensure whole word match
                        if re.search(r'\b' + re.escape(kw) + r'\b', text):
                            found_kws.append(kw)
                    
                    if found_kws:
                        explanation[label] = found_kws
                        labels.append(label)
                        vector[label_to_index[label]] = 1
                
                item = {
                    "text": text,
                    "labels": labels,
                    "multi_hot_vector": vector,
                    "explanation_keywords": explanation
                }
                results.append(item)
                
                if (i + 1) % 10000 == 0:
                    print(f"Processed {i + 1} records...")
                    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print(f"Total processed records: {len(results)}")
    
    print(f"Saving to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"Done! Size of output file: {os.path.getsize(output_json) / (1024*1024):.2f} MB")

if __name__ == '__main__':
    process_pipeline('../data/ML-Arxiv-Papers.csv', '../data/processed_dataset.json')
