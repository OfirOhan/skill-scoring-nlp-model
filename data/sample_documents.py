import json
import random
from collections import defaultdict
import os

def main():
    # Construct the path to documents_db.json
    db_path = os.path.join(os.path.dirname(__file__), 'documents_db.json')
    
    print(f"Loading documents from {db_path}...")
    with open(db_path, 'r', encoding='utf-8') as f:
        db = json.load(f)
        
    # Group documents by type
    docs_by_type = defaultdict(list)
    for doc_id, doc in db.items():
        doc_type = doc.get("type", "unknown")
        docs_by_type[doc_type].append(doc)
        
    print(f"Found {len(docs_by_type)} different document types: {list(docs_by_type.keys())}\n")
    
    # Sample and print 5 documents from each type
    for doc_type, docs in docs_by_type.items():
        print(f"={'='*60}")
        print(f"TYPE: {doc_type.upper()} (Total: {len(docs)})")
        print(f"={'='*60}")
        
        # Sample up to 5 documents
        sample_size = min(5, len(docs))
        sampled_docs = random.sample(docs, sample_size)
        
        for i, doc in enumerate(sampled_docs, 1):
            print(f"--- Sample {i} of {sample_size} ---")
            print(f"Document ID: {doc.get('doc_id')}")
            print(f"Persona ID: {doc.get('persona_id')}")
            # Print the full text of the document
            text = doc.get('text', '')
            print("Document Text:")
            print("-" * 40)
            print(text)
            print("-" * 40)
            print()
            
if __name__ == "__main__":
    main()
