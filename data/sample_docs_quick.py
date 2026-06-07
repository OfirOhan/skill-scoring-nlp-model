import json
import random
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

db = json.load(open('data/documents_db.json', 'r', encoding='utf-8'))
docs_by_type = {}
for d in db.values():
    docs_by_type.setdefault(d.get('type', '?'), []).append(d)

for t in ['cv', 'project_readme', 'recommendation', 'linkedin', 'blog']:
    if t not in docs_by_type:
        continue
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"TYPE: {t.upper()} — Sampling 3")
    print(f"{sep}")
    sample = random.sample(docs_by_type[t], min(3, len(docs_by_type[t])))
    for i, doc in enumerate(sample):
        text = doc['text']
        print(f"\n--- Sample {i+1} (doc_id={doc['doc_id']}) ---")
        print(text[:1000])
        if len(text) > 1000:
            print("...[TRUNCATED]")
        print()
