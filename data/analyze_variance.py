import json, random, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
db = json.load(open('data/documents_db.json', 'r', encoding='utf-8'))
docs_by_type = {}
for d in db.values():
    docs_by_type.setdefault(d.get('type', '?'), []).append(d)

# Show structure of 5 CVs
print("=" * 70)
print("CV STRUCTURE COMPARISON - First 15 lines of 5 random CVs")
print("=" * 70)
sample = random.sample(docs_by_type['cv'], 5)
for i, doc in enumerate(sample):
    text = doc['text']
    lines = text.split('\n')
    doc_id = doc['doc_id']
    print(f"\n--- CV {i+1} ({doc_id}) ---")
    for ln in lines[:15]:
        print(ln)
    print("...[REST TRUNCATED]")

# Show structure of 5 READMEs
print("\n" + "=" * 70)
print("README STRUCTURE COMPARISON - First 10 lines of 5 random READMEs")
print("=" * 70)
sample = random.sample(docs_by_type['project_readme'], 5)
for i, doc in enumerate(sample):
    text = doc['text']
    lines = text.split('\n')
    doc_id = doc['doc_id']
    print(f"\n--- README {i+1} ({doc_id}) ---")
    for ln in lines[:10]:
        print(ln)
    print("...[REST TRUNCATED]")

# Analyze common opening phrases
print("\n" + "=" * 70)
print("OPENING PHRASE ANALYSIS")
print("=" * 70)
for t in ['cv', 'recommendation', 'linkedin']:
    openings = []
    for doc in docs_by_type.get(t, []):
        text = doc['text'].strip()
        first_line = text.split('\n')[0].strip()
        if first_line:
            openings.append(first_line[:100])
    print(f"\n--- {t.upper()} opening lines (all {len(openings)}) ---")
    for o in openings:
        print(f"  {o}")
