"""Quick smoke test for SPE pipeline imports."""
from spe.generate.hyperparams import (
    get_taxonomy, get_archetypes, get_evidence_profiles,
    sample_archetype, sample_persona_hyperparams, sample_doc_count,
    sample_document_hyperparams, get_skills_by_category,
)
from spe.generate.prompts import build_skill_evidence_instructions

# Test static data loading
t = get_taxonomy()
a = get_archetypes()
e = get_evidence_profiles()
print(f"Skills: {len(t)}, Archetypes: {len(a)}, Evidence levels: {len(e)}")

# Test sampling
arch = sample_archetype()
hp = sample_persona_hyperparams(arch)
dc = sample_doc_count(hp["years_experience"])
print(f"Sampled: archetype={arch}, years={hp['years_experience']}, seniority={hp['seniority']}, docs={dc}")

# Test doc hyperparams
for doc_type in ["cv", "project_readme", "recommendation", "linkedin", "blog"]:
    dp = sample_document_hyperparams(doc_type)
    print(f"  {doc_type}: {dp}")

# Test skill evidence builder
test_evidence = {"Python": 5, "React": 1, "Docker": 3}
instructions = build_skill_evidence_instructions(test_evidence)
print(f"\nEvidence instructions:\n{instructions}")

# Test category lookup
backend = get_skills_by_category("backend_development")
print(f"\nBackend skills ({len(backend)}): {backend[:5]}...")

print("\n[OK] All imports and sampling OK")
