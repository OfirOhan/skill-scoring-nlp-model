"""
Prompt templates for SPE synthetic dataset generation.
All LLM prompts are centralized here for easy iteration.
"""

# ---------------------------------------------------------------------------
# Phase 2: Persona Generation
# ---------------------------------------------------------------------------

PERSONA_GENERATION = """\
You are generating a realistic hi-tech professional persona for a synthetic dataset.

== ARCHETYPE ==
{archetype_label} — typical skill areas: {primary_categories}

== CONSTRAINTS ==
- Name (pre-assigned, use exactly): {pre_selected_name}
- Years of experience: {years_experience}
- Seniority: {seniority}
- Industry: {industry}

== SKILL POOL (choose from these) ==
Primary skills (pick {primary_count} from these categories):
{primary_skills}

Secondary skills (pick {secondary_count} from these categories):
{secondary_skills}

== PROFICIENCY LEVEL RULES ==
1 = Awareness — skill listed, no real usage context
2 = Beginner — basic/learning usage
3 = Intermediate — regular use in real projects
4 = Advanced — production/complex usage, ownership
5 = Expert — architecture-level, mentoring, deep multi-year work

Assign levels by skill TIER — this is what keeps the profile rational:
- PRIMARY skills are this persona's specialty. Rate each within {primary_band}.
- SECONDARY skills are supporting / breadth areas they have touched but do NOT
  specialize in. Rate each within {secondary_band}. This tier is where genuine
  Beginner (2) and Awareness (1) levels belong — SEVERAL secondary skills should
  be 1 or 2. Do NOT rate every secondary skill at 3.

Seniority adjustment, applied WITHIN the bands above (never outside them):
{seniority_bias}

== PERSONA STYLE ==
Writing style: {writing_style}
Language fluency: {language_fluency}
Self-promotion level: {self_promotion_level}
Quantification tendency: {quantification_tendency}

== TASK ==
Generate a JSON persona with the following fields:
- "name": MUST be "{pre_selected_name}" (do not change it)
- "current_role": job title matching the archetype
- "company": realistic company name
- "education": {{"degree": "...", "university": "...", "year": YYYY}}
- "career_trajectory": list of 2-4 career steps leading to current role
- "skills": dict mapping skill name -> proficiency level (1-5)

RULES:
1. Assign {total_skill_count} skills total
2. Skills MUST come from the provided skill pool
3. Primary skills MUST fall within {primary_band}; secondary skills MUST fall within {secondary_band}
4. Make the skill combination internally consistent (a frontend engineer shouldn't have Expert Kubernetes)
5. Include soft skills ONLY if natural for this archetype
6. SELF-CHECK before returning: confirm every primary skill is within {primary_band},
   every secondary skill is within {secondary_band}, and that several secondary skills
   are rated 1 or 2. If any of these is not true, fix the levels before you output.

Output ONLY valid JSON, no markdown fences, no explanation.\
"""

# ---------------------------------------------------------------------------
# Phase 3: Document Planning
# ---------------------------------------------------------------------------

DOCUMENT_PLANNING = """\
You are planning which professional documents to generate for a candidate.

== CANDIDATE ==
Name: {name}
Role: {current_role} at {company}
Years of Experience: {years_experience}
Industry: {industry}
Archetype: {archetype_label}

== TASK ==
Generate {doc_count} documents for this candidate. You MUST include exactly 1 CV.
For the remaining {remaining_count} documents, choose from:
- project_readme: a GitHub project README (provide a project title and brief topic)
- recommendation: a professional recommendation letter (specify recommender role)
- linkedin: a LinkedIn summary/about section
- blog: a technical blog post (specify topic)

For each document, provide:
- "doc_type": one of cv, project_readme, recommendation, linkedin, blog
- "topic_summary": 1-sentence description of what this document is about
- "title": document title (project name for READMEs, blog title for blogs, etc.)

RULES:
1. Project READMEs should be about realistic projects matching the candidate's skills and industry
2. At most 3 READMEs, at most 2 recommendations, at most 1 linkedin, at most 1 blog
3. Blog posts only for candidates with 5+ years experience
4. Recommendations should come from different perspectives (manager, peer, professor, etc.)

Output ONLY a JSON list, no markdown fences, no explanation.\
"""

# ---------------------------------------------------------------------------
# Phase 4: Evidence Allocation
# ---------------------------------------------------------------------------

EVIDENCE_ALLOCATION = """\
You are planning how a candidate's skills appear across their professional documents.

== CANDIDATE ==
Name: {name}
Role: {current_role}
Years of Experience: {years_experience}
Industry: {industry}

== DOCUMENTS TO BE GENERATED ==
{documents_list}

== SKILLS WITH GLOBAL PROFICIENCY ==
{skills_list}

== TASK ==
For each (skill, document) pair, assign a LOCAL evidence intensity (1-5):
- 1 = not mentioned at all in this document
- 2 = barely mentioned / listed only
- 3 = mentioned in real work context, contributed meaningfully
- 4 = core usage, production impact, ownership
- 5 = central to identity, architecture-level, mentoring

== HARD RULES ==
1. For each skill, the MAXIMUM local intensity across all documents MUST EQUAL the global level
2. No document may have a local intensity HIGHER than the global level
3. Skills with global level 1 MUST have intensity 1 in ALL documents (not mentioned anywhere)
4. Be REALISTIC: a project about web scraping won't showcase Kubernetes at level 4
5. Higher global levels should generally show evidence in MORE documents (breadth)
6. Some skills naturally appear in some document types but not others (e.g., soft skills appear in recommendations, not in READMEs)
7. Not every skill appears in every document — that would be unrealistic

Output ONLY valid JSON mapping skill_name -> {{doc_id: intensity}}, no markdown fences.\
"""

# ---------------------------------------------------------------------------
# Phase 5: Document Text Generation
# ---------------------------------------------------------------------------

CV_GENERATION = """\
You are generating a realistic CV for a hi-tech professional.

== PERSONA ==
Name: {name}
Role: {current_role} at {company}
Experience: {years_experience} years | Seniority: {seniority}
Industry: {industry}
Education: {education}
Career trajectory: {career_trajectory}

== DOCUMENT STYLE ==
CV format: {cv_format}
Geographic convention: {geographic_convention}
Writing style: {writing_style}
Language fluency: {language_fluency}
Self-promotion level: {self_promotion_level}
Quantification tendency: {quantification_tendency}
Company type: {company_type}
Company size: {company_size}

== DOCUMENT STRUCTURE (follow this section order) ==
{structure_seed}

== SKILL EVIDENCE FOR THIS DOCUMENT ==
For each skill below, the CV MUST show EXACTLY this level of evidence.
This is the MOST IMPORTANT requirement — accuracy here is critical.

{skill_evidence_instructions}

== STYLE VARIANCE (CRITICAL) ==
- Do NOT start with common LLM patterns like "results-driven", "passionate", "dynamic"
- Vary sentence structure: mix simple, compound, and complex sentences
- Make this document feel UNIQUE — not like a template
{banned_phrases}

== RULES ==
1. NEVER use proficiency words: "expert", "proficient", "beginner", "familiar", "advanced", "mastery", "novice", "extensive experience"
2. SHOW skill depth through described WORK, not self-assessment labels
3. Each skill's evidence MUST match its assigned intensity — this is ground truth for ML training
4. A level-2 skill MUST have noticeably less evidence than a level-4 skill
5. Mix skills naturally — don't group by intensity
6. Include 2-4 work experiences appropriate for {years_experience} years
7. Keep between 400-800 words
8. Make it read like a REAL CV, not a template
9. Level 1 skills must NOT appear anywhere in the document
10. Follow the DOCUMENT STRUCTURE above — use those sections in that order

Output ONLY the CV text, no explanation or meta-commentary.\
"""

README_GENERATION = """\
You are generating a realistic GitHub project README for a hi-tech professional.

== PERSONA ==
Name: {name}
Role: {current_role}
Industry: {industry}

== PROJECT ==
Title: {title}
Topic: {topic_summary}
Scale: {project_scale}
Team size: {project_team_size}
Jargon density: {jargon_density}

== WRITING STYLE ==
Writing style: {writing_style}
Language fluency: {language_fluency}
Quantification tendency: {quantification_tendency}

== DOCUMENT STRUCTURE (follow this section order) ==
{structure_seed}

== SKILL EVIDENCE FOR THIS DOCUMENT ==
{skill_evidence_instructions}

== STYLE VARIANCE (CRITICAL) ==
- Make this README feel UNIQUE — not like a template
- Vary the tone and level of detail across sections
{banned_phrases}

== RULES ==
1. NEVER use proficiency words: "expert", "proficient", "beginner", "familiar", "advanced", "mastery", "novice"
2. SHOW skill depth through described USAGE, architecture decisions, and technical details
3. Each skill's evidence MUST match its assigned intensity
4. Keep between 200-500 words
5. Make it read like a REAL GitHub README
6. Level 1 skills must NOT appear anywhere in the document
7. Follow the DOCUMENT STRUCTURE above — use those sections in that order

Output ONLY the README text in markdown format, no explanation.\
"""

RECOMMENDATION_GENERATION = """\
You are generating a realistic professional recommendation letter for a hi-tech professional.

== CANDIDATE ==
Name: {name}
Role: {current_role} at {company}
Industry: {industry}

== RECOMMENDER ==
Role: {recommender_role}
Closeness: {recommender_closeness}
Company type: {company_type}
Company size: {company_size}

== WRITING STYLE ==
(These reflect the CANDIDATE's persona, not the recommender's)
Writing style context: {writing_style}
Quantification tendency: {quantification_tendency}

== DOCUMENT FLOW (follow this paragraph structure) ==
{structure_seed}

== SKILL EVIDENCE FOR THIS DOCUMENT ==
{skill_evidence_instructions}

== STYLE VARIANCE (CRITICAL) ==
- Do NOT start with "rare blend of", "standout", "I am pleased to"
- Do NOT use "exemplifies", "instrumental in", "transformative impact"
- Write from the recommender's AUTHENTIC voice — a professor writes differently than a startup CTO
- A peer recommendation is casual and specific. A manager is broader and more strategic.
{banned_phrases}

== RULES ==
1. NEVER use proficiency words: "expert", "proficient", "beginner", "familiar", "advanced", "mastery", "novice"
2. Show skill depth through SPECIFIC praise, anecdotes, and observed impact
3. Each skill's evidence MUST match its assigned intensity
4. A recommendation naturally highlights 2-5 strengths, not every skill
5. Include personality and work ethic observations
6. Keep between 150-350 words
7. Level 1 skills must NOT appear anywhere in the document
8. Follow the DOCUMENT FLOW above for paragraph structure

Output ONLY the recommendation letter text, no explanation.\
"""

LINKEDIN_GENERATION = """\
You are generating a realistic LinkedIn About/Summary section for a hi-tech professional.

== PERSONA ==
Name: {name}
Role: {current_role} at {company}
Experience: {years_experience} years
Industry: {industry}
Completeness: {linkedin_completeness}

== WRITING STYLE ==
Writing style: {writing_style}
Language fluency: {language_fluency}
Self-promotion level: {self_promotion_level}

== DOCUMENT FLOW ==
{structure_seed}

== SKILL EVIDENCE FOR THIS DOCUMENT ==
{skill_evidence_instructions}

== STYLE VARIANCE (CRITICAL) ==
- Do NOT start with "Passionate about" or "Driven by"
- Make this feel like a REAL person wrote it, not a template
{banned_phrases}

== RULES ==
1. NEVER use proficiency words as labels
2. Keep between 100-300 words depending on completeness level
3. LinkedIn summaries are personal and forward-looking
4. Level 1 skills must NOT appear anywhere
5. Follow the DOCUMENT FLOW above

Output ONLY the LinkedIn summary text, no explanation.\
"""

BLOG_GENERATION = """\
You are generating a realistic technical blog post excerpt for a hi-tech professional.

== PERSONA ==
Name: {name}
Role: {current_role}
Industry: {industry}

== BLOG ==
Title: {title}
Topic: {topic_summary}
Depth: {blog_depth}
Jargon density: {jargon_density}

== WRITING STYLE ==
Writing style: {writing_style}
Language fluency: {language_fluency}
Quantification tendency: {quantification_tendency}

== ARTICLE STRUCTURE (follow this flow) ==
{structure_seed}

== SKILL EVIDENCE FOR THIS DOCUMENT ==
{skill_evidence_instructions}

== STYLE VARIANCE (CRITICAL) ==
- Make this feel like a REAL blog post by a REAL engineer, not AI-generated
- Vary the opening: don't always start with a grand statement
{banned_phrases}

== RULES ==
1. NEVER use proficiency words as labels
2. Show depth through technical detail and original insights
3. Keep between 300-600 words
4. Level 1 skills must NOT appear anywhere
5. Follow the ARTICLE STRUCTURE above

Output ONLY the blog post text, no explanation.\
"""

# ---------------------------------------------------------------------------
# Skill evidence instruction builder (used in all generation prompts)
# ---------------------------------------------------------------------------

EVIDENCE_INTENSITY_DESCRIPTIONS = {
    1: "DO NOT MENTION in any part of this document — completely absent",
    2: "Barely mentioned — listed in a skills/tools section or mentioned once in passing only",
    3: "Mentioned in 1-2 real work contexts. Contributed meaningfully, part of regular workflow.",
    4: "Featured prominently with production impact, ownership, measurable results in 2-3 contexts.",
    5: "Central to identity. Architecture decisions, mentoring others, multi-year depth, cross-team impact.",
}


def build_skill_evidence_instructions(skill_evidence: dict) -> str:
    """Build the skill evidence section for a generation prompt.

    Args:
        skill_evidence: dict mapping skill_name -> local_intensity (1-5)

    Returns:
        Formatted instruction string for the prompt.
    """
    lines = []
    for skill, intensity in sorted(skill_evidence.items(), key=lambda x: x[1]):
        desc = EVIDENCE_INTENSITY_DESCRIPTIONS[intensity]
        lines.append(f"- {skill}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation Prompts
# ---------------------------------------------------------------------------

VALIDATION_SKILL_SHOWCASE = """\
Read the following professional document and assess the evidence level for each listed skill.

== DOCUMENT ==
{document_text}

== SKILLS TO ASSESS ==
{skills_list}

== SCORING RUBRIC ==
1 = not mentioned at all in this document
2 = barely mentioned / listed only
3 = mentioned in real work context, contributed meaningfully
4 = core usage, production impact, ownership
5 = central to identity, architecture-level, mentoring

For each skill, output ONLY a JSON dict mapping skill_name -> score (1-5).
No explanation, no markdown fences.\
"""

VALIDATION_ALLOCATION_RATIONALITY = """\
Review this evidence allocation for a candidate and assess if it makes sense.

== CANDIDATE ==
Role: {current_role}
Years of Experience: {years_experience}
Industry: {industry}

== GLOBAL SKILL LEVELS ==
{skills_list}

== DOCUMENTS AND THEIR TOPICS ==
{documents_list}

== EVIDENCE ALLOCATION (skill -> doc -> intensity) ==
{allocation_json}

== TASK ==
Assess whether the allocation is rational. Consider:
1. Do document topics match the skills allocated to them?
2. Are soft skills allocated to appropriate document types (e.g., recommendations, not READMEs)?
3. Is the breadth of evidence realistic for each global level?
4. Are there any obvious contradictions?

Output JSON with:
- "score": rationality score 1-5 (5 = perfectly rational)
- "concerns": list of specific concerns (empty if none)

No markdown fences, no explanation outside the JSON.\
"""
