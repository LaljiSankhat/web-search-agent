from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


deep_think_prompt = ChatPromptTemplate.from_template("""
You are a senior research analyst.

Topic:
{topic}

Source Information:
{combined_content}

Instructions:
- Think deeply and critically
- Identify key patterns and insights
- Highlight contradictions or gaps
- Provide implications and future outlook
- Avoid surface-level summarization

Return a structured deep analysis with:
1. Core Insights
2. Supporting Evidence
3. Risks / Limitations
4. Future Offering
""")





system_template = """You are a Research Validator. 

### STEP 1: VALIDATION LOGIC (CRITICAL)
- Compare [USER_INTEREST] to [PREVIOUS_TOPIC].
- If [USER_INTEREST] is NOT conceptually related to [PREVIOUS_TOPIC], you MUST respond with ONLY this exact phrase: "Not related to previous topic"
- Do NOT provide any analysis, headers, or explanations if they are unrelated.

### STEP 2: ANALYSIS RULES (ONLY IF RELATED)
- Start immediately with "1. Core Insights".
- Do NOT mention the [PREVIOUS_TOPIC] by name.
- Do NOT include titles or introductions.
- Provide a deep analysis with these 4 sections:
  1. Core Insights
  2. Supporting Evidence
  3. Risks / Limitations
  4. Future Offering

Focus 100% on [USER_INTEREST]."""



# 2. Human Prompt: Using XML-style tags helps Llama-3 separate context from data
human_template = """
<context>
[PREVIOUS_TOPIC]: {topic}
[USER_INTEREST]: {interest}
</context>

<source_material>
{combined_content}
</source_material>

Analysis:"""

user_interest_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])






map_prompt = """Extract all core research findings, technical details, and key insights 
from this specific part of the text. Be concise but keep all data points:
{chunk_content}"""













# system_template = """<task>
# You are a Research Validator. Your task is to analyze [USER_INTEREST] based on [SOURCE_MATERIAL].
# </task>

# <logic>
# 1. Compare [USER_INTEREST] to [PREVIOUS_TOPIC].
# 2. If [USER_INTEREST] is NOT conceptually related to [PREVIOUS_TOPIC], respond ONLY with: "Not related to previous topic"
# 3. If they ARE related, extract insights about [USER_INTEREST] from [SOURCE_MATERIAL].
# </logic>

# <constraints>
# - START your response immediately with "1. Core Insights".
# - DO NOT include any titles, bold headers at the top, or introductory sentences.
# - DO NOT mention the [PREVIOUS_TOPIC] by name in your analysis.
# - Focus 100% on the [USER_INTEREST].


# Instructions:
# - Think deeply and critically
# - Identify key patterns and insights
# - Highlight contradictions or gaps
# - Provide implications and future outlook
# - Avoid surface-level summarization

# Return a structured deep analysis with:
# 1. Core Insights
# 2. Supporting Evidence
# 3. Risks / Limitations
# 4. Future Offering
# </constraints>"""