from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


deep_think_prompt = ChatPromptTemplate.from_template("""
You are a senior research analyst.

Topic:
{topic}

Source Information:
{combined_content}

Instructions:
1. Deeply analyze the new source information specifically through the lens of the "{topic}".
2. Use the "Previous Research Findings" to ensure this analysis doesn't repeat basic facts, but instead builds upon them or identifies specific nuances related to the subtopic.
3. Identify patterns, contradictions, or gaps between the previous findings and the new data.
4. Structure the output organically. Use headers, bullet points, or narrative sections that best fit the data—do not follow a fixed template.
5. Provide a sophisticated, high-level synthesis that serves as a specialized extension of the previous work.

Return a deep, critical analysis.
""")





system_template = """You are a Research Validator. 

### STEP 1: VALIDATION LOGIC
- Compare [USER_INTEREST] (Subtopic) to [PREVIOUS_TOPIC] (Context).
- If the [USER_INTEREST] is entirely unrelated to the [PREVIOUS_TOPIC], respond with ONLY: "Not related to previous topic"

### STEP 2: REFINEMENT RULES
- Your goal is to provide a "Deep Dive" into the subtopic using the previous research as a foundation.
- Do NOT use a generic 1-4 numbered list.
- Organize the findings logically (e.g., by thematic relevance, technical depth, or chronological impact).
- Use the [PREVIOUS_TOPIC] summary to provide context, but focus 100% on the new [USER_INTEREST].
- Ensure the output feels like a specialized "Chapter 2" or an "Appendix" to the original research.
"""



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