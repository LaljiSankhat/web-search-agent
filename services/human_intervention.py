from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from services.content import contents
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.0, # Reduced to 0 for maximum consistency
    max_tokens=800
)

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

# Process
result_text = " ".join(contents)

response = llm.invoke(
    user_interest_prompt.format_messages(
        topic="Python and ML",
        interest="Deep Learning",
        combined_content=result_text
    )
)

print(response.content)