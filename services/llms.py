# services/llm.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from services.content import contents
from dotenv import load_dotenv
from services.prompts import deep_think_prompt
from services.text_splitter import split_text_into_chunks
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()



# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="llama-3.3-70b-versatile",
#     temperature=0.2,
#     max_tokens=800
# )


map_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",  # cheap & fast
    callbacks=[langfuse_handler],
    temperature=0,
    max_tokens=512,
)

deep_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",  #  deep reasoning
    callbacks=[langfuse_handler],
    temperature=0.5,
    max_tokens=2048,
)


sentiment_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    callbacks=[langfuse_handler],
    temperature=0  # VERY important for classification
)


sentiment_prompt = ChatPromptTemplate.from_template("""

USER_INPUT (given by user):
{user_text}

You are a sentiment classification system.
Classify the USER_INPUT as either POSITIVE or NEGATIVE.
Respond with ONLY one word: positive or negative.
""")


# response = sentiment_llm.invoke(
#     sentiment_prompt.format_messages(
#         user_text="i am not interested to know about sub topic"
#     )
# )

# print(response.content)yes

# result = " ".join(contents)

# chunks = split_text_into_chunks(result)

# import time
# from langchain_core.messages import HumanMessage

# # ... (your existing setup code) ...

# # 1. Define a "Summary" prompt for the chunks
# # This shrinks each chunk into its most important facts
# map_prompt = """Extract all core research findings, technical details, and key insights 
# from this specific part of the text. Be concise but keep all data points:
# {chunk_content}"""

# # 2. STEP 1: Process each chunk (The "Map" phase)
# chunk_summaries = []

# print(f"Total chunks to process: {len(chunks)}")

# for i, chunk in enumerate(chunks):
#     print(f"Analyzing chunk {i+1}...")
    
#     # We create a simple prompt for the individual chunk
#     chunk_response = llm.invoke(map_prompt.format(chunk_content=chunk))
    
#     chunk_summaries.append(chunk_response.content)
    
#     # IMPORTANT: Wait 2-3 seconds between chunks to avoid Groq's Rate Limits (TPM)
#     time.sleep(2)

# # 3. STEP 2: Combine the summaries for the Final Analysis (The "Reduce" phase)
# # Now, 'final_combined_content' will be much smaller (well under 6k tokens)
# final_combined_content = "\n\n".join(chunk_summaries)

# print("Starting final Deep Think analysis...")

# # 4. Final Deep Think Call
# response = llm.invoke(
#     deep_think_prompt.format_messages(
#         topic="Python and ML",
#         combined_content=final_combined_content
#     )
# )

# print("\n--- FINAL OUTPUT ---\n")
# print(response.content)