from langfuse import Langfuse
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
 
# Initialize Langfuse client
langfuse = Langfuse()

langfuse_prompt = langfuse.get_prompt("deep_think_prompt")


print(langfuse_prompt.prompt)

langfuse.create_prompt(
    name="event-planner",
    prompt=
    "Plan an event titled {{Event Name}}. The event will be about: {{Event Description}}. "
    "The event will be held in {{Location}} on {{Date}}. "
    "Consider the following factors: audience, budget, venue, catering options, and entertainment. "
    "Provide a detailed plan including potential vendors and logistics.",
    config={
        "model":"gpt-4o",
        "temperature": 0,
    },
    labels=["production"]
);

langfuse_created_prompt = langfuse.get_prompt("event-planner")

print(langfuse_created_prompt.prompt)