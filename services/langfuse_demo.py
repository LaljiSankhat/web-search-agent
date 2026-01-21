from dotenv import load_dotenv
from langfuse import observe

load_dotenv()

@observe(name="health-check")
def health_check():
    return "It works!"

health_check()
