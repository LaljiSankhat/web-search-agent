import requests
from bs4 import BeautifulSoup

# Step 2: Define the URL
urls = ["https://www.geeksforgeeks.org/python/python-programming-language-tutorial/", "https://www.geeksforgeeks.org/machine-learning/machine-learning/"]

contents = []

try:

    for url in urls:
        # Step 3: Fetch the HTML content
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Step 4: Parse the HTML
        # Use 'html.parser' or 'lxml' (if installed)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Step 5: Extract specific data (e.g., all paragraphs in the main article content)
        # You will need to inspect the target website's HTML structure to find the correct tags/classes
        main_content_container = soup.find('div', class_='article--viewer_content')

        if main_content_container:
            paragraphs = main_content_container.find_all('p')
            s = ""
            for para in paragraphs:
                # print(para.text.strip()) # .strip() removes leading/trailing whitespace
                s += para.text.strip()
            contents.append(s)
        else:
            print("Could not find the main content container. Check the website's HTML structure.")
    print(contents)
    print(len(contents))
        

except requests.exceptions.RequestException as e:
    print(f"An error occurred while fetching the URL: {e}")
