"""The first code did most the classes. The few that remain can be done manually through
a url search instead of a query for more precise results"""
#Insert the wikipedia URL in the URL variable to fetch the text data from that article

import wikipedia

def get_wikipedia_article(url):
    # Extract the title from the URL
    title = url.split("/")[-1].replace("_", " ")

    # Set the language if the URL is not English
    wikipedia.set_lang(url.split("/")[2].split(".")[0])

    # Get the page content
    content = wikipedia.page(title).content

    return content

def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

# Usage
url = "https://en.wikipedia.org/wiki/Yorkshire_Terrier"
content = get_wikipedia_article(url)
save_to_file(content, 'Yorkshire Terrier.txt')
