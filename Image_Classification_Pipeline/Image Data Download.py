"""The script requires you to create your own Google account, then go to https://developers.google.com/custom-search/v1/overview to create your own custom image search.
Once the image search is created, you will receive a cse_id and you will be able to create an API key. If you want to hide the ID and key, you will also require
a separate config file, such as in the script below."""
#Once youve gotten a cse_id and an API key and the config file, just fill the  parameters. It will create a new folder with the wuery as the name


import os
import requests
from PIL import Image
from io import BytesIO
import logging
import datetime
import config

#Set up logging
logging.basicConfig(filename='Image Changes.txt', level=logging.INFO, format='%(message)s')

#Parameters
search_query = 'puppy' #Google Search query
first_image = 1 #Google search api requres you set a range of images. The max difference between the two allowed is 100 images
last_image = 100 #Google allows max of 100 per run, It will fininsh the last page (set of 10 images)
                  #Start where you left off last run. The code logs this number in a text file in the main workfolder
                  

folder_path = search_query.replace(' ', '-')

# Log the dog breeds and the current time
logging.info('Image Download from Google')
logging.info('Time: ' + str(datetime.datetime.now()))
logging.info('Search Query: ' + search_query)
logging.info('First Image: ' + str(first_image))
logging.info('Last Image: ' + str(last_image))
logging.info('\n')

# Set up your API key and Custom Search Engine ID
api_key = config.google_api_key
cse_id = config.google_cse_id

try:
    def google_image_search(query, api_key, cse_id, start_index, num_images=10):
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': api_key,
            'cx': cse_id,
            'searchType': 'image',
            'start': start_index,
            'num': num_images
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        return response.json()

    def download_image(url, folder_path, image_name, timeout=10):
        #Downloads an image from a URL, checks the response status, opens the image, and saves it to a specified folder with a given name.
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image_path = os.path.join(folder_path, image_name)
            image.save(image_path)
        except requests.exceptions.Timeout:
            print(f"Request for {url} timed out.")
        except requests.exceptions.RequestException as e:
            print(f"Could not download {url} - {e}")
        except Exception as e:
            print(f"Could not save {url} - {e}")

  # Creates a folder for the new images of the new breed
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# A loop that establishes the range of images to download (from above parameters). 10 is the max number of images per request, and it loops to the next ten each request.
#Calculates an index for the current image based on the start index and the current index.
    for start_index in range(first_image, last_image + 1, 10):
        search_results = google_image_search(search_query, api_key, cse_id, start_index, num_images=10)
        for i, item in enumerate(search_results.get('items', [])):
            img_url = item['link']
            image_index = start_index + i
            download_image(img_url, folder_path, f'image_{image_index}.jpg')
except Exception as e:
    logging.error('Error occurred: {}'.format(e))
    logging.info('\n')
print("Images downloaded successfully!")

