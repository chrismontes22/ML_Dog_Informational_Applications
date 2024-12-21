Below are the main script sets related to this repository, along with instructions on how to run them. Some of them require accounts for API calls or compute usage, but everything in this repository, including training and data gathering, was done at ZERO COST. Because of this, Google Colab is used often in this repository due to its free GPU access. Please note that a Google account is required in order to use their free GPU tier, which is highly recommended.

Also remember to check the [Dog_List.txt](https://github.com/chrismontes22/Dog-Classification/blob/main/Dog_List.txt) file in order to see the full list of available dog breeds!

-Inference-

1. Multi Agent RAG for dog breeds - The best way to run the multi agent RAG is by heading over to my [Google Colab notebook](https://colab.research.google.com/drive/1QF40xb7qqraKYBwJpOyZNVXq2AepJPUM#scrollTo=fOPqmHy0ruwv), and changing the runtime type to a GPU. The free T4 GPU is sufficient. Once you have done so, all you have to do is select the Runtime > Run All.

2. From data colletion to a basic (not multi agent) RAG application - This application includes the full process from data collection all the way to RAG output. You can get the script from my github [here](URL) and run it after installing the necessary packages. Or you can run it from Google Colab [here](URL), then selecting Runtime > Run All. A GPU helps for the LLM portion of it, but can be ran for inference as well in about two minutes.

3. Image Classification App - The best way to see my image classification application is to head on over to my Hugging Face Space, [chrismontes/Dog_Breed_Identifier](https://huggingface.co/spaces/chrismontes/Dog_Breed_Identifier). Another way is to run a Docker Image to create a container with the necessary dependencies and inference script. If interested in using docker, please scroll to the bottom for details.

4. Fine tuned language model for fun fact output - To see the fine tuned LLM that outputs random facts about a dog breed, the best way is to see it in this [Colab notebook.](URL) This notebook has the Image Classification from above, and passes the result (a dog breed) to the language model to output random facts about the breed.


Above are the Machine learning applications and ways to run them. Feel free to also my data processing scripts, such as training the Image classification model. 


Load the Image Classification as a docker container.

In order to use docker for this application, you must mount the picture file of your dog as you run the container. For both Bash and Powershell, simply replace INSERT_ABSOLUTE_PATH_TO_DOG_PHOTO_HERE
 with the absolute filepath of the dog picture, keeping the quotation marks. It uses a Linux filesystem, so make sure to format your directory if you are using another OS. Spaces are ok to have in the file paths.

Bash
First execute in a Bash terminal:
```
export IMAGE_PATH="/mnt/INSERT_ABSOLUTE_PATH_TO_DOG_PHOTO_HERE"
```

Then execute exactly the following:
```
docker run --rm -it -e IMAGE_PATH="$IMAGE_PATH" -v "${IMAGE_PATH}:${IMAGE_PATH}" chrismontes22/dog_image_classifier
```

Powershell
Simply replace INSERT_ABSOLUTE_PATH_TO_DOG_PHOTO_HERE with your file path in Linux format and execute in a terminal:
```
$IMAGE_PATH="INSERT_ABSOLUTE_PATH_TO_DOG_PHOTO_HERE"; docker run --rm -it -e IMAGE_PATH=$IMAGE_PATH -v "${IMAGE_PATH}:${IMAGE_PATH}" chrismontes22/dog_image_classifier

```
