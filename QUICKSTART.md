Below are the main script sets related to this repository, along with instructions on how to run them. Some of them require accounts for API calls or compute usage, but everything in this repository, including training and data gathering, was done at ZERO COST. Because of this, Google Colab is used often in this repository due to its free GPU access. Please note that a Google account is required in order to use their free GPU tier, which is highly recommended.


Also remember to check the [Dog_List.txt](https://github.com/chrismontes22/Dog-Classification/blob/main/Dog_List.txt) file in order to see the full list of available dog breeds!


**-Inference-**

1. Multi Agent RAG for dog breeds - The best way to run the multi agent RAG is by heading over to my [Google Colab notebook](https://colab.research.google.com/drive/1QF40xb7qqraKYBwJpOyZNVXq2AepJPUM#scrollTo=fOPqmHy0ruwv), and changing the runtime type to a GPU. The free T4 GPU is sufficient. Once you have done so, all you have to do is select the Runtime > Run All.

2. From data colletion to a basic (not multi agent) RAG application - This application includes the full process from data collection, to processing, all the way to RAG output. You can get the .py script from my github [here](https://github.com/chrismontes22/Dog-Classification/blob/main/Multi_Agent_RAG/RAG_Pipeline_from_Data_Gathering_to_Inference.py) and run it after installing the necessary packages. Or you can run it from [Google Colab here](https://colab.research.google.com/drive/1by5UTMttZwW6xGGo89hmVNu2b90V3-HV#scrollTo=qpSPqD1AcNiH), then simply selecting Runtime > Run All. A GPU helps for the LLM portion of it, but can be ran on a CPU for inference as well in about two minutes, after the data has been embedded and the models have been loaded.

3. Image Classification App - The best way to see my image classification application is to head on over to my Hugging Face Space, [chrismontes/Dog_Breed_Identifier](https://huggingface.co/spaces/chrismontes/Dog_Breed_Identifier). Another way is to run a Docker Image to create a container with the necessary dependencies and inference script. If interested in using docker, please scroll to the bottom for details.

4. Fine tuned language model for fun fact output - To see the fine tuned model that outputs random facts about a dog breed, the best way is to see it in this [Colab notebook.](https://colab.research.google.com/drive/1mDUgQ--ztyFNzUG4O0S4WNlp8vnD-u-H#scrollTo=TXbi_oPFZ0EB) This notebook has the Image Classification from number 3 above, and passes the result (a dog breed) to the language model to output random facts about the breed.


**-Training-**

Above are the machine learning applications and ways to run them. Feel free to also check out my model training scripts:  -[Image Classification](https://github.com/chrismontes22/Dog-Classification/blob/main/Image_Classification_Pipeline/Training%20an%20Image%20Classification%20Model.ipynb)  -[Using Unsloth to fine tune an language model with custom data](https://github.com/chrismontes22/Dog-Classification/blob/main/Tuning_a_Language_Model/Tuning_the_Model.ipynb)


**-Other Data Collection Scripts-**

-[Text Data Pipeline for tuning an LLM. This is a modified version of the text data collection portion from the Basic RAG Pipeline. It downloads the data, and organizes it into JSON format ready to be used for fine tuning an LLM. It also uploads it to Hugging Face.](https://github.com/chrismontes22/Dog-Classification/blob/main/Tuning_a_Language_Model/Text%20Data%20Pipeline.py)  -[Use Google's API to download images by search term](https://github.com/chrismontes22/Dog-Classification/blob/main/Image_Classification_Pipeline/Image%20Data%20Download.py)

**-Load the Image Classification as a Docker Container-**

In order to use docker for this application, you must mount the picture file of your dog as you run the container. For both Bash and Powershell, simply replace INSERT_ABSOLUTE_PATH_TO_DOG_PHOTO_HERE with the absolute filepath of the dog picture, keeping the quotation marks.  It uses a Linux filesystem, so make sure to format your directory if you are using another OS. Spaces are ok to have in the file paths.

-Bash
First execute in a Bash terminal:
```
export IMAGE_PATH="/mnt/INSERT_ABSOLUTE_PATH_TO_DOG_PHOTO_HERE"
```

Then execute exactly the following:
```
docker run --rm -it -e IMAGE_PATH="$IMAGE_PATH" -v "${IMAGE_PATH}:${IMAGE_PATH}" chrismontes22/dog_image_classifier
```

-Powershell
Simply replace INSERT_ABSOLUTE_PATH_TO_DOG_PHOTO_HERE with your file path in Linux format and execute in a terminal:
```
$IMAGE_PATH="INSERT_ABSOLUTE_PATH_TO_DOG_PHOTO_HERE"; docker run --rm -it -e IMAGE_PATH=$IMAGE_PATH -v "${IMAGE_PATH}:${IMAGE_PATH}" chrismontes22/dog_image_classifier

```
