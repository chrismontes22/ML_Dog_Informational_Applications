**Welcome to my Dog Breed Informational repository!** 🐶

🐕 The goal of this repository is to provide various machine learning methods for information on dogs, over 70 breeds! Types of models include a multi agent RAG application, an image classifier, a full RAG pipeline and a language model that has been fine tuned with specific facts about these breeds. The image classification is hosted on an EC2 instance with AWS. 🐕

QUICKSTART

Below are the main script sets related to this repository, along with instructions on how to run them. Some of them require accounts for API calls or compute usage, but everything in this repository, including training and data gathering, can be done by anyone at ZERO COST. Even hosting my image classifier on AWS was free (for 12 months). Because of the zero cost, Google Colab is used often in this repository due to its free GPU access. Please note that a Google account is required in order to use their free GPU tier, which is highly recommended.


Also remember to check the [Dog_List.txt](https://github.com/chrismontes22/Dog-Classification/blob/main/Dog_List.txt) file in order to see the full list of available dog breeds.


**-Inference-**
1. Multi Agent RAG for dog breeds - To run the multi agent RAG, simply head over to my [Google Colab notebook](https://colab.research.google.com/drive/1QF40xb7qqraKYBwJpOyZNVXq2AepJPUM#scrollTo=fOPqmHy0ruwv) and change the runtime type to a GPU. The free T4 GPU is sufficient. Once you have done so, all you have to do is select the Runtime > Run All.

2. Image Classification App with AWS - The best way to see my image classification application is to head on over to the [dog classifier website](https://mldog.mooo.com). This model has been trained through transfer learning on thousands of images, over multiple training and data cleaning cycles. The website was deployed on a free tier EC2 instance, utilizing the FastAPI framework for the backend.

3. From data colletion to a basic (single agent) RAG application - This application includes the full process from data collection, to processing, all the way to RAG output. You can get the .py file from my github [here](https://github.com/chrismontes22/Dog-Classification/blob/main/Multi_Agent_RAG/RAG_Pipeline_from_Data_Gathering_to_Inference.py) and run it after installing the necessary packages. Or you can run it from [Google Colab here](https://colab.research.google.com/drive/1by5UTMttZwW6xGGo89hmVNu2b90V3-HV#scrollTo=qpSPqD1AcNiH), then simply selecting Runtime > Run All. A GPU helps for the LLM portion of it, but can be ran on a CPU for inference as well in about two minutes, after the data has been embedded and the models have been loaded.

4. Full model for inference* - The easiest way to run the full application is by pulling it from my Docker Hub. I created several options in regards to the LLM used because LLM's cause the docker image to grow exponentially. Only the CPU version of Torch and Torchvision are packaged due to large image sizes as well:

    The first two images create containers that are fully packaged with the model already included from Hugging Face. All you do is paste the line to a terminal, wait for the image to be pulled and container to be ran, then type "localhost" in a browser. Given the image sizes grow large when dealing with LLM's and Pytorch, the first one ("doglite") is recommended. Model from [prithivMLmods/Llama-SmolTalk-3](https://huggingface.co/prithivMLmods/Llama-SmolTalk-3.2-1B-Instruct). The second uses a slightly larger model, [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).

    Llama SmolTalk 1B
    
        docker run -it -p 80:80/tcp chrismontes22/doglite

    SmolLM2 1.7B

        docker run -it -p 80:80/tcp chrismontes22/full_dog_app_fastapi_inference`

    The following image comes without a model packaged, but you can use it by simply mounting a text generation model of your choice form Hugging Face when runing a container. Save the necessary LLM files to a folder and mount that folder to the container. Then go to "localhost" in a browser. Some recommended models to mount are the ones mentioned above:

    Bash

        docker run -p 80:80/tcp -e MODEL_DIR=/models -v /mnt/<path to your HF model folder>:/models chrismontes22/full_dog_app_fastapi:v1

    Powershell

        docker run -it -p 80:80/tcp -e MODEL_DIR=/models -v <path to your HF model folder>:/models chrismontes22/full_dog_app_fastapi:v1

5. Fine tuned language model for fun fact output - To see the fine tuned model that outputs random facts about a dog breed, the best way is to see it in this [Colab notebook.](https://colab.research.google.com/drive/1mDUgQ--ztyFNzUG4O0S4WNlp8vnD-u-H#scrollTo=TXbi_oPFZ0EB) This notebook has the Image Classification from number 2 above, and passes the result (a dog breed) to the language model to output random facts about the predicted breed. It also saves the user's input do gather data on the model's accuracy in csv form.


**-Training-**

Above are the machine learning applications and ways to run them. Feel free to also check out my model training scripts:  
-[Image Classification](https://github.com/chrismontes22/Dog-Classification/blob/main/Image_Classification_Pipeline/Training%20an%20Image%20Classification%20Model.ipynb)  
-[Using Unsloth to fine tune an language model with custom data](https://github.com/chrismontes22/Dog-Classification/blob/main/Tuning_a_Language_Model/Tuning_the_Model.ipynb)



**-Other Data Collection Scripts-**

-[Text Data Pipeline for tuning an LLM. This is a modified version of the text data collection portion from the Basic RAG Pipeline. It downloads the data, and organizes it into JSON format ready to be used for fine tuning an LLM. It also uploads it to Hugging Face.](https://github.com/chrismontes22/Dog-Classification/blob/main/Tuning_a_Language_Model/Text%20Data%20Pipeline.py)  

-[Use Google's API to download images by search term](https://github.com/chrismontes22/Dog-Classification/blob/main/Image_Classification_Pipeline/Image%20Data%20Download.py)



*The FullApp.py file can run as long as you have the required files alongside it as well (embed folder, vectordb folder, Dogrun2.Pth). If there is no Hugging Face Text Generation Model compatible model present, the script will pull the recommended model when not using Docker, HuggingFaceTB/SmolLM-1.7B. Making sure you have the necessary installations, which can be found in therequirements.txt plus any version of torch and torchvision, you can run the app on a terminal using Uvicorn, specifically uvicorn FullApp:app --reload. Then follow the numerical address provided by Uvicorn, usually http://127.0.0.1:8000.