1. The best way to run the full application is by pulling it from my dockerhub. There are two versions:

-First, there is a smaller version of the AI dog informational application. Unlike the other one which is fully packaged, this one requires a Text Generation model from Hugging Face mounted to the container. The recommended model is HuggingFaceTB/SmolLM-1.7B. In order to do so, first save a Text Generation model from HF into a folder. Then run the one of the following in a terminal-
    Bash: docker run -p 80:80/tcp -e MODEL_DIR=/models -v /mnt/<path to your HF model folder>:/models chrismontes22/full_dog_app_fastapi:v1
    Powershell: docker run -it -p 80:80/tcp -e MODEL_DIR=/models -v <path to your HF model folder>:/models chrismontes22/full_dog_app_fastapi:v1

-There is a second, much larger prebuilt docker image. It already includes the HF model and everything necessary, so you simply copy and paste the line below inside a terminal and go to "localhost" in your browser.
    docker run -it -p 80:80/tcp chrismontes22/full_dog_app_fastapi_inference


2. The FullApp.py file can run as long as you have the required files alongside it (embed folder, vectordb folder, Dogrun2.Pth). If there is no Hugging Face Text Generation Model compatible model present, the script will pull the recommended model, HuggingFaceTB/SmolLM-1.7B. Making sure you have the necessary installations, which can be found in therequirements.txt plus any version of torch and torchvision, you can run the app on a terminal using Uvicorn, specifically uvicorn FullApp:app --reload. Then follow the address provided by Uvicorn, usually http://127.0.0.1:8000.

3. Everything in this subfolder is sufficient to recreate the docker image by building it with the provided dockerfile. However, you will need to pull a compatible Hugging Face Text Generation model and mount the folder to the built container, recommended model mentioned above