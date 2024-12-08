There are several ways to get the Image and Language models going.

1. The recommended way is to run it on the [public Google Colab notebook.](https://colab.research.google.com/drive/1mDUgQ--ztyFNzUG4O0S4WNlp8vnD-u-H#scrollTo=5VLagOTr_b4w) When you arrive, make sure to first change the runtime to one of the GPU's available in order to use the langauage model. Then install the extra dependencies, LoRA adapters, and pytorch dictionary by simply running the first two cells. Some test images and the dog list (.txt) also come with the download, but you can upload your own images for analysis by clicking on the folder (left-hand side) then the up arrow to upload. *Note that you might have to change the file to a .jpg extension. Any uploaded file in colab usually saves to the /content folder, but make sure to copy the path of the file. Finally replace the first line with your image's path and run the remaining cells.

2. You can also run a Docker Image to create a container with the necessary dependencies and inference script. You will need to modify the filepaths below (anything inside a <>) to your dog image, then copy the Bash or powershell script to your terminal.This will also create a csv file with data of the user answers saved to the local drive!

Bash
```
cd </absolute/path/to/the/folder/make/sure/to/omit/the/file/name/itself>;

IMAGE_PATH="<filename_of_image_without_folder_path>"; #make sure to include quotes


docker run --rm -it \
  -e IMAGE_PATH=$IMAGE_PATH \
  -v "${PWD}:/dogapp" \
  chrismontes22/dog_project_inference:latest \
  python /dogapp/Inference_Script.py
```

Powershell
```
$IMAGE_PATH="</absolute/path/to/file/including/filename>"; docker run --rm -it -e IMAGE_PATH=$IMAGE_PATH -v "${IMAGE_PATH}:${IMAGE_PATH}" chrismontes22/dog_project_inference:latest python Inference_Script.py  #make sure to include quotes around the full file directory
```

3. You may also access the Image Classification model easily (No Language Model) by simply going to my [Hugging Face Gradio Space.](https://huggingface.co/spaces/chrismontes/Dog_Breed_Identifier)



Remember to check the [Dog_List.txt](https://github.com/chrismontes22/Dog-Classification/blob/main/Dog_List.txt) file in order to see the full list of available dog breeds!