"""After you have downloaded the image data for the selected breeds, this script creates the test, 
train, and val folders in the main image folder. Next, it moves the appropriate number of images to the corresponding folders, then reorganizes them in 000 numerical format.
Finally, it zips the entire image dataset. If the label already exists, it instead moves all of the images to the respective label subfolder in the training folder."""
"""Please note that the dog_breeds here are file paths to folders, so if the folders are saved in a different path, you must change it in the
dog_breeds list accordingly."""
# Just fill in the list of breeds with the breeds that you ran for the image download pipeline.

"""The script assumes the dataset is organized in folders in the following manner:

* Parent
	+ test
		- labels
	+ train
		- labels
	+ val
		- labels

"""


import os
import shutil
import random
import logging
import datetime
import matplotlib.pyplot as plt
import pandas as pd

MOVE_NEW_DATA = True #This script is for moving data into the format above after downloading the images. If you want to just see the number of labels and images in each label, set both to False
ZIP_DATA = False #If you want to zip the new data for easier movement

# Define the path to the main folder and zip folder
main_folder = 'DD - Copy' #Parent directory
zipped_folder = 'DD3' #Output of the zip folder

#Turn the path to the subfolders into strings, which will be used later
train_dir = os.path.join(main_folder, "train")
test_dir = os.path.join(main_folder, "test")
valid_dir = os.path.join(main_folder, "valid")

if MOVE_NEW_DATA:
    logging.basicConfig(filename='Image Changes.txt', level=logging.INFO, format='%(message)s') #Config for logging
    
    dog_breeds = ['Afghan-Hound']  # Add all dog breeds to this list, make sure to put them at the same level as the workfolder and outside the dataset parent folder as they themselves are folder directories
    # Log the dog breeds and the current time, and how many breeds with the added data
    logging.info('***********************Data Folder Change***********************')
    logging.info('Time: ' + str(datetime.datetime.now()))
    logging.info('Dog Breeds: ' + str(dog_breeds))
    

    for dog_breed in dog_breeds:
        try:
            # Check if the dog breed folder already exists in the train directory
            if os.path.exists(os.path.join(main_folder, 'train', dog_breed)):
                # If the folder exists, move the images directly to the train folder
                image_files = [f for f in os.listdir(dog_breed) if os.path.isfile(os.path.join(dog_breed, f))]
                for file in image_files:
                    filename = os.path.basename(file)
                    dst_folder = os.path.join(main_folder, 'train', dog_breed)
                    dst_file = os.path.join(dst_folder, filename)
                    num = 1
                    while os.path.exists(dst_file):
                        filename, file_extension = os.path.splitext(filename)
                        new_filename = f"{filename}_{num}{file_extension}"
                        dst_file = os.path.join(dst_folder, new_filename)
                        num += 1
                    shutil.move(os.path.join(dog_breed, file), dst_file)
            else:
                # If the folder does not exist, split the data into test, valid, and train
                # Get the list of all image files in the dog_breed
                image_files = [f for f in os.listdir(dog_breed) if os.path.isfile(os.path.join(dog_breed, f))]

                # Shuffle the list to ensure randomness
                random.shuffle(image_files)

                # Split the data into test, valid, and train
                test_files = image_files[:10] #Sends the first 10 images to the test folder directory
                valid_files = image_files[10:20]  #Sends the next 10 to the valid directory
                train_files = image_files[20:]  #sends the rest to train directory

                # Define the new label name
                new_label_name = os.path.basename(dog_breed)

                # Create new label folders inside test, train, and valid folders
                os.makedirs(os.path.join(main_folder, 'test', new_label_name), exist_ok=True)
                os.makedirs(os.path.join(main_folder, 'valid', new_label_name), exist_ok=True)
                os.makedirs(os.path.join(main_folder, 'train', new_label_name), exist_ok=True)

                # Function to move files
                def move_files(files, dst_folder):
                    for file in files:
                        filename = os.path.basename(file)
                        dst_file = os.path.join(dst_folder, filename)
                        num = 1
                        while os.path.exists(dst_file):
                            filename, file_extension = os.path.splitext(filename)
                            new_filename = f"{filename}_{num}{file_extension}"
                            dst_file = os.path.join(dst_folder, new_filename)
                            num += 1
                        shutil.move(os.path.join(dog_breed, file), dst_file)
                # Move the files to the appropriate folders
                move_files(test_files, os.path.join(main_folder, 'test', new_label_name))
                move_files(valid_files, os.path.join(main_folder, 'valid', new_label_name))
                move_files(train_files, os.path.join(main_folder, 'train', new_label_name))

                # Sometimes image filenames are all over the place. This renames the files in numerical order in a folder. Loops for each breed
                for folder in ['test', 'valid', 'train']:
                    num = 1
                    for filename in os.listdir(os.path.join(main_folder, folder, new_label_name)):
                        # Check if the file is an image
                        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                            new_filename = "{:03d}.{}".format(num, filename.split(".")[-1])
                            while os.path.exists(os.path.join(main_folder, folder, new_label_name, new_filename)):
                                num += 1
                                new_filename = "{:03d}.{}".format(num, filename.split(".")[-1])
                            os.rename(os.path.join(main_folder, folder, new_label_name, filename), os.path.join(main_folder, folder, new_label_name, new_filename))
                            num += 1
    # Renames the files in the train folder to a numerical order
            train_folder = os.path.join(main_folder, 'train', dog_breed)
            image_files = [f for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))]
            for file in image_files:
                filename, file_extension = os.path.splitext(file)
                num = 1
                new_filename = "{:03d}{}".format(num, file_extension)
                dst_file = os.path.join(train_folder, new_filename)
                while os.path.exists(dst_file):
                    num += 1
                    new_filename = "{:03d}{}".format(num, file_extension)
                    dst_file = os.path.join(train_folder, new_filename)
                os.rename(os.path.join(train_folder, file), dst_file)
        except Exception as e:
            logging.error('Error occurred: {}'.format(e))
            raise
            #Useful for logging an error occurred and test train split folders were not able to be created
    
    logging.info('Number of labels dataset: ' + str(len(os.listdir(train_dir))))

def plot_histogram(directory, title):
    classes = os.listdir(directory)
    counts = [len(os.listdir(os.path.join(directory, cls))) for cls in classes]
    df = pd.DataFrame(list(zip(classes, counts)), columns=['Class', 'Count'])
    df = df.sort_values('Class')
    plt.figure(figsize=(12,6))
    bars = plt.bar(df['Class'], df['Count'])
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation='vertical')
    
    # Add the data value on top of each bar (utilitarian, not for a good looking graph)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom', fontsize=8)
    
    plt.show()

# Now you can call the function for each directory to graph the labels and the quantity in each
plot_histogram(train_dir, 'Number of images per class in training set')
plot_histogram(test_dir, 'Number of images per class in test set')
plot_histogram(valid_dir, 'Number of images per class in validation set')

print(f'Number of labels in training set: {len(os.listdir(train_dir))}')
print(f'Number of labels in test set: {len(os.listdir(test_dir))}')
print(f'Number of labels in validation set: {len(os.listdir(valid_dir))}')

#Makes a zip copy from the above data saved to main_folder variable
def zip_folder(folder_path, zip_name):
    zip_file_path = zip_name + '.zip'
    if os.path.exists(zip_file_path): #If the zip file already exists, it will not create or overwrite the file and log this
        logging.warning(f'Zip file {zip_file_path} already exists. Skipping...')
    else:
        shutil.make_archive(zip_name, 'zip', folder_path)
        logging.info('Zip file name: ' + zip_name + '.zip')
        logging.info('\n')
# Usage
if ZIP_DATA:    
    zip_folder(main_folder, zipped_folder)