import os, shutil
from midi2img import *
import random 

os.chdir(r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\dataset')

# Converting midi files to 106x106 images

for folder in os.listdir(os.getcwd()):    
    for midi_file in os.listdir(folder):
        midi_path = r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\dataset\\' + folder + r'\\' + midi_file
        if midi_file.split('.')[-1] == 'mid':
            midi2image(midi_path)
    
# Transferring generated images

os.chdir(r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\dataset')

for folder in os.listdir(os.getcwd()): 
    destination_path = r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\other_dataset\\' + folder
    for image_file in os.listdir(folder):
        if image_file.split('.')[-1] == 'png':
            image_path = r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\dataset\\' + folder + r'\\' + image_file
            
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
                
            shutil.move(image_path, destination_path)
    
    print("\n Process completed for folder " + folder)

home_dir = r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\DATASETS\Working\midi_images'
os.chdir(home_dir)

for artist in os.listdir(os.getcwd()):
    source_dir = os.getcwd() + r'\\' + artist
    total_imgs = len(os.listdir(artist))
    test_artist = int(total_imgs * 0.30)
    destination_path = r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\DATASETS\Working\Test'
    
    os.chdir(source_dir)
    for i in range(0, test_artist):
        shutil.move(random.choice(os.listdir(source_dir)), destination_path)
    os.chdir(home_dir)


