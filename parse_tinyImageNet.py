'''
Move images in the tinyImageNet validation set to separate folder with class names
by using the annotation text file
'''
import pandas as pd
import os
import shutil

class_df = pd.read_csv('assets/tiny-imagenet-200/val/val_annotations.txt', sep="\t", header=None, index_col=0)[1]
class_df = class_df.sort_values()

root_path = 'assets/tiny-imagenet-200/val'
cur_class_name = ''

for image_name, a_class_name in class_df.items():
    if cur_class_name != a_class_name:
        cur_class_name = a_class_name
        if not os.path.exists(f"{root_path}/{cur_class_name}/"):
            os.makedirs(f"{root_path}/{cur_class_name}/")

    source_path = f'{root_path}/images/{image_name}'
    destination_path = f"{root_path}/{cur_class_name}/{image_name}"

    # check if it is an existing directory
    if os.path.isfile(source_path):
        shutil.move(source_path, destination_path)

