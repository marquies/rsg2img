import os
import shutil

# go through all files in subdirectories of output_path
# if file starts with generated and ends with .jpg copy it to target_path

output_path = "output"
target_path = "archive/2"

# make
if not os.path.exists(target_path):
    os.makedirs(target_path)


for root, dirs, files in os.walk(output_path):
    for file in files:
        if file.startswith('generated') and file.endswith('.jpeg'):
            folder = root.split('/')[-1]
            tfile = folder + "_" + file
            t_path = os.path.join(target_path, tfile)
            shutil.move(os.path.join(root, file), t_path)
            print(f"copied {file} to {t_path}")

        if file.startswith('prompt') and file.endswith('.txt'):
            folder = root.split('/')[-1]
            tfile = folder + "_" + file
            t_path = os.path.join(target_path, tfile)
            shutil.move(os.path.join(root, file), t_path)
            print(f"copied {file} to {t_path}")
