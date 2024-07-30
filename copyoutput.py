
# read directory output and subdirs, copy all files callesd frame_*.jpg to data/metaverse/original
import os
import shutil

output_path = "output"
target_path = "data/metaverse/original"
target_path_generated = "data/metaverse/generated"



# get all jpg files in output directory and subdirectories
for root, dirs, files in os.walk(output_path):
    for file in files:
        if file.startswith('frame') and file.endswith('.jpg'):
            # Get directory
            #folder = os.path.join(target_path, )
            folder = root.split('/')[-1]
            # get full path of file to copy

            #source_file_path = os.path.join(root, file)


            # add folder to filename folder_filename
            tfile = folder + "_" + file
            t_path = os.path.join(target_path, tfile)
            shutil.copy(os.path.join(root, file), t_path)

            #t_path = os.path.join(target_path_generated, tfile)
            #shutil.copy(os.path.join(root, file), t_path)
            print(f"copied {file} to {t_path}")
        if file.startswith('generated_') and file.endswith('.jpeg'):
            # get filename to frame_*.jpg
            frame = file.split('_')[2]
            #frame = frame.split('.')[0]
            target_file = f"frame_{frame}.jpg"            
            # Get directory
            folder = root.split('/')[-1]
            tfile = folder + "_" + target_file
            t_path = os.path.join(target_path_generated, tfile)
            # add folder to filename folder_filename
            #target_path = os.path.join(folder, tfile)
            shutil.copy(os.path.join(root, file), t_path)
            print(f"copied {file}, renamed target_file to {t_path}")

print("done")