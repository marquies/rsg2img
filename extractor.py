
import os
import subprocess
import json
import time
import datetime
import re

def get_video_info(video_path):
    # Führe ffprobe aus, um Videoinformationen zu erhalten
    #String command = "ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 " + f.getAbsolutePath();
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration,avg_frame_rate,width,height", "-of", "json", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Parse die JSON-Ausgabe
    info = json.loads(result.stdout)
    duration = float(info['streams'][0]['duration'])
    frame_rate = eval(info['streams'][0]['avg_frame_rate'])
    width = info['streams'][0]['width']
    height = info['streams'][0]['height']
    
    return duration, frame_rate, (width, height)


def extract_frames(video_path, output_path):
    matcher = re.compile("[ ]Unity[ ]*: [1-3-]")
    files = os.listdir(video_path)
    
    # Dateien filtern, die mit 'ohd' beginnen und '.txt' enden
    ohd_files = [f for f in files if f.startswith('odh') and f.endswith('.txt')]
    
    # Gefilterte Dateinamen ausgeben
    if len(ohd_files) == 0:
        print("Keine Dateien gefunden in " + video_path + " die mit 'ohd' beginnen und '.txt' enden.")
        return
    if len(ohd_files) > 1:
        print("Mehr als eine Datei gefunden in " + video_path)
        return
    logfile = ohd_files[0]
    # odh_logs_2024-03-22 09.19.32.909.txt
    #Extract timestamp from logfile name
    timestamp = logfile[9:32]
    #print(timestamp)

    # Öffne die Logdatei und filtere die Zeilen
    with open(os.path.join(video_path, logfile), 'r') as file:
        lines = file.readlines()
    
    filtered_lines = [line for line in lines if "--- Start Scene Graph" in line]

    scene_graphs = []
    
    frame_timestamps = []
    # Gefilterte Zeilen ausgeben
    for line in filtered_lines:
        #print(line.strip())
        frame_timestamps.append(line[0:12])
        # find line in lines
        print(lines.index(line))
        #find next line containing --- End Scene Graph
        end_line = lines.index(line) + 1
        while not "--- End Scene Graph" in lines[end_line]:
            end_line += 1
            if end_line >= len(lines):
                break
        #print(end_line)
        #print(lines[end_line])
        #print(lines[end_line+1])
        #extract the lines between the start and end line
        graph = lines[lines.index(line):end_line+1]
        #filter out all lines not containing the word "Unity"
        #unity_lines = [line for line in graph if "Unity" in line]
        #21:09:24.921 16887 16924 I Unity         : 3 GameObject{ "state":"3", "depth":"0", "name":"GameObject", "path":"/GameObject", "t_x":"0", "t_y" : "1.48644", "t_z": "2.93",  "s_x":"1", "s_y" : "1", "s_z": "1",  "r_x":"0", "r_y" : "0", "r_z": "0" , "r_w": "1",  "b_c_x":"0", "b_c_y" : "1.48644", "b_c_z": "2.93", "b_s_x":"0", "b_s_y" : "0", "b_s_z": "0"}

        # strip all of the line until the third occurence of ":" + 1

        unity_lines = [line[line.find(":", line.find(":", line.find(":")+1)+1)+1:].strip() for line in graph if matcher.search(line)]

        
        
        
        #unity_lines = [line[line.find(":")+1:].strip() for line in unity_lines]
        
        # add unity lines as a single entry to the scene_graphs list
        scene_graphs.append(unity_lines)

        




    # Dateien filtern, die mit '.mp4' enden
    mp4_files = [f for f in files if f.endswith('.mp4')]
    
    if len(mp4_files) == 0:
        print("Keine Dateien gefunden in " + video_path + " die mit '.mp4' enden.")
        return
    if len(mp4_files) > 1:
        print("Mehr als eine Video Datei gefunden in " + video_path)
        return
    videofile = mp4_files[0]
    print(videofile)
    # com.oculus.vrshell-20240322-101812.mp4
    # com.DefaultCompany.MyProject-20240502-195431.mp4
    #Extract timestamp from videofile name
    video_timestamp = videofile[videofile.find('-')+1:videofile.find('.mp4')]

    #video_timestamp = videofile[19:34]

    #convert string to time 
    video_timestamp = time.mktime(datetime.datetime.strptime(video_timestamp, "%Y%m%d-%H%M%S").timetuple())
    print(video_timestamp)

    

    
    # get length of the video
    video_length = 0
    # get the frame rate of the video
    frame_rate = 0
    # get the resolution of the video
    resolution = (0,0)
    
    # Führe ffprobe aus, um Videoinformationen zu erhalten
    video_length, frame_rate, resolution = get_video_info(os.path.join(video_path, videofile))
    
    #print(f"Videolänge: {video_length} Sekunden")
    #print(f"Framerate: {frame_rate} fps")
    #print(f"Auflösung: {resolution[0]}x{resolution[1]}")
    
    # substract the video_length from the timestamp to get the start time of the video as time object
    #start_time = datetime.datetime.fromtimestamp(video_timestamp - video_length)
    start_time = datetime.datetime.fromtimestamp(video_timestamp)
    #print(start_time)
    #print if start time is before log file time
    #print(start_time > datetime.datetime.strptime(timestamp, "%Y-%m-%d %H.%M.%S.%f"))

    # Fix timzone differences in the files.
    # set start_time hour to the same as timestamp hour
    #start_time = start_time.replace(hour=int(timestamp[11:13]))
    #print(str(start_time) + " - " +  frame_timestamps[0])
    
    mappings = []

    i = -1

    for frame_timestamp in frame_timestamps:
        i += 1
        # convert frame timestamp to time object
        frame_time = datetime.datetime.strptime(frame_timestamp, "%H:%M:%S.%f")
        frame_time = frame_time.replace(year=start_time.year, month=start_time.month, day=start_time.day)
        # calculate the time difference between the frame and the start time
        time_diff = frame_time - start_time
        #print(time_diff)

        # calculate the frame number
        frame_number = int(time_diff.total_seconds() * frame_rate)
        #print(f"Frame {frame_number} at {frame_timestamp}")
        if (frame_number < 0): 
            print(f"Frame {frame_number} at {frame_timestamp} is before the video start time")
            continue
        if (frame_number > video_length * frame_rate):
            print(f"Frame {frame_number} at {frame_timestamp} is after the video end time")
            continue

        # Extract the frame as jpeg from the video
        frame_filename = f"frame_{frame_number}.jpg"
        frame_path = os.path.join(output_path, frame_filename)
        result = subprocess.run(
            #["ffmpeg", "-i", os.path.join(video_path, videofile), "-vf", f"select=gte(n\\,{frame_number})", "-vframes", "1", frame_path],
            ["ffmpeg", "-i", os.path.join(video_path, videofile), "-vf", f"select=gte(n\\,{frame_number})", "-vframes", "1", frame_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        mappings.append((frame_number, frame_timestamp))

        graph = scene_graphs[i]
        #write scene graph to file
        with open(os.path.join(output_path, f"frame_{frame_number}.txt"), 'w') as file:
            for line in graph:
                file.write(f"{line}\n")

        #prtint subprocess output
        #print(result.stdout)
        #print(result.stderr)
        #print(f"Frame {frame_number} gespeichert als {frame_path}")	
    #write mappings to file
    with open(os.path.join(output_path, "mappings.txt"), 'w') as file:
        for mapping in mappings:
            file.write(f"{mapping[0]} {mapping[1]}\n")

def main():
    # Hauptlogik hier einfügen
    #video_folder = "/Users/breucking/Library/CloudStorage/OneDrive-Persönlich/@Fernuni/Dissertation/Experimente/MVR Prototyp/v5/"
    video_folder = "/Volumes/Storage/Diss-Dataset/v5"
    # for each folder in the video_path, extract the frames
    for folder in os.listdir(video_folder):
        if not os.path.isdir(os.path.join(video_folder, folder)):
            continue
        if folder != "recording6":# and folder != "recording44":
            continue
        video_path = os.path.join(video_folder, folder)
        output_path = "output/" + video_path.split('/')[-1]
        if os.path.exists(output_path):
            # Empty the output directory
            for file in os.listdir(output_path):
                os.remove(os.path.join(output_path, file))
        else:
            os.makedirs(output_path)
        
        extract_frames(video_path, output_path)


if __name__ == "__main__":
	main()