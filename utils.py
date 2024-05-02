import cv2
import numpy as np
import yaml
import os

def get_config(type = ''):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if type == 'model':
        return config['model_settings']
    elif type == 'data':
        return config['data_settings']
    else:
        return config

def pad_dataset(unclean_dir, clean_dir, fraction = 1):
    original_wd = os.getcwd()
    unclean_data_dir = original_wd + f'\\{unclean_dir}\\'
    clean_data_dir = original_wd + f'\\{clean_dir}\\'
    targets = [target for target in os.listdir(unclean_data_dir)]
    most_frames = 0
    frame_width = 0
    frame_height = 0
    fps = 0

    for target in targets:
        files = [file for file in os.listdir(unclean_data_dir + target)]
        print(f'Checking Label: {target}')
        for index, file in enumerate(files):
            print(f'Checking file: {file}                    ---           {round(((index+1)*100)/len(files), 2)}%', end='\r', flush=True)
            if index/len(files) > fraction:
                break
            else:
                frames = []
                cap=cv2.VideoCapture(f'{unclean_data_dir}\\{target}\\{file}')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    else:
                        frame_width = frame.shape[1]
                        frame_height = frame.shape[0]
                        frames.append(frame)
                        
                if len(frames) > most_frames:
                        most_frames = len(frames)

    print(frame_width, frame_height)

    if not os.path.exists(f'{os.getcwd()}\\dataset'):
        os.makedirs(f'{os.getcwd()}\\dataset')

    for target in targets:
        files = [file for file in os.listdir(unclean_data_dir + target)]
        if not os.path.exists(f'{original_wd}\\dataset\\{target}'):
            os.makedirs(f'{original_wd}\\dataset\\{target}')

        print(f'Padding Label: {target}')

        os.chdir(f'{original_wd}\\dataset\\{target}')
        for index, file in enumerate(files):
            print(f'Padding file: {file}                    ---           {round(((index+1)*100)/len(files), 2)}%', end='\r', flush=True)
            if index/len(files) > fraction:
                break
            else:
                frames = []
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output = cv2.VideoWriter(f'{file}_padded.avi', fourcc, fps, (frame_width, frame_height))
                print(f'padding {file} in {target}')
                print(f'saving to {os.getcwd()}\\{file}_padded.avi')
                cap=cv2.VideoCapture(f'{unclean_data_dir}\\{target}\\{file}')
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    else:
                        frame_dims = frame.shape
                        frames.append(frame)

                for frame in frames:
                    output.write(frame)

                if len(frames) < most_frames:
                    padding_frames = np.zeros((most_frames - len(frames), 240, 320, 3))
                    for padding in padding_frames:
                        output.write(padding)

            cap.release()
            output.release()
            cv2.destroyAllWindows()

        os.chdir(f'{original_wd}\\dataset')

    os.chdir(f'{original_wd}')
