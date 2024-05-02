import cv2
import numpy as np
import yaml
import os
import pandas as pd
from pathlib import Path
import shutil
import random

def get_config(type = ''):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if type == 'model':
        return config['model_settings']
    elif type == 'data':
        return config['data_settings']
    else:
        return config


def load_metadata(csv_name):
    data = pd.read_csv(f'{csv_name}.csv')

    max_frames = data['num_frames'].max()
    fps = data['fps'].max()
    frame_dimensions = (data['width'].max(), data['height'].max())
    file_names = data['file_name'].values
    labels = data['label'].values

    metadata = {'max_frames': max_frames,
        'fps': fps,
        'frame_dimensions': frame_dimensions,
        'file_names': file_names,
        'labels': labels
        }

    return metadata


# Gerenate csv of dataset
def gen_csv(target_folder_name, csv_name):
    original_dir = os.path.join(os.getcwd(), target_folder_name)
    targets = [target for target in os.listdir(original_dir)]
    
    data = pd.DataFrame(columns=['file_name', 'fps', 'width','height', 'num_frames', 'label'])

    for target in targets:
        files = [file for file in os.listdir(os.path.join(original_dir, target))]
        print(f'\nExtracting videos under label: {target}')
        for index, file in enumerate(files):
            print(f'Extracting file: {file}                    ---           {round(((index+1)*100)/len(files), 2)}%', end='\r', flush=True)
            frames = []
            cap=cv2.VideoCapture(os.path.join(original_dir, target, file))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    frame_width = frame.shape[1]
                    frame_height = frame.shape[0]
                    frames.append(frame)

            data = pd.concat([data, pd.DataFrame({
                'file_name': file,
                'fps': fps,
                'width': frame_width,
                'height': frame_height,
                'num_frames': len(frames),
                'label': target
            }, index=[0])],)

    data_to_csv = data.to_csv(f'{csv_name}.csv', index = False)



def remove_frames(dataset_dir, output_dir, divide_by, metadata_file=''):
    original_wd = os.getcwd()
    original_dir = os.path.join(original_wd, dataset_dir)
    generated_dir = os.path.join(original_wd, output_dir)
    targets = [target for target in os.listdir(original_dir)]

    if metadata_file != '':
        print(f'Retrieving metadata from {metadata_file}.csv')
        metadata = load_metadata(metadata_file)
        frame_width, frame_height = metadata['frame_dimensions']
        fps = metadata['fps']
        print(f'Found: \n - Frame dimensions: {(frame_width, frame_height)}\n - FPS: {fps}')
    else:
        files = [file for file in os.listdir(os.path.join(original_dir, targets[0]))]
        for file in files:
            print(f'No metadata passed, analysing {file} as a reference.')
            cap=cv2.VideoCapture(os.path.join(original_dir, targets[0], file))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    frame_width = frame.shape[1]
                    frame_height = frame.shape[0]
                    break

            print(f'Found: \n - Frame dimensions: {(frame_width, frame_height)}\n - FPS: {fps}')
            break

    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)

    for target in targets:
        files = [file for file in os.listdir(os.path.join(original_dir, target))]
        if not os.path.exists(os.path.join(generated_dir, target)):
            os.makedirs(os.path.join(generated_dir, target))

        print(f'\nRemoving frames of videos within: {os.path.join(original_dir, target)}')

        os.chdir(os.path.join(generated_dir, target))
        for index, file in enumerate(files):
            print(f' - Removing frames from video: {file}                    ---           {round(((index+1)*100)/len(files), 2)}%', end='\r', flush=True)
            frames = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output = cv2.VideoWriter(f'{Path(file).stem}_t_{divide_by}.avi', fourcc, fps, (frame_width, frame_height))
            cap=cv2.VideoCapture(os.path.join(original_dir, target, file))
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    if (frame_num % divide_by == 0):
                        frames.append(frame)
                    frame_num += 1

            for frame in frames:
                output.write(frame)

            cap.release()
            output.release()
            cv2.destroyAllWindows()

        os.chdir(generated_dir)

    os.chdir(original_wd)




def pad_dataset(dataset_dir, output_dir, metadata_filename):
    original_wd = os.getcwd()
    original_dir = os.path.join(original_wd, dataset_dir)
    generated_dir = os.path.join(original_wd, output_dir)
    targets = [target for target in os.listdir(original_dir)]

    print(f'Reading metadata...')
    metadata = load_metadata(metadata_filename)
    most_frames = metadata['max_frames']
    frame_width, frame_height = metadata['frame_dimensions']
    fps = metadata['fps']
    print(f'Found:\n - Max frames: {most_frames}\n - Frame dimensions: ({frame_width, frame_height})\n - FPS: {fps}')

    # for target in targets:
    #     files = [file for file in os.listdir(unclean_data_dir + target)]
    #     print(f'\nChecking Label: {target}')
    #     for index, file in enumerate(files):
    #         print(f'Checking file: {file}                    ---           {round(((index+1)*100)/len(files), 2)}%', end='\r', flush=True)
    #         if index/len(files) > fraction:
    #             break
    #         else:
    #             frames = []
    #             cap=cv2.VideoCapture(f'{unclean_data_dir}\\{target}\\{file}')
    #             fps = int(cap.get(cv2.CAP_PROP_FPS))
    #             while cap.isOpened():
    #                 ret, frame = cap.read()
    #                 if not ret:
    #                     break
    #                 else:
    #                     frame_width = frame.shape[1]
    #                     frame_height = frame.shape[0]
    #                     frames.append(frame)
                        
    #             if len(frames) > most_frames:
    #                     most_frames = len(frames)

    # print(frame_width, frame_height)

    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)

    for target in targets:
        files = [file for file in os.listdir(os.path.join(original_dir, target))]
        if not os.path.exists(os.path.join(generated_dir, target)):
            os.makedirs(os.path.join(generated_dir, target))

        print(f'\nPadding videos found in : {os.path.join(generated_dir, target)}')


        os.chdir(os.path.join(generated_dir, target))
        for index, file in enumerate(files):
            print(f' - Padding file: {file}                    ---           {round(((index+1)*100)/len(files), 2)}%       ', end='\r', flush=True)
            frames = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output = cv2.VideoWriter(f'{Path(file).stem}_padded.avi', fourcc, fps, (frame_width, frame_height))
            # print(f'padding {file} in {target}')
            # print(f'saving to {os.getcwd()}\\{file}_padded.avi')
            cap=cv2.VideoCapture(os.path.join(original_dir, target, file))
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

        os.chdir(generated_dir)

    os.chdir(original_wd)
    print(f'\nDeleting {original_dir}.')
    shutil.rmtree(original_dir)


def split_train_val_test(train_fraction, val_fraction, test_fraction, original_folder, folder_to_generate, metadata_filename):
    if train_fraction + val_fraction + test_fraction != 1:
        raise Exception("Sum of Training/Validation/Test split does not equal 1")
    else:
        original_dir = os.path.join(os.getcwd(), original_folder)
        generated_dir = os.path.join(os.getcwd(), folder_to_generate)
        metadata = load_metadata(metadata_filename)
        targets = [target for target in os.listdir(original_dir)]

        files = metadata['file_names']
        labels = metadata['labels']

        print("Shuffling files")
        files_labels = list(zip(files, labels))
        random.shuffle(files_labels)

        shuffled_files, shuffled_labels = zip(*files_labels)

        print("Checking if target directory exists.")
        for target in targets:
            if not os.path.exists(os.path.join(generated_dir, "train", target)):
                print(f'Generating {os.path.join(generated_dir, "train", target)}')
                os.makedirs(os.path.join(generated_dir, "train", target))
            else:
                print(f'Confirmed the existence of {os.path.join(generated_dir, "train", target)}')
            if not os.path.exists(os.path.join(generated_dir, "val", target)):
                print(f'Generating {os.path.join(generated_dir, "train", target)}')
                os.makedirs(os.path.join(generated_dir, "val", target))
            else:
                print(f'Confirmed the existence of {os.path.join(generated_dir, "train", target)}')
            if not os.path.exists(os.path.join(generated_dir, "test", target)):
                print(f'Generating {os.path.join(generated_dir, "train", target)}')
                os.makedirs(os.path.join(generated_dir, "test", target))
            else:
                print(f'Confirmed the existence of {os.path.join(generated_dir, "train", target)}')            

        print(f'Moving {len(shuffled_files)} files from: {original_dir} to {generated_dir}')
        for idx, file in enumerate(shuffled_files):
            if idx/len(shuffled_files) < train_fraction:
                print(f' - Moving training file: {file}                    ---           {round(((idx+1)*100)/len(shuffled_files), 2)}%         ', end='\r', flush=True)
                shutil.move(os.path.join(original_dir, shuffled_labels[idx], file), os.path.join(generated_dir, "train", shuffled_labels[idx]))
            elif idx/len(shuffled_files) < train_fraction + val_fraction:
                print(f' - Moving validation file: {file}                    ---           {round(((idx+1)*100)/len(shuffled_files), 2)}%       ', end='\r', flush=True)
                shutil.move(os.path.join(original_dir, shuffled_labels[idx], file), os.path.join(generated_dir, "val", shuffled_labels[idx]))
            else:
                print(f' - Moving testing file: {file}                    ---           {round(((idx+1)*100)/len(shuffled_files), 2)}%          ', end='\r', flush=True)
                shutil.move(os.path.join(original_dir, shuffled_labels[idx], file), os.path.join(generated_dir, "test", shuffled_labels[idx]))
        print("\n Files split and moved")

        print(f'\nDeleting {original_dir}.')
        shutil.rmtree(original_dir)