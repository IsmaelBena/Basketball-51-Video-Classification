from torch.utils.data import Dataset
import cv2
import os
import torch
import numpy as np
import random

class LocalDataset():
    def __init__(self, dir, gray_scale = False, start_segment=0, end_segment=1):
        self.dir = dir
        self.targets = [target for target in os.listdir(self.dir)]
        self.x = []
        self.y = []
        self.start_segment = start_segment
        self.end_segment = end_segment
        self.gray_scale = gray_scale

    def load_dataset(self):
        print(f'Loading local cleaned data from segments {self.start_segment} -> {self.end_segment}...')
        for t_idx, target in enumerate(self.targets):
            files = [file for file in os.listdir(os.path.join(self.dir, target))] # if os.path.isfile(self.dir + target + file)]
            files_segment = files[int(self.start_segment*len(files)):int(self.end_segment*len(files))]
        
            for idx, file in enumerate(files_segment):
                print(f' - Label: {t_idx+1}/{len(self.targets)} | File: {idx+1}/{len(files_segment)} [{round(((idx+1)*100)/len(files_segment), 2)}%]      ', end='\r', flush=True)
                frames = []
                cap=cv2.VideoCapture(os.path.join(self.dir, target, file))
                while cap.isOpened():
                    ret, frame = cap.read()
                    # Once Video is over, break the loop
                    if not ret:
                        #print("ret is false?")
                        break
                    else:
                        if self.gray_scale:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                        else:
                             frames.append(frame)

                # print(np.array(frames).shape)
                if self.gray_scale:
                    self.x.append(np.array(frames))
                else:
                    self.x.append(np.transpose(np.array(frames), [3, 0, 1, 2]))
                self.y.append(t_idx)

            print(f' - Label: {t_idx+1}/{len(self.targets)} | File: {len(files_segment)}/{len(files_segment)} [100.00%]      ', end='\n')

        return self.x, self.y

class BasketballVideos(Dataset):
        def __init__(self, data):
            self.data = data
            self.videos = data[0]
            #print(len(self.videos))
            self.labels = data[1]

        def __getitem__(self, index):
                return {"videos": torch.tensor(self.videos[index], dtype = torch.float32), 
                        "labels": torch.tensor(self.labels[index], dtype = torch.int64)
                        }
        
        def __len__(self):
                return len(self.videos)
