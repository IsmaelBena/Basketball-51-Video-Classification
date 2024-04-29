from torch.utils.data import Dataset
import cv2
import os
import torch
import numpy as np

class LocalDataset():
    def __init__(self, dir, fraction=1):
        self.dir = dir
        self.targets = [target for target in os.listdir(self.dir)]
        self.x = []
        self.y = []
        self.fraction = fraction
        self.most_frames = 0

    def load_dataset(self):

        for target in self.targets:
            files = [file for file in os.listdir(self.dir + target)] # if os.path.isfile(self.dir + target + file)]
            #print(self.dir + target)
        

            for index, file in enumerate(files):
                if index/len(files) > self.fraction:
                     break
                else:
                    frames = []
                    cap=cv2.VideoCapture(f'{self.dir}\\{target}\\{file}')
                    while cap.isOpened():
                        ret, frame = cap.read()
                        # Once Video is over, break the loop
                        if not ret:
                            #print("ret is false?")
                            break
                        else:
                            #print('ret is true?')
                            #print(frame)
                            frames.append(frame)
                            
                    self.y.append(self.targets.index(target))
                    if len(frames) > self.most_frames:
                         self.most_frames = len(frames)
                    frames = np.array(frames)
                    # print(frames.shape)
                    self.x.append(frames)                    
                    # print(f'{target}: {len(frames)}')
                    # (3x320x240xmissing_frames)

        print(len(self.x))
        return self.x, self.y, self.most_frames


class BasketballVideos(Dataset):
        def __init__(self, data):
            self.data = data
            self.videos = data[0]
            print(len(self.videos))
            self.labels = data[1]
            self.most_frames = data[2]

        def __getitem__(self, index):

                if len(self.videos[index]) < self.most_frames:
                    # self.videos[index] = self.videos[index] + np.zeros((self.most_frames - len(self.videos[index]), 240, 320, 3))
                    self.videos[index] = np.append(self.videos[index], np.zeros((self.most_frames - len(self.videos[index]), 240, 320, 3)), axis=0)
                    # print(f'new video shape: {self.videos[index].shape}')
                    #print(self.videos[index].shape())

                    #self.videos[index].append()

                self.videos[index] = np.moveaxis(self.videos[index], -1, 0)
                # print(self.videos[index].shape)

                return {"videos": torch.tensor(self.videos[index], dtype = torch.float32), 
                        "labels": torch.tensor(self.labels[index], dtype = torch.int64)
                        }
        
        def __len__(self):
                return len(self.videos)
