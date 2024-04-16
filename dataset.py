from torch.utils.data import Dataset
import cv2
import os
import torch

class LocalDataset():
    def __init__(self, dir):
        self.dir = dir
        self.targets = [target for target in os.listdir(self.dir)]
        self.x = []
        self.y = []

    def load_dataset(self):

        for target in self.targets:
            files = [file for file in os.listdir(self.dir + target)] # if os.path.isfile(self.dir + target + file)]
            #print(self.dir + target)
        

            for file in files:
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
                self.x.append(frames)
                break
            break
        return self.x, self.y


class BasketballVideos(Dataset):
        def __init__(self, data):
            self.data = data
            self.videos = data[0]
            self.labels = data[1]


        def __getitem__(self, index):

                return {"videos": torch.tensor(self.videos[index], dtype = torch.long), 
                        "labels": torch.tensor(self.labels[index], dtype = torch.long)
                        }
        
        def __len__(self):
                return len(self.data)
