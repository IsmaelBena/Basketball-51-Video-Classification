# Basketball 51 Video Classification (INM705)

## About:
Type: Video Classification.

Input: 6-second clips of NBA streams where a player takes a shot.

The dataset can be found here: 
Place the dataset folder in the root directory of the repository.

Possible outputs: Player makes a free throw, Player misses a free throw, Player makes 2 points, Player misses 2 points, Player makes a midrange shot, Player misses a midrange shot, Player makes a 3-point shot, Player misses a 3-point shot.

Methods/Models:
3DConvnet -> LSTM
2DConvnet -> LSTM
3DConvnet -> Dense Layers
Two Stream 3DConvnet -> LSTM

# How To Run
Python Version: 3.9.5

## City Hyperion [linux server]
```
sh setup.sh
```

## Windows:
```
.\setup.ps1
```

Sometimes the scripts will not properly install all required libraries and a manual pip install is required for said libraries.

# Checkpoints
Final Model checkpoints are included in the repo under the trained_models folder.

# Training

Parameters controlable from config.yaml are:
 - Number of epochs
 - Batch size
 - Learning rate
 - Rate of decay

The wand api key should be added to the run_job.sh on linux or entered as requested by Windows.

During training, you will see the most recent epoch in a folder called checkpoints, and the final model will generate a pkl file in the trained models folder.

# Inference

There is an inference for each model in the inference.ipynb file, editing the video_directory variable at the top of the notebook allows you to choose a video from the dataset to run through the model. A video and results have been preselected.


