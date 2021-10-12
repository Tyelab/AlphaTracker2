# AlphaTracker2


## Basic Setup
To start, `cd` into the `AlphaTracker2` folder. For now, the installation process will be a bit disjointed, but will be easier when it is on PyPi. 

#### 1. Type `conda env create -f environment.yml`
#### 2. Type `conda activate alphatracker-test2`
##### 3. Type `pip install . --use-feature=in-tree-build`
#### 4. See below:

   If on Linux or Windows AND have GPU: `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch`
   
   If on macOS: `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch`
   
   If on CPU machine only: `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch`
   
#### 5. If no errors, yay! It is installed. Now, `cd` into a random directory that is not anywhere near the `AlphaTracker2` repo, and then do the following:
   
   `python`
   
   `import alphatracker2 as at2`
   
   You will see some PyTorch print statements pop up. If no errors, yay!
   
   

## Basic Usage

#### 1. To begin, use the Demo dataset with the images and labeled json file for the sake of demonstration.
Please follow instructions to download the original Demo dataset from the Alphatracker 1.0 repository:    
   https://github.com/ZexinChen/AlphaTracker/blob/main/Manual/Tracking.md#demo-data-for-training
   The location where you save this data is what you will use for variables `image_filepaths` and `json_filepaths` in step 4 below.
   
*Very important!* If you are on a Windows machine, open Anaconda prompt as administrator, or it won't work!!

#### 2. `python`

#### 3. `import alphatracker2 as at2`

#### 4. Define your experiment variables:

`save_loc = r'C:\Users\hali\Desktop\alphasave' # directory where experiment will be saved...ensure this directory exists!`

`exp_name = 'test1' # the name of the new directory that will be created in save_loc`

`import os; expname = os.path.join(save_loc, exp_name) # just to make it easier`

`image_filepaths = [r'C:\Users\hali\Desktop\alphasave\single_line'] # enter a list of the location of your image directory`

`json_filepaths = [r'C:\Users\hali\Desktop\alphasave\single_line\multi_person.json'] # enter a list of the location of your json files`

`num_obj = 2 # number of animals..keep this an integer! (there are 2 mice in the demo training data)`

`num_parts = 4 # number of poses`

`split = 0.9 # train/validation split`

`extension = 'jpg'` # this should match the type of image found in your image_filepaths folder, e.g., 'jpg' or 'png'

`at2.create_experiment(save_loc, exp_name, image_filepaths, json_filepaths, num_obj, num_parts, split, extension) # a new folder should be created inside save_loc with your experiment name
 `


#### 5. Start training: 

Train pose estimator: `at2.train_pose_estimator(expname, epochs=50, nThreads=0)`

note, this will train from scratch so don't be surprised with low accuracy. to use pretrained model, we will have to figure out where to store it and download it when the train function is called

Train object detector: `at2.train_object_detector(expname, model_type='yolov5s', epochs=200)`

##### 6. Inference/Tracking:

`vidpath = 'path/to/video.mp4'`

`results, runtime = at2.predict(vidpath, num_parts, experiment_name=expname)` 

note, calling this function requires a path to the video, and the experiment name will utilize the most recent model trained for both YOLO and sppe. 

Returns two objects: `results` (equivalent to alphapose-results.json file, untracked outputs), and `runtime` (a profile of the time it took to run inference)

#### 7. Run Tracking:

`tracked, no_detections = at2.track(results, num_obj)`

Returns two objects: `tracked` (equivalent to forvis-tracked), and `no_detections` (frames in video where no detections were made)



