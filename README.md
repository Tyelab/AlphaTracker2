# AlphaTracker2

## First Steps
We recommend running AlphaTracker2 from within a conda environment. This lets you keep all your installed libraries and programs separate.  Miniconda is a lightweight version of conda and we recommend using this one.  If you do not yet have a version of conda set up on your machine, please go here to find the latest installer for Miniconda3: https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links
Follow the installation instructions on the page to install Miniconda3.  

If you already have conda installed, please verify that it is version 3.0 or higher: `conda --version`.  If your version of conda is 2.x, please install version 3 as directed above.

## Installing AlphaTracker2
To start, `cd` into the `AlphaTracker2` folder. For now, the installation process will be a bit disjointed, but will be easier when it is on PyPi. 

#### 1. Type `conda env create -f environment.yml`
#### 2. Type `conda activate alphatracker-test2`
##### 3. Type `pip install .`
##### 4. If you have a GPU on your machine, check your cuda version:

   On Linux:  type `nvidia-smi` at the command prompt. 

   On Windows:  type `nvidia-smi.exe` in the Anaconda cmd window.  
   
   At the top of the print out, there should be a CUDA Version.  If your cuda version is less than 10.2, please head to https://pytorch.org/ to get the correct command for the step 5 below.  

#### 5. Install Pytorch.  You can try letting 

   If on Linux or Windows AND have GPU: `conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge`
   
   If on macOS: `conda install pytorch torchvision torchaudio -c pytorch`
   
   If on CPU machine only: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
   
#### 6. If no errors, yay! It is installed. Now, `cd` into a random directory that is not anywhere near the `AlphaTracker2` repo, and then do the following:
   
   `python`
   
   `import alphatracker2 as at2`
   
   You will see some PyTorch print statements pop up. If no errors, yay!
   
   

## Download sample data

#### If this is your first time running AlphaTracker2, please use the Demo dataset with the images and labeled json file for the sake of demonstration.  This will help troubleshoot if there were any errors in your installation.
Please follow instructions to download our sample data by clicking on the links and saving as directed:

Download the full set of 600 images and labels for 2 mice with 4 body parts each:

https://drive.google.com/file/d/15dR-vVCEsg2z7mEVzJOF9YDW6YioEU3N'

Save the file to 'sample_annotated_data.zip'
Extract all to the folder `sample_annotated_data\`; it should contain a directory called `\train9`.  The path where you have saved this data is used for `image_filepaths`  and `json_filepaths` in step 4 below.

Download the sample 3 min video for tracking on 2 mice:

https://drive.google.com/file/d/1N0JjazqW6JmBheLrn6RoDTSRXSPp1t4K

Save the file to `sample_annotated_data\demo.mp4`.  The path where you save this data will be used below in step 6, `vidpath`.

## Running AlphaTracker2


*Very important!* If you are on a Windows machine, open Anaconda prompt as administrator, or it won't work!!  
#### 1. Open a terminal/power shell or command window.  Type `conda activate alphatracker2`

#### 2. `python` 

#### 3. `import alphatracker2 as at2`

#### 4. Define your experiment variables.  In this example, the Windows user is named 'hali'.  You should replace these paths to the proper folders on your machine.

`save_loc = r'C:\Users\hali\Desktop\alphasave' # directory where experiment will be saved...ensure this directory exists!`

`exp_name = 'test1' # the name of the new directory that will be created in save_loc`

`import os; expname = os.path.join(save_loc, exp_name) # just to make it easier`

`image_filepaths = [r'C:\Users\hali\Desktop\alphasave\sample_annotated_data\train9\'] # enter a list of the location of your image directory`

`json_filepaths = [r'C:\Users\hali\Desktop\alphasave\sample_annotated_data\train9\train9.json'] # enter a list of the location of your json files`

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

`vidpath = 'path/to/demo.mp4'`

`results, runtime = at2.predict(vidpath, num_parts, experiment_name=expname)` 

note, calling this function requires a path to the video, and the experiment name will utilize the most recent model trained for both YOLO and sppe. 

Returns two objects: `results` (equivalent to alphapose-results.json file, untracked outputs), and `runtime` (a profile of the time it took to run inference)

#### 7. Run Tracking:

`tracked, no_detections = at2.track(results, num_obj)`

Returns two objects: `tracked` (equivalent to forvis-tracked), and `no_detections` (frames in video where no detections were made)



