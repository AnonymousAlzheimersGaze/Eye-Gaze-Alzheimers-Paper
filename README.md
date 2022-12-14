# Generating Attention Maps from Eye-gaze for the Diagnosis of Alzheimer's Disease

If you want to use our code please cite the paper ["Generating Attention Maps from Eye-gaze for the Diagnosis of Alzheimer's Disease"](https://openreview.net/forum?id=yL1qcv2Q0bC) 


# Folders

Scans data: code to generate scans visualization and their metadata

Attention Maps Generation: creation of the attention maps from fixation points

Data: pandas dataframe with metadata of scans

Networks: Code of the networks explored

Ready to Execute: Google colab notebooks with models

images: diagrams of this ReadMe file


# Dataset

allData.pkl - A Pandas dataframe contains data of each scan, including the fixation points and their duration. (It's the union of nc_allData.pkl with mci_allData.pkl and ad_allData.pkl). The scans are excluded.

![Explanation of dataframe](https://github.com/AnonymousAlzheimersGaze/Eye-Gaze-Alzheimers-Paper/blob/main/images/Explanation_Dataframe.png)


## Architecture of the Networks used 

### ResNet18

![Explanation of dataframe](https://github.com/AnonymousAlzheimersGaze/Eye-Gaze-Alzheimers-Paper/blob/main/images/ResNet.jpg)

### Deep Multiscale Network 

![Explanation of dataframe](https://github.com/AnonymousAlzheimersGaze/Eye-Gaze-Alzheimers-Paper/blob/main/images/DMS.jpg)

### U-Net

![Explanation of dataframe](https://github.com/AnonymousAlzheimersGaze/Eye-Gaze-Alzheimers-Paper/blob/main/images/U-Net.jpg)
