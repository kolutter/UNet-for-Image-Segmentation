## U-Net for GIRAFE dataset (Tensorflow)

This repository provides an implementation of the basic **U-Net architecture** in Tensorflow for medical image segmentation on the GIRAFE dataset. 


## Installation

Clone the repository and install the dependencies:

```
git clone git@gitlab.gwdg.de:cvmd/deep-learning/unet_girafe.git
cd unet_girafe
pip install -r requirements.txt
```


## Project Structure

```
unet_girafe/
├── GIRAFE_dataset      # the complete dataset
├── model.py            # U-Net model architecture
├── train.py            # Training pipeline
├── dataloader.py       # Dataset & DataLoader utilities
├── utils.py            # Helper functions
├── main.py             # run this one
├── requirements.txt    # Dependencies
├── checkpoints/        # Saved model weights
└── model_summary.txt   # Model architecture summary
```

## Usage
As you can see in the structure, the dataset should be located on the same hierarchy level as the scripts and should have the name GIRAFE_dataset, otherwise it won't work. Since it is not part of the Repo, you have to downloaded it seperately and then add it there. Downloadlink: https://zenodo.org/records/13773163


Just run the main.py and a complete training workflow  + producing evalutation should be done autmatically. This shouldn't take longer than a few minutes... Consider changing device (cuda/mps/cpu)

```
python main.py 
```







