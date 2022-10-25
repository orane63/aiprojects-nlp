# ACM AI Projects Skeleton Code

## Setup

1. Create a new conda environment. (Follow instructions given to you by officers or refer to the Conda reference doc.)

2. Install PyTorch. Go to https://pytorch.org/. Then, scroll down to 'Install Pytorch' and select the right configuration for your machine. Lastly, copy the command that is provided and run it when you have the correct Conda environment activated.

3. As you work on the project, you will end up installing many more packages.

## Running the Skeleton Code

### Running the Code Locally

After activating your conda environment, run the following command:

```
conda activate <NAME OF ENVIRONMENT>
python main.py
```

### Running the Code on Google Colab

[This notebook](https://colab.research.google.com/drive/1cyIbRoubY0ZpXoGvOVSgO8kM4gFM31Op?usp=sharing) is the original made by ACM AI.

[This notebook](https://colab.research.google.com/drive/1JRmrT_FSVYsozpP4P_xjOBR7rERmaRP7?usp=sharing) is Joshua's copy, you should have edit access.

**Note:** Google Colab may terminate your session after a few hours, so be careful. Consider adding torch.save to your training loop to save model weights regularly.

### Running the Code on Kaggle

[This notebook](https://www.kaggle.com/franktzheng/acm-ai-projects-kaggle-skeleton) will walk you through setting the skeleton code up on Kaggle.

1. Navigate to the [code tab of the Kaggle competition](https://www.kaggle.com/c/cassava-leaf-disease-classification/code). Click on the "New Notebook" button to create a new notebook. The dataset should be automatically loaded in the `/kaggle/input` folder.

2. To use the GPU, click the three dots in the top-right corner and select Accelerator > GPU.

3. To access your code, run the following command (replacing the URL):

   ```
   !git clone "<my-github-repo-url-here>"
   ```

   This should clone your repository into the `/kaggle/working` folder.

4. Change directories into your repository by running the command:

   ```
   cd <name of your repository>
   ```

5. You should now be able to import your code normally. For instance, the following code will import the starting code:

   ```python
   import constants
   from datasets.StartingDataset import StartingDataset
   from networks.StartingNetwork import StartingNetwork
   from train_functions.starting_train import starting_train
   ```

6. If you want your code to run without keeping the tab open, you can click on "Save version" and commit your code. Make sure to save any outputs (e.g. log files) to the `/kaggle/working`, and you should be able to access them in the future.

**IMPORTANT:** If you want to pull new changes in the Kaggle notebook, first run `!git pull`, and then RESTART your notebook (Run > Restart & clear all outputs).

## Downloading the Dataset From Kaggle

### Method 1: Downloading from kaggle.com

1. Go to [kaggle.com](kaggle.com) and create anaccount.

2. Join either the [Quora NLP Competition](https://www.kaggle.com/competitions/quora-insincere-questions-classification/data).

3. In the data tab, you should be able to download the data as a zip file.
