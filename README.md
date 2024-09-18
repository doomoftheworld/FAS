# Supplementary materials for: 
# EFOMS: Effective, Efficient, and Environmentally Friendly Out-of-Model-Scope Detection Methodology

## This repo contains:
- The supplementary material document: [supplementary.pdf](supplementary_materials/FSE_supplementary_materials.pdf).
- EFOMS implementation: Please refer to the [instructions](#instruction-for-executing-efoms-methodology)  for information on how to install it.
- The artifacts and results to reproduce our empirical evaluations: [evaluation](#evaluation-artifacts).

## Instruction for executing EFOMS methodology:
### Prerequisite:

To execute the "EFOMS" methodology, you must have the following software installed (the versions are the recommended ones):

- Python 3.11.4
- R 4.3.3
- Jupyter Notebook 6.5.4
- CUDA with cudnn (To support the corresponding pytorch platform), you should have an NVIDIA graphics card (e.g., CUDA 11.8 with NVIDIA GeForce RTX 4080 Laptop GPU).

For the R, you must download the following libraries:
- parallel
- doParallel
- foreach
- gtools
- boot
- RANN
- mvtnorm
- sensitivity

The required libraries for Python are listed in the file requirements.txt in the folder "code."

### Training and testing datasets
It is important to dowload the following datasets before initiating the experiments:
- Download the CIFAR-10-C dataset from https://zenodo.org/records/2535967#.XncuG5P7TUJ and extract it in the folder "src".
- Download the Tiny-Imagenet dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip and extract it in the folder "src".
- Download the Tiny-Imagenet-C dataset from https://zenodo.org/records/2536630 and extract it in the folder "src".

### Pretrained models
It is also important to dowload the pretrained models to be able to launch the experiments:
- Download the "experim_models_resnet.zip," which contains the pre-trained ResNet models from https://zenodo.org/records/13761609, and extract it in the folder "src."
- Download the "experim_models_swin.zip," which contains the pre-trained SwinTransformer models from https://zenodo.org/records/13761240, and extract it in the folder "src."
- Download the "experim_resnet_attack.zip," which contains the adversarial attacks for the pre-trained models from https://zenodo.org/records/13761609, and extract it in the folder "src."

Remark: For CIFAR-10-C, you should keep the folder structure with the repeated name after directly unzipping it (i.e., keep the structure as "src\\CIFAR-10-C\\CIFAR-10-C\\...").
However, for Tiny-Imagenet, Tiny-Imagenet-C, the pre-trained models and their corresponding adversarial attacks, you should guarantee that there are no repeated names (e.g., "src\\experim_models_resnet\\...", "src\\experim_models_swin\\..." and "src\\experim_resnet_attack\\...")

### Launch EFOMS

To launch the experiment, you should:

1. Go to the "src" folder and open a terminal.
2. Type the command "jupyter notebook."
3. Open the notebook you would like to run (e.g., EFOMS_CIFAR10_CPND_OMS_detection_final.ipynb).
4. Configure the parameters required in the notebook (You can keep the default experiment in the notebook).
5. Launch the notebook.

All the output results will be saved in the folder "output" in the notebook's working directory.

We additionally provide notebooks for the training and evaluation of different ResNet models and the training of SwinTransformers.

## Evaluation artifacts

The evaluation results addressing our three research questions can be found in the [evaluation](evaluation_results) folder. The results are organized based on the dataset (CIFAR10 or Tiny_Imagenet) and the out-of-model-scope detection approach (CPND or KNN) used.

The evaluation folder follows this structure:
- EFOMS_(Dataset name)_(Tested OMS detection method)
    - (Tested network architecture)
        - Folders containing the result figures combining the applications of the EFOMS methodology using Sobol index thresholds
        - Folders containing the CSV files and figures for the application of the EFOMS methodology using one specific Sobol index threshold
    - (Another tested network architecture)
        - Same structure as above

