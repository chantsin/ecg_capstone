# ECG Signal Classification via Fourier Transforms

**Author**: Calvin Chan 

**Data**: The data set used for this project is called **PTB-XL** from **PhysioNet** and can be found [here](https://physionet.org/content/ptb-xl/1.0.3/).

## Table of Contents
- [Project Overview](#overview)
- [Workflow Timeline](#workflow)
- [Notebooks](#notebooks)
- [Project Status](#status)
- [Setup Requirements](#setup)

<a id='overview'></a>
## 1 - Project Overview

### Problem Area

Biomedical signals has always been an important aspect of the medical field. From medical diagnosis and treatment of patients to researching and developing drugs, it has helped many of those who were in need. An important aspect of why it works is because of its ability to capture many valuable information about the physiological state of the body that medical professions would not have known otherwise. However, whether this information is transferred to good use or not solely depends on the physicians ability to read the signal. In this project, we will be focusing on electrocardiograms (ECGs or EKGs) which is responsible for recording electrical activity in the heart. A systematic review and meta-analysis published in 2020 looked at 78 studies that assessed the accuracy of ECG interpretations by physicians with different levels of training and specialization, including medical students, residents, physicians in non-cardiology practice, and cardiologists. It was found that the accuracy scores varied significantly between the different physician levels, ranging from 4% to 95% [[1]](#1). This poses a significant problem for patients and specialists as different ECG results can lead to vastly different decision and outcomes. 

### Those Affected

This project can be significant if it is scaled to a larger setting. Being able to have access to this tool in a hospital setting should definitely help physicians of different levels to have better and more accurate diagnosis. It is possible to help medical students and residents in training when they are still learning about how to interpret ECG signals. In terms of business value, I believe this could also be implemented in current technologies. Imagine having a smart watch that would take your ECG on a regular basis and have feedback about your heart condition. This would be significant for people who may have unknown underlying heart conditions that have not seen a doctor yet. To have a device tell you that you may have an underlying condition advising you to see a physician can be lifesaving.

### Data Science Solution

In this project, we will attempt to utilize data science techniques to read ECG data and provide a diagnostic outcome, giving physicians and specialist an additional support for interpretation. More specifically, we will look into utilizing classification techniques used in data science to classify ECG signals into different diagnostic categories. This project can have a great impact on medical professionals as well as patients since it can act as an additional diagnostic confirmation, potentially decreasing diagostic errors made physicians allowing for more accurate and better treatment to patients. 

### Data Description

The dataset used for this project is called PTB-XL taken from PhysioNet. It contains 21799 12-lead ECG entries of 18869 unique patients collected using devices from Schiller AG. The signals were taken over the course of nearly seven years between October 1989 and June 1996. The dataset also contains a metadata file which includes patient information such as their sex, age, weight, as well as an annotation file which encodes the type of diagnosis for each ECG. 

### Data Dictionary
The dataset we are using includes two main metadata files, `ptbxl_database.csv` which we will call **Metadata** and it contains the patients information, and `scp_statements.csv` which we will call **Annotation** containing the official SCP-ECG statements for diagnosis. Below we present the data dictionary for each file.

- **Metadata** (`ptbxl_database.csv`):
<div align='center'>

| Column                       | Data Types | Description                                                     |
|------------------------------|------------|-----------------------------------------------------------------|
| patient_id                   | float      | Unique patient ID                                               |
| age                          | float      | Age in years                                                    |
| sex                          | integer    | Sex (M/F)                                                       |
| height                       | float      | Height in centimeters                                           |
| weight                       | float      | Weight in kilograms                                             |
| nurse                        | float      | Nurse that took the ECG (placeholder value)                     |
| site                         | float      | Site in which ECG was recorded                                  |
| device                       | string     | Device used to record ECG                                       |
| recording_date               | datetime   | Date time when ECG was recorded                                 |
| report                       | string     | Report by cardiologist or automatic intepretation by ECG device |
| scp_codes                    | dictionary | Standardized SCP-ECG statements                                 |
| heart_axis                   | string     | Hearts overall electrical activity direction                    |
| infarction_stadium1          | string     | First infarction stadium                                        |
| infarction_stadium2          | string     | Second infarction stadium                                       |
| validated_by                 | float      | Cardiologist that validated signal (placeholder value)          |
| second_opinion               | boolean    | Second opinion required                                         |
| initial_autogenerated_report | boolean    | Autogenerated report by ECG device                              |
| validated_by_human           | boolean    | Results validated by human                                      |
| baseline_drift               | string     | Leads with baseline drift                                       |
| static_noise                 | string     | Leads with static noise                                         |
| burst_noise                  | string     | Leads with burst noise                                          |
| electrodes_problems          | string     | Leads with electrode problem                                    |
| extra_beats                  | string     | Extra beats present                                             |
| pacemaker                    | string     | Pacemaker present in patient                                    |
| strat_fold                   | integer    | Suggested stratified folds for train test split                 |
| filename_lr                  | string     | Low sampling rate file path                                     |
| filename_hr                  | string     | High sampling rate file path                                    |
</div>

- **Annotation** (`scp_statements.csv`):
<div align='center'>

| Column                        | Data Types | Description                     |
|-------------------------------|------------|---------------------------------|
| description                   | string     | Annotation description          |
| diagnostic                    | float      | Related to diagnosis            |
| form                          | float      | Related to signal form          |
| rhythm                        | float      | Related to rhythm               |
| diagnostic_class              | string     | Superclass for diagnosis        |
| diagnostic_subclass           | string     | Subclass for diagnosis          |
| statement category            | string     | Medical description             |
| SCP-ECG statement description | string     | Official SCP-ECG description    |
| AHA code                      | float      | Unique AHA ID                   |
| aECG REFID                    | string     | Annotated ECG standard notation |
| CDISC code                    | string     | Controlled terminology          |
| DICOM code                    | string     | DICOM tags                      |
</div>

<a id='workflow'></a>
## 2 - Project Workflow
1. **Data cleaning after combining Metadata and Annotation**
    - Pulling SCP-ECG codes for diagnostic superclass
2. **ECG denoising using Fourier Analysis**
    - Removal of baseline wandering and powerline interference
3. **Baseline modeling**
    - Logistic Regression
    - Simple ANN
    - Autoencoder
4. **Advanced modeling**
    - RNN
    - CNN
5. **Model Evalutation**
    - Choosing best model

<a id='notebooks'></a>
## 3 - Notebooks
This project is divided into 5 main notebooks and are structured as follows. The notebooks are ordered by the number assigned to each and should be read in that order. 

1. **Metadata Annotations Cleaning**: 
This first notebook includes combining two data files, one with patient information and the other with the patients ECG diagnostic. We also implement some preliminary exploratory data analysis, taking a look inside what kind of data we are dealing with. 

2. **ECG Cleaning**:
In this next notebook, we go over the ECG signals and perform cleaning on them. The main tool we are using to clean our signals is Fourier Transforms, by implementing them into functions, we allow the user to explicitly remove frequency ranges in Fourier Space, effectively translating to signal denoising in time.

3. **Baseline Modeling**:
For baseline modeling, we use Logistic Regression and a Simple Neural Network as our choice. We utilize the method of binning our ECG signals based on amplitude before feeding them into our models. We also adjusted our target classes by dividing it into either multiclass classification or binary classification. 

4. **Autoencoders**:
As a detour, we experimented our works in autoencoders, utilizing the neural network architecture to see how much it can learn about ECG signals and whether it can reconstruct our input signals or not. Results indicate that it has trouble extracting important features in our data set without overfitting.

5. **Recurrent Neural Networks**:
For advanced modeling this notebook contains our advanced modeling where we use Recurrent Neural Networks to classify our data. Without using binning, we use the raw signals as RNNs process data in sequence allowing effect learning for time series signals like ours.

6. **Convolutional Neural Networks**:
Another algorithm we test out is a convolutional neural network. We process the time series signals using 1D convolution layers instead of the typical 2D for images. 

<a id='status'></a>
## 4 - Project Status

### Current Progress
\*\*\***UPDATES**\*** : By fine-tuning the RNN model to compensate for imbalanced data, I was able to increase the models performance to an accuracy of 85%. Additionally, I have implement a CNN model using PyTorch in a new notebook. Previously, there have been research groups who have achieved great results using CNN models. My goal here is to get close if not exceed their results and compare how this model performs compared to the RNN model. 

### Next steps 
Currently my CNN model sits at approximately 83% accuracy. My next goal is to fine-tune the model by adjust hyperparameters such as regularization, learning rate, batch size, and optimizations. For further goals, I have plans to test out multi-class classification using the CNN model.

<a id='setup'></a>
## 5 - Setup

### Environment Download
To run the notebooks, the appropriate packages needs to be installed. We have included all the packages used for this project in the environment file `ecgcap.yml`. To download the environment on your computer, run the following code in your terminal
```
conda env create -n ecgcap -f ecgcap.yml
```

### Data Set Download
The data used for this project can be found in the PhysioNet link above, although the data folder can also be downloaded from [here](https://drive.google.com/drive/folders/1Ju1yhHguvVVcAEAf3lyz-NFchx71L54s?usp=drive_link). With the data folder beside the other directories, the code cells in the notebooks should be running without issue.  

## References
<a id='1'></a>
[1] Cook, D. A., Oh, S. Y., & Pusic, M. V. (2020). Accuracy
of Physicians’ Electrocardiogram Interpretations: A Systematic Review
and Meta-analysis. JAMA internal medicine, 180(11), 1461–1471.
https://doi.org/10.1001/jamainternmed.2020.3989 

<a id='2'></a>
[2] Wagner, P., Strodthoff, N., Bousseljot, RD. et al. PTB-XL, a large publicly available electrocardiography dataset. Sci Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6

<a id='3'></a>
[3] Śmigiel, S., Pałczyński, K., & Ledziński, D. (2021). ECG Signal Classification Using Deep Learning Techniques Based on the PTB-XL Dataset. Entropy (Basel, Switzerland), 23(9), 1121. https://doi.org/10.3390/e23091121

<a id='4'></a>
[4] Meek, S., & Morris, F. (2002). ABC of clinical electrocardiography.Introduction. I-Leads, rate, rhythm, and cardiac axis. BMJ (Clinical research ed.), 324(7334), 415–418. https://doi.org/10.1136/bmj.324.7334.415

<a id='5'></a>
[5] Dave, T., & Pandya, U. (2018). Simultaneous monitoring of motion ECG of two subjects using Bluetooth Piconet and baseline drift. Biomedical engineering letters, 8(4), 365–371. https://doi.org/10.1007/s13534-018-0081-4
