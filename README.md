## Reference
![rPPG-Toolbox Logo](./toolbox_logo.png)

```
@article{liu2022deep,
  title={Deep physiological sensing toolbox},
  author={Liu, Xin and Zhang, Xiaoyu and Narayanswamy, Girish and Zhang, Yuzhe and Wang, Yuntao and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}
```


# Setup

STEP1: `bash setup.sh` 

STEP2: `conda activate rppg-toolbox` 

STEP3: `pip install -r requirements.txt` 

# Example of using pre-trained models 

Please use config files under `./configs/infer_configs`

For example, if you want to run The model trained on PURE and tested on UBFC, use `python main.py --config_file ./configs/infer_configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml`

If you want to test unsupervised signal processing  methods, you can use `python main.py --config_file ./configs/infer_configs/UBFC_UNSUPERVISED.yaml`

# Example of neural network training

Please use config files under `./configs/train_configs`

## Train on PURE and test on UBFC with TSCAN 

STEP1: Download the PURE raw data by asking the [paper authors](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure).

STEP2: Download the UBFC raw data via [link](https://sites.google.com/view/ybenezeth/ubfcrppg)

STEP3: Modify `./configs/train_configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml` 

STEP4: Run `python main.py --config_file ./configs/train_configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml` 

Note1: Preprocessing requires only once; thus turn it off on the yaml file when you train the network after the first time. 

Note2: The example yaml setting will allow 80% of PURE to train and 20% of PURE to valid. 
After training, it will use the best model(with the least validation loss) to test on UBFC.

## Training on SCAMPS and testing on UBFC with DeepPhys

STEP1: Download the SCAMPS via this [link](https://github.com/danmcduff/scampsdataset) and split it into train/val/test folders.

STEP2: Download the UBFC via [link](https://sites.google.com/view/ybenezeth/ubfcrppg)

STEP3: Modify `./configs/train_configs/SCAMPS_SCAMPS_UBFC_DEEPPHYS_BASIC.yaml` 

STEP4: Run `python main.py --config_file ./configs/train_configs/SCAMPS_SCAMPS_UBFC_DEEPPHYS_BASIC.yaml`

Note1: Preprocessing requires only once; thus turn it off on the yaml file when you train the network after the first time. 

Note2: The example yaml setting will allow 80% of SCAMPS to train and 20% of SCAMPS to valid. 
After training, it will use the best model(with the least validation loss) to test on UBFC.

# Predicting BVP signal and calculate heart rate on UBFC with POS/CHROME/ICA/GREEN/PBV/LGI

STEP1: Download the UBFC via [link](https://sites.google.com/view/ybenezeth/ubfcrppg)

STEP3: Modify `./configs/infer_configs/UBFC_UNSUPERVISED.yaml` 

STEP4: Run `python main.py --config_file ./configs/infer_configs/UBFC_UNSUPERVISED.yaml`

# Yaml File Setting
The rPPG-Toolbox uses yaml file to control all parameters for training and evaluation. 
You can modify the existing yaml files to meet your own training and testing requirements.

Here are some explanation of parameters:
* #### TOOLBOX_MODE: 
  * `train_and_test`: train on the dataset and use the newly trained model to test.
  * `only_test`: you need to set INFERENCE-MODEL_PATH, and it will use pre-trained model initialized with the MODEL_PATH to test.
* #### TRAIN / VALID / TEST / UNSUPERVISED DATA:
  * `USE_EXCLUSION_LIST`: If `True`, utilize a provided list to exclude preprocessed videos
  * `SELECT_TASKS`: If `True`, explicitly select tasks to load 
  * `DATA_PATH`: The input path of raw data
  * `CACHED_PATH`: The output path to preprocessed data. This path also houses a directory of .csv files containing data paths to files loaded by the dataloader. This filelist (found in default at CACHED_PATH/DataFileLists). These can be viewed for users to understand which files are used in each data split (train/val/test)
  * `EXP_DATA_NAME` If it is "", the toolbox generates a EXP_DATA_NAME based on other defined parameters. Otherwise, it uses the user-defined EXP_DATA_NAME.  
  * `BEGIN" & "END`: The portion of the dataset used for training/validation/testing. For example, if the `DATASET` is PURE, `BEGIN` is 0.0 and `END` is 0.8 under the TRAIN, the first 80% PURE is used for training the network. If the `DATASET` is PURE, `BEGIN` is 0.8 and `END` is 1.0 under the VALID, the last 20% PURE is used as the validation set. It is worth noting that validation and training sets don't have overlapping subjects.  
  * `DATA_TYPE`: How to preprocess the video data
  * `DATA_AUG`: If present, the type of generative data augmentation applied to video data
  * `LABEL_TYPE`: How to preprocess the label data
  *  `USE_PSUEDO_PPG_LABEL`: If `True` use POS generated PPG psuedo labels instead of dataset ground truth heart singal waveform
  * `DO_CHUNK`: Whether to split the raw data into smaller chunks
  * `CHUNK_LENGTH`: The length of each chunk (number of frames)
  * `DO_CROP_FACE`: Whether to perform face detection
  * `DYNAMIC_DETECTION`: If `False`, face detection is only performed at the first frame and the detected box is used to crop the video for all of the subsequent frames. If `True`, face detection is performed at a specific frequency which is defined by `DYNAMIC_DETECTION_FREQUENCY`. 
  * `DYNAMIC_DETECTION_FREQUENCY`: The frequency of face detection (number of frames) if DYNAMIC_DETECTION is `True`
  * `USE_MEDIAN_FACE_BOX`: If `True` and `DYNAMIC_DETECTION` is `True`, use the detected face boxs throughout each video to create a single, median face box per video.
  * `LARGE_FACE_BOX`: Whether to enlarge the rectangle of the detected face region in case the detected box is not large enough for some special cases (e.g., motion videos)
  * `LARGE_BOX_COEF`: The coefficient to scale the face box if `LARGE_FACE_BOX` is `True`.

  
* #### MODEL : Set used model (Deepphys, TSCAN, Physnet, EfficientPhys, and BigSmall and their paramaters are supported).
* #### UNSUPERVISED METHOD: Set used unsupervised method. Example: ["ICA", "POS", "CHROM", "GREEN", "LGI", "PBV"]
* #### METRICS: Set used metrics. Example: ['MAE','RMSE','MAPE','Pearson']
* #### INFERENCE:
  * `USE_SMALLER_WINDOW`: If `True`, use an evaluation window smaller than the video length for evaluation.

# Dataset
The toolbox supports four datasets, which are SCAMPS, UBFC, PURE, and MMPD (COHFACE support will be added shortly). Cite corresponding papers when using.
For now, we only recommend training with PURE or SCAMPS due to the level of synchronization and volume of the dataset.

* [MMPD](https://github.com/McJackTang/MMPD_rPPG_dataset)
    * Jiankai Tang, Kequan Chen, Yuntao Wang, Yuanchun Shi, Shwetak Patel, Daniel McDuff, Xin Liu.  
 "MMPD: Multi-Domain Mobile Video Physiology Dataset", arxiv, 2023
    * In order to use this dataset in a deep model, you should organize the files as follows:
    
    -----------------
         data/MMPD/
         |   |-- subject1/
         |       |-- p1_0.mat
         |       |-- p1_1.mat
         |       |...
         |       |-- p1_19.mat
         |   |-- subject2/
         |       |-- p2_0.mat
         |       |-- p2_1.mat
         |       |...
         |...
         |   |-- subjectn/
         |       |-- pn_0.mat
         |       |-- pn_1.mat
         |       |...
    -----------------
    
* [SCAMPS](https://arxiv.org/abs/2206.04197)
  
    * D. McDuff, M. Wander, X. Liu, B. Hill, J. Hernandez, J. Lester, T. Baltrusaitis, "SCAMPS: Synthetics for Camera Measurement of Physiological Signals", Arxiv, 2022
    * In order to use this dataset in a deep model, you should organize the files as follows:
    -----------------
         data/SCAMPS/Train/
            |-- P00001.mat
            |-- P00002.mat
            |-- P00003.mat
         |...
         data/SCAMPS/Val/
            |-- P00001.mat
            |-- P00002.mat
            |-- P00003.mat
         |...
         data/SCAMPS/Test/
            |-- P00001.mat
            |-- P00002.mat
            |-- P00003.mat
         |...
    -----------------

* [UBFC](https://sites.google.com/view/ybenezeth/ubfcrppg)
  
    * S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
    * In order to use this dataset in a deep model, you should organize the files as follows:
    
    -----------------
         data/UBFC/
         |   |-- subject1/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |   |-- subject2/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |...
         |   |-- subjectn/
         |       |-- vid.avi
         |       |-- ground_truth.txt
    -----------------
   
* [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
    * Stricker, R., Müller, S., Gross, H.-M.Non-contact "Video-based Pulse Rate Measurement on a Mobile Service Robot"
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
    * In order to use this dataset in a deep model, you should organize the files as follows:
    
    -----------------
         data/PURE/
         |   |-- 01-01/
         |      |-- 01-01/
         |      |-- 01-01.json
         |   |-- 01-02/
         |      |-- 01-02/
         |      |-- 01-02.json
         |...
         |   |-- ii-jj/
         |      |-- ii-jj/
         |      |-- ii-jj.json
    -----------------
    
* [BP4D+](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
    * Zhang, Z., Girard, J., Wu, Y., Zhang, X., Liu, P., Ciftci, U., Canavan, S., Reale, M., Horowitz, A., Yang, H., Cohn, J., Ji, Q., Yin, L. "Multimodal Spontaneous Emotion Corpus for Human Behavior Analysis", IEEE International Conference on Computer Vision and Pattern Recognition (CVPR) 2016.   

    -----------------
        RawData/
         |   |-- 2D+3D/
         |       |-- F001.zip/
         |       |-- F002.zip
         |       |...
         |   |-- 2DFeatures/
         |       |-- F001_T1.mat
         |       |-- F001_T2.mat
         |       |...
         |   |-- 3DFeatures/
         |       |-- F001_T1.mat
         |       |-- F001_T2.mat
         |       |...
         |   |-- AUCoding/
         |       |-- AU_INT/
         |            |-- AU06/
         |               |-- F001_T1_AU06.csv
         |               |...
         |           |...
         |       |-- AU_OCC/
         |           |-- F00_T1.csv 
         |           |...
         |   |-- IRFeatures/
         |       |-- F001_T1.txt
         |       |...
         |   |-- Physiology/
         |       |-- F001/
         |           |-- T1/
         |               |-- BP_mmHg.txt
         |               |-- microsiemens.txt
         |               |--LA Mean BP_mmHg.txt
         |               |--LA Systolic BP_mmHg.txt
         |               |-- BP Dia_mmHg.txt
         |               |-- Pulse Rate_BPM.txt
         |               |-- Resp_Volts.txt
         |               |-- Respiration Rate_BPM.txt
         |       |...
         |   |-- Thermal/
         |       |-- F001/
         |           |-- T1.mv
         |           |...
         |       |...
         |   |-- BP4D+UserGuide_v0.2.pdf
    -----------------

* [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys)
    * Sabour, R. M., Benezeth, Y., De Oliveira, P., Chappe, J., & Yang, F. (2021). Ubfc-phys: A multimodal database for psychophysiological studies of social stress. IEEE Transactions on Affective Computing.  

    -----------------
          RawData/
          |   |-- s1/
          |       |-- vid_s1_T1.avi
          |       |-- vid_s1_T2.avi
          |       |-- vid_s1_T3.avi
          |       |...
          |       |-- bvp_s1_T1.csv
          |       |-- bvp_s1_T2.csv
          |       |-- bvp_s1_T3.csv
          |   |-- s2/
          |       |-- vid_s2_T1.avi
          |       |-- vid_s2_T2.avi
          |       |-- vid_s2_T3.avi
          |       |...
          |       |-- bvp_s2_T1.csv
          |       |-- bvp_s2_T2.csv
          |       |-- bvp_s2_T3.csv
          |...
          |   |-- sn/
          |       |-- vid_sn_T1.avi
          |       |-- vid_sn_T2.avi
          |       |-- vid_sn_T3.avi
          |       |...
          |       |-- bvp_sn_T1.csv
          |       |-- bvp_sn_T2.csv
          |       |-- bvp_sn_T3.csv
    -----------------
    
## Add A New Dataloader

* Step1 : Create a new python file in dataset/data_loader, e.g. MyLoader.py

* Step2 : Implement the required functions, including:

  ```python
  def preprocess_dataset(self, config_preprocess):
  ```
  ```python
  @staticmethod
  def read_video(video_file):
  ```
  ```python
  @staticmethod
  def read_wave(bvp_file):
  ```

* Step3 :[Optional] Override optional functions. In principle, all functions in BaseLoader can be override, but we **do not** recommend you to override *\_\_len\_\_, \_\_get\_item\_\_,save,load*.
* Step4 :Set or add configuration parameters.  To set paramteters, create new yaml files in configs/ .  Adding parameters requires modifying config.py, adding new parameters' definition and initial values.

# Motion Augmented Training

The usage of synthetic data in the training of machine learning models for medical applications is becoming a key tool that warrants further research. In addition to providing support for the fully synthetic dataset [SCAMPS](https://arxiv.org/abs/2206.04197), we provide provide support for synthetic, motion-augmented versions of the [UBFC](https://sites.google.com/view/ybenezeth/ubfcrppg), [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure), [SCAMPS](https://arxiv.org/abs/2206.04197), and [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys) datasets for further exploration toward the use of synthetic data for training rPPG models. The synthetic, motion-augmented datasets are generated using the [MA-rPPG Video Toolbox](https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox), an open-source motion augmentation pipeline targeted for increasing motion diversity in rPPG videos. You can generate and utilize the aforementioned motion-augmented datasets using the steps below.

* STEP 1: Follow the instructions in the [README](https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox/blob/main/README.md) of the [MA-rPPG Video Toolbox](https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox) GitHub repo to generate any of the supported motion-augmented datasets. NOTE: You will have to have an original, unaugmented version of a dataset and driving video to generate a motion-augmented dataset. More information can be found [here](https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox#file_folder-datasets). 

* STEP 2: Using any config file of your choice in this toolbox, modify the `DATA_AUG` parameter (set to `'None'` by default) to `'Motion'`. Currently, only `train_configs` that utilize the UBFC-rPPG or PURE datasets have this parameter visible, but you can also modify other config files to add the `DATA_AUG` parameter below the `DATA_TYPE` parameter that is visible in all config files. This will enable the proper funciton for loading motion-augmented data that is in the `.npy` format.

* STEP 3: Run the corresponding config file. Your saved model's filename will have `MA` appended to the corresponding data splits that are motion-augmented.

# Extending The Toolbox To Multitasking

## BigSmall

We implement [BigSmall](https://girishvn.github.io/BigSmall/) as an example to show how this toolbox may be extended to support physiological multitasking. If you use this functionality please cite the following publication: 
* Narayanswamy, G., Liu, Y., Yang, Y., Ma, C., Liu, X., McDuff, D., Patel, S. "BigSmall: Efficient Multi-Task Learning For Physiological Measurements" https://arxiv.org/abs/2303.11573

The BigSmall mode multi-tasks pulse (PPG regression), respiration (regression), and facial action (multilabel AU classification). The model is trained and evaluated (in this toolbox) on the AU label subset (described in the BigSmall publication) of the BP4D+ dataset, using a 3-fold cross validation method (using the same folds as in the BigSmall publication).

* STEP 1: Download the BP4D+ by emailing the authors found [here](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html).

* STEP 2: Modify `./configs/train_configs/BP4D_BP4D_BIGSMALL_FOLD1.yaml` to train the first fold (config files also exist for the 2nd and 3rd fold).

* STEP 3: Run `python main.py --config_file ./configs/train_configs/BP4D_BP4D_BIGSMALL_FOLD1.yaml `

