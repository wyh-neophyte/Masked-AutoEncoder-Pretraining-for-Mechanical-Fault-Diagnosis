### 📣 Introduction
The method , as shown in figure below.
![image](https://github.com/wyh-neophyte/Masked-AutoEncoder-Pretraining-for-Mechanical-Fault-Diagnosis/edit/main/assets/method.png)
### Experiment results
We train the classifier on LinGang dataset under vaiable working conditions, e.g., using 1800rpm data as training domain and 1200 rpm as testing domain. Without MAE pretraining, the results is trained with 20 repetitive experiments.
![image](https://github.com/wyh-neophyte/Masked-AutoEncoder-Pretraining-for-Mechanical-Fault-Diagnosis/blob/main/assets/Without-MAE-train1800-test1200.png)
With MAE pretrained parameters, the encoder is frozen except for the last two encoder layers. The experiments show a 2-3 percent improvement in average accuracy compared to the baseline.  
![image](https://github.com/wyh-neophyte/Masked-AutoEncoder-Pretraining-for-Mechanical-Fault-Diagnosis/blob/main/assets/MAE-pretrained-train1800-test1200.png)

### 🚀 Quick Start
#### 1. Installation
You may install the dependencies by the following command.
```
pip install -e .
```
#### 2. Data Preparation
Download the mechanical fault diagnosis dataset, i.e., CRWU, HUST, LinGang。

#### 3. Set PYTHONPATH
```
export PYTHONPATH=/path/to/project
```

#### 4. Run MAE Pretraining
```
python tools/train_mae.py --datapath /path/to/your/data
```

#### 5. Train Classifier with MAE pre-trained parameters
```
python tools/train_classifier.py --datapath /path/to/your/data
```
