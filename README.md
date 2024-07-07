### Introduction

### 🚀 Quick Start
#### 1. Installation
You may install the dependencies by the following command.
```
pip install -e .
```
#### 2. Data Preparation
Download the mechanical fault diagnosis dataset, i.e., [CRWU](), [HUST](), [LinGang]()

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
