# AMLS_II_assignment20_21
This repo provides the solutions for AMLS_II project on cassava disease classfication, details of the kaggle competition are available at 
https://www.kaggle.com/c/cassava-disease
## Library and enviroment
Scikit-Learn, Pandas, Tensorflow 2.3.0, CUDA, keras, Python 3.7

## File structure
- AMLS_II_20-21_SN20073066
  - A1 (Baseline model selection)
    - EfficientNet.ipynb (training code contains model construction and parameter tuning)
    - Resnet.ipynb (training code contains model construction and parameter tuning)
    - EfficientNet_best.h5 (saved model)
    - Resnet_best.h5 (saved model)
  - A2 (Drop rate parameter tuning)
    - dropout0.25.ipynb (training code contains model construction and parameter tuning)
    - drop_best0.1.h5 (saved model)
    - drop_best0.25.h5 (saved model)
    - drop_best0.5.h5 (saved model)
  - A3 (Batch Normalization)
    - EfficientNet.ipynb (training code contains model construction and parameter tuning)
    - BN_best.h5 (saved model)
  - A4 (Final model with mix-up)
    - mixup_generator.py (code for mix up two images)
    - mixup.py (library for main.py)
    - EfficientNetMixUp.ipynb (training code contains model construction and parameter tuning)
    - mixup.h5 (saved model)
  - Datasets
    - train_img.csv
    - valid_img.csv
    - test_img.csv
    - labels.csv
    - train (file contains raw images)
  - main.py (display final model performance)
  - README.md
## Reminder
- use Tensorflow 2.3.0 is preferrable
- main.py can run in both terminal and jupyter notebook. In jupyter notebook use following command:
  > %run main.py
