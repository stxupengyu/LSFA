# LSFA  
The code for our paper "Label-Specific Feature Augmentation for Long-Tailed Multi-Label Text Classification”.   

## Requirements  
* python==3.7.10
* numpy==1.21.2
* scipy==1.7.2
* scikit-learn==0.22 
* pytorch==1.10.0
* gensim==4.1.2
* nltk==3.6.5
* tqdm==4.62.3

## Reproducibility   
### Pre-processing
If you want to use other datasets, please use the following command for preprocessing first:  
```bash
preprocess.py
```

### Data Path
Please confirm the corresponding configuration file. Make sure the data path parameters (train_texts, train_labels and etc.) are right in:   
```bash
main.py
```

### LSFA Experiments  
Train and evaluate as follows: 
```bash
python main.py 
```
