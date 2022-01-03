# Nlp_Emotion_Detector
Emotion detector in text using natural language processing


## installation
Use the package manager pip to install these packages.
```bash
pip install streamlit neattext eli5 lime pandas spacy numpy altair seaborn scikitlearn
```

To implement this project you will need these libraries:

**To load your data import these libraries**
```python

import pandas as pd
import numpy as np

#**Load Data Visualization**
import seaborn as sns

#**Load Text Cleaning Pkgs**
import neattext.functions as nfx

#**Load Machine Learning packages**
from sklearnex import patch_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

#**Transformers**
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

```
## Running the app interface

after making sure you installed the pre-requisits of the above packages now you can go to the app file directory and run in your preferred terminal by writing the below command: 

```bash
>> \App> streamlit run app.py
```
