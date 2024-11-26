import pandas as pd
import numpy as np
import joblib, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame, Series
from dataclasses import dataclass
import fire, os, sys
from pathlib import Path
import tqdm

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

#from tqdm import tqdm

@dataclass
class Buildmodels:
    nbfile: str

    def exec_nbs(self):
        # Load the notebook
        with open(f'{self.nbfile}') as f:
            notebook = nbformat.read(f, as_version=4)

        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(notebook, {'metadata': {'path': './'}})

        # Save the executed notebook
        with open(f'{self.nbfile}', 'w') as f:
            nbformat.write(notebook, f)



if __name__ == '__main__':
    Buildmodels('models/notebooks/imit.ipynb').exec_nbs()
    Buildmodels('models/notebooks/nlp.ipynb').exec_nbs()

