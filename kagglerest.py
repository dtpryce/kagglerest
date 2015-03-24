# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:56:08 2015

@author: dpryce
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

os.chdir('C:\Users\dpryce\Documents\DataAnalytics\Kaggle - restaurant')
train_df = pd.read_csv('train.csv',sep=",")

print train_df