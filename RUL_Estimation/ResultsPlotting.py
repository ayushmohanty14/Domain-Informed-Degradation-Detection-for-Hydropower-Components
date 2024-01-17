import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use(['science','ieee'])
import os
import scipy.stats as stats
import pylab
import seaborn as sns

#### Percentile of lifetime where FLD estimation and Bayesian Updating to be done
percentile_of_lifetime = [10, 20, 30, 40, 50, 60, 70, 80, 90]

raw_data = pd.read_csv("DailySampled_denoised.csv")
num_cols = len(raw_data.columns)

df_results = pd.read_csv("pred_error.csv")

percentile_of_lifetime_all = np.array([])
pred_error_all = np.array([])
df_prederror = pd.DataFrame([])
for i in range(num_cols):
    percentile_of_lifetime_all = np.concatenate([percentile_of_lifetime_all, [10, 20, 30, 40, 50, 60, 70, 80, 90]])
    pred_error_all = np.concatenate((pred_error_all, df_results.iloc[:, i]), axis=None)

df_prederror['% of Failure Time'] = percentile_of_lifetime_all
df_prederror['Prediction Error'] = pred_error_all

plt.figure()
sns.boxplot(data=df_prederror, x='% of Failure Time', y='Prediction Error')
plt.ylim(0, 100)
plt.title('Exponential Degradation Model for Failure Threshold = 16 Mils')
plt.show()