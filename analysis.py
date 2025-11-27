import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("data.csv")

# Basic stats
print(df.describe())

# Correlation
print("Correlation matrix:")
print(df.corr())

# Histogram
plt.hist(df['Sales'])
plt.title("Sales Distribution")
plt.savefig("sales_hist.png")
plt.close()

# T-test (example: compare high vs low marketing spend groups)
median_spend = df['Marketing_Spend'].median()
group1 = df[df['Marketing_Spend'] >= median_spend]['Sales']
group2 = df[df['Marketing_Spend'] < median_spend]['Sales']

t_stat, p_val = stats.ttest_ind(group1, group2)
print("T-test results:", t_stat, p_val)

# Confidence interval for sales
sales_mean = df['Sales'].mean()
sales_sem = stats.sem(df['Sales'])
ci = stats.t.interval(0.95, len(df)-1, loc=sales_mean, scale=sales_sem)
print("95% CI for Sales:", ci)
