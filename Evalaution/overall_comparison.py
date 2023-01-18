#!/usr/bin/env python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

lr_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/lr_model_metrics.csv')
lr_acc_=lr_model_metrics_read['Accuracy'].item()
nb_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/nb_model_metrics.csv')
nb_acc_=nb_model_metrics_read['Accuracy'].item()
rf_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/rf_model_metrics.csv')
rf_acc_=rf_model_metrics_read['Accuracy'].item()

accuracy_list=[lr_acc_*100,nb_acc_*100,rf_acc_*100]
model_list=["Logistic Regression","Naive Bayes","Random Forest"]
lr_pr_=lr_model_metrics_read['Precision'].item()
nb_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/nb_model_metrics.csv')
nb_pr_=nb_model_metrics_read['Precision'].item()
rf_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/rf_model_metrics.csv')
rf_pr_=rf_model_metrics_read['Precision'].item()

pr_list=[lr_pr_*100,nb_pr_*100,rf_pr_*100]
lr_acc_br_=lr_model_metrics_read['Balanced_Accuracy'].item()
nb_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/nb_model_metrics.csv')
nb_acc_br_=nb_model_metrics_read['Balanced_Accuracy'].item()
rf_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/rf_model_metrics.csv')
rf_acc_br_=rf_model_metrics_read['Balanced_Accuracy'].item()

br_list=[lr_acc_br_*100,nb_acc_br_*100,rf_acc_br_*100]
lr_r_=lr_model_metrics_read['Recall'].item()
nb_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/nb_model_metrics.csv')
nb_r_=nb_model_metrics_read['Recall'].item()
rf_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/rf_model_metrics.csv')
rf_r_=rf_model_metrics_read['Recall'].item()

recall_list=[lr_r_*100,nb_r_*100,rf_r_*100]
lr_f_=lr_model_metrics_read['F1Score'].item()
nb_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/nb_model_metrics.csv')
nb_f_=nb_model_metrics_read['F1Score'].item()
rf_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/rf_model_metrics.csv')
rf_f_=rf_model_metrics_read['F1Score'].item()

f1_list=[lr_f_*100,nb_f_*100,rf_f_*100]
lr_t_=lr_model_metrics_read['Time'].item()
nb_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/nb_model_metrics.csv')
nb_t_=nb_model_metrics_read['Time'].item()
rf_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/rf_model_metrics.csv')
rf_t_=rf_model_metrics_read['Time'].item()
time_list=[lr_t_*100,nb_t_*100,rf_t_*100]
df_models = pd.DataFrame({'Models': model_list, 'Accuracy': accuracy_list, 'F1 Score': f1_list,'Precison':pr_list,
                          'Recall':recall_list,'Balanced Accuracy' : br_list, 'Time' : time_list})
df_models = df_models.replace(r'\n',' ', regex=True)
df_models = df_models.sort_values(by=["Accuracy", "Balanced Accuracy"], ascending=False)
df_models = df_models.reset_index(drop=True)

print(df_models)
