#!/usr/bin/env python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

lr_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/lr_model_metrics.csv')
lr_acc_=lr_model_metrics_read['Time'].item()
nb_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/nb_model_metrics.csv')
nb_acc_=nb_model_metrics_read['Time'].item()
rf_model_metrics_read=pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/rf_model_metrics.csv')
rf_acc_=rf_model_metrics_read['Time'].item()
accuracy_list=[lr_acc_*100,nb_acc_*100,rf_acc_*100]
model_list=["Logistic Regression","Naive Bayes","Random Forest"]
df_models = pd.DataFrame({'Models': model_list, 'Accuracy': accuracy_list})
df_models = df_models.sort_values(by="Accuracy")
plt.figure(figsize=(20,7))
sns.set_style('ticks')
clrs = ['grey' if (x < max(df_models.Accuracy)) else '#800813' for x in df_models.Accuracy ]
ax = sns.barplot(x=df_models.Models, y=df_models.Accuracy, saturation = 3.0, palette = clrs)
plt.title('\nTime of Models\n', fontsize = 18)
ax.set_frame_on(False)
ax.set_yticks([])
plt.ylabel('')
plt.xlabel('')

#plt.xlabel('\nModels', fontsize = 16)
#plt.ylabel('% of Accuracy', fontsize = 16)
#plt.xticks(fontsize = 12, horizontalalignment = 'center')

plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'medium')
plt.savefig('./time_comparison.png')
