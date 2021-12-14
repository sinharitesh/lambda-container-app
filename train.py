#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import boto3
import os
from pathlib import Path


# In[2]:


# Downloading data from s3 bucket.
# from pathlib import Path
# ECG_DATA_BUCKET = "ritesh-s3-ecg-annotated-data"
# s3 = boto3.client("s3")
# for fl in lst_images:
#     fn = Path(fl).name
#     print(fn
#     s3.download_file(ECG_DATA_BUCKET,fn,f'./ecg-annotated-data/{fn}')


# In[21]:


new_file_name = 'annotation-info.csv'
df_labels = pd.read_csv(new_file_name); df_labels.head()
# Label cleansing required a bit
df_labels['choice'].value_counts()


# In[24]:


def cleanse_labels(str_labels):
    try:
        str_labels = str_labels.replace('"choices":', "").replace('{ ["','').replace('"]}','').replace('"', '').replace(", ", ";")
    except:
        pass
    return(str_labels)
df_labels['labels'] = list(map(cleanse_labels, df_labels['choice']))
df_labels['labels'].value_counts()


# In[25]:


# Considering Normal Others and Myocardial Infarction.
df_training_labels =  df_labels[df_labels['labels'].isin(['Normal','Others','Myocardial Infarction'])].reset_index()
df_training_labels['labels'].value_counts()


# In[3]:


get_ipython().system(' pip install fastai==2.3.1 > /dev/null')


# In[4]:


get_ipython().system(' pip install pillow==8.2 ')


# In[5]:


import fastai
from fastai.vision.all import *


# In[29]:


def get_name_of_file(x):
    return(Path(x).name)
df_training_labels['image_name'] = list(map(get_name_of_file, df_training_labels['image']))
df_train = df_training_labels[['image_name', 'labels']]
df_train


# In[18]:


# def get_training_files(path = './ecg-annotated-data'):
#     fnames = get_image_files(path)
#     training_files = []
#     for f in fnames:
#         df_res =  df_training_labels[df_training_labels['image_name'] ==  Path(f).name]
#         if df_res.shape[0] == 1:
#             training_files.append(f)
#     return(training_files)
# ft = get_training_files()
# len(ft)


# In[21]:





# In[22]:


# DO NOT DELETE
# Extra step to copy all files which are part of data frame only, this should solve the problem.
# NEW_IMAGE_DIR = './ecg-subset-annotated-data/'
# os.makedirs(NEW_IMAGE_DIR)
# import shutil
# for fl in df_train['image_name']:
#     src_file = IMAGE_DIR + fl
#     targ_file = NEW_IMAGE_DIR + fl
#     try:
#         shutil.copyfile(src_file, targ_file)
#     except:
#         pass
# print("done")


# In[30]:


IMAGE_DIR = '../ecg-annotated-data/'
# Need to remove those files which are in the dataset but not on the disk.
avlfiles = []
for fl in df_train['image_name']:
    if os.path.exists(IMAGE_DIR+ fl):
        avlfiles.append(fl)
df_avl = pd.DataFrame({'image_name':avlfiles}) ; df_avl.head()
df_train_final = pd.merge(df_train,df_avl,on='image_name')
df_train_final.head()


# In[24]:


# This is important as using a label_func causes the lambda function to look for it and 
# it does not find this. Using ImageDataLoaders.from_df is safe.
dls = ImageDataLoaders.from_df(df_train_final
                               , IMAGE_DIR
                               , label_col = 'labels'
                               , item_tfms = RandomResizedCrop(128, min_scale=0.35)
                               , bs = 32)


# In[25]:


dls.show_batch()


# In[26]:


learn = cnn_learner(dls, resnet34, metrics=error_rate)


# In[27]:


learn.fit_one_cycle(30)


# In[28]:


learn.export('/home/ec2-user/SageMaker/model-dec14-30-epochs.pkl')


# In[29]:


print("Done upto here.")


# In[30]:


learn.show_results()


# In[ ]:


# Carry on from here if this succeeds.


# In[31]:


learn.fit_one_cycle(30)


# In[ ]:


learn.export('/home/ec2-user/SageMaker/model2-14dec21-60-epochs.pkl')


# In[ ]:


# Testing


# In[7]:


mdl = load_learner("../model-dec14-30-epochs-error-rate-36.pkl")


# In[8]:


mdl


# In[9]:


# test_local_files = Path('../ecg-annotated-data/').ls()


# In[31]:


def get_test_prediction(fl):
  d = {}
  try:
      out = mdl.predict(fl)
      d['image_name'] = fl.name
      d['predict_label'] = out[0]
      d['predict_prob'] = out[2].numpy().max()
  except:
    pass
  return( d )
test_predicted = list(map(get_test_prediction, test_local_files))
df_res = pd.DataFrame(test_predicted)
df_res.head()


# In[34]:


df_consolidated_results = pd.merge(df_training_labels, df_res, on = ["image_name"])[['image_name', 'choice', 'predict_label']]
df_consolidated_results = df_consolidated_results.rename(columns =  {"choice": "actual_label", "image_name": "file"})
df_consolidated_results


# In[35]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[36]:


matrix = confusion_matrix(df_consolidated_results['actual_label'],df_consolidated_results['predict_label'])
cr = classification_report(y_true = df_consolidated_results['actual_label']
                      ,y_pred = df_consolidated_results['predict_label'])

print(cr)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=mdl.dls.vocab)
disp.plot() 


# In[38]:


df_training_labels['choice'].value_counts()


# In[ ]:




