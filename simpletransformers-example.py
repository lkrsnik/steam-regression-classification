#!/usr/bin/env python
# coding: utf-8

# In[21]:


import polars as pl
import numpy as np
from simpletransformers.classification import ClassificationModel, ClassificationArgs


manual_seed = 23

np.random.seed(manual_seed)
pl.set_random_seed(manual_seed)


# # Load Data

# In[22]:


df_train = pl.read_parquet('data/complete_train.parquet')
df_dev = pl.read_parquet('data/complete_dev.parquet')
df_test = pl.read_parquet('data/complete_test.parquet')


# ## Downsize

# The obtained dataset contains ~45M reviews. Training on this amount would take too long, so I decided to train models on smaller chunks of data. I aimed at something that wouldn't take more than 6 hours of training. For Roberta, this meant training on 500k randomly selected reviews. I decided to evaluate data on 50k reviews, which means 10% of the size of the training data. While the amount of training data might change, this evaluation set will be used for all models.

# In[23]:


df_train = df_train.sample(500000, seed=manual_seed, shuffle=True)
df_dev = df_dev.sample(50000, seed=manual_seed, shuffle=True)
df_test = df_test.sample(50000, seed=manual_seed, shuffle=True)

# TODO DELETE THIS BECAUSE PREPROCESSING!
df_train = df_train.cast({'recommended': pl.Int8})
df_dev = df_dev.cast({'recommended': pl.Int8})
df_test = df_test.cast({'recommended': pl.Int8})


# ## Selecting relevant columns

# In[24]:


df_train = df_train.select(['review_text', 'recommended'])
df_dev = df_dev.select(['review_text', 'recommended'])
df_test = df_test.select(['review_text', 'recommended'])


# # Training

# ## Setup roberta training

# In[25]:


# setup classification arguments
classification_args = {
    'num_train_epochs': 1,
    'manual_seed': manual_seed,
    'save_steps': -1,
    'train_batch_size': 32
}
model_args = ClassificationArgs(**classification_args)


# In[26]:


# setup model
model_args = {
    'model_type': 'roberta',
    'model_name': 'roberta-large',
    'num_labels': 2,
    'args': model_args
}
model_args = {
    'model_type': 'distilbert',
    'model_name': 'distilbert/distilbert-base-uncased',
    'num_labels': 2,
    'args': model_args
}
model = ClassificationModel(**model_args)


# ## Train roberta

# In[ ]:

model.train_model(df_train.to_pandas(), output_dir='models/roberta-large')


# # Evaluating & Predicting

# In[19]:


df_train.sample(10)


# In[ ]:




