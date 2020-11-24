# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:04:40 2019

@author: Yeshwanth
"""

import os
import glob
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from osgeo import gdal

os.chdir('H:\\M_Tech\\Lab_backup\\HP_Z4_workstation\\SC18M031\\Semester_3\\Outreach_Programme\\IDRB') #replace with your own directory.

#LOADING DATASET
kpip_watershed_dataset = gdal.Open("H:\\M_Tech\\Lab_backup\\HP_Z4_workstation\\SC18M031\\Semester_3\\Outreach_Programme\\IDRB\\Data\\Sentinel2_Imagery\\new_data\\mosaiced\\envi\\tiles_mosaicked_clipped.tif") #replace with your own raster directory.
#kpip_watershed_dataset = gdal.Open("Data\\Sentinel2_Imagery\\final_imagery_kpip\\kpip_wshed_clipped.tif") #replace with your own raster directory.
kpip_img = kpip_watershed_dataset.ReadAsArray()
kpip_img1 = np.zeros((kpip_img.shape[1],kpip_img.shape[2],kpip_img.shape[0]))

for i in range(kpip_img.shape[0]):
    kpip_img1[:,:,i] = kpip_img[i,:,:]
#    kpip_img1[:,:,i] = (kpip_img1[:,:,i] - np.min(kpip_img1[:,:,i]))/(np.max(kpip_img1[:,:,i]) - np.min(kpip_img1[:,:,i]))

kpip_img2 = kpip_img1.reshape(kpip_img.shape[1]*kpip_img.shape[2],kpip_img.shape[0])

scaler = StandardScaler()
scaler.fit(kpip_img2)
kpip_img2 = scaler.transform(kpip_img2)


#LOADING LABELLED SAMPLES
tr_sample_files = glob.glob('ROIs\\new_trial\\*.csv')
tr_sample_paths = []
for string in tr_sample_files:
    path = os.path.abspath(string)
    path1 = Path(path)
    tr_sample_paths.append(path1)

labelled_samples = {}
cols = []
for i in range(1,11):
    cols.append('B'+str(i))
for i in range(len(tr_sample_paths)):
    class_name = os.path.splitext(os.path.basename(tr_sample_files[i]))[0]
    labelled_samples[class_name] = pd.read_csv(tr_sample_paths[i],
                    skiprows=[0,1,2,3,4,5,6,8],
                    skipinitialspace=True,
                    usecols=cols,
                    )

#Number of labelled samples, features and classes.
ns = 0
df_list = []
for key, df in labelled_samples.items():
    ns += len(df)
    df_list.append(df)

nf = labelled_samples['ayacut'].shape[1]
nc = len(labelled_samples)

#Sample Matrix.
data_x = np.array(pd.concat(df_list,ignore_index=True))
data_x = scaler.transform(data_x)

#Target Matrix.
data_y = np.zeros((ns,nc))

df = None
start = 0
end = 0
cc = []
for i in range(len(df_list)):
    df = df_list[i]
    end += len(df)
    data_y[start:end,i] = 1
    cc.append((start,end))
    start += len(df)

#Splitting into training and testing data.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)

#Defining Classifier.
iters = 2000
clf = MLPClassifier(hidden_layer_sizes=(8,16,32,64,32,16,8),
                    max_iter=iters,
                    n_iter_no_change=50,
                    verbose=True)

st = time.time()
#Training classifier.
clf.fit(x_train, np.argmax(y_train,axis=1))
test_predictions = clf.predict(x_test)

print('Confusion Matrix:\n',confusion_matrix(np.argmax(y_test,axis=1), test_predictions))
print('Classification Report:\n', classification_report(np.argmax(y_test,axis=1), test_predictions))
print('Accuracy Score:',accuracy_score(np.argmax(y_test,axis=1), test_predictions))

#Making Predictions.
preds = clf.predict(kpip_img2)
preds2 = preds.reshape(kpip_img.shape[1],kpip_img.shape[2])
plt.figure(figsize=(10,10))
plt.imshow(preds2)

#Writing as a Geotiff.
geo_trf = kpip_watershed_dataset.GetGeoTransform()
proj = kpip_watershed_dataset.GetProjection()

def write_geotiff(fname,data,geotransform,projection):
    driver = gdal.GetDriverByName('GTiff')
    rows,cols = data.shape
    dataset = driver.Create(fname,cols,rows,1,gdal.GDT_UInt32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None
    
write_geotiff("Results\\new_trial1\\envi_mlp\\kpip_classified_mlp_{0}_iters.tiff".format(iters),
              preds2,
              geo_trf,
              proj)

et = time.time()

print('Time elapsed: {0} seconds'.format(et-st))