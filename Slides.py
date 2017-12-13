
# coding: utf-8

# In[63]:


get_ipython().magic(u'matplotlib notebook')

import scipy.io as spio
from scipy.signal import firwin,freqz,filtfilt
from scipy.fftpack import fft,fftfreq
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib as mpl
import csv
from pykalman import KalmanFilter
from jupyterthemes import jtplot
jtplot.style()
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
mpl.rcParams['agg.path.chunksize'] = 10000


# # Gesture Recognition Of An Arm Using A 9-DOF Inertial Measurement Unit
# <br>
# <center>Shanmugam Muruga Palaniappan</center>

#  <center>![](images/orthotic1.png)</center> 
#  

#  <center>![](images/orthotic2.png)</center> 
#  

# In[78]:


n_timesteps = 150
t = np.linspace(0, 1, n_timesteps)

data = []
dataOI = []
dataUD = []

for i in range (1,81):
    data.append(np.genfromtxt('OI/OI'+str(i)+'.csv', dtype=float, delimiter=','))
    dataOI.append(np.genfromtxt('OI/OI'+str(i)+'.csv', dtype=float, delimiter=','))

for i in range (1,81):
    data.append(np.genfromtxt('UD/UD'+str(i)+'.csv', dtype=float, delimiter=','))
    dataUD.append(np.genfromtxt('UD/UD'+str(i)+'.csv', dtype=float, delimiter=','))


sensors = {'Accelx':0,'Accely':1,'Accelz':2,'Magnetx':3,'Magnety':4,'Magnetz':5,'Gyrox':6,'Gyroy':7,'Gyroz':8}
sensorLabel = ['Accelx, G','Accely, G','Accelz, G','Magnetx, Gauss','Magnety, Gauss','Magnetz, Gauss','Gyrox, deg/s','Gyroy, deg/s','Gyroz, deg/s']

sensorsAll = sensors
sensorAllLabel=sensorLabel
dataAll = data[0]
data = np.array(data)
dataOI = np.array(dataOI)
dataUD = np.array(dataUD)
dataOI = dataOI.astype(float)
dataUD = dataUD.astype(float)


# In[98]:


dataOITraining = dataOI[0:(len(dataOI)/2)-1]
dataUDTraining = dataUD[0:(len(dataUD)/2)-1]
dataOITest = dataOI[(len(dataOI)/2):-1]
dataUDTest = dataUD[(len(dataUD)/2):-1]

X = np.concatenate((dataOITraining,dataUDTraining),axis=0) #putting them on top of each other
X = X.reshape(len(X),-1)

TestData = np.concatenate((dataOITest,dataUDTest),axis=0) #putting them on top of each other
TestData = TestData.reshape(len(TestData),-1)

OI = np.zeros(dataOITraining.shape[0])
UD = np.ones(dataUDTraining.shape[0])
y = np.concatenate((OI,UD),axis=0)
Testy= np.concatenate((OI,UD),axis=0) #labels are the same for test and training (first bits are OI and rest are UD)


# In[79]:


def f(x):
    return x
def pltsensor(f):
    plt.close()
    plt.plot(t, dataAll[:,f]);
    plt.ylabel(sensorAllLabel[f]);
    plt.xlabel('Time, seconds');
    plt.show();


# # Pick channels of interest for classification

# In[81]:


interact(pltsensor,f=sensorsAll);


# In[82]:


data = np.delete(data,[0,1,2,5,7,9,10,11], 2)
dataOI = np.delete(dataOI,[0,1,2,5,7,9,10,11], 2)
dataUD = np.delete(dataUD,[0,1,2,5,7,9,10,11], 2)
#update labels
sensors = {'Magnetx':0,'Magnety':1,'Gyrox':2,'Gyroz':3}
sensorLabel = ['Magnetx, Gauss','Magnety, Gauss','Gyrox, deg/s','Gyroz, deg/s']


# In[83]:


def kalman(f):
    plt.close()
    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),transition_covariance=0.01 * np.eye(2))
    observations = dataUD[0][:,f]
    states_pred = kf.em(observations).smooth(observations)[0]
    obs_scatter = plt.scatter(t, observations, marker='x', color='b',
                             label='observations')
    position_line = plt.plot(t, states_pred[:, 0],
                            linestyle='-', marker='o', color='r',
                            label='position est.')
    plt.ylabel(sensorLabel[f])
    plt.xlabel('Time, seconds')
    plt.legend()


# # Kalman Filtering

# In[84]:


interact(kalman,f=sensors);


# # Principal Component Analysis
# ## Dimensionality Reduction

# In[99]:


# dataflat = data.reshape(len(data),150*data.shape[2])
# dataflat = dataflat.astype(float)

# dataflatmat = np.matrix(dataflat)

#cov = (1.0/(150*data.shape[2]))*(dataflatmat.T*dataflatmat)

dataflat = X.astype(float)

dataflatmat = np.matrix(dataflat)

cov = (1.0/(150*data.shape[2]))*(dataflatmat.T*dataflatmat)


# In[100]:


plt.close()
plt.imshow(cov, cmap='seismic', interpolation='nearest');
plt.title('Covariance');
plt.colorbar();


# In[101]:


w,v = np.linalg.eig(cov)
sortedw =np.sort(w)
max1 = sortedw[len(sortedw)-1] #eigenvector corresponding to the largest eigenvalue
max2 = sortedw[len(sortedw)-2] #eigenvector corresponding to the second largest eigenvalue

vec1 = (v[:,0])
vec2 = (v[:,1])


# In[102]:


plt.close()
plt.plot(np.arange(len(vec1)),vec1);
plt.plot(np.arange(len(vec2)),vec2);
plt.title('Largest 2 Eigenvectors');


# In[155]:


def threshold(f):
    plt.close()
    akOI = (vec1.reshape(-1)) * dataOITraining.reshape(len(dataOITraining),-1).T
    bkOI = (vec2.reshape(-1)) * dataOITraining.reshape(len(dataOITraining),-1).T
    plt.scatter([akOI],[bkOI],color='magenta');
    
    akUD = (vec1.reshape(-1)) * dataUDTraining.reshape(len(dataUDTraining),-1).T
    bkUD = (vec2.reshape(-1)) * dataUDTraining.reshape(len(dataUDTraining),-1).T
    plt.scatter([akUD],[bkUD],color='cyan');
    plt.axvline(f)
    
def showTest(b):
    aktestOI = (vec1.reshape(-1)) * dataOITest.reshape(len(dataOITest),-1).T
    bktestOI = (vec2.reshape(-1)) * dataOITest.reshape(len(dataOITest),-1).T
    plt.scatter([aktestOI],[bktestOI],color='r');
    aktestUD = (vec1.reshape(-1)) * dataUDTest.reshape(len(dataUDTest),-1).T
    bktestUD = (vec2.reshape(-1)) * dataUDTest.reshape(len(dataUDTest),-1).T
    plt.scatter([aktestUD],[bktestUD],color='b');

button = widgets.Button(description = "Plot Test Data")


# # Lets Visualize this in 2-D

# In[184]:


display(button)
interact(threshold,f=(-1000,10,10));
button.on_click(showTest);


# # 97.4% Accuracy

# # Machine Learning

# In[ ]:


from sklearn import svm
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


# In[163]:


clf = RandomForestClassifier(min_samples_leaf=10)
clf.fit(X, y)
Prediction= clf.predict(TestData)


# In[180]:


def MLaccuracy():
    falsepos= 0
    accuracy= 1-np.logical_xor(Prediction,y)
    similarsum= np.sum(accuracy)
    similarity_perc= 1.0*similarsum/len(Prediction)
    falsehood= Prediction- y
    falsehood
    for i in xrange(0, len(falsehood)):
        if falsehood[i]>0:
            falsepos=falsepos+1

    display("Accuracy")        
    display(similarity_perc *100)
    display("False Positives")
    display(falsepos/float(len(accuracy)))


# In[181]:


MLaccuracy()

