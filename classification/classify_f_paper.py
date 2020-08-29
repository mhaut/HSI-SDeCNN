import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


image_or  = sio.loadmat("datasets/indian_pines.mat")['test_data']
labels_or = sio.loadmat("datasets/indian_pines_GT.mat")['indian_pines_GT']
sizeor = labels_or.shape
image_or  = image_or.reshape(-1,image_or.shape[-1])
labels_or = labels_or.reshape(-1)
image_or = MinMaxScaler().fit_transform(image_or)
image  = image_or[labels_or!=0]
labels = labels_or[labels_or!=0]
labels -= 1
train_x, test_x, train_y, test_y = train_test_split(image, labels, train_size=0.1, stratify=labels, random_state=16)
model = SVC(C=10, kernel="linear").fit(train_x, train_y)
print("------ INDIAN PINES -------")
print("----------- raw -----------")
print("OA",    100*accuracy_score(test_y, model.predict(test_x)))
print("Kx100", 100*cohen_kappa_score(test_y, model.predict(test_x)))
del model

image_or = sio.loadmat("denoised_indian/denoised_indian_noiselevel_50.mat")['new_img']
labels_or = sio.loadmat("datasets/indian_pines_GT.mat")['indian_pines_GT']
sizeor = labels_or.shape
image_or  = image_or.reshape(-1,image_or.shape[-1])
labels_or = labels_or.reshape(-1)
image_or = MinMaxScaler().fit_transform(image_or)
image  = image_or[labels_or!=0]
labels = labels_or[labels_or!=0]
labels -= 1
train_x, test_x, train_y, test_y = train_test_split(image, labels, train_size=0.1, stratify=labels, random_state=16)
model = SVC(C=1000, kernel="linear").fit(train_x, train_y)
print("-------- denoised ---------")
print("OA",    100*accuracy_score(test_y, model.predict(test_x)))
print("Kx100", 100*cohen_kappa_score(test_y, model.predict(test_x)))
del model

print()
print()
print()
print()
print()
print()

image_or = sio.loadmat("datasets/pavia_u.mat")['test_data'][:,:,:20]
labels_or = sio.loadmat("datasets/pavia_u_GT.mat")['GT_data']
sizeor = labels_or.shape
image_or  = image_or.reshape(-1,image_or.shape[-1])
labels_or = labels_or.reshape(-1)
image_or = MinMaxScaler().fit_transform(image_or)
image  = image_or[labels_or!=0]
labels = labels_or[labels_or!=0]
labels -= 1
train_x, test_x, train_y, test_y = train_test_split(image, labels, train_size=0.1, stratify=labels, random_state=1)
model = SVC(C=100, kernel="linear").fit(train_x, train_y)
print("---------- PAVIA ----------")
print("----------- raw -----------")
print("OA",    100*accuracy_score(test_y, model.predict(test_x)))
print("Kx100", 100*cohen_kappa_score(test_y, model.predict(test_x)))

del model
image_or = sio.loadmat("denoised_pavia/denoised_pavia_noiselevel_50.mat")['new_img'][:,:,:20]
labels_or = sio.loadmat("datasets/pavia_u_GT.mat")['GT_data']
sizeor = labels_or.shape
image_or  = image_or.reshape(-1,image_or.shape[-1])
labels_or = labels_or.reshape(-1)
image_or = MinMaxScaler().fit_transform(image_or)
image  = image_or[labels_or!=0]
labels = labels_or[labels_or!=0]
labels -= 1
train_x, test_x, train_y, test_y = train_test_split(image, labels, train_size=0.1, stratify=labels, random_state=1)
model = SVC(C=220, kernel="linear").fit(train_x, train_y)
print("---------- denoised ----------")
print("OA",    100*accuracy_score(test_y, model.predict(test_x)))
print("Kx100", 100*cohen_kappa_score(test_y, model.predict(test_x)))
del model
