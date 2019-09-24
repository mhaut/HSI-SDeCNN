import numpy as np
import scipy.io as sio
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')


def random_unison(a,b, rstate=None):
	assert len(a) == len(b)
	p = np.random.RandomState(seed=rstate).permutation(len(a))
	return a[p], b[p]

def train_test_split(image, labels, test_size=0.7, random_state=None):
	pixels_number = np.unique(labels, return_counts=1)[1]
	train_set_size = [round(round(a*(1-test_size), 1),0) for a in pixels_number]
	tr_size = int(sum(train_set_size))
	te_size = int(sum(pixels_number)) - int(sum(train_set_size))
	train_x = np.empty((tr_size,image.shape[-1])); train_y = np.empty((tr_size)); test_x = np.empty((te_size,image.shape[-1])); test_y = np.empty((te_size))
	trcont = 0; tecont = 0;
	for cl in np.unique(labels):
		pixels_cl = image[labels==cl]
		labels_cl = labels[labels==cl]
		pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=random_state)
		for cont, (a,b) in enumerate(zip(pixels_cl, labels_cl)):
			if cont < train_set_size[cl]:
				train_x[trcont,:] = a
				train_y[trcont] = b
				trcont += 1
			else:
				test_x[tecont,:] = a
				test_y[tecont] = b
				tecont += 1
	train_x, train_y = random_unison(train_x, train_y, rstate=random_state)
	test_x, test_y   = random_unison(test_x, test_y, rstate=random_state)
	return train_x, test_x, train_y, test_y


def my_metrics_classfier(real_labels, predicted_labels, labels):
	overall_accuracy = accuracy_score(predicted_labels, real_labels)
	confusion_matrix_ = confusion_matrix(real_labels, predicted_labels, labels=labels)
	list_diag = np.diag(confusion_matrix_); list_raw_sum = np.sum(confusion_matrix_, axis=1)
	average_accuracy = np.average(np.nan_to_num(truediv(list_diag, list_raw_sum)))
	kappa            = cohen_kappa_score(real_labels, predicted_labels)
	return [overall_accuracy, average_accuracy, kappa]


for namedset in ['indian','pavia']:
	for tipoim in ['RAW', 'DEN']:
		ev_oa = []
		noise_levels = [0] if tipoim == "RAW" else [5,25,50,75,100]
		for noise_level in noise_levels:
			bests_results = []
			for idtest in range(1,6):
				if namedset == "indian":
					if tipoim == "RAW":
						image_or = sio.loadmat("datasets/indian_pines.mat")['test_data']
					else:
						image_or = sio.loadmat("denoised_indian/" +"/denoised_indian_noiselevel_"+str(noise_level)+".mat")['new_img']
					labels_or = sio.loadmat("datasets/indian_pines_GT.mat")['indian_pines_GT']
				elif namedset == "pavia":
					if tipoim == "RAW":
						image_or = sio.loadmat("datasets/pavia_u.mat")['test_data']
					else:
						image_or = sio.loadmat("denoised_pavia/" +"/denoised_pavia_noiselevel_"+str(noise_level)+".mat")['new_img']
					labels_or = sio.loadmat("datasets/pavia_u_GT.mat")['GT_data']
					image_or = image_or[:,:,:20]
					#plt.imshow(image_or[:,:,10], cmap="gray")
					#plt.imshow(labels_or, cmap="jet", alpha=0.45)
					#plt.show()
					#exit()

				sizeor = labels_or.shape
				image_or  = image_or.reshape(-1,image_or.shape[-1])
				labels_or = labels_or.reshape(-1)
				if tipoim == "RAW": image_or = StandardScaler().fit_transform(image_or)
				image  = image_or[labels_or!=0]
				labels = labels_or[labels_or!=0]
				labels -= 1

				train_x, test_x, train_y, test_y = train_test_split(image, labels, test_size=0.90, random_state=idtest)

				bestc = -100000; bestacc = 0
				for c in range(-8, 8):
					model = LinearSVC(C=10**c, dual=0).fit(train_x, train_y)
					ytestp = accuracy_score(test_y, model.predict(test_x))
					if ytestp > bestacc:
						bestc = c
						bestacc = ytestp
					del model
				model = LinearSVC(C=10**bestc, dual=0).fit(train_x, train_y)

				ytestp = np.array(my_metrics_classfier(test_y, model.predict(test_x), np.unique(test_y)))
				bests_results.append(ytestp)
				del model, train_x, test_x, train_y, test_y, image, labels, labels_or, image_or, ytestp
			bests_results = np.average(np.array(bests_results), axis=0)
			print(namedset, tipoim, noise_level, "SVC LIN with C="+str(bestc),"ACC="+str(np.round(bests_results*100, 2)))
			ev_oa.append(bests_results[0])
		
		if tipoim == "DEN":
			plt.plot([5,25,50,75,100], ev_oa, "x-", label="AfterDEN")
			plt.legend(); plt.xticks([5,25,50,75,100], [5,25,50,75,100]);
			plt.xlabel("Denoising level"); plt.ylabel("Overall Accuracy (%)"); plt.show()
		else:
			plt.plot([5,25,50,75,100], ev_oa*5, label="Original")
		
		
