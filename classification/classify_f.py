import numpy as np
import scipy.io as sio
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from operator import truediv
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')



for namedset in ['indian','pavia']:
	for tipoim in ['RAW', 'DEN']:
		ev_oa = []
		noise_levels = [0] if tipoim == "RAW" else [5,25,50,75,100]
		for noise_level in noise_levels:
			bests_results_oa = []
			bests_results_k = []
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

				sizeor = labels_or.shape
				image_or  = image_or.reshape(-1,image_or.shape[-1])
				labels_or = labels_or.reshape(-1)
				image_or = MinMaxScaler().fit_transform(image_or)
				image  = image_or[labels_or!=0]
				labels = labels_or[labels_or!=0]
				labels -= 1

				train_x, test_x, train_y, test_y = train_test_split(image, labels, test_size=0.90, stratify=labels, random_state=idtest)

				bestc = -100000; bestacc = 0
				for c in range(-8, 8):
					model = LinearSVC(C=10**c, dual=0).fit(train_x, train_y)
					ytestp = accuracy_score(test_y, model.predict(test_x))
					if ytestp > bestacc:
						bestc = c
						bestacc = ytestp
					del model
				model = LinearSVC(C=10**bestc, dual=0).fit(train_x, train_y)

				bests_results_oa.append(100*accuracy_score(test_y, model.predict(test_x)))
				bests_results_k.append(100*cohen_kappa_score(test_y, model.predict(test_x)))
				del model, train_x, test_x, train_y, test_y, image, labels, labels_or, image_or
			bests_results_oa = np.average(np.array(bests_results_oa), axis=0)
			bests_results_k = np.average(np.array(bests_results_k), axis=0)
			print(namedset, tipoim, noise_level, "SVC LIN with C="+str(bestc),"(ACC="+str(np.round(bests_results_oa, 2)) + ", K(x100)="+str(np.round(bests_results_k, 2))+")")
			ev_oa.append(bests_results_oa)
		
		if tipoim == "DEN":
			plt.plot([5,25,50,75,100], ev_oa, "x-", label="AfterDEN")
			plt.legend(); plt.xticks([5,25,50,75,100], [5,25,50,75,100]);
			plt.xlabel("Denoising level"); plt.ylabel("Overall Accuracy (%)"); plt.show()
		else:
			plt.plot([5,25,50,75,100], ev_oa*5, label="Original")

