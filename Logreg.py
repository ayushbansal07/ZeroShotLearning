from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
import numpy as np
def logreg_model(train,seen_classes,classes):
	models = {}
	for x in seen_classes:
		print(x)
		clf = LinearSVC(random_state=0, tol=1e-5)
		clf = CalibratedClassifierCV(clf,cv=2)
		Y = classes[:,x]
		Y.shape = (train.shape[0],1)
		try:
			model = clf.fit(train, Y)
		except:
			Y[0] = 1 - Y[0]
			print("Retraining..")
			model = clf.fit(train, Y)
		models[x]=model
	return models
