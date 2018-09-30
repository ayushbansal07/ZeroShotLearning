from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import numpy as np
def logreg_model(train,seen_classes,classes):
	models = {}
	for x in seen_classes:
		clf = LinearSVC(random_state=0, tol=1e-5)
		Y = classes[:,x]
		Y.shape = (train.shape[0],1)
		model = clf.fit(train, Y)
		models[x]=model
	return models