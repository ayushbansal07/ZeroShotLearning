from select_classes import select_classes
from Logreg import logreg_model
from RBM import train_RBM_and_compute_simiarity
from evaluate import compute_unseen_class_scores, compute_precision
from scipy import sparse
import numpy as np
import os

NUM_SEEN_CLASSES = 20
y_data = np.load('tags_one_hot.npy')

if os.path.exists('similarity_matrix.npy'):
	similarity_matrix = np.load('similarity_matrix.npy')
else:
	similarity_matrix = tran_RBM_and_compute_simiarity(y_data)
similarity_matrix = 1/similarity_matrix
#selected_classes = np.random.randint(0, 700, (20))
selected_classes = select_classes(similarity_matrix,NUM_SEEN_CLASSES,'max-deg-uu')
#selected_classes = [22, 104, 146, 204, 237, 290, 345, 399, 417, 425, 440, 511, 527, 565, 570, 606, 697, 708, 715, 741]
print(selected_classes)
X_data = sparse.load_npz('tfifdf_transformed.npz')
X_data = X_data.todense()
#y_data = np.load('tags_one_hot.npy')

NUM_TRAIN = int(y_data.shape[0]*.70)
perm = np.random.permutation(NUM_TRAIN)
X_train = X_data[:NUM_TRAIN]
X_test = X_data[NUM_TRAIN:]
y_train = y_data[:NUM_TRAIN]
y_test = y_data[NUM_TRAIN:]

models = logreg_model(X_train,selected_classes,y_train)

y_pred  = np.zeros((y_test.shape[0],NUM_SEEN_CLASSES))
i = 0
for key,model in models.items():
	y_pred[:,i] = model.predict_proba(X_test)[:,1]
	i += 1

unseen_classes = list(set(range(y_data.shape[1])) - set(selected_classes))
score_unseen = compute_unseen_class_scores(y_pred,similarity_matrix,selected_classes,unseen_classes)

precision = compute_precision(y_test[:,unseen_classes],score_unseen)

print("Precision @ 5 : %.6f" % (precision))
