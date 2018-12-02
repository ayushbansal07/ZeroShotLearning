from optparse import OptionParser
import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("--data_dir", dest="data_dir")
parser.add_option("--start_seen", dest="start_seen_classes")
parser.add_option("--end_seen", dest="end_seen_classes")
parser.add_option("--plot_file",dest="plot_file_name")

(options, args) = parser.parse_args()

data_dir = options.data_dir
start_seen_classes = int(options.start_seen_classes)
end_seen_classes = int(options.end_seen_classes)
plot_file_name = options.plot_file_name


from select_classes import select_classes
from Logreg import logreg_model
from RBM1 import train_RBM_and_compute_simiarity
from evaluate import compute_unseen_class_scores, compute_precision
from scipy import sparse
import numpy as np
import os


y_train = sparse.load_npz(data_dir+'tags_one_hot_train.npz')
y_train = y_train.todense()
y_test = sparse.load_npz(data_dir+'tags_one_hot_test.npz')
y_test = y_test.todense()

y_data = np.concatenate((y_train,y_test),axis=0)

X_train = sparse.load_npz(data_dir+'tfifdf_transformed_train.npz')
X_test = sparse.load_npz(data_dir+'tfifdf_transformed_test.npz')

if os.path.exists(data_dir + 'similarity_matrix.npy'):
	similarity_matrix = np.load(data_dir + 'similarity_matrix.npy')
else:
	similarity_matrix = train_RBM_and_compute_simiarity(y_data,target_filename=data_dir + 'similarity_matrix.npy')

plot_x_ent = []
plot_y_ent = []

plot_x_deg = []
plot_y_deg = []

modes = ['max-ent-uu', 'max-deg-uu']

for mode in modes:
	for num_seen_classes in range(start_seen_classes, end_seen_classes, 5):

		selected_classes = select_classes(similarity_matrix, num_seen_classes, mode)

		to_remove = []

		for class_idx in selected_classes:
			if np.sum(y_train[:, class_idx] < 0.5) < 5 or np.sum(y_train[:, class_idx] > 0.5) < 5:
				to_remove.append(class_idx)

		for class_idx in to_remove:
			selected_classes.remove(class_idx)

		models = logreg_model(X_train,selected_classes,y_train)

		y_pred  = np.zeros((y_test.shape[0],len(selected_classes)))

		i = 0
		for key in selected_classes:
			model = models[key]
			y_pred[:,i] = model.predict_proba(X_test)[:,1]
			i += 1

		unseen_classes = list(set(range(y_data.shape[1])) - set(selected_classes))
		score_unseen = compute_unseen_class_scores(y_pred,similarity_matrix,selected_classes,unseen_classes)

		precision = compute_precision(y_test[:,unseen_classes],score_unseen)

		print("%.6f" % (precision), num_seen_classes)
		if mode == 'max-ent-uu':
			plot_x_ent.append(num_seen_classes)
			plot_y_ent.append(precision)
		else:
			plot_x_deg.append(num_seen_classes)
			plot_y_deg.append(precision)

plot_x_ent = np.array(plot_x_ent)
plot_y_ent = np.array(plot_y_ent)
plot_x_deg = np.array(plot_x_deg)
plot_y_deg = np.array(plot_y_deg)

np.save(plot_file_name+"_x_ent.list",plot_x_ent)
np.save(plot_file_name+"_y_ent.list",plot_y_ent)
np.save(plot_file_name+"_x_deg.list",plot_x_deg)
np.save(plot_file_name+"_y_deg.list",plot_y_deg)

plt.plot(plot_x_ent,plot_y_ent,color='red')
plt.plot(plot_x_deg,plot_y_deg,color='blue')
plt.xlabel('Number of Seen Classes')
plt.ylabel('Precision @ 5')
plt.legend(loc='best')
plt.savefig(plot_file_name+".png")
