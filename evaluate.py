import numpy as np

def compute_unseen_class_scores(seen_class_scores, K, seen_classes, unseen_classes):

    return np.matmul(K[unseen_classes, :][:, seen_classes], seen_class_scores.T).T

# computes precision@5
def compute_precision(scores_gt, scores_pred):
    precision = 0
    num_questions = np.shape(scores_pred)[0]

    for i in range(num_questions):
        n_tags = np.sum(scores_gt[i,:])
        for j in np.argsort(scores_pred[i,:])[-5:]:
            if scores_gt[i, j]:
                precision += 1.0# * 5 / min(5, n_tags)

    print(precision)
    return precision/num_questions