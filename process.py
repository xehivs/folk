import os
import h
import exposing
import numpy as np
from sklearn import neighbors, tree, svm, naive_bayes, datasets, model_selection, neural_network

# Set grain
grain = 16

# Prepare clfs
clfs = {
    ' EE': exposing.EE,
    'DTC': tree.DecisionTreeClassifier,
    'kNN': neighbors.KNeighborsClassifier,
    'SVC': svm.SVC,
    'NBC': naive_bayes.GaussianNB
}

# Select groups of datasets
ds_groups = [
    "imb_IRlowerThan9",
    "imb_multiclass"
]

# Point db directory
ds_dir = "datasets"

for ds_group in ds_groups:
    group_path = "%s/%s" % (ds_dir, ds_group)
    print("Group %s" % group_path)
    for ds_name in sorted(os.listdir(group_path)):
        if ds_name[0] == '.' or ds_name[0] == '_':
            continue
        print("\n/\n| %s dataset" % ds_name)

        scores = np.zeros((len(clfs), 5))

        for i in range(1,6):
            tra_path = "%s/%s/%s-5-fold/%s-5-%itra.dat" % (
                group_path, ds_name, ds_name, ds_name, i
            )
            tst_path = "%s/%s/%s-5-fold/%s-5-%itst.dat" % (
                group_path, ds_name, ds_name, ds_name, i
            )
            X_train, y_train = h.load_keel(tra_path)
            X_test, y_test = h.load_keel(tst_path)

            for j, clf_name in enumerate(clfs):
                clf = clfs[clf_name]()
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores[j,i-1] = score

        mean_scores = np.mean(scores, axis = 1)
        std_scores = np.std(scores, axis = 1)

        for i, clf in enumerate(clfs):
            print("| %s | %.2f [+-%.2f]" % (clf, mean_scores[i], std_scores[i]))
