import os
import h
import exposing
import numpy as np
from sklearn import neighbors, tree, svm, naive_bayes, datasets, model_selection, neural_network
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def notify(title, text):
    os.system("""
              osascript -e 'display notification "{}" with title "{}"'
              """.format(text, title))

# Prepare clfs
clfs = {
    'SEE': None,
    'OEE': None,
    'DTC': tree.DecisionTreeClassifier,
    'kNN': neighbors.KNeighborsClassifier,
    'SVC': svm.SVC,
    'NBC': naive_bayes.GaussianNB
}

# Select groups of datasets
ds_groups = [
    "imb_IRhigherThan9p1",
    "imb_IRhigherThan9p2",
    "imb_IRlowerThan9",
    "imb_multiclass"
]

# Point db directory
ds_dir = "datasets"

# Iterating groups
for group_idx, ds_group in enumerate(ds_groups):
    notify(ds_group, "%i/%i" % (group_idx + 1,len(ds_groups)))
    group_path = "%s/%s" % (ds_dir, ds_group)
    print("## Group %s" % ds_group)

    # Iterating datasets in group
    ds_list = sorted(os.listdir(group_path))
    for ds_idx, ds_name in enumerate(ds_list):
        if ds_name[0] == '.' or ds_name[0] == '_':
            continue
        notify(ds_name, "%i/%i" % (ds_idx + 1,len(ds_list)))

        #if ds_name != 'ecoli1':
        #    continue
        print("\n### %s dataset" % ds_name)
        scores = np.zeros((len(clfs), 5))

        # Grid Search
        parameters = {
            'focus':[0,1,2,3,4,5],
            'a_steps':[0,1,2,3,4,5]
        }

        X, y = h.load_keel("%s/%s/%s.dat" % (
            group_path, ds_name, ds_name
        ))
        print("\nBest parameters")
        ee = exposing.EE(approach='brute')
        gs = GridSearchCV(ee, parameters)
        gs.fit(X, y)
        params = gs.best_params_
        print(params)

        for i in range(1,6):
            tra_path = "%s/%s/%s-5-fold/%s-5-%itra.dat" % (
                group_path, ds_name, ds_name, ds_name, i
            )
            tst_path = "%s/%s/%s-5-fold/%s-5-%itst.dat" % (
                group_path, ds_name, ds_name, ds_name, i
            )
            X_train, y_train = h.load_keel(tra_path)
            X_test, y_test = h.load_keel(tst_path)

            ee = None
            for j, clf_name in enumerate(clfs):
                # SEE
                if clf_name == 'SEE':
                    clf = exposing.EE()
                elif clf_name == 'OEE':
                    clf = exposing.EE(approach='brute',
                                      focus = params['focus'],
                                      a_steps = params['a_steps'])
                    ee = clf
                else:
                    clf = clfs[clf_name]()
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores[j,i-1] = score

        mean_scores = np.mean(scores, axis = 1)
        std_scores = np.std(scores, axis = 1)


        fig, ax = plt.subplots(1, figsize = (6,3))
        figname = 'figures/%s%s.png' % (ds_group,ds_name)
        ax.bar(clfs.keys(), mean_scores, yerr=std_scores)
        ax.set_ylim([0, 1])
        plt.savefig(figname)
        plt.savefig("bar.png")
        plt.close(fig)

        #print(len(ee.ensemble_))
        v = np.ceil(len(ee.ensemble_)/4).astype(int)
        #print(v)

        fig, ax = plt.subplots(v,4, figsize = (8, 2*v))
        fignameb = 'figures/%s%se.png' % (ds_group,ds_name)
        for e in range(v*4):
            if e < len(ee.ensemble_):
                ex = ee.ensemble_[e]
                ax[e // 4,e % 4].imshow(ex.rgb())
                ax[e // 4,e % 4].set_title("%s - %.3f" % (
                    ex.given_subspace, ex.theta_
                ), fontsize=8)
            ax[e // 4,e % 4].axis('off')
        plt.tight_layout()
        plt.savefig(fignameb)
        plt.savefig("foo.png")
        plt.close(fig)

        print("\n|CLF|ACC|STD|")
        print("|---|---|---|")
        for i, clf in enumerate(clfs):
            print("| %s | %.2f | +-%.2f|" % (clf, mean_scores[i], std_scores[i]))

        print("\n![](%s)" % figname)
        print("\n![](%s)" % fignameb)
