import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation
import json
import numpy as np
from sklearn.cluster import KMeans

f = open("fixed_people_with_vectors_234")
people = json.load(f)

final_people = []
for p in people:
    #print p["name"], p["vector"] # this is a 20D vector
    print p["vector"]
    if p["party"] in ["D", "R"] and not (0 in p["vector"]): # and 0 in p["vector"][p["vector"].index(0)+1:]):
        final_people.append(p)

print len(final_people)
people = final_people

    

X = np.array([p["vector"] for p in people])
Y = np.array(["blue" if p['party'] == "D" else "red" for p in people])
clf = SVC()

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)
#clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
#print clf.score(X_test, y_test)
#exit(0)

#pca = PCA(n_components=3)
#X_new = pca.fit_transform(X)
#for x, x_new in zip(X, X_new):
#  print x, x_new
print people[37]


# import some data to play with

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)

kk = KMeans(n_clusters=3, random_state=1).fit(X_reduced)
kkk = {0: [], 1: [], 2: []}
w1 = []
w2 = []
w3 = []
for p in people:
    #kkk[kk.predict(X_reduced[people.index(p)])[0]].append(p["party"])
    #print p["party"], kk.predict(X_reduced[people.index(p)])[0]
    cooc = kk.predict(X_reduced[people.index(p)])[0]
    if cooc == 0:
        w1.append(p["vector"])
    elif cooc == 1:
        w2.append(p["vector"])
    else:
        w3.append(p["vector"])
print np.round(np.mean(np.array(w1),  axis=0), 1)
print np.round(np.mean(np.array(w2),  axis=0), 1)
print np.round(np.mean(np.array(w3),  axis=0), 1)


print np.round(np.mean(np.array(w1),  axis=0), 1) - np.round(np.mean(np.array(w3),  axis=0), 1)
print np.round(np.mean(np.array(w2),  axis=0), 1) - np.round(np.mean(np.array(w3),  axis=0), 1)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
           cmap=plt.cm.Paired, label="smth")
#ax.legend([blue_proxy,red_proxy],['cars','bikes'])
avg_0 = 0
count_0 = 0
avg_1 = 0
count_1 = 0
avg_0 = -9.73681457235
avg_1 = 11.6958733602
for i in range(len(X_reduced)):
    if Y[i] == 0:
        if abs(X[i][0] - avg_0) -  abs(X[i][0] - avg_1) > 12:
            pass
            #print people[i]
    else:
        if abs(X[i][0] - avg_1) - abs(X[i][0] - avg_0) > 11.955:
            #pass
            print abs(X[i][0] - avg_1) - abs(X[i][0] - avg_0), people[i]
        
    
#ax.set_title("OnThisIssue dataset in 3D(Rep=red, Dem=blue)")
#ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
#ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
#ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

plt.show()

