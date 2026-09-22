import numpy as np
import torch
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt


train_in = np.loadtxt("train_in - Copy.csv", delimiter=",")
train_out = np.loadtxt("train_out - Copy.csv", delimiter=",")
## test if reading the train set correcctly 
# print(train_in.shape)
# print(train_out.shape)


def get_vectors_of_target_number(x):
    'target number as input,'
    'get an index list that will apply to train_in and to extract corresponding vectors'
    "return the list of vectors"
    index_list=[];
    ## get index list of target number
    for i in range(len(train_out)):
        if train_out[i] == x:
            #print(i)
            index_list.append(i)

    ## get vectors
    vectors_list_x=[];
    for i in index_list:
       vectors_list_x.append(train_in[i])

    # conversion list -> tensors
    vectors_x= torch.tensor(vectors_list_x, dtype=torch.float32)
    #print(vectors_x.shape)
    return vectors_x


# to test vectors
#print(get_vectors_of_target_number(6))
#get_vectors_of_target_number(6)

# get cloud 0,1,2,.....9 and their centers 
centers_list=[]
for i in range(0,10):
    vectors_x=get_vectors_of_target_number(i)
    centers_x = vectors_x.mean(dim=0)
    centers_list.append(centers_x)

centers= torch.stack(centers_list)
print("centers.shape: ", centers.shape)

#calculate the distances between the centers of the 10 clouds, to know which pairs are more difficult to seperate
# d= distance 
d_centers= torch.cdist(centers, centers)
print("centers distances: ", d_centers.shape)
# test centers distances calculation function
#print(d_centers[2,7])

# look for the closest pair 
pairs =[]

for i in range(0,10):
    for j in range(i+1,10):
        distance=d_centers[i,j].item();
        pairs.append((distance, i,j))
pairs.sort()
 
#print(pairs) 7,9


# -----question 2 ---------

pca = PCA(n_components=2)
train_pca = pca.fit_transform(train_in)

print(train_in.shape)
print(train_pca.shape) # after pca
#print(train_pca[0])

plt.figure()
scatter=plt.scatter(
    train_pca[:,0],
    train_pca[:,1],
    c=train_out,# color 
    cmap="tab10",
    s=10,
    alpha=0.7
)

plt.title("PCA visualisation")
plt.show()










