import numpy as np
import pickle
import glob

def metric2_cos(a,b):
    """Compute cosine similarities between features"""
    a_norm = a/(a * a).sum(1, keepdims=True) ** .5
    b_norm = b/(b * b).sum(1, keepdims=True) ** .5
    similarity_matrix = a_norm @ b_norm.T
    return similarity_matrix

def get_cross_covariance(maml, x_spt, x_qry, y_spt, y_qry):
    """
    The feature (\phi(s) or \phi(q)) of data is obtained 
        through maml.net.forward_nolinear which returns
        the embedded data (i.e., the data before the linear layer).
    """
    # Obtain the features
    for i_batch in range(1):
        spt_feature = maml.net.forward_nolinear(x_spt[i_batch]).detach().cpu().numpy()
        qry_feature = maml.net.forward_nolinear(x_qry[i_batch]).detach().cpu().numpy()
        break

    memory = np.zeros((10,10,))
    for i in range(10):
        for j in range(10):
            # Concatenate the features
            if i <5: A = spt_feature[y_spt[i_batch].cpu().numpy()==i]
            else: A = qry_feature[y_qry[i_batch].cpu().numpy()==(i-5)]
            if j <5: B = spt_feature[y_spt[i_batch].cpu().numpy()==j]
            else: B = qry_feature[y_qry[i_batch].cpu().numpy()==(j-5)]
            # Compute cosine similarity
            memory[i, j] = np.mean(metric2_cos(A, B))
    return np.round(memory, 4)

def shuffle(y_spt, y_qry):
    """
    Since the model is overtrained on a single task, 
        it can easily suffer from the channel-memorization problem. 
    Thus, we perform a simple channel permutation to mitigate this problem. 
    """
    perm = np.random.permutation(5)
    y_spt += 5
    for i in range(5):
        y_spt[y_spt==(i+5)] = perm[i]
    y_qry += 5
    for i in range(5):
        y_qry[y_qry==(i+5)] = perm[i]
    return y_spt, y_qry


def get_averaged_matrix(name):
    matrix = np.zeros((11,10,10))
    count = 0
    for filename in glob.glob("./pickles/"+name+"*"):
        with open(filename, "rb") as input_file:
            matrix += np.array(pickle.load(input_file))
        count += 1
    return matrix/count

def get_map(data):
    temp = np.zeros((5,11))
    temp[:5,:5] = data[5:,:5]
    temp[:5,6:] = data[5:,5:]
    return temp