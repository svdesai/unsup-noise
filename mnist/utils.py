import numpy as np
from scipy.optimize import linear_sum_assignment

def shuffle_together(array1, array2):
    assert(array1.shape[0]==array2.shape[0])
    randomize = np.arange(array1.shape[0])
    np.random.shuffle(randomize)
    array1 = array1[randomize]
    array2 = array2[randomize]
    return array1, array2

def calc_optimal_target_permutation(reps, targets):
    # Compute cost matrix
    cost_matrix = np.zeros([reps.shape[0],targets.shape[0]])
    for i in range(reps.shape[0]):
        cost_matrix[:,i] = np.sum(np.square(reps-targets[i,:]),axis=1)

    _, col_ind = linear_sum_assignment(cost_matrix)
    # Permute
    targets[range(reps.shape[0])] = targets[col_ind]
    return targets

def generateTargetReps(n, z):
    # Use Marsaglias algorithm to generate targets on z-unit-sphere
    samples = np.random.normal(0, 1, [n, z]).astype(np.float32)
    radiuses = np.expand_dims(np.sqrt(np.sum(np.square(samples),axis=1)),1)
    reps = samples/radiuses
    return reps
