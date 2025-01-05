import numpy as np
import random as rnd
from matplotlib import pyplot as plt
from numpy import array, zeros, diag, diagflat, dot
import scipy


""" ----- Groundtruth kernels we want to estimate. ----- """


class GroundtruthKernels:
    """Compute the groundtruth kernel evaluations."""
    def __init__(self):
        self.functions = {
            1: self.d_regularised_1,
            2: self.d_regularised_2,
            3: self.p_step_rw_2,
            4: self.diffusion,
            5: self.cosine
        }
    
    def d_regularised_1(self, U: array) -> array:
        """d regularised Lapacian kernel with d=1, (1-U)^{-1}"""
        return np.linalg.inv((np.eye(len(U)) - U))

    def d_regularised_2(self, U: int) -> float:
        """d regularised Lapacian kernel with d=2, (1-U)^{-2}"""
        return np.linalg.inv(np.matmul(np.eye(len(U)) - U, np.eye(len(U)) - U))

    def p_step_rw_2(self, U: int) -> float:
        """p step RW kernel with d=2, (1+U)^2"""
        return np.matmul(np.eye(len(U)) + U, np.eye(len(U)) + U)

    def diffusion(self, U: int) -> float:
        """Diffusion kernel, exp(U)"""
        return scipy.linalg.expm(U)

    def cosine(self, U: int) -> float:
        """Cosine kernel, sqrt{2}cos(pi/4-U) = cos(U) + sin(U)."""
        return scipy.linalg.sinm(U) + scipy.linalg.cosm(U) 

    def get_groundtruth_kernel(self, func_type: int, x: int) -> float:
        """
        Get the result of the modulation function of the given type.

        :param func_type: The type of modulation (1 to 5).
        :param x: The input integer.
        :return: The result of the modulation function as a float.
        """
        if func_type not in self.functions:
            raise ValueError(f"Invalid function type {func_type}. Must be between 1 and 5.")
        return self.functions[func_type](x)
        

""" ----- Functions to do with sampling random walks. ----- """


def get_regular_walk_path(adj_lists, base_vertex,
                     p_halt):
    "Do a simple random walk and get vertex history"
    vertex_log = [base_vertex]  # Set starting vertex
    nb_vertices = len(adj_lists)
    current_vertex = base_vertex
    while (rnd.uniform(0,1) > p_halt): # Terminate with probability p
        rnd_index = int(rnd.uniform(0,1) * len(adj_lists[current_vertex]))  # Choose a new vertex
        current_vertex = adj_lists[current_vertex][rnd_index]
        vertex_log.append(current_vertex)
    return vertex_log

def get_walks_list(adj_lists, base_vertex, p_halt, nb_random_walks):
    "Get a list of random walks out of a specified base vertex"
    all_walks_log = []
    for _ in range(nb_random_walks):
        all_walks_log.append(get_regular_walk_path(adj_lists, base_vertex, p_halt))
    return all_walks_log

def get_all_walks_list(adj_lists, p_halt, nb_random_walks):
    "Gets the walk lists out of every vertex"  
    nb_vertices = len(adj_lists)
    all_walks_list = []

    for vertex in range(nb_vertices):
        all_walks_list.append(get_walks_list(adj_lists, vertex, p_halt, nb_random_walks))
    
    return all_walks_list

def get_U_matrix(W, sigma):
    "Normalise an adjacency matrix based on its degree and a regulariser sigma"
    nb_vertices = len(W)
    U = np.zeros((nb_vertices,nb_vertices))
    degrees = np.sum(W,axis=1)
    for i in range(nb_vertices):
        for j in range(i):
            U[i,j] = W[i,j]/np.sqrt(degrees[i] * degrees[j]) 
    U += U.T
    U *= sigma**2
    return U

def adj_matrix_to_lists(A):
    "Get adjacency lists and weight lists for a weighted adjacency matrix"
    adj_lists = []
    weight_lists = []
    for i in range(len(A)):
        neighbors = []
        weights = []
        for j in range(len(A[i])):
            if A[i][j] != 0.0:
                neighbors.append(j)
                weights.append(A[i][j])
        adj_lists.append(neighbors)
        weight_lists.append(weights)
    return adj_lists, weight_lists

def frob_norm_error(true, approx, relative=True):
    """Get the (relative) Frobenius norm error between a pair of matrices"""
    assert np.shape(true) == np.shape(approx)
    frob_norm = np.sum((true - approx)**2)
    if relative:
        frob_norm /= np.sum(true**2)
    return frob_norm


""" ----- Functions to do with modulation functions. ----- """


def alpha_func_cosine(i):
    """Taylor expansion coefficients of cos(U)+sin(U).
    Helper function for computing the corresponding modulation function."""
    if i%2 == 0:
        return (-1)**(i/2) / scipy.special.factorial(i)
    elif i%2 != 0:
        return (-1)**((i-1)/2) / scipy.special.factorial(i)

def get_next_f_cosine(f_vec,g_eval):
    """Helper function for computing next f function evaluation, using Eq. 6."""
    f0 = f_vec[0]
    f1 = f_vec[1:]
    f1r = f1[::-1]
    f1dot = np.dot(f1,f1r)
    fnext = (g_eval - f1dot) / (2*f0)
    f_vec = list(f_vec)
    f_vec.append(fnext)
    return np.asarray(f_vec)

class ModulationFunctions:
    """Modulation functions to generate GRFs, based on inverse convolution of Taylor expansion.
    You can replace these with your own to approximate a different kernel, or learn the coefficients (Secs 3.4 and 3.5)"""
    def __init__(self):
        self.functions = {
            1: self.d_regularised_1,
            2: self.d_regularised_2,
            3: self.p_step_rw_2,
            4: self.diffusion,
            5: self.cosine
        }
    
    def d_regularised_1(self, x: int) -> float:
        """d regularised Lapacian kernel with d=1, (1-U)^{-1}"""
        if x == 0:
            return 1
        else:
            return scipy.special.factorial2(2*x-1) / (scipy.special.factorial2(2*x))

    def d_regularised_2(self, x: int) -> float:
        """d regularised Lapacian kernel with d=2, (1-U)^{-2}"""
        return 1

    def p_step_rw_2(self, x: int) -> float:
        """p step RW kernel with d=2, (1+U)^2"""
        return scipy.special.binom(1,x)

    def diffusion(self, x: int) -> float:
        """Diffusion kernel, exp(U)"""
        return 1 / (2**x * scipy.special.factorial(x))

    def cosine(self, x: int) -> float:
        """Cosine kernel, sqrt{2}cos(pi/4-U) = cos(U) + sin(U).
        Here, there isn't a convenient closed form so we use the iterative formula in Eq. 6"""
        f_vec = [1.0]
        for ind in range(int(x)):
            f_vec = get_next_f_cosine(f_vec, alpha_func_cosine(ind+1))
        return f_vec[-1]

    def get_modulation(self, func_type: int, x: int) -> float:
        """
        Get the result of the modulation function of the given type.

        :param func_type: The type of modulation (1 to 5).
        :param x: The input integer.
        :return: The result of the modulation function as a float.
        """
        if func_type not in self.functions:
            raise ValueError(f"Invalid function type {func_type}. Must be between 1 and 5.")
        return self.functions[func_type](x)


""" ----- Functions to actually construct GRFs to approximate graph kernels ----- """


def create_rf_vector_from_walks_list(U, adj_lists, p_halt, walks_list, f): 
    """Function to create an RF vector from a list of walks. 
    U is the weighted adjacency matrix."""

    nb_walks = len(walks_list)
    nb_vertices = len(adj_lists)
    rf_vector = np.zeros(nb_vertices)  

    # Find the longest walk.
    lengths = []
    for walk in walks_list:
        lengths.append(len(walk))
    longest_walk = max(lengths) 

    # Evaluate modulation function f up to longest walk length.
    f_vec = []
    for length in range(longest_walk):
        f_vec.append(f(length))
 
    # Store product of weights and marginal probabilities.
    for walk in walks_list:
        weights_product = 1.
        marginal_proby = 1.
        for step, node_pos in enumerate(walk):
            rf_vector[node_pos] += (weights_product  /  marginal_proby ) * f_vec[step]
            if step < len(walk)-1:
                weights_product *= U[walk[step]][walk[step+1]]
                marginal_proby *= (1 - p_halt) / len(adj_lists[node_pos])
    
    # Normalise by number if walks.
    rf_vector /= nb_walks

    return rf_vector
    
def create_lr_fact(U, adj_lists, p_halt, all_walks_list, f):
    """Construct a GRF for every graph node.
    Use it to get 'low rank' decomposition of kernel."""
  
    rf_vectors = []

    # Stack up GRF vectors for each start node.
    for walks_list in all_walks_list: 
        rf_vector = create_rf_vector_from_walks_list(U, adj_lists, p_halt, walks_list, f)
        rf_vectors.append(rf_vector)   
    
    A_matrix = np.asarray(rf_vectors)
    B_matrix = A_matrix.T

    return A_matrix, B_matrix

def get_approx_gram_mat(U, adj_lists, p_halt, all_walks_list, f):
    "Combine the GRFs to get a kernel estimate."
    A_matrix,B_matrix = create_lr_fact(U, adj_lists, p_halt, all_walks_list, f)
    # Take the matrix product between GRFs to get approximate graph kernel. 
    approximate_matrix = np.matmul(A_matrix, B_matrix)
    return approximate_matrix
