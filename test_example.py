from plot_utils import *
from math_utils import *
from low_rank_approximation import *

# initialization method for the V matrix
methods = 'sketching_g'
# termination tolerance on objective function values between iterations
epsilon = 1e-08
# values of k
ks = [
    1,
    2,
    4,
    8,
    16,
    32,
    64
]

# path to the 3 dataset
h_dir = Path('./data/img/horse')
m_dir = Path('./data/img/mnist_50')
p_dir = Path('./data/img/pokemon')
i = 0

for horse, mnist, pok in  zip(h_dir.iterdir(), m_dir.iterdir(), p_dir.iterdir()):
    
    # loading of images and scaling
    H = load_image_as_grayscale_matrix(horse)
    M = upscale_bilinear(load_image_as_grayscale_matrix(mnist), H.shape[0], H.shape[1]) # rank < 20
    P = upscale_bilinear(load_image_as_grayscale_matrix(pok), H.shape[0], H.shape[1]) # rank <= 80
    
    # random generation of the normal matrix
    G = np.random.uniform(np.min(H), np.max(H), H.shape) # max rank
    
    # for each k values an execution is performed for every image of the datasets
    for k in ks:
        
        # the algorithm starting method
        # the first 2 parameters are the A matrix to be approximated and the inner dimention of U and V.
        # the third, fourth and fifth are rispectively test class name, matrix name and specific test name
        # the sixth is the initialization method for the matrix V
        # the last is the epsilon value to set the termination tolerance
        # the start method compute the aproximation and save all the input values, output values and execution stats in 
        # files inside the folder: data_folder / c_name / m_name / t_name / 
        start(H, k, 'test_class', 'h_' + horse.name,  f'{k}', methods, epsilon=epsilon, data_folder='./data/test')
        start(M, k, 'test_class', 'm_' + mnist.name,  f'{k}', methods, epsilon=epsilon, data_folder='./data/test')
        start(P, k, 'test_class', 'p_' + pok.name,  f'{k}', methods, epsilon=epsilon, data_folder='./data/test')
        start(G, k, 'test_class', f'n_{i}',  f'{k}', methods, epsilon=epsilon, data_folder='./data/test')
    
    i +=1