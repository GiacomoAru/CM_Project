from plot_utils import *
from math_utils import *
from low_rank_approximation import *

methods = 'sketching_g'
epsilon = 1e-08
ks = [
    1,
    2,
    4,
    8,
    16,
    32,
    64
]


h_dir = Path('./data/img/horse')
m_dir = Path('./data/img/mnist_50')
for i, horse, mnist in enumerate(zip(h_dir.iterdir(), m_dir.iterdir())):
    H = load_image_as_grayscale_matrix(horse)
    M = upscale_bilinear(load_image_as_grayscale_matrix(mnist), H.shape)
    G = np.random.uniform(np.min(H), np.max(H), H.shape)
    
    for k in ks:
        start(H, k, 'img', horse.name,  f'{k}', methods, epsilon=epsilon)
        start(M, k, 'img', mnist.name,  f'{k}', methods, epsilon=epsilon)
        start(G, k, 'img', f'simil_horse_{i}',  f'{k}', methods, epsilon=epsilon)