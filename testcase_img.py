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
p_dir = Path('./data/img/pokemon')
i = 0
for horse, mnist, pok in  zip(h_dir.iterdir(), m_dir.iterdir(), p_dir.iterdir()):
    H = load_image_as_grayscale_matrix(horse)
    M = upscale_bilinear(load_image_as_grayscale_matrix(mnist), H.shape[0], H.shape[1]) # rank < 20
    P = upscale_bilinear(load_image_as_grayscale_matrix(pok), H.shape[0], H.shape[1]) # rank <= 80
    G = np.random.uniform(np.min(H), np.max(H), H.shape) # rank max
    
    for k in ks:
        #start(H, k, 'g_img', 'h_' + horse.name,  f'{k}', methods, epsilon=epsilon)
        #start(M, k, 'g_img', 'm_' + mnist.name,  f'{k}', methods, epsilon=epsilon)
        start(P, k, 'g_img', 'p_' + pok.name,  f'{k}', methods, epsilon=epsilon)
        #start(G, k, 'g_img', f'n_{i}',  f'{k}', methods, epsilon=epsilon)
    
    i +=1