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
p_dir = Path('./data/img/pokemon')
i = 0

# TODO: TEST 448 ha dato problemi, ripeterlo k=64
for horse, pok in zip([Path('./data/img/horse/horse216.png')],[Path('./data/img/pokemon/448.png')]):
    
    H = load_image_as_grayscale_matrix(horse)
    P = upscale_bilinear(load_image_as_grayscale_matrix(pok), H.shape[0], H.shape[1]) # rank <= 80
    
    for k in ks:
        start_OTS_Householder(H, k, 'g_ots', 'h_' + horse.name, 'Householder_' + f'{k}', methods, epsilon=epsilon)
        start_OTS_No_Householder(H, k, 'g_ots', 'h_' + horse.name, 'No_Householder_' + f'{k}', methods, epsilon=epsilon)
        
        start_OTS_Householder(P, k, 'g_ots', 'p_' + pok.name, 'Householder_' + f'{k}', methods, epsilon=epsilon)
        start_OTS_No_Householder(P, k, 'g_ots', 'p_' + pok.name, 'No_Householder_' + f'{k}', methods, epsilon=epsilon)
    
    i +=1   