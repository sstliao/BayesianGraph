######################

###### required modules ######
exec(open('import_modules.py').read())
exec(open('MCMC_run.py').read())
exec(open('multi_stage_lasso.py').read())
exec(open('MCMC_prediction.py').read())

###### Spectral Decomposition of Graph Laplacian ######
def graph_matrix_eigen(G):
    n = len(list(G.nodes))
    W = np.zeros((n,n))
    w = list(G.edges.data('weight'))
    for i in w:
        W[i[0],i[1]],W[i[1],i[0]] = i[2],i[2] 
    D = np.diag(W.dot(np.ones((n,1))).reshape((n,)))
    L = D-W
    eign_val, eign_vec = LA.eig(L) # eign_vec[:,i] corresponds to eign_val[i]
    ## order the eigenvalues
    idx = eign_val.argsort()[::1]#1: increasing, -1: decreasing 
    eign_val = eign_val[idx]
    eign_vec = eign_vec[:,idx]
    return L,D,W,eign_val,eign_vec  
    
###### Construction and visualization of graph wavelets and scaling function #######
def s(x):
    return -5+11*x-6*(x**2)+(x**3)

def g(x, alpha=2, beta=2, x1=1, x2=2):
    if x<x1:
        return (x1**(-alpha))*(x**(alpha))
    elif x>x2:
        return (x2**(beta))*(x**(-beta))
    else:
        return s(x)

def h(x,gamma,lambda_min):
    return gamma*np.exp(-(x/(0.6*lambda_min))**4)

def kernal_gh(lambda_set,J=5,lambda_max=10,K=20, x1=1, x2=2):

    lambda_min = lambda_max/K
    t1 = x2/lambda_min
    tJ = x1/lambda_max
    tj = np.logspace(np.log(t1),np.log(tJ),num=J,base = np.exp(1))
    
    g_list = np.array([[g(tj[i]*x) for x in lambda_set] for i in range(len(tj))])
    
    gamma = np.max(g_list)
    h_val = np.array([h(x,gamma = gamma,lambda_min = lambda_min) for x in lambda_set])
    
    g_part = np.sum(g_list**2,axis = 0)
    G = h_val**2+g_part
    
    return g_list, h_val, G
  
def get_kernel_plot(lambda_max,J,K):
    g_list, h_val, G = kernal_gh(lambda_set = np.linspace(0, lambda_max,num=1000),
                                 J=J,lambda_max=lambda_max,K=K, x1=1, x2=2) 
    lambda_set = np.linspace(0, lambda_max,num=1000)
    plt.xlim((0,lambda_max ))
    for j in range(J):
        plt.plot(lambda_set,g_list[j], label = "g (j = "+str(j+1)+")")
    plt.plot(lambda_set,h_val, label = "h")
    plt.plot(lambda_set,G, label = "G",color='black')
    plt.legend(loc = 'upper right')
    plt.title('kernel functions')
    plt.show()   

    
## scaling function
def phi(v,u,L_lambda_set,L_lambda_vec,h_vals):
    n = len(L_lambda_set)
    eigen_vec_comp = np.array([L_lambda_vec[v,l]*L_lambda_vec[u,l] for l in range(n)])
    
    return sum(eigen_vec_comp*h_vals)

## wavelets
def psi(j,v,u,L_lambda_set,L_lambda_vec,g_vals): # j starting from 0 to J-1
    n = len(L_lambda_set)
    eigen_vec_comp = np.array([L_lambda_vec[v,l]*L_lambda_vec[u,l] for l in range(n)])
    g_j_vals = g_vals[j,:]
    
    return sum(eigen_vec_comp*g_j_vals)



def get_scaling_plot(vertex_center, scaling_func):
    vi = vertex_center
    cent = scaling_func[vi,:]
    abs_max = max(np.abs(cent.min()), np.abs(cent.max()))
    normalize = mcolors.Normalize(vmin=-abs_max, vmax=abs_max) 
    colormap = cm.jet

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(cent)
    
    colors = [scalarmappaple.to_rgba(i) for i in cent]

    labels = {}
    for i in range(n):
        labels[i] = str(i)
        
    plt.colorbar(scalarmappaple)
    nx.draw_networkx(G_sim, pos, node_color= colors, cmap=colormap, with_labels=False)#, node_size=sizes
    nx.draw_networkx_labels(G_sim, pos, labels, font_size=15, font_color="whitesmoke")
    plt.title('scaling function '+ r"$\phi$"+' centered at vertex '+str(vi))
    plt.show()


def get_wavelets_plot(vertex_center,j, wavelet_func):   
    vi = vertex_center
    cent = wavelet_func[j][vi,:]
    #sizes = cent / np.max(cent) * 200
    abs_max = max(np.abs(cent.min()), np.abs(cent.max()))
    normalize = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
    colormap = cm.jet

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(cent)
    
    colors = [scalarmappaple.to_rgba(i) for i in cent]
    labels = {}
    for i in range(n):
        labels[i] = str(i)

    plt.colorbar(scalarmappaple)
    nx.draw_networkx(G_sim, pos, node_color= colors, cmap=colormap, with_labels=False)#, node_size=sizes
    nx.draw_networkx_labels(G_sim, pos, labels, font_size=15, font_color="whitesmoke")
    plt.title('wavelets function '+ r"$\psi$"+' centered at vertex '+str(vi)+' (scale j='+str(j+1)+')')
    plt.show()
    

def get_eigenvec_plot(lambda_ind,L_lambda_vec):
    vi = lambda_ind
    cent = L_lambda_vec[:,vi]
    abs_max = max(np.abs(cent.min()), np.abs(cent.max()))
    normalize = mcolors.Normalize(vmin=-abs_max, vmax=abs_max) 
    colormap = cm.jet

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(cent)
    
    colors = [scalarmappaple.to_rgba(i) for i in cent]

    labels = {}
    for i in range(n):
        labels[i] = str(i)
        
    plt.colorbar(scalarmappaple)
    nx.draw_networkx(G_sim, pos, node_color= colors, cmap=colormap, with_labels=False)#, node_size=sizes
    nx.draw_networkx_labels(G_sim, pos, labels, font_size=15, font_color="whitesmoke")
    plt.title('eigen vector of lambda ind '+str(vi))
    plt.show()

    
    
