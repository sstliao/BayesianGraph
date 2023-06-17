######################

###### required modules ######
exec(open('import_modules.py').read())
exec(open('Graph_decomposition.py').read())
exec(open('MCMC_run.py').read())
exec(open('multi_stage_lasso.py').read())
exec(open('MCMC_prediction.py').read())


###### generate undirected graphs ######
def generate_random_geometric_graph(n, r, seed):
    '''
    sample nodes uniformly from [0,1]^2
    '''
    
    pos = {i: (np.random.uniform(0, 1), np.random.uniform(0, 1)) for i in range(n)} 
    g = nx.random_geometric_graph(n, r, pos=pos)
    for e in g.edges(): ## add weight
        w=np.round(np.random.uniform(1,5),3)
        g[e[0]][e[1]]['weight']=w
    return g,pos


###### generate process Z ######
def generate_Z(scaling_func, wavelet_func,K, sigma2, tau2, sigma2_0,s):#,para_prior):
    
    psi_set = wavelet_func
    phi_set = scaling_func
    
    n = phi_set.shape[0]
    J = psi_set.shape[0]

    A = np.column_stack([psi_set[i,:,:] for i in range(J)])

    P = phi_set.T

    R = np.eye(n)

    D = np.random.multivariate_normal(mean = np.zeros(n), cov = np.eye(n)*sigma2_0,size=K).T
    
    V = np.array([np.array([invgamma.rvs(s/2, scale = s/2*sigma2[i],  size = n) for i in range(J)]).flatten() 
                  for x in range(K)]).T

    G = np.array([np.array([np.random.normal(0, scale = 1,  size = n) for i in range(J)]).flatten() 
              for x in range(K)]).T
    C = np.array([np.sqrt(V[:,k])*G[:,k] for k in range(K)]).T
    
    E = np.array([np.random.normal(0, scale = np.sqrt(tau2),  size = n).flatten() 
              for x in range(K)]).T
    
    Z = P.dot(D) + R.dot(A.dot(C))+E
    
    return Z, dict(P=P, D=D, R=R, A=A, C=C, E=E, V=V, G=G)
 
