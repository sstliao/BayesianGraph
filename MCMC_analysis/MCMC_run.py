######################

###### required modules ######
exec(open('import_modules.py').read())
exec(open('Graph_decomposition.py').read())
exec(open('multi_stage_lasso.py').read())
exec(open('MCMC_prediction.py').read())


 
    
###### Computing of MCMC #######
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = LA.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(LA.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(LA.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = LA.cholesky(B)
        return True
    except LA.LinAlgError:
        return False

def my_test(x):
    if isPD(x) == False:
        raise  Exception('Error: Matrix is not PD!')


        
def adaptive_MCMC_one_new(Z,design_mat,old_para,para_prior, 
                      n_obs,
                      true_step1=False, true_step2=False,
                      true_step3=False, true_step4=False,
                      true_step5=False, true_step6=False,print_info=False):
    '''
    MCMC run for one iteration
    true_step1 - true_step5: True is used for algorithm check only
    '''
    
    ### the following components are a list of matrices 
    R = design_mat['R'].copy()
    P = design_mat['P'].copy()
    A = design_mat['A'].copy()
    
    
    sigma2 = old_para['sigma2'].copy()
    sigma2_0 = old_para['sigma2_0'].copy()
    tau2 = old_para['tau2'].copy()
    C = old_para['C'].copy()
    V = old_para['V'].copy()
    D = old_para['D'].copy()
    
    
    alpha_0 = para_prior['alpha_0'].copy()
    beta_0 = para_prior['beta_0'].copy()
    s = para_prior['s'].copy()
    alpha_j = para_prior['alpha_j'].copy() ## length J vector
    beta_j = para_prior['beta_j'].copy() ## length J vector
    alpha_tau2 = para_prior['alpha_tau2'].copy()
    beta_tau2 = para_prior['beta_tau2'].copy()
    
    n = Z.shape[0]
    K = Z.shape[1]
    J = old_para['sigma2'].shape[0]

    ## Step 1 -- c_j(t_k)
    if true_step1==False:
        for k in range(K):
            for j in range(J):

                c_tk = C[:,k].copy()

                A_j = A[k][:, (n*j):(n*(j+1))].copy()
                A_wo_j = np.delete(A[k],range(n*j,n*(j+1)),1)

                V_j_tk = V[range(n*j,n*(j+1)),k].copy()

                c_wo_j_tk = np.delete(c_tk,range(n*j,n*(j+1)),0)

                Sigma_j_tk_C =((tau2)*LA.inv((A_j.T.dot(R[k].dot(R[k])).dot(A_j)+(tau2)*(np.diag(1/V_j_tk)))))

                M_j_tk = (Z[:,k] - P[k].dot(D[:,k]) - R[k].dot(A_wo_j).dot(c_wo_j_tk))

                Mu_j_tk_C = (1/(tau2)*Sigma_j_tk_C.dot(A_j.T).dot(R[k]).dot(M_j_tk))

                Sigma_j_tk_C_comp = LA.cholesky(Sigma_j_tk_C)
                c_j_tk_update = Sigma_j_tk_C_comp.dot(np.random.standard_normal((n,1)))+Mu_j_tk_C.reshape((n,1))

                C[range(n*j,n*(j+1)),k] = c_j_tk_update.reshape((n,)).copy()
    else:
        C = old_para['C'].copy()
        if print_info is True:
            print('true C is used')
    
    if print_info is True:
        print('C-range:',C.min(), C.max())
        print('C2-range:',(C**2).min(), (C**2).max())

    
    ## Step 2 -- d_tk
    if true_step2 == False:
        
        for k in range(K):
            Sigma_D = (tau2)*LA.inv(P[k].T.dot(P[k])+(tau2)*((1/sigma2_0)*np.eye(n)))
            H_tk = Z[:,k] - R[k].dot(A[k]).dot(C[:,k])
            Mu_tk_D = 1/(tau2)*Sigma_D.dot(P[k].T).dot(H_tk)
            
            Sigma_D_comp = LA.cholesky(Sigma_D)
            d_tk_update = Sigma_D_comp.dot(np.random.standard_normal((n,1)))+Mu_tk_D.reshape((n,1))

            D[:,k] = d_tk_update.reshape((n,)).copy()
    else: 
        D = old_para['D'].copy()
        if print_info is True:
            print('true D is used')
    
    if print_info is True:
        print('D-range:',D.min(), D.max())

    
    ## Step 3 -- V
    if true_step3 == False:
        V = np.array([np.array([[1/np.random.gamma(shape=(s+1)/2, scale = 1/(s/2*sigma2[j]+ C[(n*j+v),k]**2/2),  size = 1) 
                                 for v in range(n)] for j in range(J)]).flatten() for k in range(K)]).T

    else:
        V = old_para['V'].copy()
        if print_info is True:
            print('true V is used')
    
    if print_info is True:
        print('V-post:',((s+1)/2, [s/2*sigma2[j]+ C[(n*j+0),0]**2/2 for j in range(J)]))
        print('V-range:',V.min(), V.max())
    
    
    ## Step 4 -- sigma2
    if true_step4 == False:
        #np.random.seed(seed)
        sigma2 = np.array([np.random.gamma(shape = s*n*K/2+alpha_j[j], scale = 1/(s/2*np.sum(1/V[range(n*j,n*(j+1)),:])+beta_j[j]), size=1) 
                           for j in range(J)])
       # print(s*n*K/2+a[0])
    else:
        sigma2 = old_para['sigma2'].copy()
        if print_info is True:
            print('true sigma2 is used')
    
    
    ## Step 5 -- tau2
    scale_tau2 = 1/2*sum([((Z[:,k]-R[k].dot(A[k]).dot(C[:,k])-P[k].dot(D[:,k])).T)@(Z[:,k]-R[k].dot(A[k]).dot(C[:,k])-P[k].dot(D[:,k])) 
                              for k in range(K)])+beta_tau2
    if true_step5 == False:
        tau2 = 1/(np.random.gamma(shape=np.sum(n_obs)/2+alpha_tau2, scale=1/scale_tau2,size=1))
    else:
        tau2 = old_para['tau2'].copy()
        if print_info is True:
            print('true tau_2 is used')
    
    
    ## Step 6 -- sigma2_0
    scale_sigma2_0 = sum([D[:,k].T.dot(D[:,k]) for k in range(K)])*1/2+beta_0
    shape_sigma2_0 = alpha_0+n*K/2

    if true_step6==False:
       # np.random.seed(seed)
        sigma2_0 = 1/np.random.gamma(shape=shape_sigma2_0, scale = 1/scale_sigma2_0, size=1)
    else:
        sigma2_0 = old_para['sigma2_0'].copy()
        if print_info is True:
            print('true sigma2_0 is used')
    
    
    
    return dict(sigma2=sigma2, tau2=tau2, sigma2_0=sigma2_0,
                C=C, D=D, V=V)


def adaptive_MCMC_process(Z,design_mat, para_init,para_prior,n_obs,
                          num_iter,seed,
                          true_step1=False, true_step2=False,
                          true_step3=False, true_step4=False,
                          true_step5=False, true_step6=False,print_info=False):
    samples = [0]*(num_iter+1)
    samples[0] = para_init.copy()
    old_para = para_init.copy()
    for m in range(1,num_iter+1):
        new_para = adaptive_MCMC_one_new(Z=Z, design_mat=design_mat, old_para=old_para,
                                         para_prior=para_prior,
                                         n_obs = n_obs,
                                         true_step1=true_step1, true_step2=true_step2,
                                         true_step3=true_step3, true_step4=true_step4,
                                         true_step5=true_step5, true_step6=true_step6,
                                         print_info=print_info)
        samples[m] = new_para.copy()
        old_para = new_para.copy()
        if m%100 == 0:
            print('iter -- '+str(m))
    return(samples)

def thin_MCMC(samples, step_size):
    '''
    thining the MCMC samples with given a step size
    
    '''
    thin_samples = [samples[0]]+[samples[m] for m in range(1,len(samples)) if m%step_size==0]
    return(thin_samples)

###### missing value system generating ######
def generate_missing_value(Z, design_mat, rate=0.1):
    n,K = Z.shape
    n_miss = int(n*rate)
    node_miss = [np.sort(np.random.choice(n,n_miss,replace=False)) for k in range(K)]
    M = np.array([np.eye(n) for k in range(K)])
    for k in range(K):
        M[k][node_miss[k], node_miss[k]] = 0
    
    ## check if any nodes are not observed across all replicates; if so, we randomly choose one replicate as observed
    for node in range(n):
        if np.sum(M[:,node,node]) == 0:
            x = np.random.choice(K,1) 
            M[x,node,node] = 1
    n_obs = np.array([np.sum(np.diag(M[k])) for k in range(K)])
    
    ## update Z and design mat
    Z_miss = np.array([M[k]@Z[:,k] for k in range(K)]).T
    
    design_mat_miss = {mat: np.array([M[k]@val for k in range(K)]) for mat, val in design_mat.items() if mat in ['P','R']}
    design_mat_miss['A'] = np.array([design_mat['A'] for k in range(K)])


    return Z_miss, design_mat_miss, M, n_obs, node_miss
