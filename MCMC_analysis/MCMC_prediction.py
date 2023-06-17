######################

###### required modules ######
exec(open('import_modules.py').read())
exec(open('Graph_decomposition.py').read())
exec(open('multi_stage_lasso.py').read())
exec(open('MCMC_run.py').read())



def plot_chain(samples, para_title):
    plt.plot(samples)
    plt.axhline(np.mean(samples),color='yellow')
    plt.title(para_title,fontsize = 25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    
def plot_dist(samples, para_title):
    plt.hist(samples, bins=50)
    plt.axvline(np.mean(samples),color='yellow')
    plt.title(para_title+': std='+str(np.round(np.std(samples),3)))
    
    
def sample_post_Z(mcmc_thin, design_mat, L):
    
    '''design_mat: dictionary of components P(n by n),R(n by n),A(n by n*J)'''
    
    Z_pred_list = list()
    
    for i in range(L):
        
        ii = np.random.choice(len(mcmc_thin),1,replace=True)[0]
        post_dist_mu = design_mat['P']@mcmc_thin[ii]['D']+design_mat['R']@design_mat['A']@mcmc_thin[ii]['C']
        post_dist_Sig = np.eye(n)*mcmc_thin[ii]['tau2']
    
        Z_pred_list.append(np.array([np.random.multivariate_normal(mean = post_dist_mu[:,k], cov = post_dist_Sig, size = 1).reshape((n,)) 
                                     for k in range(K)]).T)
    return Z_pred_list


def summary_Z_post_pred(Z_post_pred, node_n, rep_k, alpha=0.05):
    
    L = len(Z_post_pred)
    
    Z_samples = sorted(np.array([x[node_n,rep_k] for x in Z_post_pred]))
    
    CI_up = Z_samples[np.round((L+1)*(1-alpha/2)).astype(int)-1]
    CI_low = Z_samples[np.round((L+1)*(alpha/2)).astype(int)-1]
    
    return [CI_low, CI_up, np.median(Z_samples), np.mean(Z_samples), CI_up-CI_low]

def pred_score(Z_post_pred,Z,n,K):
    
    ### median as the point-wise estimation
    
    Z_pred_summary = np.array([[summary_Z_post_pred(Z_post_pred, node_n=nn, rep_k=kk, alpha=0.05)
                       for kk in range(K)] for nn in range(n)])
    
    Z_pred_med = np.array([[Z_pred_summary[nn,kk][2] 
                        for kk in range(K)] for nn in range(n)])

    Z_pred_coverage = np.sum(np.array([[(Z[nn,kk]<Z_pred_summary[nn,kk,1]) & (Z[nn,kk]>Z_pred_summary[nn,kk,0])
                       for kk in range(K)] for nn in range(n)]))/(n*K)

    Z_pred_CI_length = np.array([[Z_pred_summary[nn,kk,1]-Z_pred_summary[nn,kk,0]
                       for kk in range(K)] for nn in range(n)])
    
    df = dict()
    df['RMSPE'] = np.sqrt(np.mean((Z_pred_med-Z)**2))
    df['RMedSPE'] = np.sqrt(np.median((Z_pred_med-Z)**2))
    df['MCILen'] = np.mean(Z_pred_CI_length.flatten())
    df['MedCILen'] = np.median(Z_pred_CI_length.flatten())
    df['CIcov'] = Z_pred_coverage
    
    return pd.DataFrame(df,index=[''])
