######################

###### required modules ######
exec(open('import_modules.py').read())
exec(open('Graph_decomposition.py').read())
exec(open('MCMC_run.py').read())
exec(open('multi_stage_lasso.py').read())
exec(open('MCMC_prediction.py').read())
exec(open('simulation_generate.py').read())



###### generate graph and non-gaussian process ######
n = 300
p = 0.28
G_sim,pos = generate_random_geometric_graph(n, p, 1)

print(len(G_sim.edges))

## compute matrix and eigenvalue/vectors
_, _, _, L_lambda_set, L_lambda_vec = graph_matrix_eigen(G_sim)

print('finished graph-------')

## kernels
J = 2
K_par= 1
lambda_max = L_lambda_set[-1]
g_vals, h_vals,_ = kernal_gh(lambda_set = L_lambda_set,
                           J=J,lambda_max=lambda_max,K=K_par, x1=1, x2=2)

print('finished kernels-------')

## wavelets and scaling
start = time.time()

phi_set2 = np.zeros((n,n))
for vi in range(n):
    for ui in range(vi+1):
        phi_set2[vi,ui] = phi(vi,ui,L_lambda_set,L_lambda_vec,h_vals)

phi_set = phi_set2 + phi_set2.T - np.diag(phi_set2.diagonal())

psi_set = np.zeros((J,n,n))
psi_set2 = np.zeros((J,n,n))
for j in range(J):
    for vi in range(n):
        for ui in range(vi+1):
            psi_set2[j,vi,ui] = psi(j,vi,ui,L_lambda_set,L_lambda_vec,g_vals)
    psi_set[j,:,:] = psi_set2[j,:,:] + psi_set2[j,:,:].T - np.diag(psi_set2[j,:,:].diagonal())

del psi_set2, phi_set2
print(time.time()-start)

print('finished wavelets-------')

## porcess Z
K=7
sigma2_true = np.array([6*(2.5)**(-(j+1)) for j in range(J)])
tau2_true = 1
sigma2_0_true = 0.5

## set prior parametres for sigma_j^2 tau^2 tau^2_d
para_prior_names = ['alpha_0','beta_0','s','alpha_j','beta_j','alpha_tau2','beta_tau2']
para_prior_values =[np.array([1.2]),np.array([4]),np.array([4]),
                    np.array([5,5]),np.array([1,1]),np.array([1.2]),np.array([4])]
para_prior = dict()
for i in range(len(para_prior_names)):
    para_prior[para_prior_names[i]] = para_prior_values[i]

Z, comp_mat = generate_Z(scaling_func = phi_set, wavelet_func = psi_set,
                         K=K, sigma2 = sigma2_true,
                         tau2=tau2_true,sigma2_0 = sigma2_0_true ,s = para_prior['s'])

design_mat = {mat: val for mat, val in comp_mat.items() if mat in ['P','R', 'A']}
print('Z dim: '+str(Z.shape[1]))
print('finished generating process-------')

true_para = dict(sigma2=sigma2_true, tau2 = np.array([tau2_true]), sigma2_0=np.array([sigma2_0_true]), 
                 C = comp_mat['C'], V=comp_mat['V'], D=comp_mat['D'])



###### multi-stage-lasso for prior determination ######
## if missing data is preferred, it can implemented by rate argument (ranging from 0 to 1)
Z_full, design_mat_full, M_full, _,_ = generate_missing_value(Z, design_mat, rate=0)
start = time.time()
summary = graph_KFold(Z_full, design_mat_full, M_full, 
                      alpha1 = np.arange(0.0005,0.0021,0.0001), 
                      alpha2 = np.arange(0.0001,0.00045,0.00005),
                      l1_ratio = 1.0,
                      s = para_prior['s'][0],
                      n_splits=10, seed_splits=1)
print(str(np.round(time.time()-start,4))+'s')

## one can determined the optimal based on certain metrics
est_var = list(summary[sub_result['MPE']==summary['MPE'].min()].iloc[0,3:7])
para_prior['beta_0'] = (para_prior['alpha_0']-1)*est_var[1]
para_prior['beta_j'] = (para_prior['alpha_j'])/est_var[2:]
para_prior['beta_tau2'] = (para_prior['alpha_tau2']-1)*est_var[0]





###### MCMC run #######
print('start MCMC-------')
start = time.time()
# mcmc_server = adaptive_MCMC_process(Z,design_mat, para_init = true_para,para_prior=para_prior,
#                                     num_iter=120000,seed=10, print_info=False)

#para_init = true_para
print('using initials var a bit deviated from the true ---')
print('the initial values of c,d and V are specified as zero vectors, and an all-ones vector, respectively.')
para_init_var = dict(sigma2=np.array(est_var[2:]), 
                     tau2 = np.array([est_var[0]]), 
                     sigma2_0=np.array([est_var[1]]))

para_init = dict(sigma2= para_init_var['sigma2']+0.1, 
                 tau2 = para_init_var['tau2']+0.1, 
                 sigma2_0= para_init_var['sigma2_0']-0.2, 
                 C = np.zeros((n*J,K)), 
                 V = np.ones((n*J,K)), 
                 D = np.zeros((n,K)))

print(para_prior)
num_iter = 200000
samples = [0]*(num_iter+1)
samples[0] = para_init.copy()
old_para = para_init.copy()
#     np.random.seed(seed)
for m in range(1,num_iter+1):
    new_para = adaptive_MCMC_one_new(Z=Z_process, design_mat=design_mat_process, 
                                     old_para=old_para,
                                     para_prior=para_prior,
                                     n_obs = n_obs,
                                     true_step1=False, true_step2=False,
                                     true_step3=False, true_step4=False,
                                     true_step5=False, true_step6=False,print_info=False)
    samples[m] = new_para.copy()
    old_para = new_para.copy()
    if m%500 == 0:#print_info is True:
        print('iter -- '+str(m))
    if m == 50000:
        mcmc_thin_p1 = thin_MCMC(samples[:(m+1)], 50)
        with open(output+'_p1.pickle', 'wb') as f:
            pickle.dump(mcmc_thin_p1,f)
    if m == 100000:
        mcmc_thin_p2 = thin_MCMC(samples[:(m+1)], 50)
        with open(output+'_p2.pickle', 'wb') as f:
            pickle.dump(mcmc_thin_p2,f)
    if m == 150000:
        mcmc_thin_p3 = thin_MCMC(samples[:(m+1)], 50)
        with open(output+'_p3.pickle', 'wb') as f:
            pickle.dump(mcmc_thin_p3,f)

print('time spent on MCMC:'+str(time.time()-start)) 
print('finished MCMC-------')

mcmc_thin = thin_MCMC(samples, skip_size)

# with open('sim_MCMC.pickle', 'wb') as f:
#     pickle.dump(mcmc_thin,f)

    
###### prediction and performance ######

start = time.time()
Z_post_pred = sample_post_Z(mcmc_thin,design_mat, L=500)
print(str(time.time()-start)+'s')

pred_score(Z_post_pred,Z,n,K) ## return a table of metrics




