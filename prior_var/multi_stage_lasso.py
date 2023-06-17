######################

###### required modules ######
exec(open('import_modules.py').read())
exec(open('Graph_decomposition.py').read())
exec(open('MCMC_run.py').read())
exec(open('MCMC_prediction.py').read())



def grid_parameters(parameters):
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))

def LS_subset_L01(Z,M,design_mat,n,K,J,s,drop_level2,alpha,l1_ratio=1.0):
    '''
    drop_level2: int; indicating the specific level we want to exclude; 
                for now, we only consider to drop level 2;
    '''
    if drop_level2 is True:
        design_X = [M[i]@(np.hstack((design_mat['P'][i], 
                                     design_mat['R'][i]@design_mat['A'][i][:,:n]))) 
            for i in range(K)]
        num_levels = J
    else:
        design_X = [M[i]@(np.hstack((design_mat['P'][i], 
                                     design_mat['R'][i]@design_mat['A'][i]))) 
                for i in range(K)]
        num_levels = J+1
        
    design_X_model = block_diag(*design_X)
    Z_model = Z.T.reshape((n*K,1))
    final_model = ElasticNet(alpha=alpha,
                             l1_ratio=l1_ratio, 
                             random_state=0,
                             tol=1e-4,
                             max_iter = 1000,
                             fit_intercept = False)
    final_model.fit(design_X_model, Z_model)
    
    est = final_model.coef_.reshape((n*K*num_levels,1)) 

    ## Linear regression
    design_X_model_red = design_X_model[:,~(est==0).flatten()]
    
    reg = LinearRegression(fit_intercept=False).fit(design_X_model_red,Z_model)
    
    est_LS = np.zeros((est.shape[0],1))
    

    est_LS[~(est==0).flatten(),0] = reg.coef_ ## fill in non-zero coefficients from LS

    Z_pred_LS = reg.predict(design_X_model_red)
    res_LS = (Z_model - Z_pred_LS)

    
    sets = np.hstack(np.vsplit(est_LS,K))
    
    if drop_level2 is True:
        set1 = sets[:n,:].flatten()
        set2 = sets[n:(2*n),:].flatten()

        summary = pd.DataFrame([alpha,l1_ratio],index=['alpha','l1_ratio']).T
        summary[r'$\tau^2$'] =  res_LS.var()
        summary[r'$\sigma^2_0(w/ 0)$'] = set1.var()#sigma2_0;
        summary[r'$\sigma^2_1(w/ 0)$'] = set2.var()*(s-2)/s ## sigma2_1;

        summary['frac_zero_L0'] = np.round(np.sum(set1==0)/(n*K),3)
        summary['frac_zero_L1'] = np.round(np.sum(set2==0)/(n*K),3)

    
    else:
        set1 = sets[:n,:].flatten()
        set2 = sets[n:(2*n),:].flatten()
        set3 = sets[(2*n):,:].flatten()

        summary = pd.DataFrame([alpha,l1_ratio],index=['alpha_1','l1_ratio']).T
        summary[r'$\tau^2$'] =  res_LS.var()
        summary[r'$\sigma^2_0(w/ 0)$'] = set1.var()#sigma2_0;
        summary[r'$\sigma^2_1(w/ 0)$'] = set2.var()*(s-2)/s ## sigma2_1;
        summary[r'$\sigma^2_2(w/ 0)$'] = set3.var()*(s-2)/s ## sigma2_2;
        
        sets = np.hstack(np.vsplit(est,K))
        
        set1 = sets[:n,:].flatten()
        set2 = sets[n:(2*n),:].flatten()
        set3 = sets[(2*n):,:].flatten()
        summary['frac_zero_L0'] = np.round(np.sum(set1==0)/(n*K),3)
        summary['frac_zero_L1'] = np.round(np.sum(set2==0)/(n*K),3)
        summary['frac_zero_L2'] = np.round(np.sum(set3==0)/(n*K),3)


    
    
    return summary, est, sets, res_LS, design_X_model_red
        
def LS_subset_L2(Z,M,design_mat,n,K,J,s,alpha,l1_ratio=1.0):


    design_X = [M[i]@(design_mat['R'][i]@design_mat['A'][i][:,n:]) 
                for i in range(K)]
    num_levels = 1

    design_X_model = block_diag(*design_X)
    Z_model = Z.T.reshape((n*K,1))
    final_model = ElasticNet(alpha=alpha,
                             l1_ratio=l1_ratio, 
                             random_state=0,
                             tol=1e-4,
                             max_iter = 1000,
                             fit_intercept = False)
    final_model.fit(design_X_model, Z_model)
    
    est = final_model.coef_.reshape((n*K*num_levels,1)) 
    
    ## Linear regression
    design_X_model_red = design_X_model[:,~(est==0).flatten()]
    
    reg = LinearRegression(fit_intercept=False).fit(design_X_model_red,Z_model)
    
    est_LS = np.zeros((est.shape[0],1))
    
    ## 
    est_LS[~(est==0).flatten(),0] = reg.coef_ ## fill in non-zero coefficients from LS
#     print(est_LS.shape)
    Z_pred_LS = reg.predict(design_X_model_red)
    res_LS = (Z_model - Z_pred_LS)
   
    
    sets = np.hstack(np.vsplit(est_LS,K))
    
    set1 = sets.flatten()
    summary = pd.DataFrame([alpha,l1_ratio],index=['alpha_2','l1_ratio']).T
    summary[r'$\tau^2$'] =  res_LS.var()
    summary[r'$\sigma^2_2(w/ 0)$'] = set1.var()#sigma2_2;

    summary['frac_zero_L2'] = np.round(np.sum(set1==0)/(n*K),3)

    
    return summary, np.hstack(np.vsplit(est_LS,K)), sets, res_LS

        
def graph_KFold(Z_process, design_mat_process, M_process,
                alpha1,alpha2,
                l1_ratio,
                s,
                n_splits=10, seed_splits=0):
    '''
    Z_process, design_mat_process, M_process: generated/processed by generate_missing_value
    '''
    Z_full, design_mat_full, M_full = Z_process, design_mat_process, M_process
    n,K = Z_full.shape
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_splits)
    print(f'Starting {n_splits}-fold:')
    
    design_X = [M_full[i]@(np.hstack((design_mat_full['P'][i],
                                          design_mat_full['R'][i]@design_mat_full['A'][i][:,:n]))) 
                for i in range(K)]

    params = dict()
    params['alpha1'] = alpha1
    params['alpha2'] = alpha2
    params_comb = [x for x in grid_parameters(params)]

    var_est_list = list()
    pred_err_dict = dict()
    for kk in range(n_splits):
        pred_err_dict[f'pred_err{kk}'] = list()
    
    for sets in params_comb:
#         print(sets)
        try:
            '''run the adaptive lasso on full data'''
            output_full,_,_,res_LS_L01_full,_ = LS_subset_L01(Z_full,M_full,
                                                     design_mat_full,
                                                     n,K,J,s=s,
                                                     drop_level2=True,
                                                     alpha=sets['alpha1'],
                                                     l1_ratio=l1_ratio)   
            output_L2_full,_,_,_= LS_subset_L2(res_LS_L01_full,M_full,
                                          design_mat_full,
                                          n,K,J,
                                          s=s,
                                          alpha=sets['alpha2'],
                                          l1_ratio=l1_ratio)
            var_est_full = {'sigma2_0':output_full.iloc[0,3],
                    'sigma2_1':output_full.iloc[0,4],
                    'tau2': output_L2_full.iloc[0,2],
                    'sigma2_2':output_L2_full.iloc[0,3],
                    'frac_zero_L0':output_full.iloc[0,5],
                    'frac_zero_L1':output_full.iloc[0,6],
                    'frac_zero_L2':output_L2_full.iloc[0,4]}

            var_est_list.append(var_est_full)
            ''''''
        except ValueError:
            print('the following penalty pair is  not working; pass')
            print(sets)
            print('\n')
            for kk in range(n_splits):
                pred_err_dict[f'pred_err{kk}'].append(np.NaN)
            
            var_est_full = {'sigma2_0':np.NaN,
                'sigma2_1':np.NaN,
                'tau2': np.NaN,
                'sigma2_2':np.NaN,
                'frac_zero_L0':np.NaN,
                'frac_zero_L1':np.NaN,
                'frac_zero_L2':np.NaN}
            var_est_list.append(var_est_full)
            continue
      
        for i, (train_index, test_index) in enumerate(kf.split(Z_full)):
            
            try:
#             print(f'Fold {i}')

                '''setup train and test'''
                node_test = [test_index for k in range(K)] 

                M_train = M_full.copy()
                for k in range(K):
                    M_train[k][node_test[k], node_test[k]] = 0

                n_obs = np.array([np.sum(np.diag(M_train[k])) for k in range(K)])

                Z_train = np.array([M_train[k]@Z_full[:,k] for k in range(K)]).T

                design_mat_train = {mat: np.array([M_train[k]@val[k] for k in range(K)]) 
                                    for mat, val in design_mat_full.items() if mat in ['P','R']}
                design_mat_train['A'] = design_mat_full['A']


                Ztest =  np.vstack([Z_full[node_test[k],k].reshape((-1,1)) 
                                    for k in range(K)])
                Xtest = block_diag(*[np.vstack(design_X[k][node_test[k],:]) for k in range(K)])
                ''''''


                '''run adaptive lasso'''

                output,est_EN,_,res_LS_L01,Xtrain_red = LS_subset_L01(Z_train,M_train,
                                                                 design_mat_train,
                                                                 n,K,J,s=s,
                                                                 drop_level2=True,
                                                                 alpha=sets['alpha1'],
                                                                 l1_ratio=l1_ratio)

                output_L2,_,_,_= LS_subset_L2(res_LS_L01,M_train,
                                              design_mat_train,
                                              n,K,J,
                                              s=s,
                                              alpha=sets['alpha2'],
                                              l1_ratio=l1_ratio)
                ''''''

                '''prediction error on test data'''
                var_est = {'sigma2_0':output.iloc[0,3],
                           'sigma2_1':output.iloc[0,4],
                           'tau2': output_L2.iloc[0,2],
                           'sigma2_2':output_L2.iloc[0,3],
                           'frac_zero_L0':output.iloc[0,5],
                           'frac_zero_L1':output.iloc[0,6],
                           'frac_zero_L2':output_L2.iloc[0,4]}


                var_mat = [block_diag(*[var_est['sigma2_0']*np.eye(n),
                           var_est['sigma2_1']*np.eye(n)*(s/(s-2))]) for k in range(K)]

                G_hat = block_diag(*var_mat)
                zero_ind = (est_EN==0).flatten()

                G_hat_update = np.delete(G_hat,zero_ind,0)
                G_hat_update = np.delete(G_hat_update,zero_ind,1)


                V_hat_inv = LA.inv(Xtrain_red@G_hat_update@Xtrain_red.T+
                                   var_est['tau2']*np.eye(n*K))

                Xtest_red = Xtest[:,~zero_ind]

                BLUP = Xtest_red@(G_hat_update@Xtrain_red.T@V_hat_inv@(Z_train.T.reshape((n*K,1))))

                num_obs_test = np.sum(Ztest!=0)
    #                 print(num_obs_test)
                pred_err_dict[f'pred_err{i}'].append(np.sqrt(np.sum((Ztest-BLUP)**2)/num_obs_test)) # RMSE
        
            except ValueError:
                print('the following penalty pair is  not working; pass')
                print(sets)
                print('\n')
                for kk in range(i,n_splits):
                    pred_err_dict[f'pred_err{kk}'].append(np.NaN)
                break

    summary = pd.DataFrame(params_comb)
    summary['MPE'] = pd.DataFrame(pred_err_dict).mean(axis=1, skipna=False)
    summary[r'$\hat \tau^2$'] = [x['tau2'] for x in var_est_list]
    summary[r'$\hat \sigma^2_0$'] = [x['sigma2_0'] for x in var_est_list]
    summary[r'$\hat \sigma^2_1$'] = [x['sigma2_1'] for x in var_est_list]
    summary[r'$\hat \sigma^2_2$'] = [x['sigma2_2'] for x in var_est_list]
    summary['frac_zero_L0'] = [x['frac_zero_L0'] for x in var_est_list]
    summary['frac_zero_L1'] = [x['frac_zero_L1'] for x in var_est_list]
    summary['frac_zero_L2'] = [x['frac_zero_L2'] for x in var_est_list]
    
    which = (summary['MPE']==summary['MPE'].min())
    return  {'df':summary,
            'opt_alpha': dict(summary[which].iloc[0,[0,1]]), 
            'opt_var':summary[which].iloc[0,range(3,7)].tolist()}
