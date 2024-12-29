import re
import os
import sys
import numpy as np
import random

from .ML import *
from .utils import *
from .vegas_S import vegas_map_samples

import sklearn
from sklearn.ensemble import RandomForestClassifier
import time

from ...utilities.try_imports import try_tensorflow
tf = try_tensorflow()
keras = try_tensorflow('keras')

ModelCheckpoint = keras.callbacks.ModelCheckpoint
EarlyStopping = keras.callbacks.EarlyStopping
LearningRateScheduler = keras.callbacks.LearningRateScheduler
ProgbarLogger = keras.callbacks.ProgbarLogger

#####################
class scan():
    def __init__(self,collected_points,L1,L,K, period,frac):
        self.collected_points= collected_points
        self.L1 = L1
        self.L =L
        self.K = K
        #self.th_value = th_value
        #self.function_dim = function_dim 
        self.period = period
        self.frac = frac


    def lr_schedule(self,epoch):
        lr = 1e-2
        if epoch >= 10:
            lr *= 0.1
        elif epoch > 20:
            lr *= 0.05
        return lr 
    
    def labeler(self,x,th):
        ll=[]
        for q,item in enumerate(x):
            if item < th:
                ll.append(1)
            else:
                ll.append(0)
        return np.array(ll).ravel()        

    def run_RFC(self,learning_rate=0.01,n_estimators=300,max_depth=50,print_output=True):
        pathS, Lesh,SPHENOMODEL,output_dir,TotVarScanned, VarMin , VarMax,VarLabel,VarNum,TotVarTarget, TargetMin , TargetMax,TargetLabel,TargetNum ,TargetResNum   = read_input() 
        check_(pathS,Lesh,SPHENOMODEL,output_dir)  
        RFC =RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        Xf,ob1=run_train(self.L1,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)

        RFC.fit(Xf, ob1)
        X_g = Xf[ob1==1]    # It is good to have some initial guide
        obs1_g = ob1[ob1==1]
        X_b =Xf[ob1==0][:len(obs1_g)]
        obs1_b =ob1[ob1==0][:(len(obs1_g))]
        print('\nNumber of initial points in the traget region:  ', len(obs1_g) )
        if len(obs1_g) < 10 : sys.exit('Number of initial collected points is too low to train the network. Please try to increase the target range or do more iteration... EXIT!')
        time.sleep(8)
        while len(X_g) < self.collected_points:
            x = generate_init_HEP(self.L,TotVarScanned,pathS,Lesh,VarLabel,VarMin,VarMax)
            pred = RFC.predict(x).flatten()
            qs = np.argsort(pred)[::-1]
            if len(x[pred>0.9]) > round(self.K*self.frac): # How to choose the good points
                xsel1 = x[pred>0.9][:round(self.K*(1-self.frac))]
            else:
                xsel1 = x[qs][:round(self.K*(1-self.frac))]
            xsel1 = np.append(xsel1,x[:round(self.K*(self.frac))],axis=0)
            xsel2,ob = refine_points(xsel1,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)
            X_g = np.append(X_g,xsel2[ob==1],axis=0)
            obs1_g = np.append(obs1_g,ob[ob==1],axis=0)
            X_b = np.append(X_b,xsel2[ob==0],axis=0)
            obs1_b = np.append(obs1_b,ob[ob==0],axis=0)
            if (q%self.period==0 or q+1 == self.iteration):
                X = np.concatenate([X_g,X_b],axis=0)
                obs = np.concatenate([obs1_g,obs1_b],axis=0)
                X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
                RFC.fit(X_shuffled, Y_shuffled)

            else:
                X = np.concatenate([xsel2[ob==1],xsel2[ob==0]],axis=0)
                obs=np.concatenate([ob[ob==1],ob[ob==0]],axis=0)
                X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
                RFC.fit(X_shuffled, Y_shuffled)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X)))

        if os.path.exists(str(output_dir)+"/Accumelated_points.txt"): os.system('rm -rf %s/Accumelated_points.txt '%str(output_dir)) 
        f= open(str(output_dir)+"/Accumelated_points.txt","x")
        header = '\t'.join(VarLabel)
        f.write(header+' \n')
        f.close()
        np.savetxt(str(output_dir)+'/a.txt',X_g, delimiter=',')
        os.system('cat %s/a.txt >> %s/Accumelated_points.txt '%(str(output_dir),str(output_dir)))
        os.system('rm -rf %s/a.txt'%str(output_dir))
        
        print('Output saved in %s' %str(output_dir))
        return
        
        
    def run_MLPC(self,num_FC_layers,neurons,learning_rate=0.01,epochs=100,batch_size=100,vegas=True,print_output=True):
        ########################################
        earlystop = EarlyStopping(monitor = 'val_loss',restore_best_weights=True, patience=5,verbose=0)
        lr_scheduler = keras.callbacks.LearningRateScheduler(self.lr_schedule)
        progress_bar = keras.callbacks.ProgbarLogger(count_mode="steps")
        callbacks_list = [earlystop, lr_scheduler, progress_bar]
        #########################################
        pathS, Lesh,SPHENOMODEL,output_dir,TotVarScanned, VarMin , VarMax,VarLabel,VarNum,TotVarTarget, TargetMin , TargetMax,TargetLabel,TargetNum ,TargetResNum   = read_input() 
        check_(pathS,Lesh,SPHENOMODEL,output_dir)  
        model = MLP_Classifier(TotVarScanned,num_FC_layers,neurons)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.BinaryCrossentropy())
        Xf,ob1=run_train(self.L1,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)

        X_g = Xf[ob1==1]    # It is good to have some initial guide
        obs1_g = ob1[ob1==1]
        X_b =Xf[ob1==0][:len(obs1_g)]
        obs1_b =ob1[ob1==0][:(len(obs1_g))]
        print('\nNumber of initial points in the traget region:  ', len(obs1_g) )
        x_t = np.concatenate((X_g,X_b))
        y_t = np.concatenate((obs1_g,obs1_b))
        model.fit(x_t,y_t,epochs=epochs, batch_size=batch_size,callbacks=callbacks_list,verbose=0)
        if len(obs1_g) < 10 : sys.exit('Number of initial collected points is too low to train the network. Please try to increase the target range or do more iteration... EXIT!')
        q= 0

        while len(X_g) < self.collected_points:
            q+=1
            x_test = generate_init_HEP(self.L,TotVarScanned,pathS,Lesh,VarLabel,VarMin,VarMax)
            x_vegas  = np.concatenate((X_g,X_b))
            y_vegas  = np.concatenate((obs1_g,obs1_b))
            limits = np.column_stack((VarMin,VarMax))
            Veg_map = vegas_map_samples(x_vegas,y_vegas,limits)
            x,_ = Veg_map(x_test)
            if vegas:
                pred = model.predict(x,verbose=0).flatten()
            else:    
                pred = model.predict(x_test,verbose=0).flatten()
            qs = np.argsort(pred)[::-1]
            if len(x[pred>0.9]) > round(self.K*self.frac): # How to choose the good points
                xsel1 = x[pred>0.9][:round(self.K*(1-self.frac))]
            else:
                xsel1 = x[qs][:round(self.K*(1-self.frac))]
            xsel1 = np.append(xsel1,x[:round(self.K*(self.frac))],axis=0)
            xsel2,ob = refine_points(xsel1,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)
            X_g = np.append(X_g,xsel2[ob==1],axis=0)
            obs1_g = np.append(obs1_g,ob[ob==1],axis=0)
            X_b = np.append(X_b,xsel2[ob==0],axis=0)
            obs1_b = np.append(obs1_b,ob[ob==0],axis=0)
            if (q%self.period==0 or q+1 == self.iteration):
                X = np.concatenate([X_g,X_b],axis=0)
                obs = np.concatenate([obs1_g,obs1_b],axis=0)
                X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
                model.fit(X_shuffled, Y_shuffled,epochs=epochs, batch_size=batch_size,callbacks=callbacks_list,verbose=0)

            else:
                X = np.concatenate([xsel2[ob==1],xsel2[ob==0]],axis=0)
                obs=np.concatenate([ob[ob==1],ob[ob==0]],axis=0)
                X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
                model.fit(X_shuffled, Y_shuffled,epochs=epochs, batch_size=batch_size,callbacks=callbacks_list,verbose=0)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X_g)))
                
          
            
        if os.path.exists(str(output_dir)+"/Accumelated_points.txt"): os.system('rm -rf %s/Accumelated_points.txt '%str(output_dir)) 
        f= open(str(output_dir)+"/Accumelated_points.txt","x")
        header = '\t'.join(VarLabel)
        f.write(header+' \n')
        f.close()
        np.savetxt(str(output_dir)+'/a.txt',X_g, delimiter=',')
        os.system('cat %s/a.txt >> %s/Accumelated_points.txt '%(str(output_dir),str(output_dir)))
        os.system('rm -rf %s/a.txt'%str(output_dir))
        
        print('Output saved in %s' %str(output_dir))
        return    
      
    def run_similarity(self,num_FC_layers,neurons,latent_dim=10,learning_rate=0.01,epochs=100,batch_size=100,vegas=True,print_output=True):
        pathS, Lesh,SPHENOMODEL,output_dir,TotVarScanned, VarMin , VarMax,VarLabel,VarNum,TotVarTarget, TargetMin , TargetMax,TargetLabel,TargetNum ,TargetResNum   = read_input() 
        check_(pathS,Lesh,SPHENOMODEL,output_dir)  
        Xf,ob1=run_train(self.L1,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)
        X_g = Xf[ob1==1]    # It is good to have some initial guide
        obs1_g = ob1[ob1==1]
        X_b =Xf[ob1==0][:len(obs1_g)]
        obs1_b =ob1[ob1==0][:(len(obs1_g))]
        print('\nNumber of initial points in the traget region:  ', len(obs1_g) )
        x_t = np.concatenate((X_g,X_b))
        y_t = np.concatenate((obs1_g,obs1_b))
        
        model = similariy_classifier(x_t,y_t,TotVarScanned,latent_dim,neurons,num_FC_layers,epochs=epochs, learning_rate=learning_rate,batch_size=batch_size,verbose=0)
        q= 0

        while len(X_g) < self.collected_points:
            q+=1
            x_test = generate_init_HEP(self.L,TotVarScanned,pathS,Lesh,VarLabel,VarMin,VarMax)
            x_vegas  = np.concatenate((X_g,X_b))
            y_vegas  = np.concatenate((obs1_g,obs1_b))
            limits = np.column_stack((VarMin,VarMax))
            Veg_map = vegas_map_samples(x_vegas,y_vegas,limits)
            x,_ = Veg_map(x_test)
            if vegas:
                pred = model.predict(x,verbose=0).flatten()
            else:
                pred = model.predict(x_test,verbose=0).flatten()
            qs = np.argsort(pred)[::-1]
            if len(x[pred>0.9]) > round(self.K*self.frac): # How to choose the good points
                xsel1 = x[pred>0.9][:round(self.K*(1-self.frac))]
            else:
                xsel1 = x[qs][:round(self.K*(1-self.frac))]
            xsel1 = np.append(xsel1,x[:round(self.K*(self.frac))],axis=0)
            xsel2,ob = refine_points(xsel1,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)
            X_g = np.append(X_g,xsel2[ob==1],axis=0)
            obs1_g = np.append(obs1_g,ob[ob==1],axis=0)
            X_b = np.append(X_b,xsel2[ob==0],axis=0)
            obs1_b = np.append(obs1_b,ob[ob==0],axis=0)
            if (q%self.period==0 or q+1 == self.iteration):
                X = np.concatenate([X_g,X_b],axis=0)
                obs = np.concatenate([obs1_g,obs1_b],axis=0)
                X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
                model = similariy_classifier(X_shuffled, Y_shuffled,TotVarScanned,latent_dim,neurons,num_FC_layers,epochs=epochs, learning_rate=learning_rate,batch_size=batch_size,verbose=0)
                #model.fit(X_shuffled, Y_shuffled,epochs=epochs, batch_size=batch_size,verbose=0)

            else:
                X = np.concatenate([xsel2[ob==1],xsel2[ob==0]],axis=0)
                obs=np.concatenate([ob[ob==1],ob[ob==0]],axis=0)
                X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
                model = similariy_classifier(X_shuffled, Y_shuffled,TotVarScanned,latent_dim,neurons,num_FC_layers,epochs=epochs, learning_rate=learning_rate,batch_size=batch_size,verbose=0)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X_g)))
        
        
        if os.path.exists(str(output_dir)+"/Accumelated_points.txt"): os.system('rm -rf %s/Accumelated_points.txt '%str(output_dir)) 
        f= open(str(output_dir)+"/Accumelated_points.txt","x")
        header = '\t'.join(VarLabel)
        f.write(header+' \n')
        f.close()
        np.savetxt(str(output_dir)+'/a.txt',X_g, delimiter=',')
        os.system('cat %s/a.txt >> %s/Accumelated_points.txt '%(str(output_dir),str(output_dir)))
        os.system('rm -rf %s/a.txt'%str(output_dir))
        
        print('Output saved in %s' %str(output_dir))
        return            

    def run_regressor(self,th_value,num_FC_layers,neurons,learning_rate=0.01,epochs=100,batch_size=100,print_output=True):
        pathS, Lesh,SPHENOMODEL,output_dir,TotVarScanned, VarMin , VarMax,VarLabel,VarNum,TotVarTarget, TargetMin , TargetMax,TargetLabel,TargetNum ,TargetResNum   = read_input() 
        check_(pathS,Lesh,SPHENOMODEL,output_dir)  
        model = MLP_Regressor(TotVarScanned,1,num_FC_layers,neurons)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.MeanAbsoluteError())    
        X_g,ll_g=run_train_reg(self.L1,th_value,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)
        model.fit(X_g,ll_g,epochs=epochs, batch_size=batch_size,verbose=0)
        q= 0
        while len(X_g) < self.collected_points:
            q+=1
            x_test = generate_init_HEP(self.L,TotVarScanned,pathS,Lesh,VarLabel,VarMin,VarMax)
            limits = np.column_stack((VarMin,VarMax))
            Veg_map = vegas_map_samples(X_g,ll_g,limits)
            x,_ = Veg_map(x_test)
            pred = model.predict(x,verbose=0).flatten()
            ll_ = likelihood(np.array(pred),np.array(TargetMax),np.array(TargetMin))
            x_new = x[ll_>th_value]
            y_new = pred[ll_>th_value]
            xsel = np.append(x_new[:round(self.K*(1-self.frac))],x[:round(self.K*(self.frac))],axis=0)
            xsel1,ob = refine_points_reg(xsel,th_value,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)
            X_g  = np.append(X_g,xsel1[ob>th_value],axis=0)
            ll_g  = np.append(ll_g,ob[ob>th_value],axis=0)
            model.fit(X_g,ll_g,epochs=epochs, batch_size=batch_size,verbose=0)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X_g)))
            
        if os.path.exists(str(output_dir)+"/Accumelated_points.txt"): os.system('rm -rf %s/Accumelated_points.txt '%str(output_dir)) 
        f= open(str(output_dir)+"/Accumelated_points.txt","x")
        header = '\t'.join(VarLabel)
        f.write(header+'\t' + 'Likelihood \n')
        f.close()
        X_final = np.column_stack((X_g,ll_g))
        np.savetxt(str(output_dir)+'/a.txt',X_final, delimiter=',')
        os.system('cat %s/a.txt >> %s/Accumelated_points.txt '%(str(output_dir),str(output_dir)))
        os.system('rm -rf %s/a.txt'%str(output_dir))
        
        print('Output saved in %s' %str(output_dir))
        return
###############################################################
# Create an instanace of the calss to access all functions    #
###############################################################
def RFC(collected_points=500,L1=100,L=1000,K=100,period=1,frac=0.2,learning_rate=0.01,n_estimators=300,max_depth=50,print_output=True):
    ''' Function to run the scan over SPheno Package using Random Forest Calssifier.
  Requirements:
                       1) Input file specifies the spheno directory, output directory, scan ranges and target ranges.
                       2) Sklearn is used to import the RandomForest Classifier
                       
  Input args: 
                  1) collected_points: Number of the  collect points
                  2) L1: Number of random scanned points at the zero step to train the network
                  3) L: Number of the generated points to the network for prediction
                  4) K: Number of the predicted points to be refined using the SPheno package
                  5) period: Number to define the period to train the network
                  6) frac: Fraction of the randomly added points to cover the full parameter space, e.g. 0.2 for 20% random points.
                  7)K_smote: Number of nearest neighbours used by SMOTE
                  8) learning_rate: Value of the learning rate of the network
                  9)n_estimator: Number of estimator used by the Random Forest. See the Sklearn manual for more details
                  10) max_depth: maximum depth of the reandom forest. See the Sklearn manual for more details
                  11) print_output: if True, the code will print information about the collected points during the run
  Output args:
                     text file contains the valid points in the output directory.
  Defalut initialization:           
    RFC(iteration=10,L1=100,L=1000,K=100,period=1,frac=0.2,K_smote=1,learning_rate=0.01,n_estimators=300,max_depth=50,print_output=True)                         
    ''' 
    model = scan(collected_points,L1,L,K,period,frac)  
    model.run_RFC(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=n_estimators,print_output=True)
    return
  
#########################################################################  
def MLPC(vegas=True,collected_points=5000,L1=100,L=1000,K=300,period=1,frac=0.2,learning_rate=0.01,num_FC_layers=5,neurons=100,print_output=True):
    ''' Function to run the scan over SPheno Package using MLP Calssifier.
  Requirements:
                       1) Input file specifies the spheno directory, output directory, scan ranges and target ranges.
                       2) Keras  is used to import the MLP dense layers.
                       
  Input args: 
                  1) collected_points: Number of the  collect points
                  2) L1: Number of random scanned points at the zero step to train the network
                  3) L: Number of the generated points to the network for prediction
                  4) K: Number of the predicted points to be refined using the SPheno package
                  5) period: Number to define the period to train the network
                  6) frac: Fraction of the randomly added points to cover the full parameter space, e.g. 0.2 for 20% random points.
                  7)K_smote: Number of nearest neighbours used by SMOTE
                  8) num_FC_layers: Number of the used dense (fully connected) layers
                  9)neurons: Number of nuerons used in each layer.
                  10) max_depth: maximum depth of the reandom forest. See the Sklearn manual for more details
                  11) print_output: if True, the code will print information about the collected points during the run
  Output args:
                     text file contains the valid points in the output directory.
  Defalut initialization:           
   MLPC(collected_points=500,L1=100,L=1000,K=100,period=1,frac=0.2,K_smote=1,learning_rate=0.01,num_FC_layers=5,neurons=100,print_output=True)                      
    ''' 
    model = scan(collected_points,L1,L,K,period,frac)  
    model.run_MLPC(num_FC_layers,neurons,learning_rate=learning_rate,vegas=vegas,print_output=print_output)
    return  
#############
def ML_SL(vegas=True,collected_points=1000,L1=100,L=1000,K=300,period=1,frac=0.2,learning_rate=0.01,num_FC_layers=5,neurons=100,print_output=True):
    ''' Function to run the scan over SPheno Package using MLP Calssifier.
  Requirements:
                       1) Input file specifies the spheno directory, output directory, scan ranges and target ranges.
                       2) Keras  is used to import the MLP dense layers.
                       
  Input args: 
                  1) collected_points: Number of the  collect points
                  2) L1: Number of random scanned points at the zero step to train the network
                  3) L: Number of the generated points to the network for prediction
                  4) K: Number of the predicted points to be refined using the SPheno package
                  5) period: Number to define the period to train the network
                  6) frac: Fraction of the randomly added points to cover the full parameter space, e.g. 0.2 for 20% random points.
                  7)K_smote: Number of nearest neighbours used by SMOTE
                  8) num_FC_layers: Number of the used dense (fully connected) layers
                  9)neurons: Number of nuerons used in each layer.
                  10) max_depth: maximum depth of the reandom forest. See the Sklearn manual for more details
                  11) print_output: if True, the code will print information about the collected points during the run
  Output args:
                     text file contains the valid points in the output directory.
  Defalut initialization:           
   MLPC(collected_points=500,L1=100,L=1000,K=100,period=1,frac=0.2,K_smote=1,learning_rate=0.01,num_FC_layers=5,neurons=100,print_output=True)                      
    ''' 
    model = scan(collected_points,L1,L,K,period,frac)  
    model.run_similarity(num_FC_layers,neurons,learning_rate=learning_rate,vegas=vegas,print_output=print_output)
    return  
#############

def ML_regressor(th_value=0.5,collected_points=500,L1=50,L=50000,K=300,period=1,frac=0.1,learning_rate=0.0001,num_FC_layers=5,neurons=100,print_output=True):
    ''' Function to run the scan over SPheno Package using MLP Calssifier.
  Requirements:
                       1) Input file specifies the spheno directory, output directory, scan ranges and target ranges.
                       2) Keras  is used to import the MLP dense layers.
                       
  Input args: 
                  0) th_value: Likelihood threshold to accept the collected points
                  1) collected_points: Number of the  collect points
                  2) L1: Number of random scanned points at the zero step to train the network
                  3) L: Number of the generated points to the network for prediction
                  4) K: Number of the predicted points to be refined using the SPheno package
                  5) period: Number to define the period to train the network
                  6) frac: Fraction of the randomly added points to cover the full parameter space, e.g. 0.2 for 20% random points.
                  7)K_smote: Number of nearest neighbours used by SMOTE
                  8) num_FC_layers: Number of the used dense (fully connected) layers
                  9)neurons: Number of nuerons used in each layer.
                  10) max_depth: maximum depth of the reandom forest. See the Sklearn manual for more details
                  11) print_output: if True, the code will print information about the collected points during the run
  Output args:
                     text file contains the valid points in the output directory.
  Defalut initialization:           
   MLPC(collected_points=500,L1=100,L=1000,K=100,period=1,frac=0.2,K_smote=1,learning_rate=0.01,num_FC_layers=5,neurons=100,print_output=True)                      
    ''' 
    model = scan(collected_points,L1,L,K,period,frac)  
    model.run_regressor(th_value,num_FC_layers,neurons,learning_rate=0.01,epochs=500,print_output=True)
    return      

