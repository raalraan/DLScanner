import re
import os
import sys
import numpy as np
import time 
from shutil import move
import vegas
import numpy as np
import random
import sklearn

from ...utilities.try_imports import try_tensorflow
tf = try_tensorflow()
keras = try_tensorflow('keras')

###################
if  os.path.exists('spheno/'):
      os.system('rm -rf spheno/')
if  os.path.exists('input.txt'):
      os.system('rm -rf input.txt')      
os.system('git clone https://gitlab.com/Ahammad115566/spheno.git')
os.system('chmod 777 spheno/bin/*')
os.system('mv spheno/input.txt  .')

####################
def read_(path_:str):
  file_ = open(path_,'r')
  return file_.readlines()
def extract_floats(s):
    # Split the string into individual words or tokens
    tokens = s.split()
    floats = []
    
    # Iterate over the tokens
    for token in tokens:
        try:
            # Try to convert the token to a float
            number = float(token)
            floats.append(number)
        except ValueError:
            # If it cannot be converted, move to the next token
            continue
    
    return floats
######################
def read_input():
  ''' Function to read the input from a given file'''
  
  file_ =  'input.txt'  #str(input('Please enter the full the path to the input file (including the file name): '))
  file_ = file_.replace(" ","")  
  if not os.path.exists(str(file_)): sys.exit (str(file_)+'   file not exist. EXIT')
  x = read_(file_)  
  VarMin=[]
  VarMax =[]
  VarLabel=[]
  VarNum= []
  TotVarScanned = 0
  
  TargetMin=[]
  TargetMax =[]
  TargetLabel=[]
  TargetNum= []
  TargetResNum=[]
  TotVarTarget = 0
  
  for line in x:
    if str('TotVarScanned') in line: 
      TotVarScanned = int(extract_floats(line)[0])
    if str('VarMin') in line:
      l = float(extract_floats(line)[0])
      VarMin.append(float(l))
    if str('VarMax') in line:
      l = float(extract_floats(line)[0])
      VarMax.append(float(l))  
    if str('VarLabel') in line:
      l = line.rsplit(":")[-1]
      l = ''.join(l).strip()
      l= l.replace("\n","")
      VarLabel.append(str(l))  
    if str('VarNum') in line:
      l = int(extract_floats(line)[0])
      VarNum.append(int(l))  
    if str('pathS') in line:
      l = line.rsplit()[-1]
      l= l.replace(" ","")
      l= l.replace("\n","")
      pathS= l.replace("'","")  
      pathS= str(os.getcwd())+'/spheno/' 
      
    if str('Lesh:') in line:
      l = line.rsplit()[-1]
      l= l.replace(" ","")
      l= l.replace("\n","")
      Lesh= l.replace("'","")    
    if str('SPHENOMODEL') in line:
      l = line.rsplit()[-1]
      l= l.replace(" ","")
      l= l.replace("\n","")
      SPHENOMODEL= l.replace("'","")   
      
    if str('output_dir') in line:
      l = line.rsplit()[-1]
      l= l.replace(" ","")
      l= l.replace("\n","")
      output_dir= str(os.getcwd())+'/Out_MSSM/'  #l.replace("'","")       
    
    if str('TotTarget') in line: 
      TotVarTarget= int(extract_floats(line)[0])
    if str('TargetMin') in line:
      l = float(extract_floats(line)[0])
      TargetMin.append(float(l))
    if str('TargetMax') in line:
      l = float(extract_floats(line)[0])
      TargetMax.append(float(l))  
    if str('TargetLabel') in line:
      l = line.rsplit(":")[-1]
      l = ''.join(l).strip()
      l= l.replace("\n","")
      TargetLabel.append(str(l))  
    if str('TargetNum') in line:
      l = int(extract_floats(line)[0])
      TargetNum.append(int(l))  
    if str('TargetResNum') in line:
      l = int(extract_floats(line)[0])
      TargetResNum.append(int(l))      
      
  VarMin = VarMin[:TotVarScanned] 
  VarMax = VarMax[:TotVarScanned]   
  VarLabel = VarLabel[:TotVarScanned]    
  VarNum = VarNum[:TotVarScanned]  
  
  TargetMin = TargetMin[:TotVarTarget] 
  TargetMax = TargetMax[:TotVarTarget]   
  TargetLabel = TargetLabel[:TotVarTarget]    
  TargetNum = TargetNum[:TotVarTarget]   
  TargetResNum = TargetResNum[:TotVarTarget]   
  return pathS, Lesh,SPHENOMODEL,output_dir,TotVarScanned, VarMin , VarMax,VarLabel,VarNum,TotVarTarget, TargetMin , TargetMax,TargetLabel,TargetNum ,TargetResNum  
  
def generate_init_HEP(n,TotVarScanned,paths,Lesh,VarLabel,VarMin,VarMax):
    AI_2 = np.empty(shape=[0,TotVarScanned])
    for i in range(n):
        LHEfile = open(str(paths)+str(Lesh),'r+')
        AI_1 = []
        for line in LHEfile: 
            NewlineAdded = 0
            for yy in range(0,TotVarScanned):
                if str(VarLabel[yy]) in line:
                    value = VarMin[yy] + (VarMax[yy] - VarMin[yy])*random.random()
                    AI_1.append(value)
        AI_1= np.array(AI_1).reshape(1,TotVarScanned)   
        AI_2 = np.append(AI_2,AI_1,axis=0)    
    return AI_2       
    
###########################    
def check_(pathS:str,Lesh:str,SPHENOMODEL:str,output:str):
  if not os.path.exists(str(pathS)+('bin/SPheno')):
    sys.exit ('"/bin/SPheno" NOT EXIST, PLEASE TYPE make.')
  if not  os.path.exists(str(pathS)+str(Lesh)):
    sys.exit (str(pathS)+str(Lesh)+' NOT EXIST.')
  if not  os.path.exists(str(pathS)+'/bin/SPheno%s'%(str(SPHENOMODEL))):
    sys.exit (str(paths)+'/bin/SPheno%s'%(str(SPHENOMODEL))+' NOT EXIST.')
  if  os.path.exists(str(output)+'/'):
      os.system('rm -rf %s'%(str(output)+'/'))
  if  not os.path.exists(str(output)+'/'):
    os.mkdir(str(output)+'/')  
    os.mkdir(str(output)+'/Spectrums/')   
    print(f' dir created at: {str(output)}//Spectrums/')
  return None  
#########################
def const(i,TotConstScanned,ConstLabel,ConstNum,ConstResNum,ConstMin,ConstMax):
   '''Helper function to read all targets from the input file.
   '''
   for Xz in range(0,TotConstScanned):
      null = os.system("grep '%s' %s  >/dev/null"%(str(ConstLabel[Xz]),str(i))) 
      if (null == 256):
         return 0 
   f=open(str(i), 'r')
   lim = []
   for xxx in f: 
      for zz in range(0,TotConstScanned):
          if (str(ConstNum[zz])  and str(ConstLabel[zz])) in xxx:
            r = xxx.rsplit()
            Xmm = int(ConstResNum[zz])
            if (Xmm == 1 and str(r[0])!= 'DECAY') :
               l = int(float(r[Xmm]))
               lim.append(l)
            if (Xmm != 1) :
               l =  float(r[Xmm])
               mm = float(ConstMin[zz])
               nm = float(ConstMax[zz])
               lim.append(l)
   return np.array(lim)   

def likelihood(exp_value,th_max,th_min):
    out = []
    for b in range(len(exp_value)):
        ll_st = 1
        for i in range(len(th_max)):
            th = (th_max[i] + th_min[i])/2
            std = abs(th_max[i] ) - abs(th)
            ll_c = np.exp(- (exp_value[b] - th)**2/(2*std**2))
            ll_st *= ll_c
        out.append(ll_st)    
    return np.array(out)                         
#################
def run_train(npoints,th_value,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,_output):     
    '''Function to run the Spheno scan randomly. This is the zero step in the scanning loop.
    Input Args: 
                 npoints: Number of points to be specified by the user 
                 other inputs: Inputs that extracted from the input file.
    retuerns:
              AI_X: Features of the dimensions (npoints, func_dim)
              AI_Y: labels for each point. Points in the traget region have label ==1, outside ==0             
    ''' 
    os.chdir(pathS)
    AI_X = np.empty(shape=[0,TotVarScanned])
    AI_Y = []
    xx =0
    ###########################
    while  len(AI_Y) < npoints:
        xx +=1
        sys.stdout.write('\r'+' Running the initial random scanning to collect points to train the ML network: %s / %s ' %(len(AI_Y), npoints)) 
        newrunfile = open('newfile','w')
        oldrunfile = open(str(Lesh),'r+')
        AI_L = []
        for line in oldrunfile: 
            NewlineAdded = 0
            for yy in range(0,TotVarScanned):
                if str(VarLabel[yy]) in line:
                    value = VarMin[yy] + (VarMax[yy] - VarMin[yy])*random.random()
                    AI_L.append(value)
                    valuestr = str("%.4E" % value)
                    newrunfile.write(str(VarNum[yy])+'   '+valuestr +str('     ')+ VarLabel[yy]+'\n')
                    NewlineAdded = 1
            if NewlineAdded == 0:
                newrunfile.write(line)
        newrunfile.close()
        oldrunfile.close()
        os.remove(str(Lesh))
        AI_L= np.array(AI_L).reshape(1,TotVarScanned)
        ############################    
        os.rename('newfile',str(Lesh))
        os.system('./bin/SPheno'+str(SPHENOMODEL)+' '+str(Lesh)+' spc.slha'+' >  out.txt ')
        out = open(str(pathS)+'out.txt','r+')
        for l in out:
            if str('Finished!') in l:
              label = const('spc.slha',TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax)
              ll_1 = likelihood(label,TargetMax,TargetMin)
              if ll_1 [0]> th_value:
                  AI_Y.append(ll_1 [0])
                  AI_X = np.append(AI_X,AI_L,axis=0)
                  st = time.time()
                  os.rename('spc.slha','SPheno.spc.%s_%s'%(str(SPHENOMODEL),(st)))
                  move('SPheno.spc.%s_%s'%(str(SPHENOMODEL),(st)),_output+"/Spectrums/")
        os.remove('out.txt')
    return np.array(AI_X),np.array(AI_Y)  
#################################
def refine_points(npoints,th_value,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,_output):      
    ''' Function to refine the generated points
  Return: 
           AI_X: refined points. Collected points that tested by Spheno
           AI_Y: refined labels. labels that tested by Spheno
   '''
 
    os.chdir(pathS)
    AI_X = np.empty(shape=[0,TotVarScanned])
    AI_Y = []
  ######
    for xx in range(0,npoints.shape[0]):
          #sys.stdout.write('\r'+'Correcting the predicted points: %s / %s ' %(xx+1, npoints.shape[0])) 
          newrunfile = open('newfile','w')
          oldrunfile = open(str(Lesh),'r+')
          AI_L = []
          for line in oldrunfile: 
              NewlineAdded = 0
              for yy in range(0,TotVarScanned):
                  if str(VarLabel[yy]) in line:
                      value = npoints[xx,yy]
                      AI_L.append(value)
                      valuestr = str("%.4E" % value)
                      newrunfile.write(str(VarNum[yy])+'   '+valuestr +str('     ')+ VarLabel[yy]+'\n')
                      NewlineAdded = 1
              if NewlineAdded == 0:
                  newrunfile.write(line)
          newrunfile.close()
          oldrunfile.close()
          os.remove(str(Lesh))
          AI_L= np.array(AI_L).reshape(1,TotVarScanned)
          ############################    
          os.rename('newfile',str(Lesh))
          os.system('./bin/SPheno'+str(SPHENOMODEL)+' '+str(Lesh)+' spc.slha'+' >  out.txt')
          out = open(str(pathS)+'out.txt','r+')
          for l in out:
              if str('Finished!') in l:
                label = const('spc.slha',TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax)
                ll_1 = likelihood(label,TargetMax,TargetMin)
                if ll_1 [0]> th_value:
                  AI_Y.append(ll_1 [0])
                  AI_X = np.append(AI_X,AI_L,axis=0)
                  st = time.time()
                  os.rename('spc.slha','SPheno.spc.%s_%s'%(str(SPHENOMODEL),(st)))
                  move('SPheno.spc.%s_%s'%(str(SPHENOMODEL),(st)),_output+"/Spectrums/")
          os.remove('out.txt')
    return np.array(AI_X),np.array(AI_Y)  

##################===========================#########
############# VEGAS ###############################
##################===========================#########
def convert_to_unit_cube(x, limits):
    ndim = x.shape[1]
    new_x = np.empty(x.shape)

    for k in range(ndim):
        width = limits[k][1] - limits[k][0]
        new_x[:, k] = (x[:, k] - limits[k][0])/width

    return new_x


def convert_to_limits(x, limits):
    ndim = x.shape[1]
    new_x = np.empty(x.shape)

    for k in range(ndim):
        width = limits[k][1] - limits[k][0]
        # new_x[:, k] = (x[:, k] - limits[k][0])/width
        new_x[:, k] = x[:, k]*width + limits[k][0]

    return new_x



def vegas_map_samples(
        xtrain, ftrain, limits,
        ninc=100,
        nitn=5,
        alpha=1.0,
        nproc=1
):
    
    ndim = xtrain.shape[1]
    _xtrain = convert_to_unit_cube(xtrain, limits)
    vg_AdMap = vegas.AdaptiveMap([[0, 1]]*ndim, ninc=ninc)
    vg_AdMap.adapt_to_samples(
        _xtrain, ftrain,
        nitn=nitn, alpha=alpha, nproc=nproc
    )

    def _vegas_sample(npts):
        '''Obtain an array of points from a trained vegas map.
        Input Args:
            npts: int
                Number of points
        Returns:
            sample: array
                Sample of points created according to mapping
            jacobian: array
                Jacobian corresponding to mapping of the points
        '''
        xrndu = np.random.uniform(0, 1, (int(_xtrain.shape[0]),int(_xtrain.shape[1])))
        xrndmap = np.empty(xrndu.shape, xrndu.dtype)
        jacmap = np.empty(xrndu.shape[0], xrndu.dtype)
        vg_AdMap.map(xrndu, xrndmap, jacmap)

        return convert_to_limits(xrndmap, limits), jacmap
        

    return _vegas_sample
    

#################################
def MLP_Regressor(function_dim,output_dim, num_FC_layers,neurons):
 
    inp =keras.layers.Input((function_dim, ))
    x = keras.layers.Dense(neurons,activation=None)(inp)
    for _ in range(num_FC_layers):
      x = keras.layers.Dense(neurons,activation='relu')(x)
    output = keras.layers.Dense(output_dim,activation='linear')(x)
    model = keras.Model(inp,output)
    return model 

 
##################===========================#########    
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

    

    def labeler(self,x,th):
        ll=[]
        for q,item in enumerate(x):
            if item < th:
                ll.append(1)
            else:
                ll.append(0)
        return np.array(ll).ravel()        

    
    def run_regressor(self,th_value,num_FC_layers,neurons,learning_rate=0.01,epochs=100,batch_size=100,print_output=True):
        pathS, Lesh,SPHENOMODEL,output_dir,TotVarScanned, VarMin , VarMax,VarLabel,VarNum,TotVarTarget, TargetMin , TargetMax,TargetLabel,TargetNum ,TargetResNum   = read_input() 
        check_(pathS,Lesh,SPHENOMODEL,output_dir)  
        model = MLP_Regressor(TotVarScanned,1,num_FC_layers,neurons)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.MeanAbsoluteError())    
        X_g,ll_g=run_train(self.L1,th_value,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)
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
            xsel1,ob = refine_points(xsel,th_value,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,output_dir)
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
#############
ML_regressor()
