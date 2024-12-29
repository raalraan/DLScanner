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
   lim = 1
   for xxx in f: 
      for zz in range(0,TotConstScanned):
          if (str(ConstNum[zz])  and str(ConstLabel[zz])) in xxx:
            r = xxx.rsplit()
            Xmm = int(ConstResNum[zz])
            if (Xmm == 1) :
               l = int(float(r[Xmm]))
               if (l not in range(int(ConstMin[zz]),int(ConstMax[zz])) and str(r[0])!= 'DECAY'):
                  lim *= 0
               else: lim *=1   
            if (Xmm != 1) :
               l =  float(r[Xmm])
               mm = float(ConstMin[zz])
               nm = float(ConstMax[zz])
               if (l < mm) or (l>nm):
                  lim *=0
               else: lim *=1
   return lim   
               
#################
def run_train(npoints,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,_output):     
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
    while  sum(AI_Y) < npoints:
        xx +=1
        sys.stdout.write('\r'+' Running the initial random scanning to collect points to train the ML network: %s / %s ' %(sum(AI_Y), npoints)) 
        newrunfile = open('newfile','w')
        oldrunfile = open(str(Lesh),'r+')
        AI_L = []
        for line in oldrunfile: 
            NewlineAdded = 0
            for yy in range(0,TotVarScanned):
                if str(VarLabel[yy]) in line:
                    value = VarMin[yy] + (VarMax[yy] - VarMin[yy])*random.random()
                    #print(f' VarMin:  {VarMin[yy]}      VarMax:   {VarMax[yy]}             VarLabel:  {str(VarLabel[yy])}  ')
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
              AI_Y.append(label)          
              AI_X = np.append(AI_X,AI_L,axis=0)
              if label == 1:    
                  st = time.time()
                  os.rename('spc.slha','SPheno.spc.%s_%s'%(str(SPHENOMODEL),(st)))
                  move('SPheno.spc.%s_%s'%(str(SPHENOMODEL),(st)),_output+"/Spectrums/")
        os.remove('out.txt')
    return np.array(AI_X),np.array(AI_Y)  
#################################
def refine_points(npoints,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,_output):      
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
                AI_Y.append(label)          
                AI_X = np.append(AI_X,AI_L,axis=0)
                if label == 1:    
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
    '''Train a mapping of the parameter space using vegas and a sample of points.
    Input Args:
        xtrain: array
            Coordinates of the sample. All the coordinates must be normalized to the [0, 1] range
        ftrain: array
            Result of evaluating a function on xtrain.
        ninc: int, optional
            number of increments used in the mapping (see vegas documentation for AdaptiveMap)
        nitn: int, optional
            number of iterations used to refine mapping (see vegas documentation for AdaptiveMap.adapt_to_samples())
        alpha: float, optional
            Damping parameter (see vegas documentation for AdaptiveMap.adapt_to_samples())
        nproc: int, optional
            Number of processes/processors to use (see vegas documentation for AdaptiveMap.adapt_to_samples())
    Returns:
        Callable function to create a random sample using the trained mapping
    '''
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
    
##################===========================#########
############# ML  ###############################
##################===========================#########    
#################################
def MLP_Classifier(function_dim,num_FC_layers,neurons):
    ''' Function to create MLP classfier.
    Input args:
    function_dim: dimensions of the input
    num_FC_layers: Number of the fully connected layers
    neurons: number of neurons in each FC layer
    output args:
             MLP classifier network
    '''
    inp =keras.layers.Input((function_dim, ))
    x = keras.layers.Dense(neurons,activation=None)(inp)
    for _ in range(num_FC_layers):
      x = keras.layers.Dense(neurons,activation='relu')(x)
    output = keras.layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model(inp,output)
    return model
#################################
def MLP_Regressor(function_dim,num_FC_layers,neurons):
    ''' Function to create MLP classfier.
    Input args:
    function_dim: dimensions of the input
    num_FC_layers: Number of the fully connected layers
    neurons: number of neurons in each FC layer
    output args:
             MLP classifier network
    '''
    inp =keras.layers.Input((function_dim, ))
    x = keras.layers.Dense(neurons,activation=None)(inp)
    for _ in range(num_FC_layers):
      x = keras.layers.Dense(neurons,activation='relu')(x)
    output = keras.layers.Dense(function_dim,activation='linear')(x)
    model = keras.Model(inp,output)
    return model 
    
###########################################
### The following is related to the siilarity learning classifier#    
###########################################
def make_pairs(x, y):
    '''Function to create positive and negative paris of input data.
   Positive pairs has label =1 and negative pairs have labels =0.
   Input Args:
        x: input data of dimension (n, function dimension)
        y: vector label of each input with dimension (n)
   Output Args:
        x: pairs of the input with dimension (n,2,function dimension)
        y: labels of the positive and negative pairs.
                  
   Exampel to run:   pairs_train, labels_train = make_pairs(Xf, obs1f)
    '''
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(int(num_classes))]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[int(label1)])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), abs(np.array(labels).astype("float32")-1)
########################
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
###############
def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be c
                lassified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )
    return contrastive_loss
#############
def similariy_classifier(x_train, y_train,function_dim,latent_dim,neurons,num_layers,epochs, learning_rate,batch_size,verbose=0):
  ''' Function to create similarity classfier-First training part of the network.
    Input args:
    function_dim: dimensions of the input
    latent_dim: Dimension of the latent space
    output args:
             similarity classifier network
  '''
  xtrain,ytrain = make_pairs(x_train, y_train)
  input_ = keras.layers.Input((function_dim, ))   
  x = keras.layers.Dense(128, activation="tanh")(input_)
  x = keras.layers.Dense(64, activation="tanh")(x)
  x = keras.layers.Dense(32, activation="tanh")(x)
  x = keras.layers.Dense(latent_dim, activation="linear")(x)
  embedding_network = keras.Model(input_, x)
  input_1 = keras.layers.Input((function_dim, ))
  input_2 = keras.layers.Input((function_dim, ))
  tower_1 = embedding_network(input_1)
  tower_2 = embedding_network(input_2)
  merge_layer = keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
  output_layer = keras.layers.Dense(1, activation="linear")(merge_layer)
  model_S = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
  model_S.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss(margin=1))
  model_S.fit([xtrain[:,0],xtrain[:,1]], ytrain,epochs=epochs, batch_size=batch_size,verbose=0)
  ####### Second training step ######
  inp = keras.layers.Input((function_dim, ))
  x = (embedding_network(inp))
  x = keras.layers.Dense(neurons,activation=None)(inp)
  x = keras.layers.Dense(neurons,activation='relu')(x)
  output = keras.layers.Dense(1,activation="sigmoid")(x)
  model = keras.Model(inp,output)
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.BinaryCrossentropy())
  model.fit(x_train,y_train,epochs=epochs, batch_size=batch_size,verbose=0)
  return model          
##################===========================#########
############# SPheo  ###############################
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

    
        
    def run_MLPC(self,num_FC_layers,neurons,learning_rate=0.01,epochs=100,batch_size=100,print_output=True):
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
        model.fit(x_t,y_t,epochs=epochs, batch_size=batch_size,verbose=0)
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
            pred = model.predict(x,verbose=0).flatten()
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
                model.fit(X_shuffled, Y_shuffled,epochs=epochs, batch_size=batch_size,verbose=0)

            else:
                X = np.concatenate([xsel2[ob==1],xsel2[ob==0]],axis=0)
                obs=np.concatenate([ob[ob==1],ob[ob==0]],axis=0)
                X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
                model.fit(X_shuffled, Y_shuffled,epochs=epochs, batch_size=batch_size,verbose=0)
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
      
    def run_similarity(self,num_FC_layers,neurons,latent_dim=10,learning_rate=0.01,epochs=100,batch_size=100,print_output=True):
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
            pred = model.predict(x,verbose=0).flatten()
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
        
###############################################################
# Create an instanace of the calss to access all functions    #
###############################################################
#########################################################################  
def MLPC_trial(collected_points=5000,L1=100,L=1000,K=300,period=1,frac=0.2,learning_rate=0.01,num_FC_layers=5,neurons=100,print_output=True):
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
    model.run_MLPC(num_FC_layers,neurons,learning_rate=0.01,print_output=True)
    return  
#############

def ML_SL_trial(collected_points=5000,L1=100,L=1000,K=300,period=1,frac=0.2,learning_rate=0.01,num_FC_layers=5,neurons=100,print_output=True):
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
    model.run_similarity(num_FC_layers,neurons,learning_rate=0.01,print_output=True)
    return  
#############
