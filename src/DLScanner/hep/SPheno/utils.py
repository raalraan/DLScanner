import re
import os
import sys
import numpy as np
import random
import time 
from shutil import move
###################
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
  
  file_ = str(input('Please enter the full the path to the input file (including the file name): '))
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
      pathS= str(pathS)+'/'
      
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
      output_dir= l.replace("'","")       
    
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

def run_train_reg(npoints,th_value,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,_output):     
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
def refine_points_reg(npoints,th_value,TotVarScanned,Lesh,VarMin,VarMax,VarNum,VarLabel,SPHENOMODEL,pathS,TotVarTarget,TargetLabel,TargetNum,TargetResNum,TargetMin,TargetMax,_output):      
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





