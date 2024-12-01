import re
import os
import sys
import numpy as np
import random
import time 
from shutil import move
###################
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

    if str('path_micromegas') in line:
      l = line.rsplit()[-1]
      l= l.replace(" ","")
      l= l.replace("\n","")
      pathS= l.replace("'","")  
      pathS= str(pathS)+'/'
    
    if str('input:') in line:
      l = line.rsplit()[-1]
      l= l.replace(" ","")
      l= l.replace("\n","")
      Lesh= l.replace("'","")      
      
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
  
      
  VarMin = VarMin[:TotVarScanned] 
  VarMax = VarMax[:TotVarScanned]   
  VarLabel = VarLabel[:TotVarScanned]    
  VarNum = VarNum[:TotVarScanned]  
  
  TargetMin = TargetMin[:TotVarTarget] 
  TargetMax = TargetMax[:TotVarTarget]   
  TargetLabel = TargetLabel[:TotVarTarget]    
  TargetNum = TargetNum[:TotVarTarget]   
  TargetResNum = TargetResNum[:TotVarTarget]   
  return pathS,Lesh,output_dir,TotVarScanned, VarMin , VarMax,VarLabel,TotVarTarget, TargetMin , TargetMax,TargetLabel 
#####################################################################################  
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
def check_(pathS:str,Lesh:str,output:str):
  if not os.path.exists(str(pathS)):
    sys.exit('Micromegas direcotry  NOT EXIST, please provide the correct path')
  
  if not os.path.exists(str(pathS)+str(Lesh)):
    sys.exit(str(pathS)+str(Lesh)+' NOT EXIST.')
  
  if  os.path.exists(str(output)+'/'):
    os.system('rm -rf %s'%(str(output)+'/'))
  if  not os.path.exists(str(output)+'/'):
    os.mkdir(str(output)+'/')  
  return None  
  
 ###############################
def const(i,TotConstScanned,ConstLabel,ConstMin,ConstMax):
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
      #if (str(ConstLabel[zz])) in xxx:
      match = re.search(rf"{re.escape(str(ConstLabel[zz])+'=')}\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", xxx)
      if match:
        l = float(match.group(1))
        #print(f'Value of Omega:   {l}')
        mm = float(ConstMin[zz])
        nm = float(ConstMax[zz])
        if (l < mm) or (l>nm):
          lim *=0
        else: lim *=1
      #else: lim *=0  
  return lim    
  
##################################

def run_train_megas(npoints,TotVarScanned,Lesh,VarMin,VarMax,VarLabel,pathS,TotVarTarget,TargetLabel,TargetMin,TargetMax,_output):     
    os.chdir(pathS)
    AI_X = np.empty(shape=[0,TotVarScanned])
    AI_Y = []
    xx =0
    ###########################
    while  sum(AI_Y) < npoints:
        xx +=1
        sys.stdout.write('\r'+' Running the initial random scanning to collect points to train the ML network: %s / %s ' %(sum(AI_Y),npoints)) 
        newrunfile = open('newfile','w')
        oldrunfile = open(str(Lesh),'r+')
        AI_L = []
        for line in oldrunfile: 
            NewlineAdded = 0
            for yy in range(0,TotVarScanned):
                if str(VarLabel[yy]) in line:
                    value = VarMin[yy] + (VarMax[yy] - VarMin[yy])*random.random()
                    #print(f' Var{str("%.4E" % value)}            VarLabel:  {str(VarLabel[yy])}  ')
                    AI_L.append(value)
                    valuestr = str("%.4E" % value)
                    newrunfile.write(str(VarLabel[yy])+'   '+valuestr +'\n')
                    NewlineAdded = 1
            if NewlineAdded == 0:
                newrunfile.write(line)
        newrunfile.close()
        oldrunfile.close()
        os.remove(str(Lesh))
        AI_L= np.array(AI_L).reshape(1,TotVarScanned)
        ############################    
        os.rename('newfile',str(Lesh))
        os.system('./main   '+' '+str(Lesh)+' >  out.txt ')
        out = open(str(pathS)+'out.txt','r+')
        label = const('out.txt',TotVarTarget,TargetLabel,TargetMin,TargetMax)
        AI_Y.append(label)          
        AI_X = np.append(AI_X,AI_L,axis=0)
            
        os.remove('out.txt')
    return np.array(AI_X),np.array(AI_Y)    
#####################################  

def refine_points_megas(npoints,TotVarScanned,Lesh,VarMin,VarMax,VarLabel,pathS,TotVarTarget,TargetLabel,TargetMin,TargetMax,_output):      
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
                      newrunfile.write(str(VarLabel[yy])+'   '+valuestr +'\n')
                      NewlineAdded = 1
              if NewlineAdded == 0:
                  newrunfile.write(line)
          newrunfile.close()
          oldrunfile.close()
          os.remove(str(Lesh))
          AI_L= np.array(AI_L).reshape(1,TotVarScanned)
          ############################    
          os.rename('newfile',str(Lesh))
          os.system('./main   '+' '+str(Lesh)+' >  out.txt ')
          out = open(str(pathS)+'out.txt','r+')
          label = const('out.txt',TotVarTarget,TargetLabel,TargetMin,TargetMax)
          AI_Y.append(label)          
          AI_X = np.append(AI_X,AI_L,axis=0)
            
          os.remove('out.txt')
    return np.array(AI_X),np.array(AI_Y)    
###################################################
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
##################################################
#path, Lesh,output_dir,TotVarScanned, VarMin, VarMax,VarLabel,TotVarTarget, TargetMin, TargetMax,TargetLabel= read_input() 
#check_(path,Lesh,output_dir)  
#Xf,ob1=run_train_megas(50,TotVarScanned,Lesh,VarMin,VarMax,VarLabel,path,TotVarTarget,TargetLabel,TargetMin,TargetMax,output_dir)
#Xf1,ob2=refine_points_megas(Xf,TotVarScanned,Lesh,VarMin,VarMax,VarLabel,path,TotVarTarget,TargetLabel,TargetMin,TargetMax,output_dir)


