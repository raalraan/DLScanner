def plot_contour_scatter(file_:str, x_col:int, y_col:int, x_label:str,y_label:str):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import kde
    data = np.loadtxt(file_,delimiter=',')
    x = data[:,x_col]
    y = data[:,y_col]
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi  = zi/zi.max()
    # Plot contour
    fig = plt.figure(figsize=(15,6))
    fig.add_subplot(121)
    plt.contourf(xi, yi, zi.reshape(xi.shape), levels=10, cmap="RdBu_r")
    plt.tick_params(labelsize=15)
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    cbar = plt.colorbar()
    cbar.set_label('Density', fontsize=20)
    cbar.ax.tick_params(labelsize=15) 

    fig.add_subplot(122)
    plt.scatter(x,y,s=2);
    plt.tick_params(labelsize=15)
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    plt.tight_layout()
    plt.savefig('contour_%s_%s.png'%(x_label,y_label))
    plt.show()
    
    
def plot_contour(file_:str, x_col:int, y_col:int, x_label:str,y_label:str):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import kde
    data = np.loadtxt(file_,delimiter=',')
    x = data[:,x_col]
    y = data[:,y_col]
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi  = zi/zi.max()
    # Plot contour
    fig = plt.figure(figsize=(10,6))
    plt.contourf(xi, yi, zi.reshape(xi.shape), levels=10, cmap="RdBu_r")
    plt.tick_params(labelsize=15)
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    cbar = plt.colorbar()
    cbar.set_label('Density', fontsize=20)
    cbar.ax.tick_params(labelsize=15) 
    plt.savefig('contour_%s_%s.png'%(x_label,y_label))
    plt.show()    
    
def plot_scatter(file_:str, x_col:int, y_col:int, x_label:str,y_label:str):
    import numpy as np
    import matplotlib.pyplot as plt
    data = np.loadtxt(file_,delimiter=',')
    x = data[:,x_col]
    y = data[:,y_col]
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi  = zi/zi.max()
    # Plot contour
    fig = plt.figure(figsize=(10,6))

    plt.scatter(x,y,s=2);
    plt.tick_params(labelsize=15)
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    plt.tight_layout()
    plt.savefig('contour_%s_%s.png'%(x_label,y_label))
    plt.show()
    
    
    
def plot_hist_all(file_:str,width= 15,hight=10, bins=100, Normalize=True, grid=True):
  import numpy as np
  import matplotlib.pyplot as plt
  f= open(file_, 'r')
  label = f.readlines()
  label1 = label[0]
  labels = label1.rsplit('\t')
  data = np.loadtxt(file_,delimiter=',')
  dim = int(np.ceil(np.sqrt(data.shape[1])))
  fig = plt.figure(figsize=(width,hight))
  for i in range(data.shape[1]):
    y = fig.add_subplot(dim, dim, i+1)
    y.hist(data[:,i], bins=bins,histtype='step', fill=False,density=Normalize);
    y.set_ylabel('Density', fontsize=20);
    y.set_xlabel(labels[i], fontsize=20);
    plt.tick_params('both', labelsize=15);
    if grid:
      plt.grid(linestyle='--', color='k',linewidth=1.1)
  plt.tight_layout()
  plt.savefig('plot_hist_all.pdf')
  plt.show()  



def plot_hist(file_:str,width= 15,hight=10, bins=100, Normalize=True, grid=True,col:int):
  import numpy as np
  import matplotlib.pyplot as plt
  f= open(file_, 'r')
  label = f.readlines()
  label1 = label[0]
  labels = label1.rsplit('\t')
  data = np.loadtxt(file_,delimiter=',')
  fig = plt.figure(figsize=(width,hight))
  plt.hist(data[:,col], bins=bins,histtype='step', fill=False,density=Normalize);
  plt.ylabel('Density', fontsize=20);
  plt.xlabel(labels[col], fontsize=20);
  plt.tick_params('both', labelsize=15);
  if grid:
    plt.grid(linestyle='--', color='k',linewidth=1.1)
  plt.tight_layout()
  plt.savefig('plot_hist.pdf')
  plt.show()  
