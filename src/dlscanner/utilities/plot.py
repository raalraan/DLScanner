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
    plt.colorbar(label='Density')

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
    plt.colorbar(label='Density')
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
    
    
    
#plot_contour('Accumelated_points.txt',1,4,'M1','M2')    
