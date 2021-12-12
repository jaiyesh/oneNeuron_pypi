import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib  # For storing model for sklearn as well as keras
import os
from matplotlib.colors import ListedColormap # For selecting Colors
import logging

plt.style.use('fivethirtyeight')



def prepare_data(df):
    """It is used to separate the dependent and iindependent features

    Args:
        df (pd.DataFrame): It is a dataframe

    Returns:
        tuple: returning dependent and independent variable tuples
    """
    logging.info("preparing the data by segregating the independent and depenedent variables")
    X = df.drop('y',axis = 1)
    y = df['y']

    return X,y



def save_model(model,filename):
    """This saves the trained model

    Args:
        model (object): Its a trained model
        filename (string): path to model save
    """
    logging.info("Saving models")
    model_dir = 'NewModels' #Craeting directory for model saving
    os.makedirs(model_dir,exist_ok = True)  #Only create when model directory doesn't exist
    filePath = os.path.join(model_dir,filename) #model\filename
    joblib.dump(model,filePath)
    logging.info(f"Saved the trainde model at {filePath}")




def save_plot(df,file_name,model):
    def _create_base_plot(df): ##Internal function, can't use it outside of funcytion
        df.plot(kind = 'scatter',x = 'x1',y = 'x2',c = 'y',s=100,cmap = 'winter')
        plt.axhline(y=0,color='black',linestyle='--',linewidth = 1)
        plt.axvline(x=0,color='black',linestyle='--',linewidth = 1)
        figure = plt.gcf() ##Get current figure
        figure.set_size_inches(10,8)
    
    
    def _plot_decision_regions(X,y,classifier,resolution = 0.02):
        colors = ('red','blue','lightgreen','gray')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        
        
        X = X.values  #X as array
        
        
        x1_min,x1_max = X[:,0].min() -1, X[:,0].max()+1
        x2_min,x2_max = X[:,1].min() -1, X[:,1].max()+1
        
        #MeshGrid : Creating x1 and x2 values and all these values we do prediction
        xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                             np.arange(x2_min,x2_max,resolution))
        
        Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1,xx2,Z,alpha = 0.2,cmap = cmap)  #Contourf for x1 and x2 and color will be based on Z value
        plt.xlim(xx1.min(),xx1.max())
        plt.ylim(xx2.min(),xx2.max())
        
        plt.plot()
        
        
        
        
    
    X,y = prepare_data(df)
    
    _create_base_plot(df)
    _plot_decision_regions(X,y,model)
    
    
    plot_dir ='newplots'
    os.makedirs(plot_dir,exist_ok=True) 
    plotPath = os.path.join(plot_dir,file_name)
    plt.savefig(plotPath)
    logging.info('Saving the plots')