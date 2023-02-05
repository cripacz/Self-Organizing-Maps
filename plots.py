import numpy as np
import pandas as pd
#SOMToolbox Parser
from SOMToolBox_Parse import SOMToolBox_Parse
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
from ChernoffFace import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.cm as cm

 
def barplots(data, sel_unit, sel_attr, true_vis, mapsize, fig_size):
    
    '''
    The variables are the input data, the selected units to show, the attributes to display.
    The true_vis variable allows to show the map in the correct topology, according to mapsize.
    With fig_size the user select the size of the figure, which is convenient in a jupyter notebook.
    If true_vis == False, the units are displayed sequentially and the number of rows/cols is computed
    by rounding the square root of the total number of selected cells in order to mimic the shape of a
    nxn matrix as much as possible.
    '''

    fig = plt.figure(figsize=fig_size)
    tot = len(sel_unit)
    n_rows, n_cols = np.sqrt(tot).round(), np.sqrt(tot).round()
    if (n_rows*n_cols < tot):
        n_rows+=1
    bars = [str(x) for x in sel_attr]
    data=data[:,sel_attr]
    c = cm.rainbow(np.linspace(0, 1, len(sel_attr)))
    y_pos = np.arange(len(bars))
    max = np.max(data[sel_unit])
    min = np.min(data[sel_unit])
    plt.tight_layout()

    if true_vis == False:
        for i in range(tot):
            plt.subplot(int(n_rows), int(n_cols),i+1)
            plt.bar(y_pos, data[i], color=c)
            plt.ylim(min,max)
            plt.xticks(y_pos, bars)
            plt.title(f"Unit {sel_unit[i]}")
            
    else :
        k=0
        for i in range(tot):
            plt.subplot(mapsize[0],mapsize[1],sel_unit[k]+1)
            plt.bar(y_pos, data[i], color=c)
            plt.ylim(min,max)
            plt.xticks([]) 
            plt.yticks([]) 
            k+=1
    
    plt.tight_layout(pad=7)
    plt.show()

def radarplot(data, sel_unit, sel_attr, fig_size):

    '''
    The variables are the input data, the selected units to show, the attributes to display.
    With fig_size the user select the size of the figure, which is convenient in a jupyter notebook.
    The units are displayed sequentially and the number of rows/cols is computed
    by rounding the square root of the total number of selected cells in order to mimic the shape of a
    nxn matrix as much as possible. In this case true_vis is not implemented, because it is useful 
    for a high number of selected units and radar plots would be hardly interpretable.
    '''

    fig = plt.figure(figsize=fig_size)
    tot = len(sel_unit)
    n_rows, n_cols = int(np.sqrt(tot).round()), int(np.sqrt(tot).round())
    if (n_rows*n_cols < tot):
        n_rows+=1
    categories = [str(x) for x in sel_attr]
    data=data[:,sel_attr][sel_unit, :]
    
    plt.tight_layout()

    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    c = cm.rainbow(np.linspace(0, 1, len(sel_unit)))

    fig, axs = plt.subplots(figsize=fig_size, nrows=n_rows, ncols=n_cols)
    y=[]
    for ax, i in zip(axs.flat, range(len(sel_unit))):
        
        # Initialise the spider plot
        ax = plt.subplot(n_rows,n_cols,i+1, polar=True)

        # If you want the first axis to be on top:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], categories)
      
        # Draw ylabels
        ax.set_rlabel_position(0)
        ticks = np.linspace(np.min(data), np.max(data), 5)
        
        ticks_s = [str(x.round(2)) for x in ticks]

        plt.yticks(ticks, ticks_s, color="grey", size=7)
        plt.ylim(np.min(data),np.max(data))
        
        values = data[i,:]
        values= np.insert(values, len(values), values[0])
        ax.plot(angles, values, linewidth=1, linestyle='solid', label="group A", color=c[i])
        ax.fill(angles, values, 'b', alpha=0.4, color=c[i])
        ax.tick_params(axis='x', pad=0)
        ax.set_title(f'Unit {sel_unit[i]}', loc='left')

    mod = len(sel_unit) % n_rows

    if mod != 0:
        for j in range(mod,n_cols):
            fig.delaxes(axs[n_rows-1,j]) 

    plt.tight_layout(pad=5)

    plt.show()
    
def chernoff(data, sel_unit, sel_attr):
    '''
    The variables are the input data, the selected units to show, the attributes to display.
    The units are displayed sequentially. In this case true_vis is not implemented, because it is useful 
    for a high number of selected units and radar plots would be hardly interpretable.
    '''
    #for i in range(len(sel_attr)):
    #    max, min = np.max(data[:,i]), np.min(data[:,i])      
    #    data[:,i] = (data[:,i] - min)/(max-min)
    
    fig = chernoff_face(data=data[sel_unit,:][:, sel_attr], 
                        titles = ['Unit '+str(x) for x in sel_unit], 
                        color_mapper=plt_cm.Pastel1)

    fig.tight_layout()
    plt.show()


def boxplot(data, sel_unit, sel_attr):
    ''' For each of the selected units there will be a boxplot 
        that contains the information about the selected attributes.
    '''
    sel_data = []
    for unit in sel_unit:
        units_data = []
        for attr in sel_attr:
            units_data.append(data[unit,attr])
        sel_data.append(units_data)
    ax = plt.subplot()
    ax.boxplot(sel_data)
    ax.set_xticks([i + 1 for i in range(len(sel_unit))], labels=[str(unit) for unit in sel_unit])
    ax.set_xticklabels([str(unit) for unit in sel_unit])
    ax.set_ylabel('Distribution')
    ax.set_xlabel('Unit')
   
    plt.show()


def violinplot(data, sel_unit, sel_attr):
    ''' For each of the selected units there will be a violin plot 
        that contains the information about the selected attributes.
    '''
    sel_data = []
    for unit in sel_unit:
        units_data = []
        for attr in sel_attr:
            units_data.append(data[unit,attr])
        sel_data.append(units_data)
    ax = plt.subplot()
    ax.violinplot(sel_data, showextrema=False)
    ax.yaxis.grid(True)
    ax.set_xticks([i + 1 for i in range(len(sel_unit))], labels=[str(unit) for unit in sel_unit])
    ax.set_xticklabels([str(unit) for unit in sel_unit])
    ax.set_ylabel('Distribution')
    ax.set_xlabel('Unit')

    plt.show()
    

####### tests

if __name__ == "__main__":
    # Loading data from a pre-trained map as an example
    idata = SOMToolBox_Parse("datasets/iris/iris.vec").read_weight_file()
    weights = SOMToolBox_Parse("datasets/iris/iris.wgt.gz").read_weight_file()
    data = weights['arr']

    # In order to plot attribute values per unit, we let the user select the attributes (in this case the
    # maximum number is 4) and the units (in this case a 10x10 map, so 100 units)
    selected_attributes = [0,1,2,3]
    selected_units = [3,5,7,19,10,55,88,36,44]

    # And then we just call the function for the plots
    barplots(data, selected_units, selected_attributes, False, (10,10))
    chernoff(data, selected_units, selected_attributes)
    radarplot(data, selected_units, selected_attributes)
    boxplot(data, selected_units, selected_attributes)
    violinplot(data, selected_units, selected_attributes)