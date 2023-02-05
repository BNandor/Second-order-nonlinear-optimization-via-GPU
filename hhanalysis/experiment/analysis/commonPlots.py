import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
def plot_series(series_of_data_series, titles,x_label='X-axis', y_label='Y-axis', file_name=None,scale='linear'):
    prop = fm.FontProperties(fname='./plots/fonts/times-ro.ttf')
    # Create a figure and axis
    # fig, ax = plt.subplots(figsize=(8,3))
    subplots=len(series_of_data_series)
    fig, axes = plt.subplots(nrows=subplots, ncols=1,figsize=(8,3*subplots))
    
    if subplots == 1:
        axes=[axes]
    for j in range(len(series_of_data_series)):
        axes[j].set_yscale(scale)
        # Plot each series in the data_series list
        for i, series in enumerate(series_of_data_series[j]):
            x, y, label = series
            axes[j].plot(x, y, label=label,linewidth=0.8)
        
        # Add labels and title
        axes[j].set_xlabel(x_label,fontproperties=prop)
        axes[j].set_ylabel(y_label,fontproperties=prop,size=12)
        axes[j].set_title(titles[j],fontproperties=prop,size=14)
        
        # Add a legend
        axes[j].legend(prop=prop)
    plt.tight_layout(rect=(0,0,1,1))
    fig.subplots_adjust(wspace=0, hspace=0.4)
    # Save the plot to a file if file_name is provided
    if file_name:
        plt.savefig(file_name)
    
    # Show the plot
    plt.show()
    
def plotHeatmap(Ps,rows,columns,xticks,yticks,titles,xlabelTitles,ylabelTitles,figuretitles,width_ratios,height_ratios,subfigdim,figsize=(10,10),filename=None):
    prop = fm.FontProperties(fname='./plots/fonts/times-ro.ttf')
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(subfigdim[0],subfigdim[1],wspace=0.05 )
    subplotsInFig=int((rows*columns)/(subfigdim[0]*subfigdim[1]))
    for k, figrow in enumerate(subfigs):
        for l, subfig in enumerate(figrow):
            axs=subfig.subplots(1, subplotsInFig,gridspec_kw={"width_ratios":width_ratios,"height_ratios":height_ratios,'wspace': 0.0})
            # fig, axs = plt.subplots(rows, columns, figsize=figsize,)
            for i, ax in enumerate(axs):
                    im = ax.imshow(np.array(Ps[k][2*l+i]), cmap='Greens',vmin=0,vmax=1)
                    ax.set_xticks(np.arange(Ps[k][2*l+i].shape[1]))
                    ax.set_yticks(np.arange(Ps[k][2*l+i].shape[0]))
                    ax.set_xticklabels(xticks[k][2*l+i],fontproperties=prop)
                    ax.set_yticklabels(yticks[k][2*l+i],fontproperties=prop,rotation=90)
                    ax.set_title(titles[k][2*l+i],fontproperties=prop)
                    ax.tick_params(axis='both', which='both', length=0)
                    ax.set_xlabel(xlabelTitles[k][2*l+i],fontproperties=prop, va="top")
                    ax.set_ylabel(ylabelTitles[k][2*l+i],fontproperties=prop, rotation=90, va="bottom")
                    # ax.spines['top'].set_visible(False)
                    # ax.spines['right'].set_visible(False)
                    # ax.spines['bottom'].set_visible(False)
                    # ax.spines['left'].set_visible(False)
            subfig.suptitle(figuretitles[k][l],fontproperties=prop,size=14)
    # Add colorbar
    # fig.tight_layout()
    # cbar = fig.colorbar(im, ax=axs.flat, shrink=0.6)
    # cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")
    
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # Show the plot
    if filename:
        plt.savefig(filename)
    plt.show()
