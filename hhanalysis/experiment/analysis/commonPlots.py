import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pickle,json
import os
from common import *
import pandas as pd

ROOT=f"{os.path.dirname(os.path.abspath(__file__))}"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# matplotlib.rcParams['text.usetex'] = True
def plot_series(series_of_series_of_data_series, series_of_titles,x_label='X-axis', y_label='Y-axis', file_name=None,scales='linear',colors=None,figsize=(8,3),blockPlot=True):
    prop = fm.FontProperties(fname=f'{ROOT}/plots/fonts/times-ro.ttf')
    # Create a figure and axis
    # fig, ax = plt.subplots(figsize=(8,3))
    subplotsrows=len(series_of_series_of_data_series[0])
    subplotscols=len(series_of_series_of_data_series)
    fig, axes = plt.subplots(nrows=subplotsrows, ncols=subplotscols,gridspec_kw={"wspace":0.2,"hspace":0.4},figsize=(figsize[0]*subplotscols,figsize[1]*subplotsrows))
    
    if subplotsrows == 1:
        axes=[axes]
    for i in range(len(series_of_series_of_data_series)):
        for j in range(len(series_of_series_of_data_series[i])):
            if len(series_of_series_of_data_series)>1:
                ax=axes[j,i]
            else:
                ax=axes[j]
            ax.set_yscale(scales[i])
            # Plot each series in the data_series list
            for l, series in enumerate(series_of_series_of_data_series[i][j]):
                x, y, label = series
                if colors !=None:
                    ax.plot(x, y, label=label,linewidth=0.8,color=colors[l])
                else:
                    ax.plot(x, y, label=label,linewidth=0.8)
            
            # Add labels and title
            ax.set_xlabel(x_label,fontproperties=prop)
            ax.set_ylabel(y_label,fontproperties=prop,size=12)
            ax.set_title(series_of_titles[i][j],fontproperties=prop,size=14)
            
            # Add a legend
            ax.legend(prop=prop)
        plt.tight_layout(rect=(0,0,1,1))
        fig.subplots_adjust(wspace=0.5, hspace=0)
        fig.subplots_adjust(left=0.16, right=0.96, top=0.95)  # Adjust the left and right margins
        # Save the plot to a file if file_name is provided
    if file_name:
        plt.savefig(file_name)
    if blockPlot:
        plt.show()

def plotWilcoxRanksums(df,rows,columns,labels,filename,figsize=(10,10),blockPlot=True):
    prop = fm.FontProperties(fname=f'{ROOT}/plots/fonts/times-ro.ttf')    
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=figsize, gridspec_kw={'wspace': 0.0})

    # loop over each subplot and plot a random 5x5 matrix
    fig.subplots_adjust(left=0.05, right=1.05, bottom=0.01, top=0.95,wspace=0.2, hspace=0.4)
    for  i in range(rows):
        for j in range(columns):
            # if i == 0 and j == 0:
            #     axs[i, j].legend()
            # else:
            #     axs[i, j].legend_.remove()
            wilcoxRanksum = pickle.loads(json.loads(df['wilcoxRanksums'].iloc[i*columns + j]).encode('latin-1'))
            # plot the image in the current subplot
            
            if columns>1:
                axs[i, j].imshow(wilcoxRanksum, cmap='Greys',vmin=0,vmax=1)
                # axs[i, j].axis('off')
                title=f"{(df['problemName'].iloc[i*columns + j]).replace('PROBLEM_','')}-{int(df['modelSize'].iloc[i*columns + j])}"
                # axs[i, j].set_xticks(np.arange(0,wilcoxRanksum.shape[0]))
                axs[i, j].set_yticks(np.arange(0,wilcoxRanksum.shape[0]))
                # axs[i, j].set_xticklabels(range(0,wilcoxRanksum.shape[0]),fontproperties=prop)
                axs[i, j].xaxis.set_visible(False)
                axs[i, j].set_yticklabels(labels,size=8,fontproperties=prop)
                axs[i, j].set_title(title,fontproperties=prop)
            else:
                axs[i].imshow(wilcoxRanksum, cmap='Greys',vmin=0,vmax=1)
                # axs[i, j].axis('off')
                title=f"{(df['problemName'].iloc[i*columns + j]).replace('PROBLEM_','')}-{int(df['modelSize'].iloc[i*columns + j])}"
                # axs[i, j].set_xticks(np.arange(0,wilcoxRanksum.shape[0]))
                axs[i].set_yticks(np.arange(0,wilcoxRanksum.shape[0]))
                # axs[i, j].set_xticklabels(range(0,wilcoxRanksum.shape[0]),fontproperties=prop)
                axs[i].xaxis.set_visible(False)
                axs[i].set_yticklabels(labels,size=8,fontproperties=prop)
                axs[i].set_title(title,fontproperties=prop)
    if filename:
        plt.savefig(filename)
    if blockPlot:
        plt.show()

def plotHeatmap(Ps,rows,columns,xticks,yticks,titles,xlabelTitles,ylabelTitles,figuretitles,width_ratios,height_ratios,subfigdim,figsize=(10,10),filename=None,color="Greens"):
    prop = fm.FontProperties(fname=f'{ROOT}/plots/fonts/times-ro.ttf')
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(subfigdim[0],subfigdim[1],wspace=0.05 )
    subplotsInFig=int((rows*columns)/(subfigdim[0]*subfigdim[1]))
    for k, figrow in enumerate(subfigs):
        for l, subfig in enumerate(figrow):
            axs=subfig.subplots(1, subplotsInFig,gridspec_kw={"width_ratios":width_ratios,"height_ratios":height_ratios,'wspace': 0.0})
            # fig, axs = plt.subplots(rows, columns, figsize=figsize,)
            for i, ax in enumerate(axs):
                    im = ax.imshow(np.array(Ps[k][2*l+i]), cmap=color,vmin=0,vmax=1)
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

def plotMethodsComparison(categories,subcategories,thevalues,xlabel,ylabel,title,block=False):
    values=np.array(thevalues)
    # Determine the number of subcategories
    num_subcategories = len(subcategories)
    bar_width = 0.05  # Width of each bar

    # Calculate the positions of the bars on the x-axis
    positions = np.arange(len(categories))

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Create the vertical bar plot for each subcategory
    for i in range(num_subcategories):
        ax.bar(positions + i * bar_width, values[:, i], bar_width, label=subcategories[i])

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set the x-axis tick positions and labels
    ax.set_xticks(positions + (num_subcategories - 1) * bar_width / 2)
    ax.set_xticklabels(categories)

    # Add a legend
    ax.legend()

    # Display the plot
    if block:
        plt.show()

def plotDataForWilcoxRanksumsComparisonPlot(statisticsforDimension,optimizers):
    categories=[]
    subcategories=optimizers
    values=[]
    for modelSize,statistics in statisticsforDimension.items():
        categories.append(str(modelSize))
        theseValues=[]
        for optimizer in optimizers:
            theseValues.append(statisticsforDimension[modelSize][optimizer])
        values.append(theseValues)
    return (categories,subcategories,values)

def fillStepsMinValue(steps,til):
    filled=[]
    min=steps[0][1]
    i=0
    # (0,10)
    # (3,11)
    # (5,1)
    for (step,value) in steps:
        if min > value:
            min=value
        for j in range(i,step+1):
            filled.append(min)
        i=step+1
    for j in range(i,til):
        filled.append(min)
    return filled

def createMethodsCostEvolutionPlots(methodPathsAndIds,
                                    experimentProblemsAndScales,
                                    performanceMapping,
                                    experimentFilter=[],filename=None,figuresize=(16/3,3),seriesLimit=100):
    allperformances=[]
    alltitles=[]
    allscales=[]
    for (problem,scale) in experimentProblemsAndScales:
        methodsLogs=[]
        # for method in methods:
        #     logs = open(f"{LOGS_ROOT}/{method}/{problem}")
        #     methodsLogs.append(pd.DataFrame(json.load(logs)['experiments']))

        for method in methodPathsAndIds:
            if not 'customDF' in method[2]:
                logs = open(f"{method[0].replace('/records.json','')}/{problem}")
                df=pd.DataFrame(json.load(logs)['experiments'])
            else:
                df=method[2]['customDF']
                df=method[2]['problemFilter'](df,problem)
            df['experimentId']=method[1]
            methodsLogs.append(df)
        
        allData=pd.concat(methodsLogs)
        if len(experimentFilter)>0:
                    allData=allData[selectAllMatchAtLeastOne(allData,experimentFilter)]
        allData=allData.groupby(['baseLevel-xDim'])
        series=pd.DataFrame()

        optimizers=set()
        optimizersList=[]
        
        for (dimension,groupIndex) in allData:
            serie={}
            trialSizes={}
            serie["dimension"]=dimension
            for index,row in groupIndex.iterrows():
                optimizerName=row["experimentId"]
                serie[optimizerName]=row["trials"]
                trialSizes[optimizerName]=row["trialCount"]
                if optimizerName not in optimizers:
                    optimizersList.append(optimizerName)
                    optimizers.add(optimizerName)
            series=series.append(serie,ignore_index=True)
        for optimizer in optimizersList:
            series[optimizer]=series[optimizer].map(lambda trials: list(map(performanceMapping,trials)))
            series[optimizer]=series[optimizer].map(lambda trials: fillStepsMinValue(list(zip(range(0,len(trials)),trials)),trialSizes[optimizerName]))
        performances=[]
        titles=[]
        for index,row in series.iterrows():
                performances.append([(range(0, len(row[optimizer][0:seriesLimit])),row[optimizer][0:seriesLimit], optimizer) for optimizer in optimizersList ])
                titles.append(f"{problem.replace('.json','').replace('styblinskitang','Styblinski Tang').capitalize()}-{int(row['dimension'])}")
        allperformances.append(performances)
        alltitles.append(titles)
        allscales.append(scale)
    plot_series(allperformances, alltitles, x_label='steps', y_label=f' best fitness ({allscales[0]} scale)',scales=allscales,
                    file_name=filename,figsize=figuresize)
