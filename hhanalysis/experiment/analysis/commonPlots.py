import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm

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

# This function takes in a list of data series, each series is a tuple of x values, y values, and a label. It also takes in a title, x label and y label for the plot and a file_name where the plot will be saved.
# You can call this function like this:

# data_series = [
#     (range(1, 6), [2, 4, 6, 8, 10], 'Series 1'),
#     (range(2, 7), [3, 6, 9, 12, 15], 'Series 2'),
#     (range(3, 8), [3, 1, 9, 4, -5], 'Series 3')
# ]
# plot_series(data_series, title='My Plot')