import matplotlib.pyplot as plt

def plot_series(data_series, x_label='X-axis', y_label='Y-axis', title='Plot', file_name=None,scale='linear'):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_yscale(scale)
    # Plot each series in the data_series list
    for i, series in enumerate(data_series):
        x, y, label = series
        ax.plot(x, y, label=label)
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add a legend
    ax.legend()
    
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