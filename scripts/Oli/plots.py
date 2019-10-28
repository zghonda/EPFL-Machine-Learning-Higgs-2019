# -*- coding: utf-8 -*-
"""function for plot."""
import matplotlib.pyplot as plt
import numpy as np
#from grid_search import get_best_parameters

def accuracy_plot(accuracies, degrees, lambdas):
    """gets accuracy matrix with degrees as rows and 
    lambdas as columns and produces a color contour"""
    
    accuracies = np.reshape(accuracies, [len(degrees), len(lambdas)])
    accuracies = np.transpose(accuracies)
    lambdas = np.around(np.log10(lambdas),2)
    
    #Create Plot
    fig, ax = plt.subplots()
    im = ax.imshow(accuracies, cmap='coolwarm', interpolation='gaussian')
    
    #Set axes ticks 
    ax.set_yticks(np.arange(len(lambdas)))
    ax.set_xticks(np.arange(len(degrees)))
    
    #Set axis labels
    ax.set_yticklabels(lambdas)
    ax.set_xticklabels(degrees)
    
    
    for i in range(len(lambdas)):
        for j in range(len(degrees)):
            if (i+1)%3==0 and (j)%2==0:
                text = ax.text(j, i, round(accuracies[i, j],2), ha="center", va="center", color="black")
    
    #Show the plot
    #ax.set_title("Accuracies for Subset")
    ax.set_xlabel('degrees')
    ax.set_ylabel('log(lambdas)')
    fig.tight_layout()
    plt.show()



def prediction(w0, w1, mean_x, std_x):
    """Get the regression line from the model."""
    x = np.arange(1.2, 2, 0.01)
    x_normalized = (x - mean_x) / std_x
    return x, w0 + w1 * x_normalized





    
