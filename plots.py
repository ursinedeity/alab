#!/usr/bin/env python

# Copyright (C) 2015 University of Southern California and
#                          Nan Hua
# 
# Authors: Nan Hua
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
__author__ = 'N.H.'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

def histogram(figurename, x, binNum, xlab=None, ylab=None, title=None, line=None, **kwargs):
    """Plot a frequency histogram with a given array x
    
    Parameters
    ----------
    x  : a 1-D vector
        Raw data
    binNum : int
        Number of bins to draw
    xlab : string, optional
        label for x axis
    ylab : string, optional
        label for y axis
    title : string, optional
        title of the figure
    line  : float or array, optional
        draw a vertical line at certain position(s)
    """
  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(x,binNum,**kwargs)
    if xlab != None:
        ax.set_xlabel(xlab)
    if ylab != None:
        ax.set_ylabel(ylab)
    if title != None:
        ax.title(title)
  
    if line != None:
        for l in np.array([line]).flatten():
            ax.axvline(l, color='c', linestyle='dashed', linewidth=2)
    plt.show()
    fig.savefig(figurename,dpi=600)
    plt.close(fig)

def plotxy(figurename,x,y,format='png',xlab=None,ylab=None,title=None,xticklabels=None,yticklabels=None,vline=None, hline=None, **kwargs):
    """xy plot
    Parameters:
    -----------
    x,y: dataset used to plot
    format: str
        format to save figure
    xlab/ylab : string, optional
        label for x/y axis
    title : string, optional
        title of the figure
    vline/hline: float or array, optional
        draw a vertical/horizontal line at certain position(s)
    xticks/yticks: ticks for x,y axis
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y,**kwargs)
    
    if xlab != None:
        ax.set_xlabel(xlab)
    if ylab != None:
        ax.set_ylabel(ylab)
    if title != None:
        ax.title(title)
    if xticklabels != None:
        ax.set_xticklabels(xticklabels)
    if yticklabels != None:
        ax.set_yticklabels(yticklabels)
    
    if vline != None:
        for l in np.array([vline]).flatten():
            ax.axvline(l, color='c', linestyle='dashed', linewidth=1)
    if hline != None:
        for l in np.array([hline]).flatten():
            ax.axhline(l, color='c', linestyle='dashed', linewidth=1)
    plt.show()
    if format == 'png':
        fig.savefig(figurename,dpi=600)
    elif format == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(figurename)
        pp.savefig(fig,dpi=600)
        pp.close()
  
    plt.close(fig)
  
def plotmatrix(figurename,matrix,format='png',title=None,**kwargs):
    """Plot a 2D array with a colorbar.
    
    Parameters
    ----------
    
    matrix : a 2d numpy array
        A 2d array to plot
    cmap : matplotlib color map
        Color map used in matrix, e.g cm.Reds, cm.bwr
    clip_min : float, optional
        The lower clipping value. If an element of a matrix is <clip_min, it is
        plotted as clip_min.
    clip_max : float, optional
        The upper clipping value.
    label : str, optional
        Colorbar label
    ticklabels1 : list, optional
        Custom tick labels for the first dimension of the matrix.
    ticklabels2 : list, optional
        Custom tick labels for the second dimension of the matrix.
    """
    
    clip_min = kwargs.pop('clip_min', -np.inf)
    clip_max = kwargs.pop('clip_max', np.inf)
    cmap     = kwargs.pop('cmap',cm.Reds)
    fig  = plt.figure()
    if 'ticklabels1' in kwargs:
        plt.yticks(range(matrix.shape[0]))
        plt.gca().set_yticklabels(kwargs.pop('ticklabels1'))
        
    if 'ticklabels2' in kwargs:
        plt.xticks(range(matrix.shape[1]))
        plt.gca().set_xticklabels(kwargs.pop('ticklabels2'))
    
    cax  = plt.imshow(np.clip(matrix, a_min=clip_min, a_max=clip_max),
                        interpolation='nearest',
                        cmap=cmap,
                        **kwargs)
    if title != None:
        plt.title(title)
    
    if 'label' not in kwargs:
        plt.colorbar()
    else:
        plt.colorbar().set_label(kwargs['label'])
    
    plt.show()

    if format == 'png':
        fig.savefig(figurename,dpi=600)
    elif format == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(figurename)
        pp.savefig(fig,dpi=600)
        pp.close()
  
    plt.close(fig)