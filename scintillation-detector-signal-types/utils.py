import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

from tqdm import tqdm


def exp_func(x, a, c, d):
    return a*np.exp(-c*x)+d

def exp_fit(signals):
    print('Fitting is starting')
    coeff = []
    error = []
    for num, signal in tqdm(enumerate(signals)):
        x = [i for i in range(1, len(signal) + 1)]
        y = signal
        try:
            popt, pcov = curve_fit(exp_func, x, y, p0=(1, 1e-6, 1))
            coeff.append(popt)
        except RuntimeError:
            error.append(num)
    print('Fitting has ended')
    return coeff, error


def invert(signals):
    return [(np.amax(signal)-np.array(signal) + 2) for signal in signals]


def get_signals(file, step):
    signals_ = []
    print('Signal extraction is starting')
    for row in tqdm(file.itertuples()): 
        values = row[1:]
        index_of_min = np.argmin(values)
        median_level = np.median(values[300:400])
        gap = median_level - values[index_of_min]

        start = 0
        while (values[index_of_min] + step*gap >= values[index_of_min + start]):
            start += 1

        fin = 0
        three_sigma_level = median_level - 3 * np.std(values[300:400], ddof=0)
        three_sigma_rule = True
        try:
            while three_sigma_rule:
                fin += 1
                three_sigma_rule = values[index_of_min + start + fin] <= three_sigma_level
        except KeyError: 
            signals_.append(list(values[index_of_min + start:]))
        else:
            signals_.append(list(values[index_of_min + start:index_of_min + start + fin]))
    print('Signal extraction has ended')

    return signals_


def diff_line_height(coeff, step):
    plt.figure(figsize=(10, 10))
    X = np.array([coeff[i][1] for i in range(len(coeff)) if coeff[i][1]<0.6])
    y_histogram, x_histogram, _ = plt.hist(X, bins=500);
    
    X_plot = np.linspace(0, 0.6, 1000)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.004).fit(X.reshape(-1, 1))
    log_dens = kde.score_samples(X_plot)
    dens_estim = np.exp(log_dens) * max(y_histogram) / max(np.exp(log_dens))
    
    peaks = argrelextrema(dens_estim, np.greater)
    peaks = peaks[0]
    peak_1 = peaks[np.argmax(dens_estim[peaks])]
    peaks = [x for i,x in enumerate(peaks) if i!=np.argmax(dens_estim[(peaks)])]
    peak_2 = peaks[np.argmax(dens_estim[(peaks)])]

    diff = argrelextrema(dens_estim[min(peak_1, peak_2):max(peak_1, peak_2)], np.less)[0][0] + min(peak_1, peak_2)
    
    plt.plot(X_plot[[peak_1, peak_2], 0], dens_estim[[peak_1, peak_2]], 'go', label='modal peak')
    plt.plot(X_plot[[diff], 0], dens_estim[[diff]], 'ro', label='split point: ' + str(dens_estim[diff]))
    plt.plot(X_plot[:, 0], dens_estim, label='gauss kernel density estimation')
    plt.plot([0, 0.6], [50, 50])
    plt.legend(loc='best')
    plt.xlabel('coeff value')
    plt.ylabel('N')
    plt.title('tau histogram for step = ' + str(step))
    
    return dens_estim[diff], x_histogram[int(np.round((x_histogram.shape[0] / dens_estim.shape[0]) * diff))]


def plot_principal_components(data, model, scatter=True, legend=True):
    W_pca = model.components_
    if scatter:
        plt.scatter(data[:,0], data[:,1], .8)
    #plt.plot(data[:,0], -(W_pca[0,0]/W_pca[0,1])*data[:,0], color="c")
    plt.plot(data[:,0], -(W_pca[1,0]/W_pca[1,1])*data[:,0], color="c")
    if legend:
        c_patch = mpatches.Patch(color='c', label='Principal components')
        plt.legend(handles=[c_patch], loc='lower right')
        

        

def get_signal(row):
    
    extrem = np.argmax(np.array(row))
    level = np.mean(row[:50])
    std = np.std(row[:50])
    
    fl = row[extrem:] < level + 3 * std
    try:
        ind = fl[fl==True].index[0] #  индекс первого элемента, подходящего под условия
    except:
        return False
    
    return row[extrem:ind]   

def get_PSD(signals, short_len, step_from_max):
    PSD = [] 
    for signal in signals:
        short = signal[step_from_max: step_from_max + short_len]
        long = signal[step_from_max:]
        
        short = sum([i for i in short if i>=0])
        long = sum([i for i in long if i>=0])
        
        PSD.append((long - short)/long)
    return PSD

def inv_x(signals):
    return [signals_item * (-1) + max(signals_item) for signals_item in tqdm(signals)]

def normalize(signals):
    return [signal / sum(signal) for signal in signals]