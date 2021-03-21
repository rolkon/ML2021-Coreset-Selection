import numpy as np
from os import path
import logging
logging.getLogger().setLevel(logging.CRITICAL)

import matplotlib

matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': 20,
    'text.usetex': True,
    'pgf.rcfonts': False,
})


netnames = ['Resnet', 'Mobilenet', 'VGG', 'Densenet']
percentage_strings = ['010', '030', '050']
selectors = ['Glister', 'K-Centers', 'Random']
percentages = [10, 30, 50, 100]

from matplotlib import pyplot as plt

def __get_data():
    data_path = 'data_2021_03_21/'

    data = {}

    data['epochs'] = np.arange(5, 105, 5)

    for netname in netnames:

        frac_dict = {}
        
        filename = '100_{}'.format(netname.lower())
        frac_dict[100] = np.loadtxt(data_path+filename+'_subset.csv')
        
        for percentage_string in percentage_strings:
            frac = int(percentage_string)
            
            sel_dict = {}
            for selector in selectors:
                filename = '{}_{}_{}'.format(percentage_string, netname.lower(), selector.lower())

                sel_dict[selector.lower()] = np.loadtxt(data_path+filename+'_subset.csv')
                
            frac_dict[frac] = sel_dict
        
        data[netname.lower()] = frac_dict

    return data

def plot_generalization_ability():
    data = __get_data()

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, hspace=0.25)

    axes = gs.subplots(sharex=True, sharey=True)

    #selectors = np.array(selectors)

    for i, selector in enumerate(selectors):
        resnet = []
        mobilenet = []
        densenet = []
        vgg = []
        
        for percentage in percentages[:3]:

            resnet.append(data['resnet'][percentage][selector.lower()][-1])
            mobilenet.append(data['mobilenet'][percentage][selector.lower()][-1])
            densenet.append(data['densenet'][percentage][selector.lower()][-1])
            vgg.append(data['vgg'][percentage][selector.lower()][-1])
            
        # add entry of full dataset to list
        resnet.append(data['resnet'][100][-1])
        mobilenet.append(data['mobilenet'][100][-1])
        densenet.append(data['densenet'][100][-1])
        vgg.append(data['vgg'][100][-1])

        axes[i].plot(percentages, resnet, label='Resnet', marker='o')
        axes[i].plot(percentages, mobilenet, label='Mobilenet', marker='^')
        axes[i].plot(percentages, densenet, label='Densenet', marker='*')
        axes[i].plot(percentages, vgg, label='VGG', marker = 'x')
        axes[i].set_title(selector, fontsize=16)
        axes[i].set_xlabel('Percentage of Dataset', fontsize=14)
        axes[i].set_ylabel('Accuracy', fontsize=14)
        axes[i].set_autoscaley_on(False)
        axes[i].set_ylim([0.3, 0.9])

        axes[i].label_outer()
        axes[i].legend()
        axes[i].grid()

        axes[i].label_outer()

    plt.savefig('generalization_ability.eps', bbox_inches= 'tight')
    plt.savefig('generalization_ability.png', dpi=100, bbox_inches='tight')


def plot_accuracy_comparison():
    data = __get_data()

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, hspace=0.25)

    axes = gs.subplots(sharex=True, sharey=True)

    names = np.array(netnames)

    for i, name in enumerate(names):
        glister = []
        kcenter = []
        random  = []

        for percentage in percentages[:3]:

            glister.append(data[name.lower()][percentage]['glister'][-1])
            kcenter.append(data[name.lower()][percentage]['k-centers'][-1])
            random.append(data[name.lower()][percentage]['random'][-1])

        # add entry of full dataset to list
        acc_full_data = data[name.lower()][100][-1]
        glister.append(acc_full_data)
        kcenter.append(acc_full_data)
        random.append(acc_full_data)

        axes[i].plot(percentages, glister, label='GLISTER', marker='o')
        axes[i].plot(percentages, kcenter, label='K-Center', marker='^')
        axes[i].plot(percentages, random, label='Random', marker='*')
        axes[i].set_title(name, fontsize=16)
        axes[i].set_xlabel('Percentage of Dataset', fontsize=14)
        axes[i].set_ylabel('Accuracy', fontsize=14)
        axes[i].set_autoscaley_on(False)
        axes[i].set_ylim([0.3, 0.9])

        axes[i].label_outer()
        axes[i].legend()
        axes[i].grid()

        axes[i].label_outer()

    plt.savefig('generalization_accuracy_comparison.eps', bbox_inches= 'tight')
    plt.savefig('generalization_accuracy_comparison.png', dpi=100, bbox_inches='tight')


def plot_accuracies_per_selector(selector):
    data = __get_data()
    
    percentages = [10, 30, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    fig.tight_layout()

    names = np.reshape(netnames, (2, 2))
    
    for row in range(2):
        for col in range(2):
            axes[row][col].set_title(names[row][col])
            axes[row][col].set_xlabel('Epochs')
            axes[row][col].set_ylabel('Accuracy')
            axes[row][col].set_autoscaley_on(False)
            axes[row][col].set_ylim([0.1, 0.9])

            axes[row][col].plot(data['epochs'], data[names[row][col].lower()][100], label="100\%")

            for percentage in percentages:
                axes[row][col].plot(data['epochs'], data[names[row][col].lower()][percentage][selector.lower()], label=str(percentage) + '\%')

            axes[row][col].label_outer()
            axes[row][col].legend()
            axes[row][col].grid()
            
    plt.savefig('epochs_{}'.format(selector.lower())+'.eps')
    plt.savefig('epochs_{}'.format(selector.lower()) + '.png', dpi=100, bbox_inches='tight')


def plot_accuracies_per_percentage(percentage):
    data = __get_data()

    models = ['Resnet', 'Mobilenet', 'VGG', 'Densenet']
    percentage_strings = ['010', '030', '050']
    selectors = ['Glister', 'K-Centers', 'Random']
    percentages = [10, 30, 50, 100]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    fig.tight_layout()
    
    names = np.reshape(netnames, (2, 2))
    
    for row in range(2):
        for col in range(2):
            axes[row][col].set_title(names[row][col])
            axes[row][col].set_xlabel('Epochs')
            axes[row][col].set_ylabel('Accuracy')
            axes[row][col].set_ylim([0.05, 0.9])
            
            axes[row][col].plot(data['epochs'], data[names[row][col].lower()][100], label="Full Set")
            
            for selector in selectors:
                axes[row][col].plot(data['epochs'], data[names[row][col].lower()][percentage][selector.lower()], label=selector)

            axes[row][col].label_outer()
            axes[row][col].legend()
            axes[row][col].grid()
            
    plt.savefig('epochs_{}'.format('{:03}'.format(percentage))+'.eps')
    plt.savefig('epochs_{}'.format('{:03}'.format(percentage)) + '.png', dpi=100, bbox_inches='tight')