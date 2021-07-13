import pandas as pd
import numpy as np
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scripts.model import get_prediction
from scripts.utils import create_directory

def plot_feature_distributions(feature_list:list, 
                               dataframe:pd.core.frame.DataFrame,
                               target_column:str='next_hour_good_performance',
                               labelsize:int = 20,
                               save_folder:str= None,
                               positive_label:str='good',
                               negative_label:str='bad')->None:
    """
    Given feature lists, dataframe and target boolean column this method prints distribution
    of the features assoiated with the target column.
    """

    fig = plt.figure(figsize = (20, 25))
    j = 0
    sns.set_context("paper", rc={"axes.labelsize": labelsize})
    for feature in feature_list:
        if feature == target_column:
            continue
        plt.subplot(6, 4, j+1)
        sns.distplot(dataframe[dataframe[target_column] == 1][feature], bins=50, label=positive_label)
        sns.distplot(dataframe[dataframe[target_column] == 0][feature], bins=50, label=negative_label)
        
        plt.legend(loc='best')
        j += 1
    fig.suptitle('Selected features distribution plot')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    if save_folder:
        create_directory(f"{save_folder}{os.sep}{feature}")
        plt.savefig( f"{save_folder}{os.sep}"+'selected_feature_distributions.jpg', dpi=300, bbox_inches = "tight")
    plt.show()

def without_hue(plot_object:object, feature:list) -> None:
    """
    Given a plot_object:matplotlib.axes._subplots.AxesSubplot 
    containing bar this methods prints percentage of bars w.rt
    th whole sample on top of each bar.

    """

    total = len(feature)
    for p in plot_object.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        plot_object.annotate(percentage, (x, y), size = 12)


def roc_curve_comparison( temp_model_dict:dict,
                           X_test:list,
                           y_test:list,
                           save_path:str = None)->None:
    """
    Plots roc curve of multiple models for comparison.
    Args:
        temp_model_dict:dict -> contains model name as key and model object as value
    
    """
    try:    
        for key, value in temp_model_dict.items():
            prediction = get_prediction(value, X_test)
            auc_score = roc_auc_score(y_test, prediction)
            fpr, tpr, thresholds = roc_curve(y_test, prediction)
            
            plt.plot(fpr, tpr, label='{} (auc = {})'.format(key, round(auc_score,4)) )


        plt.plot([0,1], [0,1],label='Base Rate' 'k--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Graph')
        plt.legend(loc="best")
        
        if save_path:
            create_directory(save_path)
            plt.savefig( f"{save_path}{os.sep}"+'roc_curve_comparison.png', dpi=300, bbox_inches = "tight")

        plt.show()
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def precision_recall_curve_comparison( temp_model_dict:dict,
                                       X_test:list,
                                       y_test:list,
                                       save_path:str = None)->None:
    """
    Plots PR-AUC curve to compare multiple models.
    Args:
        temp_model_dict:dict -> contains model name as key and model object as value
    
    """
    try:    
        for key, value in temp_model_dict.items():
            prediction = get_prediction(value, X_test)
            precision, recall, _ = precision_recall_curve(y_test, prediction)
            auc_score = auc(recall, precision)
            plt.plot(recall, precision, label='{} (auc = {})'.format(key, round(auc_score,4)) )


        no_skill = np.sum(y_test==1)/y_test.shape
        plt.plot([0,1], [no_skill,no_skill],label='No Skill')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Graph')
        plt.legend(loc="best")
        
        if save_path:
            create_directory(save_path)
            plt.savefig( f"{save_path}{os.sep}"+'precision_recall_curve_comparison.png', dpi=300, bbox_inches = "tight")

        plt.show()
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          save_path:str = None,
                          cmap=plt.cm.Blues)->None:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    """
    try:
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save_path:
            create_directory(save_path)
            plt.savefig( f"{save_path}{os.sep}"+'confusion_matrix.png', dpi=300, bbox_inches = "tight")

        plt.show()
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def plot_roc_curve(auc:int, fpr:list, tpr:list, save_path:str)->None:
    """
    This function plots the roc curve with given parameters.

    Args:
        auc:int
        fpr:list        -> false positive rate
        tpr:list        -> true positive rate
        save_path:str   -> save directory
    """ 
    try:   
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b',label='Model - AUC = %0.3f'% auc)
        ax.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--', label='Chance')
        ax.legend()
        ax.set_xlim([-0.1,1.0])
        ax.set_ylim([-0.1,1.01])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        if save_path:
            create_directory(save_path)
            plt.savefig( f"{save_path}{os.sep}"+'roc_curve.png', dpi=300, bbox_inches = "tight")
        plt.show()
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def plot_precision_recall_curve(auc:int, 
                                recall:list, 
                                precision:list,
                                best_threshold_pos:tuple=None,
                                no_skill:int=0.1,
                                save_path:str=None)->None:
    """
    This function plots the precision-recall curve with given parameters.

    Args:
        auc:int                   
        recall:list               
        precision:list            
        no_skill:int              -> y axis value to plot no skill  value
        best_threshold_pos:tuple  -> true positive rate
        save_path:str             -> save directory
    """ 
    try:  
        fig, ax = plt.subplots(figsize=(6,6))  
        ax.set_title('Precision-Recall Graph')
        plt.plot(recall, precision, label='Model - AUC = %0.3f'% auc)
        plt.plot([0,1], [no_skill,no_skill],label='No skill')

        if best_threshold_pos:
            ax.scatter(best_threshold_pos[0], best_threshold_pos[1], marker='o', color='black', label='Best')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc="best")

        if save_path:
            create_directory(save_path)
            plt.savefig( f"{save_path}{os.sep}"+'precision_recall.png', dpi=300, bbox_inches = "tight")

        plt.show()
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))