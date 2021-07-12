import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from scripts.utils import create_directory
from scripts.model import get_prediction

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

def precision_recall_curve_comparison(temp_model_dict:dict,
                                       X_test:list,
                                       y_test:list,
                                       save_path:str = None)->None:
    """
    Plots PR-AUC curve of multiple models for comparison.
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

def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def get_cut_off_threshold_from_precision_recall(precision:list, 
                                                recall:list,
                                                thresholds:list)->int:
    try:
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        print('PR-curve threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        return ix
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def get_cut_off_threshold_through_iteration(pos_probs:list, y_test:list)->float:
    """
    Extracts cut off thresholds by itrating all possible values up to 3 decimal places
    from 0.0001-1. Returns the value maximizes macro f1 score.
    
    """
    try:
        # define thresholds
        thresholds = np.arange(0, 1, 0.0001)
        # evaluate each threshold
        scores = [f1_score(y_test, to_labels(pos_probs, t), average='macro') for t in thresholds]


        # get best threshold
        ix = np.argmax(scores)
        print('Threshold=%.3f, Best macro F1-Score=%.5f' % (thresholds[ix], scores[ix]))
        
        return thresholds[ix]
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def plot_roc_curve(auc:int, fpr:list, tpr:list, save_path:str)->None:
 
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
                                best_threshold:tuple,
                                no_skill:int=0.1,
                                save_path:str=None)->None:

    try:  
        fig, ax = plt.subplots(figsize=(6,6))  
        ax.set_title('Precision-Recall Graph')
        plt.plot(recall, precision, label='Model - AUC = %0.3f'% auc)
        plt.plot([0,1], [no_skill,no_skill],label='No skill')
        ax.scatter(best_threshold[0], best_threshold[1], marker='o', color='black', label='Best')

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

def get_evaluation_report(test_set:list,
                          prediction_proba:list,
                          labels:list,
                          threshold:float = None,
                          plot:str='precision-recall',
                          save_path:str = None)->dict:
    """
    Args:
        test_set:list         -> original target values
        prediction_proba:list -> extension to use for serializing
        labels:list           -> target label names
        threshold:float       -> Probability cut off threshold
        plot:str              -> roc or precision-recall
        save_path:str         -> save directory
    """
    try:
        auc_score = 0

        if plot=='roc':
            fpr, tpr, _ = roc_curve(test_set, prediction_proba)
            auc_score = roc_auc_score(test_set, prediction_proba)
            plot_roc_curve(auc_score, fpr, tpr)

        elif plot=='precision-recall':

            precision, recall, thresholds = precision_recall_curve(test_set, prediction_proba)
            auc_score = auc(recall, precision)
            no_skill = np.sum(test_set==1)/test_set.shape
            ix = get_cut_off_threshold_from_precision_recall(precision, recall, thresholds)
            best_threshold_pos = (recall[ix], precision[ix])

            plot_precision_recall_curve(auc_score,
                                        recall, 
                                        precision, 
                                        best_threshold_pos, 
                                        round(no_skill[0], 2),
                                        save_path)
            #threshold = round(thresholds[ix], 3) if not threshold else None

        if not threshold:
            threshold = get_cut_off_threshold_through_iteration(prediction_proba, test_set)
        
        predictions = prediction_proba>threshold
        
        cr = classification_report(test_set, predictions, target_names=labels)
        cm = confusion_matrix(test_set,  predictions)
        mcc = matthews_corrcoef(test_set,  predictions)
        
        print('\n',cr)
        print('Matthews correlation coefficient: ', mcc)

        plot_confusion_matrix(cm,
                              labels, 
                              save_path=save_path)
        
        return {'threshold':threshold,
                'auc':auc_score, 
                'mcc':mcc, 
                'confusion_matrix': cm, 
                'classification_report':classification_report(test_set, 
                                                              predictions, 
                                                              target_names=labels,
                                                              output_dict=True)}
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))