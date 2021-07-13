import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from scripts.visualization import plot_roc_curve
from scripts.visualization import plot_precision_recall_curve
from scripts.visualization import plot_confusion_matrix

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