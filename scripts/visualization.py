import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_distributions(feature_list:list, 
                               dataframe:pd.core.frame.DataFrame,
                               target_column:str='next_hour_good_performance',
                               labelsize:int = 20,
                               save_folder:str= None)->None:
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
        sns.distplot(dataframe[dataframe[target_column] == 1][feature], bins=50, label='good')
        sns.distplot(dataframe[dataframe[target_column] == 0][feature], bins=50, label='bad')
        
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