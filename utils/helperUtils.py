import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# # ROC
from sklearn.metrics import auc

def calculateRMSE_PersentagePresision(y_test, predictionRMSE):
    return 100 - ((100 * predictionRMSE) / np.array(y_test).mean())

def plotROC(classifiers, title):
    """
    Create ROC for classifiers 
    """
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    mean_fpr = np.linspace(0, 1, 100)

    for c in classifiers:
        mean_tpr = np.mean(c["roc_tprs"], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(c["roc_aucs"])
        ax.plot(mean_fpr, mean_tpr,
            label='Mean {} ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(c["name"],mean_auc, std_auc),
            lw=2, alpha=.8)



    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title= title)
    ax.legend(loc="lower right")
    plt.show()