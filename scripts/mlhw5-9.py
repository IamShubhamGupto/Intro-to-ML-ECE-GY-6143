from numpy import log as ln
from math import e
import numpy as np

def get_stats(confusion_matrix):
    '''
    Assumption
            predicted
            1       0
    actual
    1       1       0   TP      FN
    0       1       0   FP      TN
    '''
    TP = confusion_matrix[0,0]
    TN = confusion_matrix[1,1]
    FP = confusion_matrix[1,0]
    FN = confusion_matrix[0,1]
    # TP = np.diag(confusion_matrix)
    # TN = np.sum(confusion_matrix) - (FP + FN + TP)
    

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate 1 - TNR
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # False Ommision rate 1- NPV
    FOR = FN/(FN + TN)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    # actual positivity rate
    # slightly off in calculations, unsure why
    APR = (TP+FN)/(TP+FP+FN+TN)

    # predicted positivity rate
    PPR = (TP+FP)/(TP+FP+FN+TN)
    return {'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 'NPV': NPV, 'FPR': FPR, 'FNR': FNR, 'FDR': FDR, 'FOR': FOR, 'ACC': ACC, 'APR': APR, 'PPR': PPR}
a = np.array([[13470, 3270], [69, 329]])
b = np.array([[10049, 3578], [1643, 8692]]) 
c = a+b
# T to deal with opposite convention in hw !!!!!!
stats_list = [get_stats(a.T), get_stats(b.T), get_stats(c.T)]
for smap in stats_list:
    for key, value in smap.items():
        print(f"{key} = {value}")
    print()



