from sklearn.metrics import f1_score
import numpy as np 

def ComputeClsAcc(target, pred):
    """
    target - target array, (N, cls)
    pred - prediction array (N, cls)
    """
    f1_macro = []
    f1_micro = []
    for cls in range(target.shape[1]):
        target_ = target[:, cls]
        pred_ = pred[:, cls]
        f1_macro.append(f1_score(target_, pred_, average='macro'))
        f1_micro.append(f1_score(target_, pred_, average='micro'))
    f1_macro = np.array(f1_macro)
    f1_micro = np.array(f1_micro)

    return f1_macro, f1_micro