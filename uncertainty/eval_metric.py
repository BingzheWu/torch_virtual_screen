import numpy as np


def tpr(results, threshold=0.5):
    y_prob = results['predictions'][:,1]
    y_true = results['targets']
    y_pred = np.where(y_prob>=threshold, 1, 0)
    print((y_pred==y_true).mean())
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    print("TPR at threshold is {}".format((tp/(tp+fn))))
    print("FPR at threshold is {}".format((fp/(fp+tn))))

if __name__ == '__main__':
    result_file = "logs/HIV_MPNN/HIV_HIV_activetest.npz"
    results = np.load(result_file)
    tpr(results)


