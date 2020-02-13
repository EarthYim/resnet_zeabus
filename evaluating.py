from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import argparse

ap. argparse.ArugmentParser()
ap.add_argument("-i", "--input", required=True, help='path to input csv')
args = vars(ap.parse_args())

def calmat3x3(mat):
    
    tpa = int(mat[0][0])
    tpb = int(mat[1][1])
    tpc = int(mat[2][2])
    
    fpa = sum(mat.T[0]) - tpa
    fpb = sum(mat.T[1]) - tpb
    fpc = sum(mat.T[2]) - tpc
    
    fna = sum(mat[0]) - tpa
    fnb = sum(mat[1]) - tpb
    fnc = sum(mat[2]) - tpc
    
    tna = sum(sum(mat)) - (tpa+fpa+fna)
    tnb = sum(sum(mat)) - (tpb+fpb+fnb)
    tnc = sum(sum(mat)) - (tpc+fpc+fnc)

    return np.array([tpa,tna,fpa,fna]), np.array([tpb,tnb,fpb,fnb]), np.array([tpc,tnc,fpc,fnc])

def caltprfpr(conf):
    tpr = conf[0]/(conf[0]+conf[3]) #tp/tp+fn
    fpr = conf[2]/(conf[2]+conf[1]) #fp/fp+tn 
    return (tpr, fpr)


id = {1:'gate', 0:'flare_yellow', 2:'flare_red'}
groundtruth = open('test.csv').read().split("\n")
predict = open('predicted_crop.csv').read().split("\n")

tpr = []
fpr = []
score = 0.5
for i in range(7):
    y_true = []
    y_pred = []
    count = 0
    print('score', score)
    for i in range(len(groundtruth)-1):
        if groundtruth[i].split(',')[0].split('/')[-1] != predict[count].split(',')[0]:
            continue
    
        if float(predict[count].split(',')[-1]) < score:
            count += 1
            continue

        y_true.append(groundtruth[i].split(',')[5])
        y_pred.append(predict[count].split(',')[5])
        count += 1
    
    mat = confusion_matrix(y_true, y_pred, labels=["flare_yellow", "gate", "flare_red"])
    y, g, r = calmat3x3(mat)
    py, pg, pr = caltprfpr(y), caltprfpr(g), caltprfpr(r)
    tpr.append(py[0])
    tpr.append(pg[0])
    tpr.append(pr[0])
    fpr.append(py[1])
    fpr.append(pg[1])
    fpr.append(pr[1])
    score += 0.05

print(tpr)
print(fpr)



