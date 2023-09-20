import numpy as np
# for example application
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def label_func(p0, p1, L):
    p2 = 1-p0-p1
    l01 = L[0]; l02 = L[1]; l10 = L[2]; l12 = L[3]; l20 = L[4]; l21 = L[5]
    out = np.full(len(p0), "N")
    out[np.where(((p0 - p1 > l01) & (p1 > p2)) | ((p0 - p2 > l02) & (p2 > p1)))] = "0"
    out[np.where(((p1 - p0 > l10) & (p0 > p2)) | ((p1 - p2 > l12) & (p2 > p0)))] = "1"
    out[np.where(((p2 - p0 > l20) & (p0 > p1)) | ((p2 - p1 > l21) & (p1 > p0)))] = "2"
    return out

def asymmetric_neutral_zone(p0,p1,true_labels,alpha):        
    L_final = []
    for i1 in [[0,1,2],[1,0,2],[2,0,1]]:
        cond_ind = true_labels != i1[0]
        alpha_2 = alpha[slice(2*i1[0],2*i1[0]+2)]
        L_2_list = []
        err_list = []
        area = []
        first = [999]
        second = [999]
        for i2 in range(101):
            for i3 in range(101):
                b=False
                L = [0 for x in range(6)]
                L_2 = [0,0]; L_2[0] = i2/100; L_2[1] = i3/100
                L[slice(2*i1[0],2*i1[0]+2)] = L_2
                pred_lab = label_func(p0 = p0[cond_ind], p1 = p1[cond_ind], L = L)
                pred_ind = pred_lab==str(i1[0])
                true_ind1 = true_labels[cond_ind]==i1[1]
                true_ind2 = true_labels[cond_ind]==i1[2]
                err1 = sum(pred_ind[true_ind1])/sum(true_ind1)
                err2 = sum(pred_ind[true_ind2])/sum(true_ind2)
                if err1<=alpha_2[0] and err2<=alpha_2[1]:
                    if all(x>L_2[0] for x in first) or all(x>L_2[1] for x in second):
                        first.append(L_2[0])
                        second.append(L_2[1])
                        area.append(L_2[0]/12*(2-L_2[0])/0.5 + L_2[1]/12*(2-L_2[1])/0.5)
                        err_list.append([L,err1,err2])
                    b=True; break
            if b: 
                continue

        obj_res = [(x[1]-alpha_2[0])**2 + (x[2]-alpha_2[1])**2 for x in err_list]
        L_final.append(err_list[obj_res.index(min(obj_res))])

    L_res = [sum(x) for x in zip(L_final[0][0],L_final[1][0],L_final[2][0])]
    
    pred_neut = label_func(p0 = p0, p1 = p1, L = L_res)

    conf_table = np.array((np.empty(4),np.empty(4),np.empty(4)))
    for i in [0,1,2]:
        row_sum = sum(y==i)
        conf_table[i][0] = sum((y==i) & (pred_neut=="0"))/row_sum
        conf_table[i][1] = sum((y==i) & (pred_neut=="1"))/row_sum
        conf_table[i][2] = sum((y==i) & (pred_neut=="2"))/row_sum
        conf_table[i][3] = sum((y==i) & (pred_neut=="N"))/row_sum
    conf_table
    d = dict();
    d['L'] = L_res
    d['conf_table'] = conf_table
    return d


X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
res = model.fit(X, y)
confusion_matrix(y_true=y , y_pred=res.predict(X), normalize='true') # check conditional misclassification rates
pred_probs = res.predict_proba(X)
asymmetric_neutral_zone(p0=pred_probs[:,0], p1=pred_probs[:,1], true_labels=y, alpha=[0.1,0.1,0.1,0.1,0.1,0.1]) # make all conditional misclassifiation rates less than or equal to 0.1
