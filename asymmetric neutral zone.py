import numpy as np
# for example application
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def label_func(p0, p1, L):
    p2 = 1-p0-p1
    l01 = L[0]; l02 = L[1]; l10 = L[2]; l12 = L[3]; l20 = L[4]; l21 = L[5]
    out = np.full(len(p0), "N")
    out[np.where(((p0 - p1 > l01) & (p1 > p2)) | ((p0 - p2 > l02) & (p2 > p1)))] = "0"
    out[np.where(((p1 - p0 > l10) & (p0 > p2)) | ((p1 - p2 > l12) & (p2 > p0)))] = "1"
    out[np.where(((p2 - p0 > l20) & (p0 > p1)) | ((p2 - p1 > l21) & (p1 > p0)))] = "2"
    return out

def asymmetric_neutral_zone(p0,p1,true_labels,alpha):
    all_comb = np.array(np.meshgrid(range(101),range(101))).reshape(2,101*101).T/100
    # 01, 02, 10, 12, 20, 21
    res = []
    for i in [[0,1,2],[1,0,2],[2,0,1]]:
        def cond_err_func(L_2, true_lab_vec, index):
            L = [0 for x in range(6)]
            L[slice(2*index[0],2*index[0]+2)] = L_2
            cond_ind = true_lab_vec != index[0]
            pred_lab = label_func(p0 = p0[cond_ind], p1 = p1[cond_ind], L = L)
            cond_err_probs = []
            cond_err_probs.append(L_2[0]);cond_err_probs.append(L_2[1])
            pred_ind = pred_lab==str(index[0])
            true_ind1 = true_lab_vec[cond_ind]==index[1]
            true_ind2 = true_lab_vec[cond_ind]==index[2]
            cond_err_probs.append(sum(pred_ind[true_ind1])/sum(true_ind1))
            cond_err_probs.append(sum(pred_ind[true_ind2])/sum(true_ind2))
            return cond_err_probs
        res.append(np.apply_along_axis(cond_err_func, 1, all_comb, true_lab_vec=true_labels, index=i))
    
    def obj_fun(qwer,a):
        return (qwer[2]-a[0])**2 + (qwer[3]-a[1])**2
    L_res = []
    for i in [0,1,2]:
        obj_vecs = np.apply_along_axis(obj_fun,1,res[i],a=alpha[slice(2*i,2*i+2)])
        L_pair = res[i][np.argmin(obj_vecs),[0,1]]
        L_res.append(L_pair[0]); L_res.append(L_pair[1])
    
    pred_neut = label_func(p0 = pred_probs[:,0], p1 = pred_probs[:,1], L = L_res)
    conf_table = np.array((np.empty(4),np.empty(4),np.empty(4)))
    for i in [0,1,2]:
        row_sum = sum(y==i)
        conf_table[i][0] = sum((y==i) & (pred_neut=="0"))/row_sum
        conf_table[i][1] = sum((y==i) & (pred_neut=="1"))/row_sum
        conf_table[i][2] = sum((y==i) & (pred_neut=="2"))/row_sum
        conf_table[i][3] = sum((y==i) & (pred_neut=="N"))/row_sum
    d = dict();
    d['L'] = L_res
    d['conf_table'] = conf_table
    return d


X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# model.fit(x, ylab)
res = model.fit(X, y)
pred_probs = res.predict_proba(X)
asymmetric_neutral_zone(p0=pred_probs[:,0], p1=pred_probs[:,1], true_labels=y, alpha=[0.05,0.05,0.1,0.1,0.15,0.15])
