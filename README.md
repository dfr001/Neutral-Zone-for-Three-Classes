# Neutral Zone Classification for Three Classes
R and python scripts to determine the asymmetric neutral zone developed in "[A neutral zone classifier for three classes with an application to text mining](https://doi.org/10.1002/sam.11639)." The neutral zone classifier reduces the number of misclassifications by avoiding hard classification of borderline observations. Instead of labeling using only the three possible classes, a fourth label, $N$, is given to observations whose predicted class is ambiguous because the predicted probabilities for two or more classes are close.

The R script is presented in the context of text classification as used in the aforementioned paper. However, the classifier may be extracted and used in more general situations. The python script defines the necessary functions and presents an example using simulated data and a multinomial logistic regression classifier.

## Python functions documentation
**label_func**(*p0*, *p1*, *L*)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Label observations as 0, 1, 2, or N (neutral) based on predicted probabilities of class 0, class 1, and set of $L$'s.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Parameters**:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;p0 : *array*  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Vector of the predicted probabilities each observation belongs to class 0.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;p1 : *array*  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Vector of the predicted probabilities each observation belongs to class 1.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;L : *list*  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;List with the six values for $L_{01}$, $L_{02}$, $L_{10}$, $L_{12}$, $L_{20}$, and $L_{21}$. Each value must be in [0,1]. Establishes the neutral zone. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Returns**:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;out : *ndarray*  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A character vector of the predicted labels.

<br>

**asymmetric_neutral_zone**(*p0*, *p1*, *true_labels*, *alpha*)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Determine the asymemetric neutral zone based on the desired conditional misclassification rates.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Parameters**:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;p0 : *array*    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Column vector of the predicted probabilities each observation belongs to class 0.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;p1 : *array*  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Column vector of the predicted probabilities each observation belongs to class 1.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;true_labels : *array*  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Vector of the true class labels for each observation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;alpha : *list*  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;List with the six values for the desired conditional misclassification rates $\alpha = [\alpha_{01}, \alpha_{02}, \alpha_{10}, \alpha_{12}, \alpha_{20}, \alpha_{21}]$ corresponding to $P(\hat{C}=0|C = 1)$, $P(\hat{C}=0|C = 2)$, $P(\hat{C}=1|C = 0)$, $P(\hat{C}=1|C = 2)$, $P(\hat{C}=2|C = 0)$, and $P(\hat{C}=2|C = 1)$.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Returns**:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;out : *dictionary*  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A dictionary with two elements: `L` and `conf_table`. `L` is a list of the six *L*'s ($L_{01}$, $L_{02}$, $L_{10}$, $L_{12}$, $L_{20}$, $L_{21}$) needed to achieve the desired conditional misclassification rates. `conf_table` is an array of the conditional misclassification rates. The rows represent the true labels 0, 1, and 2, respectively, and the columns represent the predictied labels 0, 1, 2, and N, respectively.
