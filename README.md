# MachineLearning_Algorithms

> This repository contains examples of popular machine learning algorithms implemented in **Python** with mathematics behind them being explained. Each algorithm has interactive **Jupyter Notebook** demo that allows you to play with training data, algorithms configurations and immediately see the results, charts and predictions **right in your browser**. 

The purpose of this repository is _not_ to implement machine learning algorithms by using 3<sup>rd</sup> party library one-liners _but_ rather to practice implementing these algorithms from **scratch** and get better understanding of the mathematics behind each algorithm. 

# Machine Learning

## Supervised Learning

In supervised learning we have a set of training data as an input and a set of labels or **correct answers** for each training set as an output. Then we're training our model (machine learning algorithm parameters) to map the input to the output correctly (to do correct prediction). The ultimate purpose is to find such model parameters that will successfully continue correct _input→output_ mapping (predictions) even for new input examples.

### Regression

In regression problems we do real value predictions. Basically we try to draw a line/plane/n-dimensional plane along the training examples. In regression we deal with continuos as well as decreate data

#### 🤖 Linear Regression

- 📗 [Math | Linear Regression](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Linear%20Regression/Linear%20Regression%20(Detailed%20Explanation%20from%20scratch).ipynb) - theory and links for further readings
- ▶️ [Demo | Univariate Linear Regression](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Linear%20Regression/Linear%20Regression%20(Detailed%20Explanation%20from%20scratch).ipynb) 


#### 🤖 Polynomial Regression

- 📗 [Math | Polynomial Regression](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Polynomial%20Regression/Polynomial%20Regression%20(%20Simple%20Explanation%20).ipynb) - theory and links for further readings
- ▶️ [Demo | Univariate Polynomial Regression](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Polynomial%20Regression/Polynomial%20Regression%20(%20From%20Scratch%20).ipynb) 

#### 🤖 Lasso/Ridge Regression

- ▶️ [Demo | Lasso/Ridge Regression](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Neural%20Network/Linear%20Regression%20with%20Neural%20Network%20(%20ANN%20from%20Scratch%20).ipynb) -Both regressions are explained with Neural Network. But you can apply in any algorithm also.

#### 🤖 Support Vector Machines (SVMs)
SVM construct a hyper plane in high dimension which can be used for Classification , regression , or outlier detection.

- 📗 [Math | SVM ](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Support%20Vector%20Machine/2.%20SVM%20with%20hard%20margin%20(from%20scratch).ipynb) - theory and code
- ▶️ [Demo | SVM Hard Margin](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Support%20Vector%20Machine/2.%20SVM%20with%20hard%20margin%20(from%20scratch).ipynb) - theory and code
- ▶️ [Demo | SVM Soft Margin](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Support%20Vector%20Machine/3.%20SVM%20with%20soft%20margin%20(from%20scratch).ipynb) - theory and code
- ▶️ [Demo | SVM Kernel Trick](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Support%20Vector%20Machine/4.%20Kernel%20Trick.ipynb) - theory and code
- ▶️ [Demo | SVM Sequential Minimal Optimization( SMO ) ](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Support%20Vector%20Machine/5.%20Sequential%20Minimal%20Optimization%20(%20SMO%20).ipynb) - theory and code
- ▶️ [Demo | SVM (Multi-class) ](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Support%20Vector%20Machine/6.%20Multi%20Class%20SVM.ipynb) - theory and code

#### 🤖 K nearest neighbors

KNN (K — Nearest Neighbors) is one of many (supervised learning) algorithms used in data mining and machine learning, it’s a classifier algorithm where the learning is based “how similar” is a data (a vector) from other

- 📗 [Math | KNN ](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/K-Nearest%20Neighbors/k-NN%20simple%20explanation.ipynb) - theory
- ▶️ [Demo | KNN](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/K-Nearest%20Neighbors/k-NN%20wine-quality%20(from%20scratch).ipynb) - code from scratch.

### Ensemble Method

Ensemble learning is a machine learning paradigm where multiple models (often called “weak learners”) are trained to solve the same problem and combined to get better results. The main hypothesis is that when weak models are correctly combined we can obtain more accurate and/or robust models.

- 📗 [Theory | Ensemble Learning ](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/01.%20Ensemble%20Method.ipynb)

- 📗 [Theory | Bootstrap Aggregating](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/02.%20Bootstrap%20Aggregating%20(Bagging).ipynb)

- 📗 [Theory | Random Forest](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/03.%20Random%20Forest%20Theory.ipynb)

- ▶️ [Demo | Random Forest](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/04.%20Random%20Forest%20Implementation.ipynb) - using sklearn

- ▶️ [Demo | Random Forest (hypertune)](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/05.%20Random%20Forest%20(Improve%20Tune%20Param)_1.ipynb) using sklearn

- 📗 [Theory | Boosting](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/06.%20Boosting%20(Theory).ipynb)

- ▶️ [Demo | Adaptative Boosting](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/07.%20Adaptative%20Boosting%20(Adaboost).ipynb) using sklearn

- 📗 [Theory | Gradient Boosting](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/08.%20Gradient%20Boosting%20(Theory).ipynb)

- ▶️ [Demo | Gradient Boosting](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/09.%20Gradient%20Boosting%20(Implementation).ipynb) using sklearn

- 📗 [Theory | Stacking ](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/10.%20Stacking%20(Theory).ipynb)

- ▶️ [Demo | eXtreme Gradient Boosting(XGBoost)](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Ensemble%20Learning/11.%20XGBoost%20(eXtreme%20Gradient%20Boosting).ipynb)


### Classification

In classification problems we split input examples by certain characteristic.

_Usage examples: benign-malignent-data, wine-quality, MNIST handwritten.

#### 🤖 Logistic Regression

- 📗 [Math | Logistic Regression](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Logistic%20Regression/Logistic%20Regression%20%5BDetailed%20Explanation%5D%20.ipynb) - theory and links for further readings
- ▶️ [Demo | Logistic Regression (Linear Boundary)](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Logistic%20Regression/Logistic%20Regression%20%5BDetailed%20Explanation%5D%20.ipynb)
- ▶️ [Demo | Multivariate Logistic Regression | Wine-quality](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Logistic%20Regression/Logistic%20Regression%20Multiclass%20(Wine-Test).ipynb)
- ▶️ [Demo | Multivariate Logistic Regression | Benign-Malignent](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Logistic%20Regression/Logistic%20Regression%20%5B%20Multi_Classes%20%5D.ipynb)


#### 🤖 Naive Bayes

- 📗 [Math | Naive Bayes Classifier](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Naive%20Bayes%20Classifier/Naive%20Bayes%20Classifier%20%5BSimple%20Explanation%20from%20Scratch%5D.ipynb) - theory and links for further readings
- ▶️ [Demo | Bivariate Naive Bayes Classifier | Benign-Malignent](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Naive%20Bayes%20Classifier/Naive%20Bayes%20Classifier%20%5BBenign_Malignent%5D%20from%20Scratch.ipynb)

#### 🤖 Decision Tree

- ▶️ [Demo | Decision Tree | Wine-quality](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Decision%20Tree/Decision%20Tree%20(Wine_quality).ipynb) from scratch

- ▶️ [Demo | Decision Tree | Multi-class](https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Decision%20Tree/Decision%20Tree%20(%20Multiclass%20simple).ipynb)


## Unsupervised Learning

Unsupervised learning is a branch of machine learning that learns from test data that has not been labeled, classified or categorized. Instead of responding to feedback, unsupervised learning identifies commonalities in the data and reacts based on the presence or absence of such commonalities in each new piece of data.

### Dimentional Reduction

In dimentional reduction we select 'K' features from given 'n' features by using some techniques. 

#### 🤖 Principal Component Analysis (PCA)

- 📗 [Math | Principal Component Analysis](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Principal%20Component%20Analysis/Principal%20Component%20Analysis%20(%20Simple%20Example%20with%20Detailed%20explanation).ipynb) - theory and explanation
- ▶️ [Demo | Principal Component Analysis](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Principal%20Component%20Analysis/Principal%20Component%20Analysis%20(%20PCA%20)%20from%20%20scratch.ipynb) - code

## Neural Networks and Deep Learning

The neural network itself isn't an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs.

### 🤖 Artificial Neural Network

- 📗 [Theory | Artificial Neural Network](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Neural%20Network/Neural%20Network%20(Explanation).ipynb) - Theory of ANN
- 📗 [Math | Regression with Artificial Neural Network](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Neural%20Network/Linear%20Regression%20with%20Neural%20Network%20(%20ANN%20from%20Scratch%20).ipynb) - theory and code
- ▶️ [Demo | Bivariate Classification with Artificial Neural Network](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Neural%20Network/BiClass%20Classification%20with%20Neural%20Network%20from%20scratch.ipynb)
- ▶️ [Demo | Multivariate Classification with Artificial Neural Network using Sigmoid function](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Neural%20Network/MultiClass%20Classification%20Neural%20Network_1(%20with%20sigmoid%20).ipynb)
- ▶️ [Demo | Multivariate Classification with Artificial Neural Network using ReLu function](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Neural%20Network/MultiClass%20Classification%20Neural%20Network_2(%20with%20ReLu%20).ipynb)
- ▶️ [Demo | Multivariate Classification with Artificial Neural Network ( DevnagariHandwrittenDataSet )](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Neural%20Network/ANN%20(Devnagri%20from%20scratch).ipynb)- complete code from scratch ( Accuracy 73 % )
- ▶️ [Demo | Multivariate Classification with Artificial Neural Network ( MNIST )](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Neural%20Network/ANN%20(MNIST%20from%20scratch).ipynb)- complete code from scratch (Accuracy 92 %)
- ▶️ [Demo | Bunch of Activations](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Neural%20Network/Bunch%20of%20Activation%20Functions.ipynb)

### 🤖 Perceptron

Perceptron is similar to SVM it also construct a hyper plane in high dimension if data is linearly seperable.

- 📗 [Theory | Perceptron](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Perceptron/Perceptron%20Theory%20Explanation.ipynb) - Theory of perceptron
- ▶️ [Demo | Perceptron](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Perceptron/Perceptron%20Code%20from%20scratch.ipynb) - code



## Optimization Algorithms

### Gradient Decent

- 📗 [Math | Gradient Decent](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Gradient%20Decent/Gradient%20Decent%20%5BSimple%20Explanation%5D.ipynb)
- ▶️ [Demo | Multivariate Gradient decent](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Gradient%20Decent/Gradient%20Decent%20for%20MultiClass.ipynb)

### Newton's Raphson Method

- 📗 [Math | Newton's Raphson Method](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Newton's%20Raphson%20Method/Newton's%20Raphson%20Method.ipynb) - Theory
- ▶️ [Demo | Newton's Raphson Method](https://github.com/Girrajjangid/MachineLearning_Algorithms/blob/master/Newton's%20Raphson%20Method/Logistic%20Regression%20with%20%5BNewton's%20Method%20AUC%2CROC%20%20%20from%20%20SCRATCH%5D.ipynb) - Implementation with **ROC and AUC Curve**

## Prerequisites

#### Installing Python

Make sure that you have [Python installed](https://realpython.com/installing-python/) on your machine.

You might want to use [venv](https://docs.python.org/3/library/venv.html) standard Python library
to create virtual environments and have Python, `pip` and all dependent packages to be installed and 
served from the local project directory to avoid messing with system wide packages and their 
versions.

#### Installing Dependencies

Install all dependencies that are required for the project by running:

## Datasets

The list of datasets that is being used for Jupyter Notebook demos may be found in [DataSet Folder](DataSets).
