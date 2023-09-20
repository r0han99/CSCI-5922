import pandas as pd 
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=1e-3, n_iters=1000):
        # init parameters
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_list = []
        
    def _init_params(self):
        self.weights = np.ones(self.n_features)
        self.bias = 0
    
    def _update_params(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def _compute_loss(self, y, y_pred):
        # Calculate the negative log loss
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def _sigmoid(self, z):

        return 1/(1 + np.exp(-z))

    
    def _get_prediction(self, X):
        return self._sigmoid(np.dot(X, self.weights) + self.bias)
    
    def _get_gradients(self, X, y, y_pred):
        # get distance between y_pred and y_true
        error = y_pred - y
        
        # compute the gradients of weight & bias
        dw = (1 / self.n_samples) * np.dot(X.T, error)
        db = (1 / self.n_samples) * np.sum(error)
        return dw, db
    
    def fit(self, X, y):
        # get number of samples & features
        self.n_samples, self.n_features = X.shape
        # init weights & bias
        self._init_params()

        # perform gradient descent for n iterations
        for _ in range(self.n_iters):
            # get y_prediction
            y_pred = self._get_prediction(X)
            # compute gradients
            dw, db = self._get_gradients(X, y, y_pred)
            # update weights & bias with gradients
            self._update_params(dw, db)

            loss = self._compute_loss(y, y_pred)
            self.loss_list.append(loss)
            if (_ + 1) % 100 == 0: # print for every hundred iterations 
                print(f"Iteration {_ + 1}/{self.n_iters}, Loss: {loss:.4f}")
                

    
    
    def predict(self, X):
        y_pred = self._get_prediction(X)
        return y_pred
    

def thresholding(predictions):

    classified = []

    for pred in predictions:
        if pred < 0.5: 
            classified.append(0)
        else:
            classified.append(1)

    return classified



# X, y = make_classification(n_samples=1000, n_features=4, random_state=101)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

data = pd.read_csv('invented_data.csv')
X = data[['Age','BMI','MinutesOfPhysicalActivity']]
y = data['StrokeRisk']



epochs = 500
logreg = LogisticRegression(learning_rate=0.01, n_iters=epochs)
logreg.fit(X, y)



predictions = logreg.predict(X)
print(thresholding(predictions))

print(classification_report(thresholding(predictions), y))
plt.title('Confusion Matrix')
sns.heatmap(confusion_matrix(thresholding(predictions),y), annot=True, fmt='d', cmap='Greens',xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.show()


plt.title('Loss Reduction Over Epochs')
plt.plot(range(epochs),logreg.loss_list, color='orange')
plt.show()


