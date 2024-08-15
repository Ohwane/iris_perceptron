import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#perceptron class
class perceptron:
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate=learning_rate
        self.n_iters=n_iters
        self.activation_func = lambda x: self.unit_step_func(x)  # Use lambda
        self.weights=None
        self.bias=None
    
    def fit(self, X,y):
        n_samples, n_features= X.shape

        # initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output= np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                 # Assign learning rate here:
                self.learning_rate = 0.01  
                update = self.learning_rate * (y_[idx]-y_predicted)
                self.weights += update * x_i
                self.bias+= update

    #predict function
    def predict(self,X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted=self.activation_func(linear_output)
        return y_predicted

    def unit_step_func(self,x):
        return np.where(x>=0, 1,0)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

if __name__ == "__main__":
    iris = datasets.load_iris()

    # Select features (Sepal length and Petal length)
    X = iris.data[:100, [0, 2]]
    y = iris.target[:100]

    # Create binary labels (Iris Setosa vs. Iris Versicolor)
    y_binary = np.where(y ==0, 1, 0)  # Setosa=0, Versicolor = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=123
    )

    ppt = perceptron(learning_rate=0.01, n_iters=1000)
    ppt.fit(X_train, y_train)
    predictions = ppt.predict(X_test)

    print("Perceptron classification accuracy:", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) 

    # Separate scatter plots for Setosa and Versicolor
    setosa_indices = np.where(y_train == 1)[0]
    versicolor_indices = np.where(y_train == 0)[0]
    jitter = np.random.uniform(-0.1, 0.1, size=len(X_train)) 
    plt.scatter(X_train[setosa_indices, 0], X_train[setosa_indices, 1] , marker="o", c='blue', s=20, label='Setosa', alpha=0.7)
    plt.scatter(X_train[versicolor_indices, 0], X_train[versicolor_indices, 1], marker="x", c='red', s=20, label='Versicolor', alpha=0.7)
    
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])
    if np.abs(ppt.weights[1]) < 1e-6:
            print("Warning: weights[1] is close to zero, skipping decision boundary plot.")
    else:
            x1_1 = (-ppt.weights[0] * x0_1 - ppt.bias) / ppt.weights[1]
            x1_2 = (-ppt.weights[0] * x0_2 - ppt.bias) / ppt.weights[1]
            ax.plot([x0_1, x0_2], [x1_1, x1_2], "k") 

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 1, ymax + 1])
    print(len(y))
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend()
    plt.show()