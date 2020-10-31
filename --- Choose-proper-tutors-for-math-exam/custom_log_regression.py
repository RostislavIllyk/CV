import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import euclidean



def sigmoid(x: float) -> float:
    """Сигмоида.
    """   
    return 1 / (1 + np.exp(-x))


def partial_derivative(j: int, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Частная производная функционала Q по переменной b_j.
    """   
    return -sum(x[i, j] * y[i] * (1 - sigmoid(b.dot(x[i]) * y[i])) for i in range(x.shape[0]))


def gradient(b: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Вектор градиента.
    """  
    return np.array([  partial_derivative(j, b, x, y) for j in range(b.shape[0]) ])


def gradient_descent_step(
        lambda_: float,
        b: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,) -> np.ndarray:
    """Один шаг градиентного спуска.
    """  
    return b - lambda_ * gradient(b, x, y)


def curent_roc_auc(b,x,y):
    b = b.reshape(-1, 1)
    r=sigmoid(x.dot(b))
    c_rocauc = roc_auc_score(y, r)
    return c_rocauc


class toy_log_reg_model():
    
    def fit(self, X_train,  y_train, X_test=None, y_test=None, batch_size = 15):      
        
        train_hist=[]
        test_hist =[]
        
        ones = np.ones((X_train.shape[0], 1))
        x = np.hstack([ones, X_train])
        y = y_train.values
    
        if X_test is not None:
            ones = np.ones((X_test.shape[0], 1))
            x_val = np.hstack([ones, X_test])
            if y_test is not None:
                y_val = y_test.values
    
    
        self.b_0 = np.zeros(x.shape[1])
        self.b = self.b_0
        crit_level = 3500            
        lambda_ = 0.03 

        
        for i in trange(1, 7000):
            idx=np.random.randint(0,len(x), batch_size)
            batch_x = x[idx]
            batch_y = y[idx]    
        
            b_new = gradient_descent_step(lambda_, self.b, batch_x, batch_y)
            c_roc_x_train=curent_roc_auc(b_new,x,y)
            
            if y_test is not None:
                c_roc_x_val  =curent_roc_auc(b_new,x_val,y_val)            
            else:
                c_roc_x_val = 0.
                
            train_hist.append(c_roc_x_train)
            test_hist.append(c_roc_x_val)
            
            if i > crit_level:
                lambda_ = lambda_/10
                crit_level = 7000
                print(f'lambda: {lambda_} on step {i}\n')     
                print(euclidean(self.b, b_new), c_roc_x_train, c_roc_x_val)                         
            self.b = b_new
                  
        self.b = b_new.reshape(-1, 1)
        print(self.b.shape, x.shape)
        
        r=sigmoid(x.dot(self.b))
        train_roc_auc_score=roc_auc_score(y, r)
        print('train_roc_auc_score:',train_roc_auc_score)

        if X_test is not None:
            c_roc_x_val  =curent_roc_auc(b_new,x_val,y_val)            
            print('test_roc_auc_score:',c_roc_x_val)
        plt.plot(train_hist, color='r')
        plt.plot(test_hist, color='g')
        plt.grid(True)
        return
    
    
    
    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])
        y_pred=sigmoid(X.dot(self.b))
        return y_pred
    
    
    
    
    