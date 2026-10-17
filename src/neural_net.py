"""Two-layer regularised back-propagation neural network, written in NumPy.

    z1 = X W1 + b1 ; a = sigmoid(z1) ; z2 = a W2 + b2 ; y_hat = softmax(z2)
    J  = -(1/B) sum_i sum_k y_k log(y_hat_k) + lam (||W1||^2 + ||W2||^2)

Trained with mini-batch gradient descent. sklearn-compatible (clone / pipelines).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=None, lr=2.0, lam=1e-4, batch_size=1000,
                 epochs=100, seed=0):
        self.hidden = hidden          # None -> sqrt(n_in * n_out)  (Madhiarasan & Deepa)
        self.lr = lr
        self.lam = lam
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)
        h = self.hidden or max(2, int(round(np.sqrt(d * k))))
        rng = np.random.default_rng(self.seed)
        self.W1_ = rng.normal(0, 1 / np.sqrt(d), (d, h)); self.b1_ = np.zeros(h)
        self.W2_ = rng.normal(0, 1 / np.sqrt(h), (h, k)); self.b2_ = np.zeros(k)
        Y = np.eye(k)[yi]
        self.loss_history_, self.acc_history_ = [], []
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                xb, yb = X[idx], Y[idx]
                a = _sigmoid(xb @ self.W1_ + self.b1_)
                p = _softmax(a @ self.W2_ + self.b2_)
                dz2 = (p - yb) / len(idx)
                dW2 = a.T @ dz2 + 2 * self.lam * self.W2_
                da = dz2 @ self.W2_.T
                dz1 = da * a * (1 - a)
                dW1 = xb.T @ dz1 + 2 * self.lam * self.W1_
                self.W2_ -= self.lr * dW2; self.b2_ -= self.lr * dz2.sum(0)
                self.W1_ -= self.lr * dW1; self.b1_ -= self.lr * dz1.sum(0)
            P = self.predict_proba(X)
            self.loss_history_.append(float(-np.mean(np.sum(Y * np.log(P + 1e-12), 1))))
            self.acc_history_.append(float((P.argmax(1) == yi).mean()))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return _softmax(_sigmoid(X @ self.W1_ + self.b1_) @ self.W2_ + self.b2_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


if __name__ == "__main__":
    print("Testing NeuralNetClassifier (NumPy 2-layer backprop MLP)...")
    rng = np.random.default_rng(42)
    # Generate 500 synthetic samples with 10 features, 3 classes
    X_demo = rng.normal(0, 1, (500, 10))
    y_demo = rng.choice([1, 2, 3], size=500)

    model = NeuralNetClassifier(epochs=25, lr=1.0, batch_size=50, seed=42)
    model.fit(X_demo, y_demo)

    print(f"Architecture: input={model.W1_.shape[0]} -> hidden={model.W1_.shape[1]} -> output={model.W2_.shape[1]}")
    print(f"Initial loss: {model.loss_history_[0]:.4f} -> Final loss (epoch 25): {model.loss_history_[-1]:.4f}")
    sample_preds = model.predict(X_demo[:8])
    print(f"Sample predictions (first 8): {sample_preds.tolist()}")
    print(f"Accuracy on demo data: {model.acc_history_[-1]*100:.2f}%")

