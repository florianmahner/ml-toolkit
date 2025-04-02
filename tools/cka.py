import numpy as np

def linear_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the linear kernel (dot-product) between matrices x and y.
    """
    return x @ y.T

def center_kernel(k: np.ndarray) -> np.ndarray:
    """
    Symmetrically center the kernel matrix K.
    
    K_centered = H K H, where H = I - 1/n * 11ᵀ
    """
    n = k.shape[0]
    ones = np.ones((n, n)) / n
    h = np.eye(n) - ones
    return h @ k @ h

def hsic(k: np.ndarray, l: np.ndarray) -> float:
    """
    Compute Hilbert-Schmidt Independence Criterion (HSIC) between
    two centered kernel matrices k and l.
    """
    n = k.shape[0]
    return np.trace(k @ l) / ((n - 1) ** 2)

def cka(x: np.ndarray, y: np.ndarray, kernel=linear_kernel) -> float:
    """
    Compute the Centered Kernel Alignment (CKA) between representations x and y.
    """
    k = kernel(x, x)
    l = kernel(y, y)    

    k_centered = center_kernel(k)
    l_centered = center_kernel(l)

    hsic_xy = hsic(k_centered, l_centered)
    hsic_xx = hsic(k_centered, k_centered)
    hsic_yy = hsic(l_centered, l_centered)

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)
