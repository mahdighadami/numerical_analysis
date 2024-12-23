import numpy as np

def householder_reflection(a):
    v = a.copy()
    v[0] += np.copysign(np.linalg.norm(a), a[0])
    v = v/np.linalg.norm(v)
    H = np.eye(len(a)) - 2*np.outer(v, v)
    return H

def qr_decomposition(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for i in range(min(m, n)):
        H = np.eye(m)
        H[i:, i:] = householder_reflection(R[i:, i])
        Q = Q @ H
        R = H @ R

    return Q, R



#Examples:
for example in range(3):
    A = 10*np.random.rand(3, 3)
    print(f"\n\n -----------Example 0{example + 1}:-----------\n")
    print(f"A:\n {A}\n")
    Q, R = qr_decomposition(A)
    print(f"Q:\n {Q}\n")
    print(f"R:\n {R}\n")
    print(f"QR:\n {Q @ R}")
    

for example in range(3):
    A = 10*np.random.rand(4, 4)
    print(f"\n\n -----------Example 0{example + 4}:-----------\n")
    print(f"A:\n {A}\n")
    Q, R = qr_decomposition(A)
    print(f"Q:\n {Q}\n")
    print(f"R:\n {R}\n")
    print(f"QR:\n {Q @ R}")
    

for example in range(3):
    A = 10*np.random.rand(5, 5)
    print(f"\n\n -----------Example 0{example + 7}:-----------\n")
    print(f"A:\n {A}\n")
    Q, R = qr_decomposition(A)
    print(f"Q:\n {Q}\n")
    print(f"R:\n {R}\n")
    print(f"QR:\n {Q @ R}")
    


A = 10*np.random.rand(6, 6)
print(f"\n\n -----------Example {example + 10}:-----------\n")
print(f"A:\n {A}\n")
Q, R = qr_decomposition(A)
print(f"Q:\n {Q}\n")
print(f"R:\n {R}\n")
print(f"QR:\n {Q @ R}")

