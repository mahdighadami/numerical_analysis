import numpy as np

def SOR(A, b, omega, max_iteration):
    tol=1e-8
    # Initiation
    n = len(b)
    x = np.zeros(n)

    for counter in range(max_iteration):
        x_old = np.copy(x)

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]

        # convergence checking
        if np.linalg.norm(x - x_old, ord = np.inf) < tol:
            print(f'Convergence reached in {counter + 1} iterations.')
            return x, counter + 1

    print('Max_iteration reached without convergence!')
    return x, max_iteration


#Examples:
for example in range(3):
    A = 10*np.random.rand(3, 3)
    B = 10*np.random.rand(3, 1)
    max_iteration = 100
    omega = 1
    print(f"\n\n -----------Example 0{example + 1}:-----------\n")
    print(f"A:\n {A}\n\nB:\n{B}\n")
    answer = SOR(A, B, omega, max_iteration)
    print(f"solution:\n {answer[0]}\n")    
    print(f"\nIterations: {answer[1]}\n")

for example in range(3):
    A = 10*np.random.rand(4, 4)
    B = 10*np.random.rand(4, 1)
    max_iteration = 100
    omega = 1
    print(f"\n\n -----------Example 0{example + 4}:-----------\n")
    print(f"A:\n {A}\n\nB:\n{B}\n")
    answer = SOR(A, B, omega, max_iteration)
    print(f"solution:\n {answer[0]}\n")    
    print(f"\nIterations: {answer[1]}\n")

for example in range(3):
    A = 10*np.random.rand(5, 5)
    B = 10*np.random.rand(5, 1)
    max_iteration = 100
    omega = 1
    print(f"\n\n -----------Example 0{example + 7}:-----------\n")
    print(f"A:\n {A}\n\nB:\n{B}\n")
    answer = SOR(A, B, omega, max_iteration)
    print(f"solution:\n {answer[0]}\n")    
    print(f"\nIterations: {answer[1]}\n")


A = 10*np.random.rand(6, 6)
B = 10*np.random.rand(6, 1)
max_iteration = 100
omega = 1
print(f"\n\n -----------Example 10:-----------\n")
print(f"A:\n {A}\n\nB:\n{B}\n")
answer = SOR(A, B, omega, max_iteration)
print(f"solution:\n {answer[0]}\n")
print(f"\nIterations: {answer[1]}\n")
