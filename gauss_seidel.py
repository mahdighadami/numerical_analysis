import numpy as np

def gauss_seidel(A, B, x0, tolerance, max_iteration):

    scale = np.shape(A)
    n = scale[0]
    m = scale[1]

    #initial value
    X = np.copy(x0)
    diff = np.ones(n, dtype = float)
    error = 2 * tolerance
    counter = 0

    while not(error <= tolerance or counter > max_iteration):
            for i in range(0, n, 1):
                    sum = 0
                    for j in range(0, m, 1):
                            if (i != j):
                                    sum = sum - A[i, j] * X[j]
                    new_ = (B[i] + sum) / A[i, i]
                    diff[i] = np.abs(new_ - X[i])
                    X[i] = new_
            error = np.max(diff)
            counter += 1


    X = np.transpose([X])
    check = np.dot(A, X)
    return(X, check, counter)



#Examples:
for example in range(3):
    A = 10*np.random.rand(3, 3)
    B = 10*np.random.rand(3, 1)
    x0  = np.array([0.0,0.0,0.0])
    tolerance = 0.01
    max_iteration = 100
    print(f"\n\n -----------Example 0{example + 1}:-----------\n")
    print(f"A:\n {A}\n\nB:\n{B}\n")
    answer = gauss_seidel(A, B, x0, tolerance, max_iteration)
    print(f"X:\n {answer[0]}\n")
    print(f"AX=B:\n {answer[1]}")
    print(f"\nIterations: {answer[2]}\n")

for example in range(3):
    A = 10*np.random.rand(4, 4)
    B = 10*np.random.rand(4, 1)
    x0  = np.array([0.0,0.0,0.0,0.0])
    tolerance = 0.01
    max_iteration = 100
    print(f"\n\n -----------Example 0{example + 4}:-----------\n")
    print(f"A:\n {A}\n\nB:\n{B}\n")
    answer = gauss_seidel(A, B, x0, tolerance, max_iteration)
    print(f"X:\n {answer[0]}\n")
    print(f"AX=B:\n {answer[1]}")
    print(f"\nIterations: {answer[2]}\n")

for example in range(3):
    A = 10*np.random.rand(5, 5)
    B = 10*np.random.rand(5, 1)
    x0  = np.array([0.0,0.0,0.0,0.0,0.0])
    tolerance = 0.01
    max_iteration = 100
    print(f"\n\n -----------Example 0{example + 7}:-----------\n")
    print(f"A:\n {A}\n\nB:\n{B}\n")
    answer = gauss_seidel(A, B, x0, tolerance, max_iteration)
    print(f"X:\n {answer[0]}\n")
    print(f"AX=B:\n {answer[1]}")
    print(f"\nIterations: {answer[2]}\n")


A = 10*np.random.rand(6, 6)
B = 10*np.random.rand(6, 1)
x0  = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
tolerance = 0.01
max_iteration = 100
print(f"\n\n -----------Example {example + 10}:-----------\n")
print(f"A:\n {A}\n\nB:\n{B}\n")
answer = gauss_seidel(A, B, x0, tolerance, max_iteration)
print(f"X:\n {answer[0]}\n")
print(f"AX=B:\n {answer[1]}")
print(f"\nIterations: {answer[2]}\n")
