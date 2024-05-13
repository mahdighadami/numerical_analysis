import numpy as np

def jacobi(A, b, x0, tolerance, max_iteration):
    scale = np.shape(A)
    n = scale[0]

    x_new = np.zeros_like(x0)
    error = 2 * tolerance
    counter = 0
    x = np.zeros_like(b)
    while not(error <= tolerance or counter > max_iteration):
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
            if x_new[i] == x_new[i-1]:
                break

        # if np.allclose(x, x_new, atol=tolerance, rtol=0.):
        #     break
        
        counter += 1
        x = x_new
    check = np.dot(A, x_new)
    return(x_new, check, counter)


#Examples:
for example in range(3):
    A = 10*np.random.rand(3, 3)
    B = 10*np.random.rand(3, 1)
    x0  = np.array([0.0,0.0,0.0])
    tolerance = 1e-10
    max_iteration = 100
    print(f"\n\n -----------Example 0{example + 1}:-----------\n")
    print(f"A:\n {A}\n\nB:\n{B}\n")
    answer = jacobi(A, B, x0, tolerance, max_iteration)
    print(f"X:\n {answer[0]}\n")
    print(f"AX=B:\n {answer[1]}")
    print(f"\nIterations: {answer[2]}\n")

for example in range(3):
    A = 10*np.random.rand(4, 4)
    B = 10*np.random.rand(4, 1)
    x0  = np.array([0.0,0.0,0.0,0.0])
    tolerance = 1e-10
    max_iteration = 100
    print(f"\n\n -----------Example 0{example + 4}:-----------\n")
    print(f"A:\n {A}\n\nB:\n{B}\n")
    answer = jacobi(A, B, x0, tolerance, max_iteration)
    print(f"X:\n {answer[0]}\n")
    print(f"AX=B:\n {answer[1]}")
    print(f"\nIterations: {answer[2]}\n")

for example in range(3):
    A = 10*np.random.rand(5, 5)
    B = 10*np.random.rand(5, 1)
    x0  = np.array([0.0,0.0,0.0,0.0,0.0])
    tolerance = 1e-10
    max_iteration = 100
    print(f"\n\n -----------Example 0{example + 7}:-----------\n")
    print(f"A:\n {A}\n\nB:\n{B}\n")
    answer = jacobi(A, B, x0, tolerance, max_iteration)
    print(f"X:\n {answer[0]}\n")
    print(f"AX=B:\n {answer[1]}")
    print(f"\nIterations: {answer[2]}\n")


A = 10*np.random.rand(6, 6)
B = 10*np.random.rand(6, 1)
x0  = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
tolerance = 1e-10
max_iteration = 100
print(f"\n\n -----------Example {example + 10}:-----------\n")
print(f"A:\n {A}\n\nB:\n{B}\n")
answer = jacobi(A, B, x0, tolerance, max_iteration)
print(f"X:\n {answer[0]}\n")
print(f"AX=B:\n {answer[1]}")
print(f"\nIterations: {answer[2]}\n")
