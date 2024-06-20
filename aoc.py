import multiprocessing
import NumPy as np
def matrix_multiply_worker(A, B, result, row_start, row_end):
    for i in range(row_start, row_end):
        for j in range(B.shape[1]):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(A.shape[1]))
def parallel_matrix_multiply(A, B, num_processes):
    assert A.shape[1] == B.shape[0]
    result = np.zeros((A.shape[0], B.shape[1]))
    chunk_size = A.shape[0] // num_processes
    processes = []
    for i in range(num_processes):
        row_start = i * chunk_size
        row_end = (i + 1) * chunk_size if i != num_processes - 1 else A.shape[0]
        process = multiprocessing.Process(target=matrix_multiply_worker, 
                                          args=(A, B, result, row_start, row_end))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    return result
if __name__ == '__main__':
    A = np.random.randint(0, 10, (4, 4))
    B = np.random.randint(0, 10, (4, 4))
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    num_processes = 4
    result = parallel_matrix_multiply(A, B, num_processes)
    print("\nResultant Matrix:")
    print(result)
