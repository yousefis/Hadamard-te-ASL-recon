import numpy as np
def bilinear_up_kernel( dim=3, kernel_size=3):
    center = kernel_size // 2
    if dim == 3:
        indices = [None] * dim
        indices[0], indices[1], indices[2] = np.meshgrid(np.arange(0, 3), np.arange(0, 3), np.arange(0, 3),
                                                         indexing='ij')
        for i in range(0, dim):
            indices[i] = indices[i] - center
        distance_to_center = np.absolute(indices[0]) + np.absolute(indices[1]) + np.absolute(indices[2])
        kernel = (np.ones(np.shape(indices[0])) / (np.power(2, distance_to_center))).astype(np.float32)

    return kernel