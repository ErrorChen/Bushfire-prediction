
# check_gpu.py

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU device list:", tf.config.list_physical_devices('GPU'))
print("GPU available (tf.test):", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
print("Default GPU name:", tf.test.gpu_device_name())
