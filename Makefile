# Define compiler and flags
NVCC = nvcc
CFLAGS = -I/usr/local/cuda/include

# Define the target executable
TARGET = matrix_multiply

# Define the source files
SRC = test_and_experiments/cuda_matrix_multiply.cpp

# Rule to build the target
$(TARGET): $(SRC)
	$(NVCC) -x cu $(CFLAGS) $(SRC) -o $(TARGET)

# Clean up the generated files
clean:
	rm -f $(TARGET)
