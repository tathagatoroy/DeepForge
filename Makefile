# Compilers and flags
NVCC = nvcc
CXX = g++

# SFML paths
SFML_PATH = /home2/tathagato/DeepForge/libraries/SFML-2.5.1
SFML_INCLUDE = $(SFML_PATH)/include
SFML_LIB = $(SFML_PATH)/lib

# Include and library paths
INCLUDES = -I./include -I/usr/local/cuda/include -I$(SFML_INCLUDE)
NVCC_FLAGS = -x cu $(INCLUDES)
CXX_FLAGS = $(INCLUDES) -std=c++11
LIBS = -L$(SFML_LIB) -lsfml-graphics -lsfml-window -lsfml-system

INCLUDES += -I/home2/tathagato/DeepForge/libraries/


# Source files
MATRIX_SRC = src/matrix_multiply.cu
MNIST_SRC = src/generate_mnist_data_from_csv.cpp
UTILS_SRC = src/utils.cpp
MAIN_SRC = src/main.cpp

# Objects and targets
UTILS_OBJ = build/utils.o
MATRIX_OBJ = build/matrix_multiply.o
MNIST_OBJ = build/generate_mnist_data_from_csv.o
MATRIX_TARGET = bin/matrix_multiply
MNIST_TARGET = bin/mnist_generator
MAIN_OBJ = build/main.o

# Default target
all: dirs $(MATRIX_TARGET) $(MNIST_TARGET)

# Create directories
dirs:
	mkdir -p build bin

# Specific compilation rule for MNIST generator
$(MNIST_OBJ): $(MNIST_SRC)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# General C++ compilation rule (for files with headers)
build/%.o: src/%.cpp include/%.hpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# CUDA compilation rule
build/%.o: src/%.cu include/%.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Matrix multiply executable
$(MATRIX_TARGET): $(MATRIX_OBJ) $(UTILS_OBJ)
	$(NVCC) $^ -o $@ $(LIBS)

# MNIST generator executable
$(MNIST_TARGET): $(MNIST_OBJ) $(UTILS_OBJ)
	$(CXX) $^ -o $@ $(LIBS)

clean:
	rm -rf build bin

.PHONY: all clean dirs