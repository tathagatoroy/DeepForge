#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include <SFML/Graphics.hpp>

// Forward declarations for stb_image
extern "C" {
    unsigned char* stbi_load(const char* filename, int* x, int* y, int* channels_in_file, int desired_channels);
    void stbi_image_free(void* retval_from_stbi_load);
    int stbi_write_png(const char* filename, int w, int h, int comp, const void* data, int stride_in_bytes);
}


/**
    * @brief Matrix Operations
    */

/**
    * @brief Initializes a matrix with random float values between 0 and 1
    * @param matrix Pre-allocated float array
    * @param rows Number of rows
    * @param cols Number of columns
    */
void initializeMatrix(float* matrix, int rows, int cols);

/**
    * @brief Prints matrix contents to stdout
    * @param matrix Float array to print
    * @param rows Number of rows
    * @param cols Number of columns
    */
void printMatrix(float* matrix, int rows, int cols);

/**
    * @brief Image Operations
    */

/**
    * @brief Read a PNG image into a 2D vector
    * @param filepath Path to PNG file
    * @return 2D vector of grayscale pixel values
    * @throws std::runtime_error if image cannot be loaded
    */
std::vector<std::vector<int>> readPNG(const std::string& filepath);

/**
    * @brief Save a 2D matrix as a PNG image
    * @param matrix 2D vector of pixel values
    * @param filepath Output filepath
    * @return bool True if successful
    * @throws std::runtime_error if matrix is empty
    */
bool savePNG(const std::vector<std::vector<int>>& matrix, const std::string& filepath);

/**
    * @brief Display a 2D matrix as an image using SFML
    * @param matrix 2D vector of pixel values
    * @param windowTitle Title for display window
    * @throws std::runtime_error if matrix is empty
    */
void displayImage(const std::vector<std::vector<int>>& matrix, 
                    const std::string& windowTitle = "Image");


#endif // UTILS_HPP