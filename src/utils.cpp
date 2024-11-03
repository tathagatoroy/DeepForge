#include <iostream>
#include <cstdlib> // for rand() and RAND_MAX
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <string>

// STB Image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// SFML for display
#include <SFML/Graphics.hpp>

using namespace std;

void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// Print the contents of a matrix
void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}


/**
* @brief Read a PNG image into a 2D vector
* 
* @param filepath Path to the PNG file
* @return 2D vector containing grayscale pixel values
* @throws runtime_error if image cannot be loaded
*/
vector<vector<int>> readPNG(const string& filepath) {
    int width, height, channels;
    // def stbi_uc *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels)
    // stbi_uc is a typedef for unsigned char used in the stb_image.h library. It's basically a byte type (8 bits) used to store image pixel data.
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 1);
    
    if (!data) {
        throw runtime_error("Failed to load image: " + filepath);
    }
    
    // Convert to 2D vector
    vector<vector<int>> matrix(height, vector<int>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            matrix[i][j] = static_cast<int>(data[i * width + j]);
        }
    }
    // release memory
    stbi_image_free(data);
    return matrix;
}

/**
* @brief Save a 2D matrix as a PNG image
* 
* @param matrix 2D vector of pixel values
* @param filepath Output filepath
* @return bool True if successful
*/
bool savePNG(const vector<vector<int>>& matrix, const string& filepath) {
    if (matrix.empty() || matrix[0].empty()) {
        throw runtime_error("Empty matrix provided");
    }
    
    int height = matrix.size();
    int width = matrix[0].size();
    
    // Convert to flat array of unsigned char
    vector<unsigned char> data;
    data.reserve(height * width);
    
    for (const auto& row : matrix) {
        for (int pixel : row) {
            // Clamp values to 0-255
            pixel = min(255, max(0, pixel));
            data.push_back(static_cast<unsigned char>(pixel));
        }
    }
    // int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
    return stbi_write_png(filepath.c_str(), width, height, 1, data.data(), 
                        width * sizeof(unsigned char)) != 0;
}

/**
* @brief Display a 2D matrix as an image
* 
* @param matrix 2D vector of pixel values
* @param windowTitle Title for the display window
*/
void displayImage(const vector<vector<int>>& matrix, const string& windowTitle) {
    if (matrix.empty() || matrix[0].empty()) {
        throw runtime_error("Empty matrix provided");
    }
    
    int height = matrix.size();
    int width = matrix[0].size();
    
    // Create window and image
    sf::RenderWindow window(sf::VideoMode(width, height), windowTitle);
    sf::Image image;
    image.create(width, height);
    
    // Set pixels
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixel = min(255, max(0, matrix[i][j]));
            image.setPixel(j, i, sf::Color(pixel, pixel, pixel));
        }
    }
    
    // Create texture and sprite
    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);
    
    // Display loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
        
        window.clear();
        window.draw(sprite);
        window.display();
    }
}
