// The format is
//     label, pix-11, pix-12, pix-13, ...

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "config.hpp"  // For BLOCK_SIZE definition
#include "utils.hpp"   // For utility functions
#include<cmath>

using namespace std;

/**
 * @struct ImageData
 * @brief Stores the label and pixel values for a single image
 * 
 * @member label   The classification label for the image
 * @member pixels  Vector containing all pixel values in row-major order
 */
struct ImageData {
    int label;
    vector<int> pixels;
};

/**
 * @class CSVImageReader
 * @brief Provides functionality to read and process image data from CSV files
 * 
 * This class contains static methods for parsing CSV files containing image data,
 * where each row represents one image with a label followed by pixel values.
 */
class CSVImageReader {
public:
    /**
     * @brief Parses a single line from the CSV file into an ImageData structure
     * 
     * @param line A string containing comma-separated values, where the first value
     *             is the label and subsequent values are pixel intensities
     * @return ImageData containing the parsed label and pixel values
     * @throws invalid_argument if the line cannot be parsed properly
     * @throws out_of_range if number conversion fails
     */
    static ImageData parseLine(const string& line) {
        stringstream ss(line);
        string value;
        ImageData image;
        
        // Read label (first column)
        // definition : istream& getline (istream& is, string& str, char delim);
        // ss is read till the first comma and the value is stored in value
        if (getline(ss, value, ',')) {
            image.label = stoi(value);
        }
        
        // Read pixel values
        while (getline(ss, value, ',')) {
            if (!value.empty()) {
                image.pixels.push_back(stoi(value));
            }
        }
        
        return image;
    }
    
    /**
     * @brief Reads all images from a CSV file
     * 
     * @param filename Path to the CSV file containing image data
     * @return Vector of ImageData objects, each containing one image
     * @throws runtime_error if the file cannot be opened
     * @throws invalid_argument if any line cannot be parsed
     * @throws out_of_range if number conversion fails
     * 
     * @note The function assumes the first line is a header and skips it
     * @note Each line should be formatted as: label,pixel1,pixel2,...,pixelN
     */
    static vector<ImageData> readCSV(const string& filename) {
        vector<ImageData> images;
        ifstream file(filename);
        string line;
        
        if (!file.is_open()) {
            throw runtime_error("Could not open file: " + filename);
        }
        
        // Skip header if it exists
        getline(file, line);
        
        // Read data lines
        while (getline(file, line)) {
            try {
                images.push_back(parseLine(line));
            } catch (const exception& e) {
                cerr << "Error parsing line: " << e.what() << endl;
                continue;
            }
        }
        
        return images;
    }
    
    /**
     * @brief Calculates the size of a square image based on the number of pixels
     * 
     * @param image The ImageData object containing the pixel values
     * @return Integer representing both width and height of the square image
     * @throws runtime_error if the number of pixels is not a perfect square
     * 
     * @note This function assumes the image is square (width = height)
     */
    static int getImageSize(const ImageData& image) {
        int totalPixels = static_cast<int>(image.pixels.size());
        int size = static_cast<int>(sqrt(totalPixels));
        
        // Verify the image is actually square
        if (size * size != totalPixels) {
            throw runtime_error("Image is not square");
        }
        
        return size;
    }
    
    /**
     * @brief Prints the image as a 2D grid to standard output
     * 
     * @param image The ImageData object to print
     * @throws runtime_error if the image is not square
     * 
     * @note Output format:
     * Label: X
     * p11 p12 p13 ...
     * p21 p22 p23 ...
     * ...
     */
    static void printImage(const ImageData& image) {
        int size = getImageSize(image);
        cout << "Label: " << image.label << endl;
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                cout << image.pixels[i * size + j] << " ";
            }
            cout << endl;
        }
    }
};

int main() {
    string filename = "mnist_sample.csv";
    
    try {
        vector<ImageData> images = CSVImageReader::readCSV(filename);
        
        for (const ImageData& image : images) {
            cout<< "Label: " << image.label << endl;
            cout << endl;
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}