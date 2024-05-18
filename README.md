# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming
 ## AIM:
The aim of this project is to demonstrate how to convert an image to grayscale using CUDA programming without relying on the OpenCV library. It serves as an example of GPU-accelerated image processing using CUDA.

## Procedure:
1.Load the input image using the stb_image library.
2.Allocate memory on the GPU for the input and output image buffers.
3.Copy the input image data from the CPU to the GPU.
4.Define a CUDA kernel function that performs the grayscale conversion on each pixel of the image.
5.Launch the CUDA kernel with appropriate grid and block dimensions.
6.Copy the resulting grayscale image data from the GPU back to the CPU.
7.Save the grayscale image using the stb_image_write library.
8.Clean up allocated memory.
## Program:
```

!apt-get update
!apt-get install -y nvidia-cuda-toolkit
!pip install opencv-python

%%writefile grayscale.cu

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHANNELS 3

__global__ 
void colorConvertToGray(unsigned char *rgb, unsigned char *gray, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows) {
        int gray_offset = row * cols + col;
        int rgb_offset = gray_offset * CHANNELS;

        unsigned char r = rgb[rgb_offset];
        unsigned char g = rgb[rgb_offset + 1];
        unsigned char b = rgb[rgb_offset + 2];

        gray[gray_offset] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void loadImageFile(unsigned char **rgb_image, int *rows, int *cols, const std::string &file) {
    cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "Error: Unable to load image %s\n", file.c_str());
        exit(EXIT_FAILURE);
    }

    *rows = img.rows;
    *cols = img.cols;

    *rgb_image = (unsigned char*) malloc(*rows * *cols * CHANNELS * sizeof(unsigned char));
    memcpy(*rgb_image, img.data, *rows * *cols * CHANNELS * sizeof(unsigned char));
}

void saveImageFile(const unsigned char *gray_image, int rows, int cols, const std::string &file) {
    cv::Mat img(rows, cols, CV_8UC1, (void*)gray_image);
    cv::imwrite(file, img);
}

int main() {
    std::string input_file = "input.jpg";
    std::string output_file = "output.jpg";

    unsigned char *h_rgb_image, *h_gray_image;
    unsigned char *d_rgb_image, *d_gray_image;
    int rows, cols;

    loadImageFile(&h_rgb_image, &rows, &cols, input_file);

    size_t image_size = rows * cols * CHANNELS * sizeof(unsigned char);
    size_t gray_image_size = rows * cols * sizeof(unsigned char);

    h_gray_image = (unsigned char*) malloc(gray_image_size);

    cudaMalloc(&d_rgb_image, image_size);
    cudaMalloc(&d_gray_image, gray_image_size);

    cudaMemcpy(d_rgb_image, h_rgb_image, image_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
    
    colorConvertToGray<<<dimGrid, dimBlock>>>(d_rgb_image, d_gray_image, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gray_image, d_gray_image, gray_image_size, cudaMemcpyDeviceToHost);

    saveImageFile(h_gray_image, rows, cols, output_file);

    cudaFree(d_rgb_image);
    cudaFree(d_gray_image);
    free(h_rgb_image);
    free(h_gray_image);

    return 0;
}

!nvcc -o grayscale grayscale.cu `pkg-config --cflags --libs opencv4`
!./grayscale

import cv2
from matplotlib import pyplot as plt
output_image = cv2.imread('/content/skzoo.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(output_image, cmap='gray')
plt.axis('off')
plt.show()
```
## Output:
![image](https://github.com/Rakshithadevi/PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD/assets/94165326/c6f34488-69b6-4559-b793-c312aaaa8658)

![image](https://github.com/Rakshithadevi/PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD/assets/94165326/f2cce9db-66d4-49b7-8d03-769e7f60b759)

## Result:
The CUDA program successfully converts the input image to grayscale using the GPU. The resulting grayscale image is saved as an output file. This example demonstrates the power of GPU parallelism in accelerating image processing tasks.
