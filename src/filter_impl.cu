#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

// Struct for RGB color (uint8)
struct rgb {
    uint8_t r, g, b;
};

// Struct for LAB color space
struct lab {
    float l, a, b;
};

__device__ float correction_gamma(float color) {
    return (color <= 0.04045f) ? color / 12.92f : powf((color + 0.055f) / 1.055f, 2.4f);
}

__device__ float f(float t) {
    return (t > 0.008856f) ? powf(t, 1.0f / 3.0f) : (7.787f * t) + (16.0f / 116.0f);
}

// Convert RGB to LAB color space
__device__ lab RGBtoLAB(const rgb& rgb) {
    float r = correction_gamma(rgb.r / 255.0f);
    float g = correction_gamma(rgb.g / 255.0f);
    float b = correction_gamma(rgb.b / 255.0f);

    float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

    float fx = f(X / 95.047f);
    float fy = f(Y / 100.0f);
    float fz = f(Z / 108.883f);

    return { (116.0f * fy) - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz) };
}

// Kernel to convert RGB frame to LAB frame
__global__ void RgbToLabKernel(const rgb* rgbFrame, lab* labFrame, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        labFrame[idx] = RGBtoLAB(rgbFrame[idx]);
    }
}

// Device function to calculate color difference
__device__ float deltaE(const lab& pixel1, const lab& pixel2) {
    return sqrtf(
            (pixel1.l - pixel2.l) * (pixel1.l - pixel2.l) +
            (pixel1.a - pixel2.a) * (pixel1.a - pixel2.a) +
            (pixel1.b - pixel2.b) * (pixel1.b - pixel2.b)
    );
}

// Kernel to compute residual difference
__global__ void ComputeResidualKernel(const lab* img1, const lab* img2, float* residual, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        residual[idx] = deltaE(img1[idx], img2[idx]);
    }
}

// Kernel to normalize the residual image
__global__ void normalizeResidualKernel(const float* residual, uint8_t* normalizedResidual, float maxResidual, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        normalizedResidual[idx] = static_cast<uint8_t>((residual[idx] / maxResidual) * 255.0f);
    }
}

// Kernel for erosion
__global__ void erodeKernel(const uint8_t* src, uint8_t* dst, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        uint8_t minPixel = 255;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uint8_t currentPixel = src[ny * width + nx];
                    if (currentPixel < minPixel) {
                        minPixel = currentPixel;
                    }
                }
            }
        }
        dst[idx] = minPixel;
    }
}

// Kernel for dilation
__global__ void dilateKernel(const uint8_t* src, uint8_t* dst, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        uint8_t maxPixel = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uint8_t currentPixel = src[ny * width + nx];
                    if (currentPixel > maxPixel) {
                        maxPixel = currentPixel;
                    }
                }
            }
        }
        dst[idx] = maxPixel;
    }
}

// Kernel for hysteresis thresholding
__global__ void hysteresisThresholdKernel(const uint8_t* src, uint8_t* dst, bool* strong_edges, int width, int height, int minThreshold, int maxThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        uint8_t pixel = src[idx];
        if (pixel > maxThreshold) {
            strong_edges[idx] = true;
            dst[idx] = 255;
       // } else if (pixel > minThreshold) {
          //  dst[idx] = 128; 
        } else {
            dst[idx] = 0;
        }
    }
}

// Kernel for propagating strong edges
__global__ void StrongEdgesPropagationKernel(const uint8_t* src, uint8_t* dst, bool* strong_edges, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height && strong_edges[idx]) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int n_idx = ny * width + nx;
                    if (src[n_idx] == 128) { // If it's a weak edge
                        strong_edges[n_idx] = true;
                        dst[n_idx] = 255;
                    }
                }
            }
        }
    }
}

// Kernel to apply a mask to the RGB image
__global__ void putMaskKernel(const rgb* input, const uint8_t* mask, rgb* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        rgb red = {255, 0, 0};
        rgb pixel = input[idx];
        uint8_t pixel_mask = mask[idx];
        rgb pixel_output;

        if (pixel_mask > 0) {
            pixel_output.r = static_cast<uint8_t>(min(255.0f, pixel.r * 0.5f + red.r * 0.5f));
            pixel_output.g = static_cast<uint8_t>(min(255.0f, pixel.g * 0.5f + red.g * 0.5f));
            pixel_output.b = static_cast<uint8_t>(min(255.0f, pixel.b * 0.5f + red.b * 0.5f));
        } else {
            pixel_output = pixel;
        }

        output[idx] = pixel_output;
    }
}

// Function to compute and normalize residuals
void computeAndNormalizeResidual(const lab* img1, const lab* img2, float* residual, uint8_t* normalizedResidual, int width, int height, cudaStream_t stream) {
    int size = width * height;
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    ComputeResidualKernel<<<gridSize, blockSize, 0, stream>>>(img1, img2, residual, width, height);

    thrust::device_ptr<float> dev_ptr_residual(residual);
    float maxResidual = *thrust::max_element(thrust::device, dev_ptr_residual, dev_ptr_residual + size);

    normalizeResidualKernel<<<gridSize, blockSize, 0, stream>>>(residual, normalizedResidual, maxResidual, width, height);
}

// Function for morphological opening (erosion followed by dilation)
void morphologicalOpening(const uint8_t* src, uint8_t* dst, int width, int height, cudaStream_t stream) {
    int radius = 3;
    uint8_t* d_temp;
    size_t size = width * height * sizeof(uint8_t);

    cudaMalloc(&d_temp, size);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    erodeKernel<<<gridSize, blockSize, 0, stream>>>(src, d_temp, width, height, radius);
    dilateKernel<<<gridSize, blockSize, 0, stream>>>(d_temp, dst, width, height, radius);

    cudaFree(d_temp);
}

// Function for hysteresis thresholding
void hysteresisThreshold(const uint8_t* src, uint8_t* dst, int width, int height, int minThreshold, int maxThreshold, cudaStream_t stream) {
    size_t bool_size = width * height * sizeof(bool);
    bool* d_strong_edges;

    cudaMalloc(&d_strong_edges, bool_size);
    cudaMemset(d_strong_edges, 0, bool_size);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    hysteresisThresholdKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, d_strong_edges, width, height, minThreshold, maxThreshold);
    cudaStreamSynchronize(stream);  

    StrongEdgesPropagationKernel<<<gridSize, blockSize, 0, stream>>>(dst, dst, d_strong_edges, width, height);
    cudaStreamSynchronize(stream);  

    cudaFree(d_strong_edges);
}

// Function to apply a mask to the image
void putMask(const rgb* input, const uint8_t* mask, rgb* output, int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    putMaskKernel<<<gridSize, blockSize, 0, stream>>>(input, mask, output, width, height);
}

// Struct for background model data
struct BackgroundModel {
    rgb* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
    bool is_initialized = false;

    ~BackgroundModel() {
        delete[] data;
    }
};

BackgroundModel bg_model;

// External function to filter the image
extern "C" {
void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    static int frame_count = 0;
    bool first_call = true;
    frame_count++;

    if (!bg_model.is_initialized) {
        bg_model.data = new rgb[width * height];
        bg_model.width = width;
        bg_model.height = height;
        bg_model.stride = stride;
        bg_model.is_initialized = true;

        for (int y = 0; y < height; ++y) {
            memcpy(bg_model.data + y * width, buffer + y * stride, width * sizeof(rgb));
        }
        return;
    }

    size_t rgb_size = width * height * sizeof(rgb);
    size_t lab_size = width * height * sizeof(lab);
    size_t float_size = width * height * sizeof(float);
    size_t uint8_size = width * height * sizeof(uint8_t);

    rgb *d_frame_rgb = nullptr;
    rgb *d_bg_rgb = nullptr;
    lab *d_frame_lab = nullptr;
    lab *d_bg_lab = nullptr;
    float *d_residual = nullptr;
    uint8_t *d_residual_normalized = nullptr;
    uint8_t *d_opened = nullptr;
    uint8_t *d_hysteresis = nullptr;
    rgb *d_output = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    if (first_call) {
        cudaMalloc(&d_frame_rgb, rgb_size);
        cudaMalloc(&d_bg_rgb, rgb_size);
        cudaMalloc(&d_frame_lab, lab_size);
        cudaMalloc(&d_bg_lab, lab_size);
        cudaMalloc(&d_residual, float_size);
        cudaMalloc(&d_residual_normalized, uint8_size);
        cudaMalloc(&d_opened, uint8_size);
        cudaMalloc(&d_hysteresis, uint8_size);
        cudaMalloc(&d_output, rgb_size);
        first_call = false;
    }

    cudaMemcpy(d_frame_rgb, buffer, rgb_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bg_rgb, bg_model.data, rgb_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    RgbToLabKernel<<<gridSize, blockSize, 0, stream>>>(d_frame_rgb, d_frame_lab, width, height);
    RgbToLabKernel<<<gridSize, blockSize, 0, stream>>>(d_bg_rgb, d_bg_lab, width, height);
    cudaStreamSynchronize(stream);

    computeAndNormalizeResidual(d_bg_lab, d_frame_lab, d_residual, d_residual_normalized, width, height, stream);
    cudaStreamSynchronize(stream);

    morphologicalOpening(d_residual_normalized, d_opened, width, height, stream);
    cudaStreamSynchronize(stream);

    hysteresisThreshold(d_opened, d_hysteresis, width, height, 4, 30, stream);
    cudaStreamSynchronize(stream);

    putMask(d_frame_rgb, d_hysteresis, d_output, width, height, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(buffer, d_output, rgb_size, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);

    // Free allocated memory
    cudaFree(d_frame_rgb);
    cudaFree(d_bg_rgb);
    cudaFree(d_frame_lab);
    cudaFree(d_bg_lab);
    cudaFree(d_residual);
    cudaFree(d_residual_normalized);
    cudaFree(d_opened);
    cudaFree(d_hysteresis);
    cudaFree(d_output);
}
} // extern "C"
