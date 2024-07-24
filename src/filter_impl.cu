#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

// Macro pour vérifier les erreurs CUDA, utile pour le débugging
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Erreur Runtime CUDA à : %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        std::exit(EXIT_FAILURE);
    }
}

// Struct pour la couleur RGB uint8 
struct rgb {
    uint8_t r, g, b;
};

struct lab {
    float l, a, b;
};

__device__ float correction_gamma(float color) {
    return (color <= 0.04045f) ? color / 12.92f : powf((color + 0.055f) / 1.055f, 2.4f);
}

__device__ float f(float t) {
    return (t > 0.008856f) ? powf(t, 1.0f / 3.0f) : (7.787f * t) + (16.0f / 116.0f);
}

// Conversion RGB vers LAB (je sais pas si les formules sont correctes)
__device__ lab RGBtoLAB(const rgb& rgb) {
    float r = correction_gamma(rgb.r / 255.0f);
    float g = correction_gamma(rgb.g / 255.0f);
    float b = correction_gamma(rgb.b / 255.0f);

    // Conversion en espace de couleur XYZ
    float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

    // Conversion en espace de couleur LAB
    float fx = f(X / 95.047f);
    float fy = f(Y / 100.0f);
    float fz = f(Z / 108.883f);

    return { (116.0f * fy) - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz) };
}

// Kernel pour convertir une frame RGB en frame LAB 
__global__ void RgbToLabKernel(const rgb* rgbFrame, lab* labFrame, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        labFrame[idx] = RGBtoLAB(rgbFrame[idx]);
    }
}

// Fonction device pour calculer la différence de couleur 
__device__ float deltaE(const lab& pixel1, const lab& pixel2) {
    return sqrtf(
            (pixel1.l - pixel2.l) * (pixel1.l - pixel2.l) +
            (pixel1.a - pixel2.a) * (pixel1.a - pixel2.a) +
            (pixel1.b - pixel2.b) * (pixel1.b - pixel2.b)
    );
}

__global__ void ComputeResidualKernel(const lab* img1, const lab* img2, float* residual, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        residual[idx] = deltaE(img1[idx], img2[idx]);
    }
}

// Kernel pour normaliser l'image résiduelle 
__global__ void normalizeResidualKernel(const float* residual, uint8_t* normalizedResidual, float maxResidual, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        normalizedResidual[idx] = static_cast<uint8_t>((residual[idx] / maxResidual) * 255.0f);
    }
}

// erosion
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

// Kernel pour la dilatation morphologique (pareil ici)
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

__global__ void hysteresisThresholdKernel(const uint8_t* src, uint8_t* dst, bool* strong_edges, int width, int height, int seuil_min, int seuil_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        uint8_t pixel = src[idx];
        if (pixel > seuil_max) {
            strong_edges[idx] = true;
            dst[idx] = 255;
        } else {
            dst[idx] = 0;
        }
    }
}


__global__ void StrongEdgesPropagationKernel(const uint8_t* src, uint8_t* dst, bool* strong_edges, int width, int height, int seuil_min) {
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
                    uint8_t neighbor_pixel = src[n_idx];
                    if (neighbor_pixel >= seuil_min && !strong_edges[n_idx]) {
                        strong_edges[n_idx] = true;
                        dst[n_idx] = 255;
                    }
                }
            }
        }
    }
}

// a fix lae rgb
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
            // //
            pixel_output.r = static_cast<uint8_t>(min(255.0f, pixel.r * 0.5f + red.r * 0.5f));
            pixel_output.g = static_cast<uint8_t>(min(255.0f, pixel.g * 0.5f + red.g * 0.5f));
            pixel_output.b = static_cast<uint8_t>(min(255.0f, pixel.b * 0.5f + red.b * 0.5f));
        } else {
            pixel_output = pixel;
        }

        output[idx] = pixel_output;
    }
}

// Fonction pour calculer et normalisefaut laa check)
void computeAndNormalizeResidual(const lab* img1, const lab* img2, float* residual, uint8_t* normalizedResidual, int width, int height, cudaStream_t stream) {
    int size = width * height;
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    ComputeResidualKernel<<<gridSize, blockSize, 0, stream>>>(img1, img2, residual, width, height);

    thrust::device_ptr<float> dev_ptr_residual(residual);
    float maxResidual = *thrust::max_element(thrust::device, dev_ptr_residual, dev_ptr_residual + size);

    normalizeResidualKernel<<<gridSize, blockSize, 0, stream>>>(residual, normalizedResidual, maxResidual, width, height);
}

// Fonction pour l'ouverture morphologique (érosion suivie de dilatation) sans fermutre, le plus simple pour optimiser le temps de calcul
void morphologicalOpening(const uint8_t* src, uint8_t* dst, int width, int height, cudaStream_t stream) {
    int radius = 3; // Rayon ste a 3 comme dans le pdf 
    uint8_t* d_temp;
    size_t size = width * height * sizeof(uint8_t);

    cudaMalloc(&d_temp, size);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    erodeKernel<<<gridSize, blockSize, 0, stream>>>(src, d_temp, width, height, radius);
    dilateKernel<<<gridSize, blockSize, 0, stream>>>(d_temp, dst, width, height, radius);

    cudaFree(d_temp);
}

// Fonction pour l''hystérésis
void hysteresisThreshold(const uint8_t* src, uint8_t* dst, int width, int height, int seuil_min, int seuil_max, cudaStream_t stream) {
    size_t bool_size = width * height * sizeof(bool);
    bool* d_strong_edges;

    cudaMalloc(&d_strong_edges, bool_size);
    cudaMemset(d_strong_edges, 0, bool_size);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    hysteresisThresholdKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, d_strong_edges, width, height, seuil_min, seuil_max);
    StrongEdgesPropagationKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, d_strong_edges, width, height, seuil_min);

    cudaFree(d_strong_edges);
}

// Fonction pour appliquer un masque à l'image
void putMask(const rgb* input, const uint8_t* mask, rgb* output, int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    putMaskKernel<<<gridSize, blockSize, 0, stream>>>(input, mask, output, width, height);
}

// Struct pour contenir les données du modèle de fond
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

// Fonction externe pour filtrer l'image
extern "C" {
void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    static int frame_count = 0;
    static bool first_call = true;
    static rgb *d_frame_rgb = nullptr, *d_bg_rgb = nullptr, *d_output = nullptr;
    static lab *d_frame_lab = nullptr, *d_bg_lab = nullptr;
    static float *d_residual = nullptr;
    static uint8_t *d_residual_normalized = nullptr, *d_opened = nullptr, *d_hysteresis = nullptr;

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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    RgbToLabKernel<<<gridSize, blockSize, 0, stream>>>(d_frame_rgb, d_frame_lab, width, height);
    RgbToLabKernel<<<gridSize, blockSize, 0, stream>>>(d_bg_rgb, d_bg_lab, width, height);

    computeAndNormalizeResidual(d_bg_lab, d_frame_lab, d_residual, d_residual_normalized, width, height, stream);
    morphologicalOpening(d_residual_normalized, d_opened, width, height, stream);
    hysteresisThreshold(d_opened, d_hysteresis, width, height, 4, 30, stream);
    putMask(d_frame_rgb, d_hysteresis, d_output, width, height, stream);

    cudaMemcpy(buffer, d_output, rgb_size, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
}
}
