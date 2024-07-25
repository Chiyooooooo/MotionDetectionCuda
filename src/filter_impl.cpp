#include "filter_impl.h"

#include <chrono>
#include <thread>
#include "logo.h"
#include <iostream>
#include <fstream>
#include <random>
#include <cstring>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <deque>

// Struct for RGB color (uint8)
struct rgb {
    uint8_t r, g, b;
};

// Struct for LAB color space
struct lab {
    double l, a, b;
};

// Struct for background model data
struct BackgroundModel {
    std::deque<std::vector<rgb>> frames;
    int width = 0;
    int height = 0;
    int stride = 0;
    bool is_initialized = false;
};

BackgroundModel bg_model;

/**
 * @brief Apply gamma correction to a color channel.
 *
 * @param channel The input color channel value.
 * @return The gamma-corrected color channel value.
 */
double gammaCorrection(double channel) {
    return (channel <= 0.04045) ? channel / 12.92 : pow((channel + 0.055) / 1.055, 2.4);
}

/**
 * @brief Convert an RGB color to XYZ color space.
 *
 * @param rgb The input RGB color.
 * @return The corresponding XYZ color.
 */
std::array<double, 3> RGBtoXYZ(const rgb& rgb) {
    double r_normalized = rgb.r / 255.0;
    double g_normalized = rgb.g / 255.0;
    double b_normalized = rgb.b / 255.0;

    double R = gammaCorrection(r_normalized);
    double G = gammaCorrection(g_normalized);
    double B = gammaCorrection(b_normalized);

    double X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375;
    double Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750;
    double Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041;

    return {X * 100 / 95.047, Y * 100 / 100.000, Z * 100 / 108.883};
}

/**
 * @brief Helper function for LAB conversion.
 *
 * @param t The input value.
 * @return The transformed value.
 */
double f(double t) {
    const double delta = 6.0 / 29.0;
    const double delta3 = delta * delta * delta;

    if (t > delta3) {
        return pow(t, 1.0 / 3.0);
    } else {
        return (t / (3 * delta * delta) + 4.0 / 29.0);
    }
}

/**
 * @brief Convert an XYZ color to LAB color space.
 *
 * @param xyz The input XYZ color.
 * @return The corresponding LAB color.
 */
lab XYZtoLAB(const std::array<double, 3>& xyz) {
    double X = f(xyz[0]);
    double Y = f(xyz[1]);
    double Z = f(xyz[2]);

    return {(116.0 * Y) - 16.0, 500.0 * (X - Y), 200.0 * (Y - Z)};
}

/**
 * @brief Convert an RGB color to LAB color space.
 *
 * @param rgb_ The input RGB color.
 * @return The corresponding LAB color.
 */
lab RGBtoLAB(const rgb& rgb_) {
    std::array<double, 3> xyz = RGBtoXYZ(rgb_);
    return XYZtoLAB(xyz);
}

/**
 * @brief Convert an array of RGB colors to LAB color space.
 *
 * @param rgbPixels The input array of RGB colors.
 * @param labPixels The output array of LAB colors.
 * @param size The size of the input array.
 */
void RGBtoLAB(const rgb* rgbPixels, lab* labPixels, int size) {
    for (int i = 0; i < size; ++i) {
        labPixels[i] = RGBtoLAB(rgbPixels[i]);
    }
}

/**
 * @brief Calculate the color difference between two LAB colors.
 *
 * @param lab1 The first LAB color.
 * @param lab2 The second LAB color.
 * @return The color difference.
 */
double deltaE(const lab& lab1, const lab& lab2) {
    return std::sqrt(
        (lab1.l - lab2.l) * (lab1.l - lab2.l) +
        (lab1.a - lab2.a) * (lab1.a - lab2.a) +
        (lab1.b - lab2.b) * (lab1.b - lab2.b)
    );
}

/**
 * @brief Compute the residual image from two LAB images.
 *
 * @param img1 The first LAB image.
 * @param img2 The second LAB image.
 * @param residual The output residual image.
 * @param size The size of the input images.
 */
void ComputeResidual(const lab* img1, const lab* img2, double* residual, int size) {
    for (int i = 0; i < size; ++i) {
        residual[i] = deltaE(img1[i], img2[i]);
    }
}

/**
 * @brief Compute and normalize the residual image.
 *
 * @param img1 The first LAB image.
 * @param img2 The second LAB image.
 * @param residual The output residual image.
 * @param normalizedResidual The output normalized residual image.
 * @param size The size of the input images.
 */
void ComputeAndNormalizeResidual(const lab* img1, const lab* img2, double* residual, uint8_t* normalizedResidual, int size) {
    double maxResidual = 0.0;
    for (int i = 0; i < size; ++i) {
        double r = deltaE(img1[i], img2[i]);
        residual[i] = r;
        if (r > maxResidual) {
            maxResidual = r;
        }
    }

    for (int i = 0; i < size; ++i) {
        normalizedResidual[i] = static_cast<uint8_t>((residual[i] / maxResidual) * 255.0);
    }
}

/**
 * @brief Apply Gaussian blur to an image.
 *
 * @param src The input image.
 * @param dst The output blurred image.
 * @param width The width of the image.
 * @param height The height of the image.
 */
void gaussianBlur(const uint8_t* src, uint8_t* dst, int width, int height) {
    int kernel_size = 5;
    double sigma = 1.0;
    std::vector<double> kernel(kernel_size * kernel_size);
    double sum = 0.0;
    int half = kernel_size / 2;

    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            double value = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[(y + half) * kernel_size + (x + half)] = value;
            sum += value;
        }
    }

    for (auto& val : kernel) {
        val /= sum;
    }

    for (int y = half; y < height - half; ++y) {
        for (int x = half; x < width - half; ++x) {
            double blurred_pixel = 0.0;
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int pixel_val = src[(y + ky) * width + (x + kx)];
                    blurred_pixel += pixel_val * kernel[(ky + half) * kernel_size + (kx + half)];
                }
            }
            dst[y * width + x] = static_cast<uint8_t>(blurred_pixel);
        }
    }
}

/**
 * @brief Perform erosion on an image.
 *
 * @param src The input image.
 * @param dst The output eroded image.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius of the erosion.
 */
void erode(const uint8_t* src, uint8_t* dst, int width, int height, int radius = 1) {
    for (int y = radius; y < height - radius; ++y) {
        for (int x = radius; x < width - radius; ++x) {
            uint8_t minPixel = 255;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    uint8_t currentPixel = src[ny * width + nx];
                    if (currentPixel < minPixel) {
                        minPixel = currentPixel;
                    }
                }
            }
            dst[y * width + x] = minPixel;
        }
    }
}

/**
 * @brief Perform dilation on an image.
 *
 * @param src The input image.
 * @param dst The output dilated image.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius of the dilation.
 */
void dilate(const uint8_t* src, uint8_t* dst, int width, int height, int radius = 1) {
    for (int y = radius; y < height - radius; ++y) {
        for (int x = radius; x < width - radius; ++x) {
            uint8_t maxPixel = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    uint8_t currentPixel = src[ny * width + nx];
                    if (currentPixel > maxPixel) {
                        maxPixel = currentPixel;
                    }
                }
            }
            dst[y * width + x] = maxPixel;
        }
    }
}

/**
 * @brief Perform morphological opening (erosion followed by dilation) on an image.
 *
 * @param src The input image.
 * @param dst The output image.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius of the structuring element.
 */
void opening(const uint8_t* src, uint8_t* dst, int width, int height, int radius = 1) {
    std::vector<uint8_t> temp(width * height);
    erode(src, temp.data(), width, height, radius);
    dilate(temp.data(), dst, width, height, radius);
}

/**
 * @brief Perform morphological closing (dilation followed by erosion) on an image.
 *
 * @param src The input image.
 * @param dst The output image.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius of the structuring element.
 */
void closing(const uint8_t* src, uint8_t* dst, int width, int height, int radius = 1) {
    std::vector<uint8_t> temp(width * height);
    dilate(src, temp.data(), width, height, radius);
    erode(temp.data(), dst, width, height, radius);
}

/**
 * @brief Perform a series of morphological operations (erosion, dilation, opening, and closing) on an image.
 *
 * @param src The input image.
 * @param dst The output image.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius of the structuring element.
 */
void morphologicalOperations(const uint8_t* src, uint8_t* dst, int width, int height, int radius = 1) {
    std::vector<uint8_t> temp1(width * height);
    std::vector<uint8_t> temp2(width * height);

    erode(src, temp1.data(), width, height, radius);
    dilate(temp1.data(), temp2.data(), width, height, radius);
    opening(temp2.data(), temp1.data(), width, height, radius);
    closing(temp1.data(), dst, width, height, radius);
}

/**
 * @brief Apply hysteresis thresholding to an image.
 *
 * @param src The input image.
 * @param dst The output image.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param minThreshold The lower threshold.
 * @param maxThreshold The upper threshold.
 */
void hysteresis_threshold(const uint8_t* src, uint8_t* dst, int width, int height, int minThreshold, int maxThreshold) {
    std::fill(dst, dst + width * height, 0);
    std::vector<bool> strong_edges(width * height, false);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            uint8_t pixel = src[idx];
            if (pixel > maxThreshold) {
                strong_edges[idx] = true;
                dst[idx] = 255;
            }
        }
    }

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            if (strong_edges[idx]) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ny = y + dy;
                        int nx = x + dx;
                        int n_idx = ny * width + nx;
                        uint8_t neighbor_pixel = src[n_idx];
                        if (neighbor_pixel >= minThreshold && !strong_edges[n_idx]) {
                            strong_edges[n_idx] = true;
                            dst[n_idx] = 255;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Apply a mask to an RGB image, highlighting masked areas in red.
 *
 * @param input The input RGB image.
 * @param mask The mask image.
 * @param output The output image.
 * @param width The width of the image.
 * @param height The height of the image.
 */
void put_mask(const rgb* input, const uint8_t* mask, rgb* output, int width, int height) {
    rgb red = {255, 0, 0};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            rgb pixel = input[idx];
            uint8_t pixel_mask = mask[idx];

            rgb pixel_output;
            if (pixel_mask > 0) {
                pixel_output.r = static_cast<uint8_t>(std::min(255.0, pixel.r * 0.5 + red.r * 0.5));
                pixel_output.g = static_cast<uint8_t>(std::min(255.0, pixel.g * 0.5 + red.g * 0.5));
                pixel_output.b = static_cast<uint8_t>(std::min(255.0, pixel.b * 0.5 + red.b * 0.5));
            } else {
                pixel_output = pixel;
            }

            output[idx] = pixel_output;
        }
    }
}

/**
 * @brief Compute the mean background from a series of frames.
 *
 * @param frames The input frames.
 * @param mean_frame The output mean frame.
 * @param width The width of the frames.
 * @param height The height of the frames.
 */
void computeMeanBackground(const std::deque<std::vector<rgb>>& frames, rgb* mean_frame, int width, int height) {
    size_t num_frames = frames.size();
    std::vector<double> r_sum(width * height, 0.0);
    std::vector<double> g_sum(width * height, 0.0);
    std::vector<double> b_sum(width * height, 0.0);

    for (const auto& frame : frames) {
        for (int i = 0; i < width * height; ++i) {
            r_sum[i] += frame[i].r;
            g_sum[i] += frame[i].g;
            b_sum[i] += frame[i].b;
        }
    }

    for (int i = 0; i < width * height; ++i) {
        mean_frame[i].r = static_cast<uint8_t>(r_sum[i] / num_frames);
        mean_frame[i].g = static_cast<uint8_t>(g_sum[i] / num_frames);
        mean_frame[i].b = static_cast<uint8_t>(b_sum[i] / num_frames);
    }
}

/**
 * @brief Main filtering function that processes the input buffer to detect and highlight changes.
 *
 * @param buffer The input buffer containing the image data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param stride The stride of the image.
 * @param pixel_stride The pixel stride of the image.
 */
extern "C" {
void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    static int frame_count = 0;
    frame_count++;

    std::vector<rgb> current_frame(width * height);
    for (int y = 0; y < height; ++y) {
        memcpy(current_frame.data() + y * width, buffer + y * stride, width * sizeof(rgb));
    }

    if (!bg_model.is_initialized) {
        bg_model.width = width;
        bg_model.height = height;
        bg_model.stride = stride;
        bg_model.is_initialized = true;
    }

    bg_model.frames.push_back(current_frame);
    if (bg_model.frames.size() > 20) {
        bg_model.frames.pop_front();
    }

    std::vector<rgb> mean_frame(width * height);
    computeMeanBackground(bg_model.frames, mean_frame.data(), width, height);

    lab* bg_lab = new lab[width * height];
    lab* frame_lab = new lab[width * height];

    RGBtoLAB(mean_frame.data(), bg_lab, width * height);
    RGBtoLAB(current_frame.data(), frame_lab, width * height);

    double* residual = new double[width * height];
    uint8_t* normalized_residual = new uint8_t[width * height];

    ComputeAndNormalizeResidual(bg_lab, frame_lab, residual, normalized_residual, width * height);

    uint8_t* blurred = new uint8_t[width * height];
    gaussianBlur(normalized_residual, blurred, width, height);

    uint8_t* morphologically_processed = new uint8_t[width * height];
    morphologicalOperations(blurred, morphologically_processed, width, height, 1);

    uint8_t* hysteresis = new uint8_t[width * height];
    hysteresis_threshold(morphologically_processed, hysteresis, width, height, 40, 100);

    rgb* output = new rgb[width * height];
    put_mask(current_frame.data(), hysteresis, output, width, height);

    for (int y = 0; y < height; ++y) {
        memcpy(buffer + y * stride, output + y * width, width * sizeof(rgb));
    }

    delete[] bg_lab;
    delete[] frame_lab;
    delete[] residual;
    delete[] normalized_residual;
    delete[] blurred;
    delete[] hysteresis;
    delete[] morphologically_processed;
    delete[] output;
}
}