// Auto-generated ESP32-P4 Vehicle Classifier
// Variant: pico
// Quantization: qat
// Generated: 2025-10-24 00:16:19

#include "dl_model_base.hpp"
#include "dl_image_jpeg.hpp"
#include "dl_image_process.hpp"
#include <cmath>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Model binary
extern const uint8_t vehicle_classifier_espdl[] asm("_binary_vehicle_classifier_espdl_start");

// Test images
extern const uint8_t not_vehicle_0_jpg_start[] asm("_binary_not_vehicle_0_jpg_start");
extern const uint8_t not_vehicle_0_jpg_end[] asm("_binary_not_vehicle_0_jpg_end");
extern const uint8_t not_vehicle_1_jpg_start[] asm("_binary_not_vehicle_1_jpg_start");
extern const uint8_t not_vehicle_1_jpg_end[] asm("_binary_not_vehicle_1_jpg_end");
extern const uint8_t vehicle_0_jpg_start[] asm("_binary_vehicle_0_jpg_start");
extern const uint8_t vehicle_0_jpg_end[] asm("_binary_vehicle_0_jpg_end");
extern const uint8_t vehicle_1_jpg_start[] asm("_binary_vehicle_1_jpg_start");
extern const uint8_t vehicle_1_jpg_end[] asm("_binary_vehicle_1_jpg_end");

#define IMG_HEIGHT 96
#define IMG_WIDTH 96
#define IMG_CHANNELS 3
#define NUM_WARMUP 2
#define NUM_ITERATIONS 10

const float MEAN[3] = {0.485f, 0.456f, 0.406f};
const float STD[3] = {0.229f, 0.224f, 0.225f};

void preprocess_image(dl::image::img_t &img, int8_t *output, int exponent)
{
    uint8_t *rgb_data = (uint8_t *)img.data;
    
    for (int i = 0; i < IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS; i++) {
        int channel = i % IMG_CHANNELS;
        float pixel_value = (float)rgb_data[i];
        float normalized = (pixel_value / 255.0f - MEAN[channel]) / STD[channel];
        output[i] = dl::quantize<int8_t>(normalized, DL_RESCALE(exponent));
    }
}

void test_single_image(dl::Model *model, const uint8_t *jpg_start, const uint8_t *jpg_end, 
                       const char *image_name, int expected_class)
{
    printf("\n=== Testing: %s ===\n", image_name);
    
    // Decode JPEG
    dl::image::jpeg_img_t jpeg_img = {
        .data = (void *)jpg_start, 
        .data_len = (size_t)(jpg_end - jpg_start)
    };
    
    auto decoded_img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    
    // Get channel count from pix_type
    int channels = dl::image::get_pix_channel_num(decoded_img.pix_type);
    printf("Decoded image: %dx%dx%d\n", decoded_img.width, decoded_img.height, channels);
    
    // Resize using ImageTransformer if needed
    dl::image::img_t resized_img;
    if (decoded_img.width != IMG_WIDTH || decoded_img.height != IMG_HEIGHT) {
        printf("Resizing to %dx%d...\n", IMG_WIDTH, IMG_HEIGHT);
        
        // Allocate output buffer
        resized_img.width = IMG_WIDTH;
        resized_img.height = IMG_HEIGHT;
        resized_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
        resized_img.data = heap_caps_malloc(dl::image::get_img_byte_size(resized_img), MALLOC_CAP_DEFAULT);
        
        // Use ImageTransformer for resizing
        dl::image::ImageTransformer transformer;
        transformer.set_src_img(decoded_img)
                   .set_dst_img(resized_img)
                   .transform();
        
        heap_caps_free(decoded_img.data);
    } else {
        resized_img = decoded_img;
    }
    
    // Get model input
    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_output = model_outputs.begin()->second;
    
    // Preprocess and quantize
    int8_t *input_ptr = (int8_t *)model_input->data;
    preprocess_image(resized_img, input_ptr, model_input->exponent);
    
    // Run inference
    TickType_t start_tick = xTaskGetTickCount();
    model->run();
    TickType_t end_tick = xTaskGetTickCount();
    uint32_t latency_ms = (end_tick - start_tick) * portTICK_PERIOD_MS;
    
    // Get results
    int8_t *output_ptr = (int8_t *)model_output->data;
    float class0_score = dl::dequantize(output_ptr[0], DL_SCALE(model_output->exponent));
    float class1_score = dl::dequantize(output_ptr[1], DL_SCALE(model_output->exponent));
    
    int predicted_class = (class1_score > class0_score) ? 1 : 0;
    const char *class_names[] = {"Not Vehicle", "Vehicle"};
    bool correct = (predicted_class == expected_class);
    
    printf("Inference time: %lu ms\n", (unsigned long)latency_ms);
    printf("Scores: Not Vehicle=%.4f, Vehicle=%.4f\n", class0_score, class1_score);
    printf("Predicted: %s\n", class_names[predicted_class]);
    printf("Expected: %s\n", class_names[expected_class]);
    printf("Result: %s\n", correct ? "CORRECT ✓" : "WRONG ✗");
    
    heap_caps_free(resized_img.data);
}

void benchmark_latency(dl::Model *model)
{
    printf("\n=== Latency Benchmark ===\n");
    
    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_output = model_outputs.begin()->second;
    
    int8_t *input_ptr = (int8_t *)model_input->data;
    int input_size = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS;
    
    srand(12345);
    for (int i = 0; i < input_size; i++) {
        float random_pixel = (float)(rand() % 256);
        int channel = i % IMG_CHANNELS;
        float normalized = (random_pixel / 255.0f - MEAN[channel]) / STD[channel];
        input_ptr[i] = dl::quantize<int8_t>(normalized, DL_RESCALE(model_input->exponent));
    }
    
    printf("Warming up (%d iterations)...\n", NUM_WARMUP);
    for (int i = 0; i < NUM_WARMUP; i++) {
        model->run();
    }
    
    printf("Running benchmark (%d iterations)...\n", NUM_ITERATIONS);
    
    TickType_t start_tick = xTaskGetTickCount();
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        model->run();
    }
    
    TickType_t end_tick = xTaskGetTickCount();
    uint32_t total_time_ms = (end_tick - start_tick) * portTICK_PERIOD_MS;
    
    int8_t *output_ptr = (int8_t *)model_output->data;
    float class0_score = dl::dequantize(output_ptr[0], DL_SCALE(model_output->exponent));
    float class1_score = dl::dequantize(output_ptr[1], DL_SCALE(model_output->exponent));
    
    int predicted_class = (class1_score > class0_score) ? 1 : 0;
    const char *class_names[] = {"Not Vehicle", "Vehicle"};
    
    float avg_latency_ms = (float)total_time_ms / (float)NUM_ITERATIONS;
    float fps = 1000.0f / avg_latency_ms;
    
    printf("\n=== Benchmark Results ===\n");
    printf("Total time: %lu ms (%d iterations)\n", (unsigned long)total_time_ms, NUM_ITERATIONS);
    printf("Average latency: %.2f ms per frame\n", avg_latency_ms);
    printf("Throughput: %.2f FPS\n", fps);
    printf("\nSample prediction:\n");
    printf("  Class 0 (Not Vehicle): %.4f\n", class0_score);
    printf("  Class 1 (Vehicle): %.4f\n", class1_score);
    printf("  Predicted: %s\n", class_names[predicted_class]);
}

void run_vehicle_classifier()
{
    printf("\n");
    printf("======================================================================\n");
    printf("  ESP32-P4 Vehicle Classifier\n");
    printf("======================================================================\n");
    printf("Variant: pico\n");
    printf("Quantization: qat\n");
    printf("Model: MobileNetV2 INT8\n");
    printf("Input size: %dx%dx%d\n", IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS);
    printf("Model size: 2.54 MB\n");
    printf("Test images: 4\n");
    printf("======================================================================\n");
    
    dl::Model *model = new dl::Model((const char *)vehicle_classifier_espdl, 
                                     fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    
    printf("\n=== Model Info ===\n");
    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_output = model_outputs.begin()->second;
    
    printf("Input shape: [");
    for (size_t i = 0; i < model_input->shape.size(); i++) {
        printf("%d%s", model_input->shape[i], i < model_input->shape.size()-1 ? ", " : "");
    }
    printf("]\n");
    
    printf("Output shape: [");
    for (size_t i = 0; i < model_output->shape.size(); i++) {
        printf("%d%s", model_output->shape[i], i < model_output->shape.size()-1 ? ", " : "");
    }
    printf("]\n");
    
    // Auto-generated test cases
    test_single_image(model, not_vehicle_0_jpg_start, not_vehicle_0_jpg_end, "not_vehicle_0.jpg", 0);
    test_single_image(model, not_vehicle_1_jpg_start, not_vehicle_1_jpg_end, "not_vehicle_1.jpg", 0);
    test_single_image(model, vehicle_0_jpg_start, vehicle_0_jpg_end, "vehicle_0.jpg", 1);
    test_single_image(model, vehicle_1_jpg_start, vehicle_1_jpg_end, "vehicle_1.jpg", 1);
    
    benchmark_latency(model);
    
    delete model;
    printf("\n=== Test Complete ===\n");
}

extern "C" void app_main(void)
{
    run_vehicle_classifier();
}