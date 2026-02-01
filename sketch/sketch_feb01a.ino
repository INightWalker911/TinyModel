#include <TensorFlowLite_ESP32.h> // Важно: эта библиотека должна быть первой!

#include <esp_camera.h>
#include <WiFi.h>
#include <esp_http_server.h>
#include "model_data.h" 

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ==========================================
// 1. НАСТРОЙКИ
// ==========================================
const char* ssid = "ВАШ_WIFI";
const char* password = "ВАШ_ПАРОЛЬ";

const char* CLASSES[] = {
  "Missing_hole",    // 0
  "Mouse_bite",      // 1
  "Open_circuit",    // 2
  "Short",           // 3
  "Spur",            // 4
  "Spurious_copper"  // 5
};
const int NUM_CLASSES = 6;

// Цвета (RGB565)
uint16_t CLASS_COLORS[] = {
  0xF800, 0x07E0, 0x001F, 0xFFE0, 0xF81F, 0x07FF
};

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

const int kImageWidth = 96;
const int kImageHeight = 96;
const int kTensorArenaSize = 160 * 1024; 
uint8_t* tensor_arena;

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ==========================================
// 2. ФУНКЦИИ
// ==========================================

void draw_box(camera_fb_t *fb, uint16_t color) {
  int x_start = (320 - 96) / 2;
  int y_start = (240 - 96) / 2;
  int box_size = 96;
  
  // Рисуем простую рамку (толщина 2 пикселя)
  for (int x = x_start; x < x_start + box_size; x++) {
      for (int t = 0; t < 2; t++) {
          int idx_top = ((y_start + t) * 320 + x) * 2;
          int idx_bot = ((y_start + box_size - 1 - t) * 320 + x) * 2;
          if (idx_top < fb->len) { fb->buf[idx_top] = color >> 8; fb->buf[idx_top + 1] = color & 0xFF; }
          if (idx_bot < fb->len) { fb->buf[idx_bot] = color >> 8; fb->buf[idx_bot + 1] = color & 0xFF; }
      }
  }
  for (int y = y_start; y < y_start + box_size; y++) {
      for (int t = 0; t < 2; t++) {
          int idx_left = (y * 320 + x_start + t) * 2;
          int idx_right = (y * 320 + x_start + box_size - 1 - t) * 2;
          if (idx_left < fb->len) { fb->buf[idx_left] = color >> 8; fb->buf[idx_left + 1] = color & 0xFF; }
          if (idx_right < fb->len) { fb->buf[idx_right] = color >> 8; fb->buf[idx_right + 1] = color & 0xFF; }
      }
  }
}

int run_inference(camera_fb_t *fb, float &confidence) {
    int start_x = (320 - kImageWidth) / 2;
    int start_y = (240 - kImageHeight) / 2;
    int index = 0;

    for (int y = 0; y < kImageHeight; y++) {
        for (int x = 0; x < kImageWidth; x++) {
            int fb_idx = ((start_y + y) * 320 + (start_x + x)) * 2;
            uint16_t pixel = (fb->buf[fb_idx] << 8) | fb->buf[fb_idx + 1];
            uint8_t r = (pixel & 0xF800) >> 8;
            uint8_t g = (pixel & 0x07E0) >> 3;
            uint8_t b = (pixel & 0x001F) << 3;
            
            // Расширяем диапазон 5bit -> 8bit
            input->data.uint8[index++] = (r * 255) / 31;
            input->data.uint8[index++] = (g * 255) / 63;
            input->data.uint8[index++] = (b * 255) / 31;
        }
    }

    if (interpreter->Invoke() != kTfLiteOk) return -1;

    int max_score = 0;
    int max_index = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (output->data.uint8[i] > max_score) {
            max_score = output->data.uint8[i];
            max_index = i;
        }
    }
    confidence = max_score / 255.0;
    return max_index;
}

esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];
  static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
  static const char* _STREAM_BOUNDARY = "\r\n--frame\r\n";
  static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) { res = ESP_FAIL; } else {
      float conf = 0;
      int idx = run_inference(fb, conf);
      
      if(idx >= 0) draw_box(fb, CLASS_COLORS[idx % 6]);
      
      if (conf > 0.5) {
        Serial.printf("DETECTED: %s (%.1f%%)\n", CLASSES[idx], conf*100);
      }

      uint8_t * jpg_buf = NULL;
      size_t jpg_len = 0;
      bool jpeg_converted = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
      esp_camera_fb_return(fb);
      
      if (!jpeg_converted) { res = ESP_FAIL; } else {
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        size_t hlen = snprintf(part_buf, 64, _STREAM_PART, jpg_len);
        res = httpd_resp_send_chunk(req, part_buf, hlen);
        res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);
        free(jpg_buf);
      }
    }
    if (res != ESP_OK) break;
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  httpd_handle_t stream_httpd = NULL;
  
  // ИСПРАВЛЕННАЯ СТРОКА ДЛЯ ESP32 v2.x
  httpd_uri_t stream_uri = {
      .uri = "/stream",
      .method = HTTP_GET,
      .handler = stream_handler, // БЫЛО .function, СТАЛО .handler
      .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

void setup() {
  Serial.begin(115200);
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565; 
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if(esp_camera_init(&config) != ESP_OK) {
    Serial.println("Cam Init Failed");
    return;
  }

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nReady! http://" + WiFi.localIP().toString() + "/stream");

  tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) tensor_arena = (uint8_t *)malloc(kTensorArenaSize);

  static tflite::AllOpsResolver resolver;
  const tflite::Model* model = tflite::GetModel(model_data);
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  startCameraServer();
}

void loop() { delay(1000); }


