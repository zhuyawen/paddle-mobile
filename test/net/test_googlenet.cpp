/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include "../test_helper.h"
#include "../test_include.h"

int main() {
#ifdef PADDLE_MOBILE_FPGA
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
#endif

#ifdef PADDLE_MOBILE_CPU
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
#endif
  char encrypted_key[] = {0xfd, 0xa9, 0xfa, 0xbe, 0xf3, 0xed, 0xe0, 0x87, 0xa7,
                          0xa4, 0xab, 0xbb, 0xeb, 0xb8, 0xfb, 0xed, 0xbf, 0xed,
                          0xa4, 0x87, 0x9e, 0x81, 0xf7, 0xa9, 0x87, 0x84, 0xef,
                          0xf8, 0xbd, 0xeb, 0x90, 0xb2, 0xb6, 0xfd, 0xb2, 0x8b,
                          0xb3, 0xa7, 0xf5, 0x95, 0xf5, 0xe5, 0x91, 0xe9, 0xab,
                          0xaf, 0x93, 0x8c, 0xac, 0xbf};
  DLOG << "sizeof(key): " << sizeof(encrypted_key);
  DLOG << "strlen(key): " << strlen(encrypted_key);
  for (int i = 0; i < strlen(encrypted_key); ++i) {
    //  DLOG<<*(encrypted_key+i);
    encrypted_key[i] ^= 0xde;
    DLOG << *(encrypted_key + i);
  }
  paddle_mobile.SetThreadNum(4);
  bool optimize = true;
  auto time1 = time();
  if (paddle_mobile.LoadEncrypt(g_googlenet, optimize, false, 1,
                                encrypted_key)) {
    auto time2 = time();
    std::cout << "load cost :" << time_diff(time1, time2) << "ms" << std::endl;
    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 224, 224};
    GetInput<float>(g_test_image_1x3x224x224, &input, dims);
    // 预热一次
    auto vec_result = paddle_mobile.Predict(input, dims);
    auto time3 = time();
    for (int i = 0; i < 10; ++i) {
      auto vec_result = paddle_mobile.Predict(input, dims);
    }
    auto time4 = time();

    std::cout << "predict cost :" << time_diff(time3, time4) / 10 << "ms"
              << std::endl;
  }
  return 0;
}
