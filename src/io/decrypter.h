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

//
// Created by 谢柏渊 on 2018/8/21.
//
#pragma once
#ifdef ENABLE_CRYPT

#include <cstddef>
#include <cstdint>

namespace paddle_mobile {
class decrypter {
 public:
  /**
   * load and decrypt file to buf
   *
   * @param file_path filepath
   * @param decrypt_output output buf
   * @param key  key
   * @return
   */
  size_t ReadBufferCrypt(const char *file_path, uint8_t **decrypt_output,
                         const char *key);

 protected:
 private:
};

}  // namespace paddle_mobile
#endif
