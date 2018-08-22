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
#include "io/decrypter.h"
#include <common/log.h>
#include <model-decrypt/include/model_crypt.h>
#include <model-decrypt/include/model_decrypt.h>
#include <string>

namespace paddle_mobile {

size_t decrypter::ReadBufferCrypt(const char *file_path,
                                  uint8_t **decrypt_output, const char *key) {
  std::string str = file_path;
  DLOG << "ReadBufferCrypt..." << str;
  DLOG << "ReadBufferCrypt...key" << std::string(key);
  DLOG << "ReadBufferCrypt...keylen" << strlen(key);
  int ret = 0;
  void *context = nullptr;
  unsigned int sign = 0;
  void *file_map = nullptr;
  unsigned int file_size = 0;
  // unsigned char *decrypt_output = *out;
  unsigned int decrypt_output_size = 0;
  // init decryption context
  ret = init_crypt_context(reinterpret_cast<const unsigned char *>(key),
                           static_cast<unsigned int>(strlen(key)), &context,
                           &sign);
  if (0 != ret) {
    DLOG << "failed to init_crypt_context.";
    return static_cast<size_t>(0);
  }
  DLOG << "init_crypt_context succeed.";

  // create file mapping for encrypted model file
  ret = open_file_map(file_path, &file_map, &file_size);
  if (0 != ret) {
    char buf[128];
    snprintf(buf, sizeof(buf), "failed to open_file_map for file %s",
             file_path);
    DLOG << "failed to open_file_map.\n";
    return static_cast<size_t>(0);
  }
  DLOG << "open_file_map succeed.  file_size: " << file_size;

  // decrypt
  ret = model_decrypt(context, sign, (unsigned char *)file_map, file_size,
                      decrypt_output, &decrypt_output_size);
  if (0 != ret) {
    printf("failed to model_decrypt.\n");
    DLOG << "failed to model_decrypt.  ret:" << ret;
    return static_cast<size_t>(0);
  }
  DLOG << "decrypt_output_size: " << decrypt_output_size;

  // release output
  //    free(decrypt_output);
  //    decrypt_output = NULL;

  // release file mapping
  close_file_map(file_map, file_size);
  file_map = nullptr;

  // release decryption context
  uninit_crypt_context(context);
  context = nullptr;

  return decrypt_output_size;
}
}  // namespace paddle_mobile
