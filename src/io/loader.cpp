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

#include "io/loader.h"
#include <model-decrypt/include/model_crypt.h>
#include <cstring>

#include "framework/lod_tensor.h"
#include "framework/program/program-optimize/program_optimize.h"
#ifdef ENABLE_CRYPT

#include "model-decrypt/include/model_decrypt.h"

static size_t ReadBufferCrypt(const char *file_path, uint8_t **decrypt_output,
                              const char *key) {
  std::string str = file_path;
  std::cout << "ReadBufferCrypt..." << str;
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
    return static_cast<size_t>(-1);
  }
  DLOG << "init_crypt_context succeed.";

  // create file mapping for encrypted model file
  ret = open_file_map(file_path, &file_map, &file_size);
  if (0 != ret) {
    char buf[128];
    snprintf(buf, sizeof(buf), "failed to open_file_map for file %s",
             file_path);
    DLOG << "failed to open_file_map.\n";
    return static_cast<size_t>(-1);
  }
  std::cout << "open_file_map succeed.  file_size: " << file_size;

  // decrypt
  ret = model_decrypt(context, sign, (unsigned char *)file_map, file_size,
                      decrypt_output, &decrypt_output_size);
  if (0 != ret) {
    printf("failed to model_decrypt.\n");
    DLOG << "failed to model_decrypt.  ret:" << ret;
    return static_cast<size_t>(-1);
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

#endif

namespace paddle_mobile {
using framework::Variable;

static size_t ReadBuffer(const char *file_name, uint8_t **out) {
  FILE *fp;
  fp = fopen(file_name, "rb");
  PADDLE_MOBILE_ENFORCE(fp != NULL, " %s open failed !", file_name);

  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  rewind(fp);

  DLOG << "model size: " << size;

  *out = reinterpret_cast<uint8_t *>(malloc(size));

  size_t cur_len = 0;
  size_t nread;
  while ((nread = fread(*out + cur_len, 1, size - cur_len, fp)) != 0) {
    cur_len += nread;
  }
  fclose(fp);
  return cur_len;
}

template <typename Dtype, Precision P>
const framework::Program<Dtype, P> Loader<Dtype, P>::Load(
    const std::string &dirname, bool optimize, bool quantification,
    bool can_add_split, const char *key) {
#ifdef ENABLE_CRYPT
  if (key && *key == 0) {
    auto program = this->LoadProgram(dirname + "/__model__", optimize,
                                     quantification, can_add_split);
    program.model_path = dirname;
    return program;
  } else {
    auto program = this->LoadProgram(dirname + "/model.mlm", optimize,
                                     quantification, can_add_split, key);
    program.model_path = dirname;
    return program;
  }

#else
  auto program = this->LoadProgram(dirname + "/__model__", optimize,
                                   quantification, can_add_split);
  program.model_path = dirname;
  return program;
#endif
}

template <typename Dtype, Precision P>
const framework::Program<Dtype, P> Loader<Dtype, P>::Load(
    const std::string &model_path, const std::string &para_path, bool optimize,
    bool quantification, const char *key) {
  auto program =
      this->LoadProgram(model_path, optimize, quantification, false, key);

  program.para_path = para_path;
  program.combined = true;
  program.quantification = quantification;
  return program;
}

template <typename Dtype, Precision P>
const framework::Program<Dtype, P> Loader<Dtype, P>::LoadProgram(
    const std::string &model_path, bool optimize, bool quantification,
    bool can_add_split, const char *key) {
  std::string model_filename = model_path;
  PaddleMobile__Framework__Proto__ProgramDesc *c_program;
  uint8_t *buf = nullptr;

  size_t read_size;
#ifdef ENABLE_CRYPT
  if (key && *key == 0) {
    read_size = ReadBuffer(model_filename.c_str(), &buf);
  } else {
    unsigned int decrypt_output_size = 0;
    read_size = ReadBufferCrypt(model_filename.c_str(), &buf, key);
  }

#else
  read_size = ReadBuffer(model_filename.c_str(), &buf);
#endif

  PADDLE_MOBILE_ENFORCE(buf != nullptr, "read from __model__ is null");

  for (int i = 0; i < read_size; ++i) {
    DLOG << "buf -" << i << *(buf + i);
  }

  c_program = paddle_mobile__framework__proto__program_desc__unpack(
      nullptr, read_size, buf);

  DLOG << "read_size: " << read_size;
  DLOG << "buf: " << *buf;
  //
  PADDLE_MOBILE_ENFORCE(c_program != nullptr, "program is null");
  //
  DLOG << "n_ops: " << (*c_program->blocks)->n_ops;
  //
  auto originProgramDesc = std::make_shared<framework::ProgramDesc>(c_program);

  framework::Program<Dtype, P> program;
  program.originProgram = originProgramDesc;
  program.quantification = quantification;

  auto scope = std::make_shared<framework::Scope>();
  program.scope = scope;

  for (const auto &block : originProgramDesc->Blocks()) {
    for (auto var_desc : block->Vars()) {
      auto var = scope->Var(var_desc->Name());

      if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
        if (var_desc->Persistable() &&
            var_desc->Type() != framework::VARTYPE_TYPE_FEED_MINIBATCH &&
            var_desc->Type() != framework::VARTYPE_TYPE_FETCH_LIST) {
          auto dim = var_desc->Tensor_desc().Dims();
          auto tensor = var->GetMutable<framework::LoDTensor>();
          tensor->Resize(framework::make_ddim(dim));
        } else {
          auto dim = var_desc->Tensor_desc().Dims();
          PADDLE_MOBILE_ENFORCE(dim.size() > 0, "dim size is 0");
          dim[0] = 1;
          auto tensor = var->GetMutable<framework::LoDTensor>();
          tensor->Resize(framework::make_ddim(dim));
        }
      } else {
        // TODO(codeWorm): some.
      }
    }
  }

  if (optimize) {
    framework::ProgramOptimize program_optimize;
    program.optimizeProgram =
        program_optimize.FusionOptimize(originProgramDesc, can_add_split);
  }
  if (optimize) {
    program.optimizeProgram->Description("optimize: ");
  } else {
    originProgramDesc->Description("program: ");
  }

  paddle_mobile__framework__proto__program_desc__free_unpacked(c_program, NULL);
  return program;
}

template class Loader<CPU, Precision::FP32>;
template class Loader<FPGA, Precision::FP32>;
template class Loader<GPU_MALI, Precision::FP32>;

}  // namespace paddle_mobile
