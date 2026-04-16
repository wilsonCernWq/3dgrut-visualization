/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "renderer_core.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "args.hxx"

int main(int argc, char *argv[]) {
  args::ArgumentParser parser("Render a 3DGS .ply scene to a PNG image using ANARI.");
  args::HelpFlag        help        (parser, "help",      "Show this help message and exit",                                    {'h', "help"});
  args::Positional<std::string> plyArg(parser, "path.ply", "Path to input .ply file");
  args::ValueFlag<std::string>  libraryArg   (parser, "NAME",  "ANARI library to load (default: visrtx)",                    {"library"});
  args::ValueFlag<std::string>  outputArg    (parser, "FILE",  "Output PNG path (default: gaussian_viewer.png)",              {"output"});
  args::ValueFlag<std::string>  resolutionArg(parser, "WxH",   "Image resolution (default: 3840x2160)",                      {"resolution"});
  args::ValueFlag<int>          sppArg       (parser, "N",     "Samples per pixel (default: 128)",                           {"spp"});
  args::ValueFlag<float>        scaleArg     (parser, "F",     "Global Gaussian scale multiplier (default: 1.0)",             {"scale-factor"});
  args::ValueFlag<float>        opacityArg   (parser, "T",     "Discard Gaussians below this opacity (default: 0.05)",       {"opacity-threshold"});
  args::ValueFlag<float>        ambientArg   (parser, "F",     "Ambient light intensity (default: 1.0)",                     {"ambient-radiance"});
  args::ValueFlag<std::string>  bgColorArg   (parser, "R,G,B", "Background colour, floats 0-1 (default: 0.1,0.1,0.1)",      {"bg-color"});
  args::Flag                    float32Arg   (parser, "float32", "Use 32-bit float framebuffer instead of uint8",            {"float32"});
  args::Flag                    noSrgbArg    (parser, "no-srgb", "Use linear uint8 output instead of sRGB (ignored with --float32)", {"no-srgb"});

  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help &) {
    std::cout << parser;
    return 0;
  } catch (const args::ParseError &e) {
    std::cerr << "Error: " << e.what() << "\n\n" << parser;
    return 1;
  }

  if (!plyArg) {
    std::cerr << "Error: missing required argument <path.ply>\n\n" << parser;
    return 1;
  }

  const std::string plyPath      = args::get(plyArg);
  const std::string libraryName  = libraryArg  ? args::get(libraryArg)  : "visrtx";
  const std::string outputPath   = outputArg   ? args::get(outputArg)   : "gaussian_viewer.png";
  const float scaleFactor        = scaleArg    ? args::get(scaleArg)    : 1.0f;
  const float opacityThreshold   = opacityArg  ? args::get(opacityArg)  : 0.05f;
  const float ambientRadiance    = ambientArg  ? args::get(ambientArg)  : 1.0f;
  const int   spp                = sppArg      ? args::get(sppArg)      : 128;
  const bool  useFloat32         = bool(float32Arg);
  const bool  useSRGB            = !bool(noSrgbArg);

  if (spp <= 0) {
    std::cerr << "Error: --spp must be a positive integer\n";
    return 1;
  }

  uvec2 imageSize = {3840, 2160};
  if (resolutionArg) {
    unsigned w = 0, h = 0;
    const std::string &res = args::get(resolutionArg);
    if (std::sscanf(res.c_str(), "%ux%u", &w, &h) != 2 || w == 0 || h == 0) {
      std::cerr << "Error: invalid resolution '" << res << "', expected WxH (e.g. 1920x1080)\n";
      return 1;
    }
    imageSize = {w, h};
  }

  vec3 bgColor = {0.1f, 0.1f, 0.1f};
  if (bgColorArg) {
    float r = 0, g = 0, b = 0;
    const std::string &col = args::get(bgColorArg);
    if (std::sscanf(col.c_str(), "%f,%f,%f", &r, &g, &b) != 3) {
      std::cerr << "Error: invalid --bg-color '" << col << "', expected R,G,B (e.g. 0.1,0.1,0.1)\n";
      return 1;
    }
    bgColor = {r, g, b};
  }

  InitOptions options;
  options.plyPath            = plyPath;
  options.libraryName        = libraryName;
  options.scaleFactor        = scaleFactor;
  options.opacityThreshold   = opacityThreshold;
  options.frameSize          = imageSize;
  options.useFloat32Color    = useFloat32;
  options.useSRGB            = useSRGB;
  options.rendererConfig.spp             = spp;
  options.rendererConfig.ambientRadiance = ambientRadiance;
  options.rendererConfig.bgColor         = {bgColor[0], bgColor[1], bgColor[2], 1.f};

  GaussianRendererCore renderer;
  std::string errorMessage;
  if (!renderer.init(options, &errorMessage)) {
    fprintf(stderr, "Renderer init failed: %s\n", errorMessage.c_str());
    return 1;
  }

  const auto &center = renderer.sceneCenter();
  printf("Scene center: (%.3f, %.3f, %.3f)  diagonal: %.3f\n", center[0], center[1], center[2], renderer.sceneDiagonal());
  printf("Scale factor: %.3f\n", scaleFactor);
  printf("Rendering %zu Gaussians...\n", renderer.gaussianCount());

  if (!renderer.run(&errorMessage)) {
    fprintf(stderr, "Render failed: %s\n", errorMessage.c_str());
    return 1;
  }

  printf("Rendered in %.2f ms\n", renderer.lastDurationSeconds() * 1000.f);

  stbi_flip_vertically_on_write(1);
  auto fb = renderer.mapColorHost(&errorMessage);
  if (!fb.data) {
    fprintf(stderr, "Framebuffer map failed: %s\n", errorMessage.c_str());
    return 1;
  }
  stbi_write_png(outputPath.c_str(), fb.width, fb.height, 4, fb.data, 4 * fb.width);
  renderer.unmapColorHost();
  printf("Saved: %s\n", outputPath.c_str());

  return 0;
}
