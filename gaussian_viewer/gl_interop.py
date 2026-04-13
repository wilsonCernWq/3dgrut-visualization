# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

logger = logging.getLogger(__name__)


def set_custom_cugl_bindings():
    import polyscope as ps

    try:
        from . import _gaussian_renderer_core as gl_ext  # type: ignore
    except ImportError:
        logger.warning("Failed to load _gaussian_renderer_core. Ensure it was built with the package.")
        return

    if not hasattr(gl_ext, "cuda"):
        logger.warning("CUDA-GL interop was not compiled into _gaussian_renderer_core.")
        return

    ps_device_func_dict = {
        "map_resource_and_get_array": lambda handle: gl_ext.cuda.map_graphics_resource_array(handle),
        "unmap_resource": lambda handle: gl_ext.cuda.unmap_graphics_resource(handle),
        "register_gl_buffer": lambda native_id: gl_ext.cuda.register_gl_buffer(native_id),
        "register_gl_image_2d": lambda native_id: gl_ext.cuda.register_gl_texture(native_id),
        "unregister_resource": lambda handle: gl_ext.cuda.unregister_cuda_resource(handle),
        "get_array_ptr": lambda input_array: (input_array.data_ptr(), None, None, None),
        "memcpy_2d": lambda dst_ptr, src_ptr, width, height: gl_ext.cuda.memcpy_2d_to_array_async(
            dst_ptr, 0, 0, src_ptr, width, width, height
        ),
    }
    ps.set_device_interop_funcs(ps_device_func_dict)


def initialize_cugl_interop():
    try:
        import polyscope

        try:
            import cuda
            import cupy

            logger.info("polyscope loaded with cupy, cuda-python for cu-opengl interop.")
            # polyscope is available with cupy / cuda-python, do nothing
        except ImportError:
            set_custom_cugl_bindings()  # polyscope is available without cupy / cuda-python, use custom bindings
            logger.info("polyscope loaded with custom cu-opengl interop bindings.")
    except ImportError:
        # polyscope is unavailable
        logger.warning("polyscope unavailable, running in headless mode.")
