import os
import sys

sys.path.append('/home/seung/Workspace/papers/2022/clora/bop_renderer/build')
os.environ['LD_LIBRARY_PATH'] = "/opt/llvm/lib:/opt/osmesa/lib"

# C++ renderer (https://github.com/thodan/bop_renderer)
# sys.path.append(config.bop_renderer_path)
import bop_renderer
