mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/speech/users/ysw/workspace/libtorch/libtorch ..
cmake --build . --config Release
