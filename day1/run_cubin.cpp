#include <cuda.h>
#include <iostream>

int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    cuModuleLoad(&module, "vector_add.cubin");
    cuModuleGetFunction(&kernel, module, "vector_add");

    std::cout << "Kernel loaded successfully!" << std::endl;

    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}


