This is the NOWLAB fork of upstream PyTorch. The key features we've added are:
1. Support for building PyTorch's distributed module with any \[CUDA/ROCm\]-aware MPI library including NOWLAB's [MVAPICH2-GDR](http://mvapich.cse.ohio-state.edu/userguide/gdr/)
2. Support for MPI + fp16 communication operations