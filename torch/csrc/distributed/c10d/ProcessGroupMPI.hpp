#pragma once

#ifdef USE_C10D_MPI

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_map>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
// Profiling support: PARAM_COMMS_DATA
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>

#ifdef USE_MPIX_STREAM
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#endif

#include <mpi.h>

namespace c10d {

constexpr const char* MPI_BACKEND_NAME = "mpi";

// WorkEntry is the state associated with a single MPI run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
  explicit WorkEntry(
      std::vector<at::Tensor>* srcPtr,
      std::vector<at::Tensor>* dstPtr,
      std::function<void(std::unique_ptr<WorkEntry>&)> run)
      : dst(dstPtr ? *dstPtr : std::vector<at::Tensor>()), run(std::move(run)) {
    if (srcPtr) {
      src = *srcPtr;
    }
  }

  // Not copyable
  WorkEntry(const WorkEntry&) = delete;
  // Not copy assignable
  WorkEntry& operator=(const WorkEntry&) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor> src;

  // Copy of user provided outputs.
  const std::vector<at::Tensor> dst;

  // src rank returned, for recv only
  int* srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry>&)> run;
};

// ProcessGroupMPI implements MPI bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes.
//
// All MPI functions provided by this class is asynchronously scheduled on a
// Worker thread. Therefore, ProcessGroupMPI requires the MPI implementation
// that is used to have a minimum thread support value of MPI_THREAD_SERIALIZED.
// That is, The process may be multi-threaded, and multiple threads may make
// MPI calls, but only one at a time: MPI calls are not made concurrently from
// two distinct threads (all MPI calls are serialized). However, with
// MPI_THREAD_SERIALIZED, ProcessGroupMPI will only support a singe process
// group. In other words, no more than 1 process group can be created globally.
//
// If you would like to use multiple ProcessGroupMPI, it requires your MPI
// implementation to have a thread support value of MPI_THREAD_MULTIPLE, that
// is, multiple threads may call MPI, with no restriction.
//
// Also note that ProcessGroupMPI only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//
// CUDA tensor can be supported if the MPI used is CUDA-aware MPI, and
// ProcessGroupMPI will automatically detect this support.
class TORCH_API ProcessGroupMPI : public Backend {
 public:
  class WorkMPI : public Work {
   public:
    explicit WorkMPI(
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt)
        : Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
          outputTensors_(std::move(outputTensors)),
          future_(c10::make_intrusive<at::ivalue::Future>(
              c10::ListType::create(c10::TensorType::get()))) {}

    std::vector<at::Tensor> result() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   protected:
    friend class ProcessGroupMPI;

   private:
    void finishWorkMPI();
    void finishWorkMPIError(const std::exception_ptr& eptr);

    std::vector<at::Tensor> outputTensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };

  class AsyncWork : public Work {
   public:
    AsyncWork(
        MPI_Request request,
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt);

    ~AsyncWork() override;

    bool isCompleted() override;

    bool isSuccess() const override;

    int sourceRank() const override;

    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

    void abort() override;

    std::vector<at::Tensor> result() override;

   protected:
    void populateException();

   private:
    const std::vector<at::Tensor> outputTensors_;
    MPI_Request request_;
    MPI_Status status_{};
  };

#ifdef USE_MPIX_STREAM
  class MPIXStreamWork : public Work {
   public:
    explicit MPIXStreamWork(
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors = std::nullopt,
        bool enableTiming = false);

    ~MPIXStreamWork() override;

    bool isCompleted() override;

    bool isSuccess() const override;

    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

    void abort() override;

    
    void synchronize() override;

    
    std::vector<at::Tensor> result() override;

    
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    
    float getDuration() const;

   private:
    
    bool finishedGPUExecutionInternal() const;

    
    void setException(std::exception_ptr exception_ptr);

    
    std::vector<at::Tensor> outputTensors_;

    // for comms duration capture
    std::shared_ptr<at::cuda::CUDAEvent> startEvent_;
    std::shared_ptr<at::cuda::CUDAEvent> endEvent_;

    
    at::Device device_;

    
    at::cuda::CUDAStream cudaStream_;

    
    c10::intrusive_ptr<at::ivalue::Future> future_;

    
    std::exception_ptr exception_;
    std::mutex mutex_;

    
    bool timingEnabled_;

    friend class ProcessGroupMPI;
  };
#endif // USE_MPIX_STREAM

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupMPI(int rank, int size, MPI_Comm pgComm);

  ~ProcessGroupMPI() override;

  // Abort the MPI program, needs to be called when exception is detected
  void abort() override;

  const std::string getBackendName() const override {
    return std::string(MPI_BACKEND_NAME);
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  // Creating a new ProcessGroupMPI, will initialize MPI if not initialized
  static c10::intrusive_ptr<ProcessGroupMPI> createProcessGroupMPI(
      std::vector<int> ranks = {});

 protected:
  using WorkType =
      std::tuple<std::unique_ptr<WorkEntry>, c10::intrusive_ptr<WorkMPI>>;
  // Worker thread loop
  void runLoop();
  // Helper function that is called by the destructor
  void destroy();

  c10::intrusive_ptr<Work> enqueue(
      std::unique_ptr<WorkEntry> entry,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors =
          std::nullopt);

  bool stop_;

  std::mutex pgMutex_;
  std::thread workerThread_;

  std::deque<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  // Global states
  static void initMPIOnce();
  static void mpiExit();

  static std::mutex pgGlobalMutex_;
  static int mpiThreadSupport_;

  MPI_Comm pgComm_;
// add USE_MPIX_STREAM guard
#ifdef USE_MPIX_STREAM
    MPI_Comm mpixStreamComm_;
    MPIX_Stream mpixStream_;
    
    // only one stream per MPIX PG 
    // TODO: multiple device-stream per PG, might need stream map impl
    at::cuda::CUDAStream mpixCudaStream_;
    
    
    bool hasMPIXStream() const { return mpixStreamComm_ != MPI_COMM_NULL; }
    at::cuda::CUDAStream& getMPIXCudaStream() { return mpixCudaStream_; }
    MPI_Comm getMPIXStreamComm() const { return mpixStreamComm_; }
    
    
    void enableCollectivesTiming() { enableTiming_ = true; }
    
    
    c10::intrusive_ptr<MPIXStreamWork> createMPIXWork(
        std::vector<at::Tensor>& tensors,
        const char* profilingTitle = nullptr,
        bool enableTiming = false);
    
    
    bool enableTiming_ = false;
#endif
};

} // namespace c10d

#endif // USE_C10D_MPI
