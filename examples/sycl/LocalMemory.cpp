#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename ValueType, size_t LocalSize>
void run(size_t n, size_t compute_iters)
{
  sycl::queue q(gpu_selector_v);

  // Launch the computation
  event e = q.submit([&](handler& h) {
    local_accessor<ValueType, 1> local_acc1{LocalSize, h};

    sycl::nd_range<1> ndr{n, LocalSize};
    h.parallel_for<class LocalMemory>(ndr, [=](sycl::nd_item<1> item) {
      int gid = item.get_global_id(0);
      int lid = item.get_local_id(0);
      ValueType r0;

      if (gid < n) {
        for (int i = 0; i < compute_iters; i++) {
          r0 = local_acc1[lid];
          local_acc1[LocalSize - lid - 1] = r0;
        }
      }
    });
  });
  e.wait_and_throw();
  uint64_t begin = e.get_profiling_info<info::event_profiling::command_start>(), end = e.get_profiling_info<info::event_profiling::command_end>();
  std::cout << "runtime: " << (end - begin) * 1e-9 << " s\n";
}

// 1000000 1000000
int main(int argc, char** argv)
{

  size_t n = atoi(argv[1]);
  size_t compute_iters = atoi(argv[2]);

  run<float, 32>(n, compute_iters);
}