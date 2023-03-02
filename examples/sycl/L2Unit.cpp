#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename ValueType>
void run(size_t n, size_t compute_iters)
{
  sycl::queue q(gpu_selector_v);

  std::vector<ValueType> a(n);
  std::vector<ValueType> c1(n);

  for (size_t i = 0; i < n; i++) {
    a[i] = (float)(i % 1) + 1;
  }

  buffer<ValueType, 1> a_buf(a.data(), {n});
  buffer<ValueType, 1> c1_buf(c1.data(), {n});

  // Launch the computation
  event e = q.submit([&](handler& h) {
    accessor<ValueType, 1, access_mode::read> a_acc{a_buf, h};
    accessor<ValueType, 1, access_mode::read_write> c1_acc{c1_buf, h};

    range<1> grid{n};

    h.parallel_for<class L2Unit>(grid, [=](sycl::id<1> id) {
      constexpr size_t unrolls = 32;
      int gid = id.get(0);
      ValueType r0;

      for (int k = 0; k < 10; k++) {
        for (int j = 0; j < compute_iters; j += unrolls) {
#pragma unroll
          for (int i = 0; i < unrolls; i++) {
            r0 = a_acc[gid];
            c1_acc[gid] = r0;
          }
        }
      }
      c1_acc[gid] = r0;
    });
  });
  e.wait_and_throw();
  uint64_t begin = e.get_profiling_info<info::event_profiling::command_start>(), end = e.get_profiling_info<info::event_profiling::command_end>();
  std::cout << "runtime: " << (end - begin) * 1e-9 << " s\n";
}

// 1000000 131072
int main(int argc, char** argv)
{

  size_t n = atoi(argv[1]);
  size_t compute_iters = atoi(argv[2]);

  run<float>(n, compute_iters);
}