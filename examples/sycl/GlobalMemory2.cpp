#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename ValueType, size_t k>
void run(size_t n, size_t compute_iters)
{
  sycl::queue q(gpu_selector_v);

  std::vector<ValueType> a1(n);
  std::vector<ValueType> a2(n);
  std::vector<ValueType> c1(n);
  std::vector<ValueType> c2(n);

  for (size_t i = 0; i < n; i++) {
    a1[i] = (float)(i % 1) + 1;
    a2[i] = (float)(i % 1) + 1;
  }

  buffer<ValueType, 1> a1_buf(a1.data(), {n});
  buffer<ValueType, 1> a2_buf(a2.data(), {n});
  buffer<ValueType, 1> c1_buf(c1.data(), {n});
  buffer<ValueType, 1> c2_buf(c2.data(), {n});

  // Launch the computation
  event e = q.submit([&](handler& h) {
    accessor<ValueType, 1, access_mode::read_write> a1_acc{a1_buf, h};
    accessor<ValueType, 1, access_mode::read_write> a2_acc{a2_buf, h};
    accessor<ValueType, 1, access_mode::read_write> c1_acc{c1_buf, h};
    accessor<ValueType, 1, access_mode::read_write> c2_acc{c2_buf, h};

    range<1> grid{n};

    h.parallel_for<class GlobalMemory2>(grid, [=](sycl::id<1> id) {
      int gid = id.get(0);
      ValueType r0, r1;

      for (int j = 0; j < compute_iters; j++) {

#pragma unroll 4
        for (size_t i = 0; i < k; i++) {
          if (gid + i < n) {
            r0 = a1_acc[gid + i];
            r1 = a2_acc[gid + i];
            c1_acc[gid + i] = r0;
            c2_acc[gid + i] = r1;
          }
        }
      }
    });
  });
  e.wait_and_throw();
  uint64_t begin = e.get_profiling_info<info::event_profiling::command_start>(), end = e.get_profiling_info<info::event_profiling::command_end>();
  std::cout << "runtime: " << (end - begin) * 1e-9 << " s\n";
}

//
int main(int argc, char** argv)
{
  size_t n = atoi(argv[1]);
  size_t compute_iters = atoi(argv[2]);

  run<float, 1024>(n, compute_iters);
  run<float, 8192>(n, compute_iters);
  run<float, 524288>(n, compute_iters);
}