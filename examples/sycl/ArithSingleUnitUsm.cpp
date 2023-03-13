#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename ValueType, size_t AddPerc, size_t MulPerc, size_t DivPerc, size_t SpecPerc>
void run(size_t n, size_t compute_iters)
{
  sycl::queue q(gpu_selector_v);

  ValueType* a = sycl::malloc_device<ValueType>(n, q);
  ValueType* c1 = sycl::malloc_device<ValueType>(n, q);
  ValueType* c2 = sycl::malloc_device<ValueType>(n, q);
  ValueType* c3 = sycl::malloc_device<ValueType>(n, q);
  ValueType* c4 = sycl::malloc_device<ValueType>(n, q);

  q.parallel_for({n}, [=](id<1> id) {
    a[id] = (ValueType)(id.get(0) % 1) + 1;
  });
  q.wait();

  // Launch the computation
  event e = q.submit([&](handler& h) {
    h.parallel_for({n}, [=](sycl::id<1> id) {
      int gid = id.get(0);
      ValueType ra, rb, rc, rd;

      if (gid < n) {
        ra = a[gid];
        rb = a[n - gid];
        rc = a[gid];
        rd = a[n - gid];

        for (int i = 0; i < compute_iters; i++) {

#pragma unroll
          for (int i = 0; i < AddPerc; i++) {
            ra = ra + rb;
            rb = rb + rc;
            rc = rc + rd;
            rd = rd + ra;
          }
#pragma unroll
          for (int i = 0; i < DivPerc; i++) {
            ra = ra / rb;
            rb = rb / rc;
            rc = rc / rd;
            rd = rd / ra;
          }
#pragma unroll
          for (int i = 0; i < MulPerc; i++) {
            ra = ra * rb;
            rb = rb * rc;
            rc = rc * rd;
            rd = rd * ra;
          }
#pragma unroll
          for (int i = 0; i < SpecPerc; i++) {
            ra = log(rb);
            rb = cos(rc);
            rc = log(rd);
            rd = sin(ra);
          }
        }

        c1[gid] = ra;
        c2[gid] = rb;
        c3[gid] = rc;
        c4[gid] = rd;
      }
    });
  });
  e.wait_and_throw();
  // uint64_t begin = e.get_profiling_info<info::event_profiling::command_start>(), end = e.get_profiling_info<info::event_profiling::command_end>();
  // std::cout << "runtime: " << (end - begin) * 1e-9 << " s\n";
}

// 1000000 500000
int main(int argc, char** argv)
{
  size_t n = atoi(argv[1]);
  size_t compute_iters = atoi(argv[2]);

  std::cout << "int\n";
  std::cout << "add\n";
  run<int, 1, 0, 0, 0>(n, compute_iters);
  run<int, 5, 0, 0, 0>(n, compute_iters);
  run<int, 10, 0, 0, 0>(n, compute_iters);

  std::cout << "mul\n";
  run<int, 0, 1, 0, 0>(n, compute_iters);
  run<int, 0, 5, 0, 0>(n, compute_iters);
  run<int, 0, 10, 0, 0>(n, compute_iters);

  std::cout << "div\n";
  run<int, 0, 0, 1, 0>(n, compute_iters);
  run<int, 0, 0, 2, 0>(n, compute_iters);
  run<int, 0, 0, 3, 0>(n, compute_iters);

  std::cout << "\nfloat\n";
  std::cout << "add\n";
  run<float, 1, 0, 0, 0>(n, compute_iters);
  run<float, 5, 0, 0, 0>(n, compute_iters);
  run<float, 10, 0, 0, 0>(n, compute_iters);

  std::cout << "mul\n";
  run<float, 0, 1, 0, 0>(n, compute_iters);
  run<float, 0, 5, 0, 0>(n, compute_iters);
  run<float, 0, 10, 0, 0>(n, compute_iters);

  std::cout << "div\n";
  run<float, 0, 0, 1, 0>(n, compute_iters);
  run<float, 0, 0, 2, 0>(n, compute_iters);
  run<float, 0, 0, 3, 0>(n, compute_iters);

  std::cout << "\nspecial\n";
  run<float, 0, 0, 0, 1>(n, compute_iters);
  run<float, 0, 0, 0, 2>(n, compute_iters);
  run<float, 0, 0, 0, 3>(n, compute_iters);
}