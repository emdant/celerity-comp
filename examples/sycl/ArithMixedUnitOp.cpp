#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename ValueType, size_t AddPerc, size_t MulPerc, size_t DivPerc, size_t SpecPerc>
void run(size_t n, size_t compute_iters)
{
  sycl::queue q(gpu_selector_v);

  std::vector<ValueType> a(n);
  std::vector<ValueType> c1(n);
  std::vector<ValueType> c2(n);
  std::vector<ValueType> c3(n);
  std::vector<ValueType> c4(n);

  for (size_t i = 0; i < n; i++) {
    a[i] = (float)(i % 1) + 1;
  }

  buffer<ValueType, 1> a_buf(a.data(), {n});
  buffer<ValueType, 1> c1_buf(c1.data(), {n});
  buffer<ValueType, 1> c2_buf(c2.data(), {n});
  buffer<ValueType, 1> c3_buf(c3.data(), {n});
  buffer<ValueType, 1> c4_buf(c4.data(), {n});

  // Launch the computation
  event e = q.submit([&](handler& h) {
    accessor<ValueType, 1, access_mode::read> a_acc{a_buf, h};
    accessor<ValueType, 1, access_mode::read_write> c1_acc{c1_buf, h};
    accessor<ValueType, 1, access_mode::read_write> c2_acc{c2_buf, h};
    accessor<ValueType, 1, access_mode::read_write> c3_acc{c3_buf, h};
    accessor<ValueType, 1, access_mode::read_write> c4_acc{c4_buf, h};

    range<1> grid{n};

    h.parallel_for<class ArithMixedUnitOp>(grid, [=](sycl::id<1> id) {
      int gid = id.get(0);
      ValueType ra, rb, rc, rd;

      if (gid < n) {
        ra = a_acc[gid];
        rb = a_acc[n - gid];
        rc = a_acc[gid];
        rd = a_acc[n - gid];

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

        c1_acc[gid] = ra;
        c2_acc[gid] = rb;
        c3_acc[gid] = rc;
        c4_acc[gid] = rd;
      }
    });
  });
  e.wait_and_throw();
  uint64_t begin = e.get_profiling_info<info::event_profiling::command_start>(), end = e.get_profiling_info<info::event_profiling::command_end>();
  std::cout << "runtime: " << (end - begin) * 1e-9 << " s\n";
}

// 1000000 100000
int main(int argc, char** argv)
{
  size_t n = atoi(argv[1]);
  size_t compute_iters = atoi(argv[2]);

  run<int, 1, 1, 1, 0>(n, compute_iters);
  run<int, 2, 2, 2, 0>(n, compute_iters);
  run<int, 3, 3, 3, 0>(n, compute_iters);

  run<int, 1, 1, 1, 1>(n, compute_iters);
  run<int, 2, 2, 2, 2>(n, compute_iters);
  run<int, 3, 3, 3, 3>(n, compute_iters);

  run<float, 1, 1, 1, 0>(n, compute_iters);
  run<float, 2, 2, 2, 0>(n, compute_iters);
  run<float, 3, 3, 3, 0>(n, compute_iters);

  run<float, 1, 1, 1, 1>(n, compute_iters);
  run<float, 2, 2, 2, 2>(n, compute_iters);
  run<float, 3, 3, 3, 3>(n, compute_iters);
}