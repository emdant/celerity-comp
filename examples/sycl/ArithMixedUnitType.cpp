#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

template <size_t AddPercFloat, size_t MulPercFloat, size_t DivPercFloat, size_t AddPercInt, size_t MulPercInt, size_t DivPercInt, size_t SpecPerc>
void run(size_t n, size_t compute_iters)
{
  sycl::queue q(gpu_selector_v);

  std::vector<float> a(n);
  std::vector<float> c1(n);
  std::vector<float> c2(n);
  std::vector<float> c3(n);
  std::vector<float> c4(n);

  for (size_t i = 0; i < n; i++) {
    a[i] = (float)(i % 1) + 1;
  }

  buffer<float, 1> a_buf(a.data(), {n});
  buffer<float, 1> c1_buf(c1.data(), {n});
  buffer<float, 1> c2_buf(c2.data(), {n});
  buffer<float, 1> c3_buf(c3.data(), {n});
  buffer<float, 1> c4_buf(c4.data(), {n});

  // Launch the computation
  event e = q.submit([&](handler& h) {
    accessor<float, 1, access_mode::read> a_acc{a_buf, h};
    accessor<float, 1, access_mode::read_write> c1_acc{c1_buf, h};
    accessor<float, 1, access_mode::read_write> c2_acc{c2_buf, h};
    accessor<float, 1, access_mode::read_write> c3_acc{c3_buf, h};
    accessor<float, 1, access_mode::read_write> c4_acc{c4_buf, h};

    range<1> grid{n};

    h.parallel_for<class ArithMixedUnitType>(grid, [=](sycl::id<1> id) {
      int gid = id.get(0);
      float raf, rbf, rcf, rdf;
      int rai, rbi, rci, rdi;

      if (gid < n) {
        rai = raf = a_acc[gid];
        rbi = rbf = a_acc[n - gid];
        rci = rcf = a_acc[gid];
        rdi = rdf = a_acc[n - gid];

        for (int i = 0; i < compute_iters; i++) {

#pragma unroll
          for (int i = 0; i < AddPercFloat; i++) {
            raf = raf + rbf;
            rbf = rbf + rcf;
            rcf = rcf + rdf;
            rdf = rdf + raf;
          }
#pragma unroll
          for (int i = 0; i < DivPercFloat; i++) {
            raf = raf / rbf;
            rbf = rbf / rcf;
            rcf = rcf / rdf;
            rdf = rdf / raf;
          }
#pragma unroll
          for (int i = 0; i < MulPercFloat; i++) {
            raf = raf * rbf;
            rbf = rbf * rcf;
            rcf = rcf * rdf;
            rdf = rdf * raf;
          }

#pragma unroll
          for (int i = 0; i < AddPercInt; i++) {
            rai = rai + rbi;
            rbi = rbi + rci;
            rci = rci + rdi;
            rdi = rdi + rai;
          }
#pragma unroll
          for (int i = 0; i < DivPercInt; i++) {
            rai = rai / rbi;
            rbi = rbi / rci;
            rci = rci / rdi;
            rdi = rdi / rai;
          }
#pragma unroll
          for (int i = 0; i < MulPercInt; i++) {
            rai = rai * rbi;
            rbi = rbi * rci;
            rci = rci * rdi;
            rdi = rdi * rai;
          }
#pragma unroll
          for (int i = 0; i < SpecPerc; i++) {
            raf = log(rbf);
            rbf = cos(rcf);
            rcf = log(rdf);
            rdf = sin(raf);
          }
        }

        c1_acc[gid] = raf + rai;
        c2_acc[gid] = rbf + rbi;
        c3_acc[gid] = rcf + rci;
        c4_acc[gid] = rdf + rdi;
      }
    });
  });
  e.wait_and_throw();
  uint64_t begin = e.get_profiling_info<info::event_profiling::command_start>(), end = e.get_profiling_info<info::event_profiling::command_end>();
  std::cout << "runtime: " << (end - begin) * 1e-9 << " s\n";
}

// 1000000 25000
int main(int argc, char** argv)
{

  size_t n = atoi(argv[1]);
  size_t compute_iters = atoi(argv[2]);

  // equal
  run<1, 1, 1, 1, 1, 1, 0>(n, compute_iters);
  run<3, 3, 3, 3, 3, 3, 0>(n, compute_iters);
  run<5, 5, 5, 5, 5, 5, 0>(n, compute_iters);

  // different
  run<3, 3, 3, 1, 1, 1, 0>(n, compute_iters);
  run<1, 1, 1, 3, 3, 3, 0>(n, compute_iters);
  run<5, 5, 5, 2, 2, 2, 0>(n, compute_iters);
  run<2, 2, 2, 5, 5, 5, 0>(n, compute_iters);

  // equal
  run<1, 1, 1, 1, 1, 1, 1>(n, compute_iters);
  run<3, 3, 3, 3, 3, 3, 1>(n, compute_iters);
  run<5, 5, 5, 5, 5, 5, 1>(n, compute_iters);

  // different
  run<3, 3, 3, 1, 1, 1, 1>(n, compute_iters);
  run<1, 1, 1, 3, 3, 3, 1>(n, compute_iters);
  run<5, 5, 5, 2, 2, 2, 1>(n, compute_iters);
  run<2, 2, 2, 5, 5, 5, 1>(n, compute_iters);
}