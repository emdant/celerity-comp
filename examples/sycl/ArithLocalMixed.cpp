#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

template <size_t LocalSize, size_t AddPercFloat, size_t MulPercFloat, size_t DivPercFloat, size_t AddPercInt, size_t MulPercInt, size_t DivPercInt, size_t SpecPerc>
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

    local_accessor<float, 1> in_float_local_acc{sycl::range<1>{LocalSize}, h};
    local_accessor<int, 1> in_int_local_acc{sycl::range<1>{LocalSize}, h};

    sycl::range<1> r{n};
    sycl::range<1> local_r{LocalSize};

    h.parallel_for<class ArithLocalMixed>(nd_range<1>{r, local_r}, [=](sycl::nd_item<1> it) {
      group<1> group = it.get_group();
      int gid = it.get_global_id(0);
      int lid = it.get_local_id(0);

      in_float_local_acc[lid] = a_acc[gid];
      in_int_local_acc[lid] = a_acc[gid];

      sycl::group_barrier(group);

      for (int i = 0; i < compute_iters; i++) {

#pragma unroll
        for (size_t j = 0; j < LocalSize; j++) {
          float raf, rbf, rcf, rdf;
          int rai, rbi, rci, rdi;

          raf = in_float_local_acc[j];
          rbf = in_float_local_acc[LocalSize - j];
          rcf = in_float_local_acc[j];
          rdf = in_float_local_acc[LocalSize - j];

          rai = in_int_local_acc[j];
          rbi = in_int_local_acc[LocalSize - j];
          rci = in_int_local_acc[j];
          rdi = in_int_local_acc[LocalSize - j];

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

          c1_acc[gid] = raf + rai;
          c2_acc[gid] = rbf + rbi;
          c3_acc[gid] = rcf + rci;
          c4_acc[gid] = rdf + rdi;
        }
      }
    });
  });
  e.wait_and_throw();
  uint64_t begin = e.get_profiling_info<info::event_profiling::command_start>(), end = e.get_profiling_info<info::event_profiling::command_end>();
  std::cout << "runtime: " << (end - begin) * 1e-9 << " s\n";
}

// 131072 20000
int main(int argc, char** argv)
{
  size_t n = atoi(argv[1]);
  size_t compute_iters = atoi(argv[2]);

  run<8, 1, 1, 1, 1, 1, 1, 0>(n, compute_iters);
  run<16, 1, 1, 1, 1, 1, 1, 0>(n, compute_iters);
  run<32, 1, 1, 1, 1, 1, 1, 0>(n, compute_iters);

  run<8, 3, 3, 3, 1, 1, 1, 0>(n, compute_iters);
  run<16, 3, 3, 3, 1, 1, 1, 0>(n, compute_iters);
  run<32, 3, 3, 3, 1, 1, 1, 0>(n, compute_iters);

  run<8, 1, 1, 1, 3, 3, 3, 0>(n, compute_iters);
  run<16, 1, 1, 1, 3, 3, 3, 0>(n, compute_iters);
  run<32, 1, 1, 1, 3, 3, 3, 0>(n, compute_iters);

  run<8, 1, 1, 1, 1, 1, 1, 1>(n, compute_iters);
  run<16, 1, 1, 1, 1, 1, 1, 1>(n, compute_iters);
  run<32, 1, 1, 1, 1, 1, 1, 1>(n, compute_iters);

  run<8, 3, 3, 3, 1, 1, 1, 1>(n, compute_iters);
  run<16, 3, 3, 3, 1, 1, 1, 1>(n, compute_iters);
  run<32, 3, 3, 3, 1, 1, 1, 1>(n, compute_iters);

  run<8, 1, 1, 1, 3, 3, 3, 1>(n, compute_iters);
  run<16, 1, 1, 1, 3, 3, 3, 1>(n, compute_iters);
  run<32, 1, 1, 1, 3, 3, 3, 1>(n, compute_iters);
}