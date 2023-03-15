#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename ValueType, int radius, size_t AddPerc, size_t MulPerc, size_t DivPerc, size_t SpecPerc>
void run(size_t n, size_t compute_iters)
{
  sycl::queue q(gpu_selector_v);

  std::vector<ValueType> a(n * n);
  std::vector<ValueType> b(n * n);
  std::vector<ValueType> c1(n * n);

  for (size_t i = 0; i < n * n; i++) {
    a[i] = (ValueType)(i % 1) + 1;
    b[i] = (ValueType)(i % 1) + 1;
  }

  buffer<ValueType, 2> a_buf(a.data(), {n, n});
  buffer<ValueType, 2> b_buf(b.data(), {n, n});
  buffer<ValueType, 2> c1_buf(c1.data(), {n, n});

  // Launch the computation
  event e = q.submit([&](handler& h) {
    auto a_acc = a_buf.template get_access<sycl::access_mode::read_write>(h);
    auto b_acc = b_buf.template get_access<sycl::access_mode::read_write>(h);
    auto c1_acc = c1_buf.template get_access<sycl::access_mode::read_write>(h);
    range<2> grid{n, n};

    h.parallel_for<class Stencil>(grid, [=](sycl::id<2> id) {
      int gidx = id.get(0);
      int gidy = id.get(1);

      for (int j = 0; j < compute_iters; j++) {
        for (int x = -radius; x < radius + 1; x++)
          for (int y = -radius; y < radius + 1; y++)
            if (gidx + x > -1 && gidx + x < n && gidy + y > -1 && gidy + y < n) {

#pragma unroll
              for (int i = 0; i < AddPerc; i++)
                c1_acc[gidx][gidy] += a_acc[gidx + x][gidy + y] + b_acc[gidx + x][gidy + y];

#pragma unroll
              for (int i = 0; i < MulPerc; i++)
                c1_acc[gidx][gidy] *= a_acc[gidx + x][gidy + y];

#pragma unroll
              for (int i = 0; i < DivPerc; i++)
                c1_acc[gidx][gidy] /= b_acc[gidx + x][gidy + y];

#pragma unroll
              for (int i = 0; i < SpecPerc; i++)
                c1_acc[gidx][gidy] = log(c1_acc[gidx][gidy]);
            }
      }
    });
  });
  e.wait_and_throw();
}

int main(int argc, char** argv)
{
  size_t n = atoi(argv[1]);
  size_t compute_iters = atoi(argv[2]);

  run<float, 2, 1, 1, 1, 0>(n, compute_iters);
  run<float, 3, 1, 1, 1, 0>(n, compute_iters);
  run<float, 4, 1, 1, 1, 0>(n, compute_iters);
  run<float, 5, 0, 1, 1, 0>(n, compute_iters);

  run<int, 2, 1, 1, 1, 0>(n, compute_iters);
  run<int, 3, 1, 1, 1, 0>(n, compute_iters);
  run<int, 4, 1, 1, 1, 0>(n, compute_iters);
  run<int, 5, 0, 1, 1, 0>(n, compute_iters);

  run<float, 2, 1, 1, 1, 1>(n, compute_iters);
  run<float, 3, 1, 1, 1, 1>(n, compute_iters);
  run<float, 4, 1, 1, 1, 1>(n, compute_iters);
  run<float, 5, 1, 0, 0, 1>(n, compute_iters);

  run<int, 2, 1, 1, 1, 1>(n, compute_iters);
  run<int, 3, 1, 1, 1, 1>(n, compute_iters);
  run<int, 4, 1, 1, 1, 1>(n, compute_iters);
  run<int, 5, 1, 0, 0, 1>(n, compute_iters);
}
