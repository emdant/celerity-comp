#include <sycl/sycl.hpp>

template <size_t Size, size_t Coarsening, size_t Iterations>
class SpecialFunctionsKernel;

template <size_t Size, size_t Coarsening, size_t Iterations>
void run()
{
  sycl::queue q;
  std::array<float, Size> in_array;
  std::array<float, Size> out_array;

  in_array.fill(0.0f);

  {
    sycl::buffer<float, 1> in_buf{in_array};
    sycl::buffer<float, 1> out_buf{out_array};

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<float, 1, sycl::access_mode::read> in_acc{in_buf, cgh};
      sycl::accessor<float, 1, sycl::access_mode::write> out_acc{out_buf, cgh};
      sycl::range<1> r{Size / Coarsening};

      cgh.parallel_for<SpecialFunctionsKernel<Size, Coarsening, Iterations>>(r, [=](sycl::id<1> id) {
        size_t base_data_index = id.get(0) * Coarsening;

#pragma unroll
        for (size_t i = 0; i < Coarsening; i++) {
          size_t data_index = base_data_index + i;

          float f0, f1, f2;

          f0 = in_acc[data_index];
          f1 = f2 = f0;

#pragma unroll
          for (size_t j = 0; j < Iterations; j++) {
            f0 = sycl::cos(f1);
            f1 = sycl::sin(f2);
            f2 = sycl::tan(f1);
          }

          out_acc[data_index] = f0;
        }
      });
    });
  }

  float sum = 0;
  for (float value : out_array) {
    assert(value == 1);
    sum += value;
  }
  std::cout << sum << std::endl;
}

int main()
{
  run<4096, 1, 1>();
  run<4096, 1, 64>();
  run<4096, 2, 1>();
  run<4096, 2, 64>();
  run<4096, 4, 1>();
  run<4096, 4, 64>();
  run<4096, 8, 1>();
  run<4096, 8, 64>();

  return 0;
}