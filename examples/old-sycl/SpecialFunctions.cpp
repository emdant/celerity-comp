#include <sycl/sycl.hpp>

template <size_t Size, size_t Coarsening, size_t Iterations>
void run()
{
  sycl::queue q;
  std::array<float, Size> in_array;
  std::array<float, Size> out_array1;
  std::array<float, Size> out_array2;

  in_array.fill(0.0f);

  {
    sycl::buffer<float, 1> in_buf{in_array};
    sycl::buffer<float, 1> out_buf1{out_array1};
    sycl::buffer<float, 1> out_buf2{out_array2};

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<float, 1, sycl::access_mode::read> in_acc{in_buf, cgh};
      sycl::accessor<float, 1, sycl::access_mode::write> out_acc1{out_buf1, cgh};
      sycl::accessor<float, 1, sycl::access_mode::write> out_acc2{out_buf2, cgh};
      sycl::range<1> r{Size / Coarsening};

      cgh.parallel_for<class SpecialFunctions>(r, [=](sycl::id<1> id) {
        size_t base_data_index = id.get(0) * Coarsening;

#pragma unroll
        for (size_t i = 0; i < Coarsening; i++) {
          size_t data_index = base_data_index + i;

          float f0, f1, f2;

          f0 = in_acc[data_index];
          f1 = f2 = f0 = in_acc[data_index + 1];

#pragma unroll
          for (size_t j = 0; j < Iterations; j++) {
            out_acc2[data_index] = sycl::cos(out_acc2[data_index]);
            f0 = sycl::sin(f2);
            f2 = sycl::tan(f0);
          }

          out_acc1[data_index] = f2;
        }
      });
    });
  }

  std::cout << out_array1[0] << " " << out_array2[0] << std::endl;
}

int main()
{
  run<4096, 1, 1>();
  run<4096, 1, 8>();
  run<4096, 1, 16>();

  run<4096, 2, 1>();
  run<4096, 2, 8>();
  run<4096, 2, 16>();

  run<4096, 4, 1>();
  run<4096, 4, 8>();
  run<4096, 4, 16>();

  run<4096, 8, 1>();
  run<4096, 8, 8>();
  run<4096, 8, 16>();

  return 0;
}