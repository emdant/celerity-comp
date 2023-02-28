#include <cstdlib>
#include <sycl/sycl.hpp>

template <typename DataT, size_t Size, size_t Coarsening>
void run()
{
  sycl::queue q;
  std::array<DataT, Size> in_array;
  std::array<DataT, Size> out_array1;
  std::array<DataT, Size> out_array2;

  in_array.fill(rand() % 1 + 1);

  {
    sycl::buffer<DataT, 1> in_buf{in_array};
    sycl::buffer<DataT, 1> out_buf1{out_array1};
    sycl::buffer<DataT, 1> out_buf2{out_array2};

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<DataT, 1, sycl::access_mode::read> in_acc{in_buf, cgh};
      sycl::accessor<DataT, 1, sycl::access_mode::write> out_acc1{out_buf1, cgh};
      sycl::accessor<DataT, 1, sycl::access_mode::write> out_acc2{out_buf2, cgh};
      sycl::range<1> r{Size / Coarsening};

      cgh.parallel_for<class GlobalMemory>(r, [=](sycl::id<1> id) {
        size_t base_data_index = id.get(0) * Coarsening;

#pragma unroll
        for (size_t i = 0; i < Coarsening; i++) {
          size_t data_index = base_data_index + i;
          out_acc1[data_index] = in_acc[data_index];
          out_acc2[data_index] = in_acc[data_index];
        }
#pragma unroll
        for (size_t i = 0; i < Coarsening; i++) {
          size_t data_index = base_data_index + i;
          out_acc2[data_index] *= out_acc1[data_index];
          out_acc2[data_index] /= out_acc1[data_index];
          out_acc2[data_index] += out_acc1[data_index];
        }
      });
    });
  }

  std::cout << out_array1[0] << " " << out_array2[0] << std::endl;
}

int main()
{
  run<int, 4096, 1>();
  run<int, 4096, 2>();
  run<int, 4096, 4>();
  run<int, 4096, 8>();

  run<float, 4096, 1>();
  run<float, 4096, 2>();
  run<float, 4096, 4>();
  run<float, 4096, 8>();

  return 0;
}