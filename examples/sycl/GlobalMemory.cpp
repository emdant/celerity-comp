#include <sycl/sycl.hpp>

template <typename DataT, size_t Size, size_t Coarsening>
void run()
{
  sycl::queue q;
  std::array<DataT, Size> in_array;
  std::array<DataT, Size> out_array;

  in_array.fill(1);

  {
    sycl::buffer<DataT, 1> in_buf{in_array};
    sycl::buffer<DataT, 1> out_buf{out_array};

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<DataT, 1, sycl::access_mode::read> in_acc{in_buf, cgh};
      sycl::accessor<DataT, 1, sycl::access_mode::write> out_acc{out_buf, cgh};
      sycl::range<1> r{Size / Coarsening};

      cgh.parallel_for(r, [=](sycl::id<1> id) {
        size_t base_data_index = id.get(0) * Coarsening;

#pragma unroll
        for (size_t i = 0; i < Coarsening; i++) {
          size_t data_index = base_data_index + i;
          out_acc[data_index] = in_acc[data_index];
        }
      });
    });
  }

  DataT sum = 0;
  for (DataT value : out_array) {
    assert(value == 1);
    sum += value;
  }
  std::cout << sum << std::endl;
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