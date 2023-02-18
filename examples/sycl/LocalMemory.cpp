#include <sycl/sycl.hpp>

template <typename DataT, size_t GlobalSize, size_t LocalSize>
void run()
{
  sycl::queue q;
  std::array<DataT, GlobalSize> in_array;
  std::array<DataT, GlobalSize> out_array;

  in_array.fill(1.0f);

  {
    sycl::buffer<DataT, 1> in_buf{in_array};
    sycl::buffer<DataT, 1> out_buf{out_array};

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<DataT, 1, sycl::access_mode::read> in_acc{in_buf, cgh};
      sycl::accessor<DataT, 1, sycl::access_mode::write> out_acc{out_buf, cgh};
      sycl::local_accessor<DataT, 1> local_acc{LocalSize, cgh};

      sycl::nd_range<1> ndr{GlobalSize, LocalSize};

      cgh.parallel_for(ndr, [=](sycl::nd_item<1> item) {
        sycl::id<1> lid = item.get_local_id();
        sycl::id<1> gid = item.get_global_id();

        local_acc[lid] = in_acc[gid];
        out_acc[gid] = local_acc[lid];
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
  run<int, 4096, 8>();
  run<int, 4096, 16>();
  run<int, 4096, 32>();
  run<int, 4096, 64>();

  run<float, 4096, 8>();
  run<float, 4096, 16>();
  run<float, 4096, 32>();
  run<float, 4096, 64>();

  return 0;
}