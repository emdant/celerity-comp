#include <cstdlib>
#include <sycl/sycl.hpp>

template <typename DataT, size_t GlobalSize, size_t LocalSize>
void run()
{
  sycl::queue q;
  std::array<DataT, GlobalSize> in_array1;
  std::array<DataT, GlobalSize> in_array2;
  std::array<DataT, GlobalSize> out_array1;

  in_array1.fill(rand() % 1 + 1);
  in_array2.fill(rand() % 1 + 1);

  {
    sycl::buffer<DataT, 1> in_buf1{in_array1};
    sycl::buffer<DataT, 1> in_buf2{in_array2};
    sycl::buffer<DataT, 1> out_buf1{out_array1};

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<DataT, 1, sycl::access_mode::read> in_acc1{in_buf1, cgh};
      sycl::accessor<DataT, 1, sycl::access_mode::read> in_acc2{in_buf2, cgh};
      sycl::accessor<DataT, 1, sycl::access_mode::write> out_acc1{out_buf1, cgh};
      sycl::local_accessor<DataT, 1> local_acc1{LocalSize, cgh};
      sycl::local_accessor<DataT, 1> local_acc2{LocalSize, cgh};

      sycl::nd_range<1> ndr{GlobalSize, LocalSize};

      cgh.parallel_for<class LocalMemory>(ndr, [=](sycl::nd_item<1> item) {
        sycl::id<1> lid = item.get_local_id();
        sycl::id<1> gid = item.get_global_id();

        local_acc1[lid] = in_acc1[gid];
        local_acc2[lid] = in_acc2[gid];

#pragma unroll
        for (size_t i = 0; i < LocalSize; i++) {
          local_acc2[lid] *= local_acc1[lid];
          local_acc2[lid] /= local_acc1[lid];
          local_acc2[lid] += local_acc1[lid];
        }

        out_acc1[gid] = local_acc1[lid] + local_acc2[lid];
      });
    });
  }

  std::cout << out_array1[0] << std::endl;
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