#include <sycl/sycl.hpp>

template<typename DataT, int Dim>
class LocalMemoryBenchmark {
  public:
  LocalMemoryBenchmark(sycl::accessor<DataT, Dim, sycl::access_mode::read> in,
    sycl::accessor<DataT, Dim, sycl::access_mode::write> out,
    sycl::local_accessor<DataT, Dim> local) 
    : m_in{in}, m_out{out}, m_local{local} {}

  void operator()(sycl::nd_item<Dim> item) const {
    sycl::id<Dim> lid = item.get_local_id();
    sycl::id<Dim> gid = item.get_global_id();
    
    m_local[lid] = m_in[gid];
    m_out[gid] = m_local[lid];
  }

  private:
    sycl::accessor<DataT, Dim, sycl::access_mode::read> m_in;
    sycl::accessor<DataT, Dim, sycl::access_mode::write> m_out;
    sycl::local_accessor<DataT, Dim> m_local;
};

int main()
{
  constexpr size_t global_size = 4096; 
  constexpr size_t local_size = 32;

  sycl::queue q;
  std::array<float, global_size> in_array;
  std::array<float, global_size> out_array;

  in_array.fill(1.0f);

  {
    sycl::buffer<float, 1> in_buf {in_array};
    sycl::buffer<float, 1> out_buf {out_array};

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<float, 1, sycl::access_mode::read> in_acc {in_buf, cgh};
      sycl::accessor<float, 1, sycl::access_mode::write> out_acc {out_buf, cgh};
      sycl::local_accessor<float, 1> local_acc {local_size, cgh};
      
      sycl::nd_range<1> ndr {global_size, local_size};

      cgh.parallel_for(ndr, LocalMemoryBenchmark<float, 1> {in_acc, out_acc, local_acc});
    });
    
  }

  for (auto value : out_array)
    assert(value == 1.0f);

  return 0;
}