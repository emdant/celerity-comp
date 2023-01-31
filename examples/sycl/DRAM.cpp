#include <sycl/sycl.hpp>

template<typename DataT, int Dim>
class DRAMBenchmark {
  public:
  DRAMBenchmark(sycl::accessor<DataT, Dim, sycl::access_mode::read> in, sycl::accessor<DataT, Dim, sycl::access_mode::write> out) 
    : m_in{in}, m_out{out} {}

  void operator()(sycl::id<Dim> id) const {
    m_out[id] = m_in[id];
  }

  private:
    sycl::accessor<DataT, Dim, sycl::access_mode::read> m_in;
    sycl::accessor<DataT, Dim, sycl::access_mode::write> m_out;
};

int main()
{
  constexpr size_t size = 4096; 
  sycl::queue q;
  std::array<float, size> in_array;
  std::array<float, size> out_array;

  in_array.fill(1.0f);

  {
    sycl::buffer<float, 1> in_buf {in_array};
    sycl::buffer<float, 1> out_buf {out_array};

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<float, 1, sycl::access_mode::read> in_acc {in_buf, cgh};
      sycl::accessor<float, 1, sycl::access_mode::write> out_acc {out_buf, cgh};
      sycl::range<1> r {size};

      cgh.parallel_for(r, DRAMBenchmark<float, 1> {in_acc, out_acc});
    });
    
  }

  for (auto value : out_array)
    assert(value == 1.0f);

  return 0;
}