#include <cstdlib>
#include <sycl/sycl.hpp>

template <size_t Size, size_t Coarsening, size_t Iterations, size_t PercFloatAddsub, size_t PercFloatMul, size_t PercFloatDiv, size_t PercIntAddsub, size_t PercIntMul, size_t PercIntDiv, size_t PercSpFunc>
void run(float f_fill_value, int i_fill_value)
{
  sycl::queue q;
  std::array<float, Size> in_float_array;
  std::array<int, Size> in_int_array;
  std::array<float, Size> out_array;

  in_float_array.fill(f_fill_value);
  in_int_array.fill(i_fill_value);

  {
    sycl::buffer<float, 1> in_float_buf{in_float_array};
    sycl::buffer<int, 1> in_int_buf{in_int_array};
    sycl::buffer<float, 1> out_buf{out_array};

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<float, 1, sycl::access_mode::read> in_float_acc{in_float_buf, cgh};
      sycl::accessor<int, 1, sycl::access_mode::read> in_int_acc{in_int_buf, cgh};
      sycl::accessor<float, 1, sycl::access_mode::write> out_acc{out_buf, cgh};
      sycl::range<1> r{Size / Coarsening};

      cgh.parallel_for(r, [=](sycl::id<1> id) {
        size_t base_data_index = id.get(0) * Coarsening;

        // clang-format off
        #pragma unroll
        for (size_t i = 0; i < Coarsening; i++) {
          size_t data_index = base_data_index + i;

          float f0 = in_float_acc[data_index];
          int i0  = in_int_acc[data_index];
          
          float f1 = i0;
          int i1 = f0;
          
          // clang-format off
          #pragma unroll
          for (size_t f_mul = 0; f_mul < PercFloatMul; f_mul++) {
            f1 = f1 * f0;
            f0 = f0 * f1;
          }
          
          // clang-format off
          #pragma unroll
          for (size_t f_div = 0; f_div < PercFloatDiv; f_div++) {
            f1 = f1 / f0;
            f0 = f0 / f1;
          }

          // clang-format off
          #pragma unroll
          for (size_t f_sp = 0; f_sp < PercSpFunc; f_sp++) {
            f1 = sycl::acos(f0);
            f0 = f0 * f0 + f1;
          }

          // clang-format off
          #pragma unroll
          for (size_t f_addsub = 0; f_addsub < PercFloatAddsub; f_addsub++) {
            f0 = f0 + f1;
            f1 = f1 + f0;
          }
          
          // clang-format off
          #pragma unroll
          for (size_t i_mul = 0; i_mul < PercIntMul; i_mul++) {
            i1 = i1 * i0;
            i0 = i0 * i1;
          }
          i0 = i0 * i1;
          
          // clang-format off
          #pragma unroll
          for (size_t i_div = 0; i_div < PercIntDiv; i_div++) {
            i1 = i1 / i0;
            i0 = i0 / i1;
          }
          i1 = i1 / i0;
          i1 = i1 / i0;

          // clang-format off
          #pragma unroll
          for (size_t i_addsub = 0; i_addsub < PercIntAddsub; i_addsub++) {
            i1 = i1 + i0;
            i0 = i0 + i1;
          }
          
          out_acc[data_index] = i0 + f0;
        }
      });
    });
  }
  
  std::cout << out_array[0] << std::endl;
}

int main()
{
  srand(1);
  float f_fill = rand() % 1 + 1;
  int i_fill = rand() % 1 + 1;

  run<4096, 1, 1, 10, 10, 10, 10, 10, 10, 10>(f_fill, i_fill);

  return 0;
}