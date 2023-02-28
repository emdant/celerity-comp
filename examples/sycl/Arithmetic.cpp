#include <cstdlib>
#include <sycl/sycl.hpp>

template <size_t Size, size_t Coarsening, size_t PercFloatAddsub, size_t PercFloatMul, size_t PercFloatDiv, size_t PercIntAddsub, size_t PercIntMul, size_t PercIntDiv, size_t PercSpFunc>
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

      cgh.parallel_for<class Arithmetic>(r, [=](sycl::id<1> id) {
        size_t base_data_index = id.get(0) * Coarsening;

// clang-format off
        #pragma unroll
        for (size_t i = 0; i < Coarsening; i++) {
          size_t data_index = base_data_index + i;

          float f0 = in_float_acc[data_index];
          int i0  = in_int_acc[data_index];
          
          float f1 = in_int_acc[(data_index + Size / 2) % Size];
          int i1 = in_float_acc[(data_index + Size / 2) % Size];
          
          // clang-format off
          #pragma unroll
          for (size_t f_mul = 0; f_mul < PercFloatMul; f_mul++) {
            f1 = f1 * f0;
            f0 = f0 * f1;
          }
          out_acc[data_index] *= f0;
          
          // clang-format off
          #pragma unroll
          for (size_t f_div = 0; f_div < PercFloatDiv; f_div++) {
            f1 = f1 / f0;
            f0 = f0 / f1;
          }
          out_acc[data_index] *= f0;

          // clang-format off
          #pragma unroll
          for (size_t f_sp = 0; f_sp < PercSpFunc; f_sp++) {
            f1 = sycl::acos(f0);
            f0 = f0 * f0 + f1;
          }
          out_acc[data_index] *= f0;

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
          out_acc[data_index] *= i0;
          
          // clang-format off
          #pragma unroll
          for (size_t i_div = 0; i_div < PercIntDiv + 1; i_div++) {
            i1 = i1 / i0;
            i0 = i0 / i1;
          }
          out_acc[data_index] *= i0;

          // clang-format off
          #pragma unroll
          for (size_t i_addsub = 0; i_addsub < PercIntAddsub; i_addsub++) {
            i1 = i1 + i0;
            i0 = i0 + i1;
          }
          
          out_acc[data_index] *= (i0 + f0);
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

  // float
  run<4096, 1, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);
  run<4096, 2, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);
  
  // int
  run<4096, 1, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);
  run<4096, 2, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);
  
  // float + sp
  run<4096, 1, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);
  run<4096, 2, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);
  
  // int + sp
  run<4096, 1, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);
  run<4096, 2, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);
  
  // equal float and int
  run<4096, 1, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);
  run<4096, 2, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);
  run<4096, 1, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);
  run<4096, 2, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);
  
  // more float than int
  run<4096, 1, 6, 7, 7, 4, 5, 5, 0>(f_fill, i_fill);
  run<4096, 2, 6, 7, 7, 4, 5, 5, 0>(f_fill, i_fill);
  run<4096, 1, 6, 7, 7, 4, 5, 5, 2>(f_fill, i_fill);
  run<4096, 2, 6, 7, 7, 4, 5, 5, 2>(f_fill, i_fill);
  
  // more int than float 
  run<4096, 1, 4, 5, 5, 6, 7, 7, 0>(f_fill, i_fill);
  run<4096, 2, 4, 5, 5, 6, 7, 7, 0>(f_fill, i_fill);
  run<4096, 1, 4, 5, 5, 6, 7, 7, 2>(f_fill, i_fill);
  run<4096, 2, 4, 5, 5, 6, 7, 7, 2>(f_fill, i_fill);

  return 0;
}