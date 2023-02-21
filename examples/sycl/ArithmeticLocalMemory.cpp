#include <cstdlib>
#include <sycl/sycl.hpp>
//TODO: controllare addresspace(3) e meorizzare i registri in una lista, fare lo stesso anche per addresspace(1) cioè global memory
//TODO: risolvere numerod i accessi alla memoria local quando il coarsening è maggiore di 1
template <size_t Size, size_t LocalSize, size_t Coarsening, size_t PercFloatAddsub, size_t PercFloatMul, size_t PercFloatDiv, size_t PercIntAddsub, size_t PercIntMul, size_t PercIntDiv, size_t PercSpFunc>
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
      // Local accessor
      sycl::local_accessor<float, 1> in_float_local_acc{sycl::range<1>{LocalSize * Coarsening}, cgh};
      sycl::local_accessor<int, 1> in_int_local_acc{sycl::range<1>{LocalSize * Coarsening}, cgh};

      sycl::accessor<float, 1, sycl::access_mode::write> out_acc{out_buf, cgh};
      sycl::range<1> r{Size / Coarsening};
      sycl::range<1> local_r{LocalSize};

      cgh.parallel_for<class LocalMemory>(sycl::nd_range<1>{r, local_r}, [=](sycl::nd_item<1> it) {
        sycl::group group = it.get_group();
        sycl::id<1> global_id = it.get_global_id();
        sycl::id<1> local_id = it.get_local_id();
        size_t global_base_data_index = global_id.get(0) * Coarsening;
        size_t local_base_data_index = local_id.get(0) * Coarsening;

        // LocalSize * Coarsening
        #pragma unroll
        for(size_t i = 0; i < Coarsening; i++){
          in_float_local_acc[local_base_data_index+i] = in_float_acc[global_base_data_index+i];
          in_int_local_acc[local_base_data_index+i] = in_int_acc[global_base_data_index+i];
        }
        
        sycl::group_barrier(group);

        // clang-format off
        #pragma unroll
        for (size_t i = 0; i < Coarsening; i++) {
          size_t data_index = local_base_data_index + i;

          float f0 = in_float_local_acc[data_index];
          int i0  = in_int_local_acc[data_index];

          #pragma unroll
          for(size_t j = 0; j < LocalSize * Coarsening; j++){
            float f1 = in_int_local_acc[j];
            int i1 = in_float_local_acc[j];
            
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
            // i0 = i0 * i1;
            
            // clang-format off
            #pragma unroll
            for (size_t i_div = 0; i_div < PercIntDiv; i_div++) {
              i1 = i1 / i0;
              i0 = i0 / i1;
            }

            // clang-format off
            #pragma unroll
            for (size_t i_addsub = 0; i_addsub < PercIntAddsub; i_addsub++) {
              i1 = i1 + i0;
              i0 = i0 + i1;
            }
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


  // float
  run<4096, 8, 1, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);
  // run<4096, 16, 1, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);
  // run<4096, 32, 1, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);
  // run<4096, 64, 1, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);


  run<4096, 8, 2, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);
  // run<4096, 16, 2, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);
  // run<4096, 32, 2, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);
  // run<4096, 64, 2, 4, 5, 5, 0, 0, 0, 0>(f_fill, i_fill);

  
  // // int
  // run<4096, 8, 1, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 16, 1, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 32, 1, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 64, 1, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);


  // run<4096, 8, 2, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 16, 2, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 32, 2, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 64, 2, 0, 0, 0, 4, 5, 5, 0>(f_fill, i_fill);

  
  // // float + sp
  // run<4096, 8, 1, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);
  // run<4096, 16, 1, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);
  // run<4096, 32, 1, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);
  // run<4096, 64, 1, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);


  // run<4096, 8, 2, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);
  // run<4096, 16, 2, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);
  // run<4096, 32, 2, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);
  // run<4096, 64, 2, 4, 5, 5, 0, 0, 0, 2>(f_fill, i_fill);

  
  // // int + sp
  // run<4096, 8, 1, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 16, 1, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 32, 1, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 64, 1, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);

  // run<4096, 8, 2, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 16, 2, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 32, 2, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 64, 2, 0, 0, 0, 4, 5, 5, 2>(f_fill, i_fill);

  
  // // equal float and int
  // run<4096, 8, 1, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 16, 1, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 32, 1, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 64, 1, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);


  // run<4096, 8, 2, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 16,2, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 32,2, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 64,2, 4, 5, 5, 4, 5, 5, 0>(f_fill, i_fill);

  // run<4096, 8, 1, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 16, 1, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 32, 1, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 64, 1, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);

  // run<4096, 8, 2, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 16, 2, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 32, 2, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 64, 2, 4, 5, 5, 4, 5, 5, 2>(f_fill, i_fill);

  
  // // more float than int
  // run<4096, 8, 1, 8, 10, 10, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 16, 1, 8, 10, 10, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 32, 1, 8, 10, 10, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 64, 1, 8, 10, 10, 4, 5, 5, 0>(f_fill, i_fill);


  // run<4096, 8, 2, 8, 10, 10, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 16, 2, 8, 10, 10, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 32, 2, 8, 10, 10, 4, 5, 5, 0>(f_fill, i_fill);
  // run<4096, 64, 2, 8, 10, 10, 4, 5, 5, 0>(f_fill, i_fill);


  // run<4096, 8, 1, 8, 10, 10, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 16, 1, 8, 10, 10, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 32, 1, 8, 10, 10, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 64, 1, 8, 10, 10, 4, 5, 5, 2>(f_fill, i_fill);

  // run<4096, 8, 2, 8, 10, 10, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 16, 2, 8, 10, 10, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 32, 2, 8, 10, 10, 4, 5, 5, 2>(f_fill, i_fill);
  // run<4096, 64, 2, 8, 10, 10, 4, 5, 5, 2>(f_fill, i_fill);

  
  // // more int than float 
  // run<4096, 8,1, 4, 5, 5, 8, 10, 10, 0>(f_fill, i_fill);
  // run<4096, 16,1, 4, 5, 5, 8, 10, 10, 0>(f_fill, i_fill);
  // run<4096, 32,1, 4, 5, 5, 8, 10, 10, 0>(f_fill, i_fill);
  // run<4096, 64,1, 4, 5, 5, 8, 10, 10, 0>(f_fill, i_fill);

  // run<4096, 8,2, 4, 5, 5, 8, 10, 10, 0>(f_fill, i_fill);
  // run<4096, 16,2, 4, 5, 5, 8, 10, 10, 0>(f_fill, i_fill);
  // run<4096, 32,2, 4, 5, 5, 8, 10, 10, 0>(f_fill, i_fill);
  // run<4096, 64,2, 4, 5, 5, 8, 10, 10, 0>(f_fill, i_fill);



  // run<4096, 8,1, 4, 5, 5, 8, 10, 10, 2>(f_fill, i_fill);
  // run<4096, 16,1, 4, 5, 5, 8, 10, 10, 2>(f_fill, i_fill);
  // run<4096, 32,1, 4, 5, 5, 8, 10, 10, 2>(f_fill, i_fill);
  // run<4096, 64,1, 4, 5, 5, 8, 10, 10, 2>(f_fill, i_fill);


  // run<4096, 8, 2, 4, 5, 5, 8, 10, 10, 2>(f_fill, i_fill);
  // run<4096, 16, 2, 4, 5, 5, 8, 10, 10, 2>(f_fill, i_fill);
  // run<4096, 32, 2, 4, 5, 5, 8, 10, 10, 2>(f_fill, i_fill);
  // run<4096, 64, 2, 4, 5, 5, 8, 10, 10, 2>(f_fill, i_fill);


  return 0;
}