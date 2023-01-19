

/* This example is a very small one designed to show how compact SYCL code
 * can be. That said, it includes no error checking and is rather terse. */
#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T>
class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC) {
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{N};
  sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);
  deviceQueue.submit([&](sycl::handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    auto kern = [=](sycl::id<1> wiID) {
      accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    };
    cgh.parallel_for<class SimpleVadd<T>>(numOfItems, kern);
  });
}

int main() {
  const size_t array_size = 4;
  std::array<int, array_size> A = {{1, 2, 3, 4}},
                                           B = {{1, 2, 3, 4}}, C;
  std::array<float, array_size> D = {{1.f, 2.f, 3.f, 4.f}},
                                             E = {{1.f, 2.f, 3.f, 4.f}}, F;
  simple_vadd(A, B, C);
  simple_vadd(D, E, F);


  for (unsigned int i = 0; i < array_size; i++) {
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
    if (F[i] != D[i] + E[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << F[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}