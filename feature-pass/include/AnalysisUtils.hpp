#pragma once

#include <iomanip>
#include <iostream>
#include <string>

#include <llvm/ADT/StringMap.h>
#include <llvm/IR/Function.h>

namespace celerity {

/// Printing utilities
/// Print all feature names in one line
template <typename T>
void print_feature_names(llvm::StringMap<T>& feature_map, llvm::raw_ostream& out_stream)
{
  auto keys = feature_map.keys();
  std::stringstream ss;
  for (llvm::StringRef f : keys) {
    ss << std::setw(7) << f.str() << " ";
  }
  ss << "\n";
  out_stream << ss.str();
}

/// Print all feature unsigned values in one line
template <typename T>
void print_feature_values(llvm::StringMap<T>& feature_map, llvm::raw_ostream& out_stream)
{
  auto keys = feature_map.keys();
  std::stringstream ss;
  for (llvm::StringRef f : keys) {
    if constexpr (std::is_same<float, T>::value) {
      ss << std::setprecision(3) << std::setfill(' ') << std::setw(7);
    } else
      ss << std::setw(7);
    ss << feature_map[f] << " ";
  }
  ss << "\n";
  out_stream << ss.str();
}

/// Demangling utilties
std::string get_demangled_name(const llvm::Function& function);

struct CoalescedMemAccess {
  int mem_access;
  int mem_coalesced;
};

/// Uses a simple heuristics to calculate how many mem access are coalesced
CoalescedMemAccess getCoalescedMemAccess(llvm::Function& fun);

/// Enum identifying OpenCL address spaces
enum class cl_address_space_type { Generic,
                                   Global,
                                   Region,
                                   Local,
                                   Constant,
                                   Private };

/// Mapping between LLVM address space qualifiers and OpenCL address space types.
/// https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-memory-model
cl_address_space_type get_cl_address_space_type(const unsigned addrSpaceId);

/// Support utility functions to deal with memory accesses
inline bool isGlobalMemoryAccess(const unsigned addrSpaceId) { return get_cl_address_space_type(addrSpaceId) == cl_address_space_type::Global; }

inline bool isLocalMemoryAccess(const unsigned addrSpaceId) { return get_cl_address_space_type(addrSpaceId) == cl_address_space_type::Local; }

inline bool isConstantMemoryAccess(const unsigned addrSpaceId) { return get_cl_address_space_type(addrSpaceId) == cl_address_space_type::Global; }

} // namespace celerity
