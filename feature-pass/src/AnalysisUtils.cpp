#include <cxxabi.h>
#include <set>
#include <stack>
#include <string>

#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/User.h>

#include "AnalysisUtils.hpp"

using namespace llvm;

namespace celerity {
std::string get_demangled_name(const llvm::Function& function)
{
  std::string function_name = function.getName().str();

  int status;
  char* demangled_c_str = abi::__cxa_demangle(function_name.data(), 0, 0, &status);

  if (demangled_c_str) {
    function_name = demangled_c_str; // string copy
    delete demangled_c_str;
  }

  return function_name;
}

/// Uses a simple heuristics to calculate how many mem access are coalesced
CoalescedMemAccess getCoalescedMemAccess(Function& fun)
{
  CoalescedMemAccess cma = {0, 0};
  raw_ostream& debug = outs(); // raw_null_ostream;
  debug.changeColor(raw_null_ostream::Colors::MAGENTA, true);
  debug << "coalesced mem access: ";
  debug.changeColor(raw_null_ostream::Colors::WHITE, false);

  // 1. Search for pointer arguments
  std::set<GetElementPtrInst*> gep_set;
  for (unsigned i = 0; i < fun.arg_size(); i++) {
    Argument* arg = fun.getArg(i);
    // if we have a pointer, it cannot be used for loop bound analysis, thus we skip it
    if (arg->getType()->isPointerTy()) {
      // 2. Collect all uses of the pointer argument as gep
      for (User* user : arg->users()) {
        if (GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(user)) {
          // outs() << "  gep:" << *gep << "\n";
          gep_set.insert(gep);
        }
      }
    }
  }
  cma.mem_access = gep_set.size();
  // 3. Find get_global_id()
  for (BasicBlock& bb : fun) {
    for (Instruction& inst : bb) {
      if (const CallInst* call_inst = dyn_cast<CallInst>(&inst)) {
        std::string fun_name = call_inst->getCalledFunction()->getGlobalIdentifier();
        if (fun_name.find("get_global_id") != std::string::npos) {
          // debug << "global_id\n";
          //  4. We collcet all uses of <get_global_id>
          std::stack<const User*> id_worklist;
          for (const User* user : call_inst->users()) {
            id_worklist.push(user);
          }
          // 5. Iterate over all uses until we found one of our presaved geps
          while (!id_worklist.empty()) {
            const User* id_use = id_worklist.top();
            id_worklist.pop();
            // 5a. we follow the use in case of shl instruction  (e.g., integer casted to int)
            if (const BinaryOperator* shl = dyn_cast<BinaryOperator>(id_use))
              if (shl->getOpcode() == Instruction::Shl) {
                // debug << "  *follow uses of int-to-unsigned cast* " << "\n";
                for (const User* shl_user : shl->users())
                  id_worklist.push(shl_user);
                continue;
              }
            // 5b. we check if the use corresponds to the pre-saved gep
            for (GetElementPtrInst* gep : gep_set) {
              if (id_use == gep->getOperand(1)) {
                // debug << "  *coalesced for global id*\n";
                cma.mem_coalesced++;
              }
            }
          } // end while worklist
        }
      }
    }
  }
  // 6. Further check gep with constant index
  for (GetElementPtrInst* gep : gep_set) {
    const Value* gep_op = gep->getOperand(1);
    if (const ConstantInt* cont_int = dyn_cast<ConstantInt>(gep_op)) {
      // debug << "  *coalesced for constant* " << "\n";
      cma.mem_coalesced++;
    }
  }
  assert(cma.mem_coalesced <= cma.mem_access);
  debug << cma.mem_coalesced << "/" << cma.mem_access << "\n";
  return cma;
}

cl_address_space_type get_cl_address_space_type(const unsigned addrSpaceId)
{
  switch (addrSpaceId) {
  case 0:
    return cl_address_space_type::Generic;
  case 1:
    return cl_address_space_type::Global;
  case 2:
    return cl_address_space_type::Region;
  case 3:
    return cl_address_space_type::Local;
  case 4:
    return cl_address_space_type::Constant;
  case 5:
    return cl_address_space_type::Private;
  default:
    errs() << "WARNING: unkwnown address space id: " << addrSpaceId << "\n";
    return cl_address_space_type::Generic;
  }
}

// /// Support utility functions to deal with memory accesses
// bool isGlobalMemoryAccess(const unsigned addrSpaceId) { return get_cl_address_space_type(addrSpaceId) == cl_address_space_type::Global; }

// bool isLocalMemoryAccess(const unsigned addrSpaceId) { return get_cl_address_space_type(addrSpaceId) == cl_address_space_type::Local; }

// bool isConstantMemoryAccess(const unsigned addrSpaceId) { return get_cl_address_space_type(addrSpaceId) == cl_address_space_type::Global; }

} // namespace celerity
