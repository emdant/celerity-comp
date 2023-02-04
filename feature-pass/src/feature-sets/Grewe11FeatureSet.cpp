
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>

#include "AnalysisUtils.hpp"
#include "FeatureSet.hpp"
#include "InstructionCollection.h"

using namespace std;
using namespace llvm;
using namespace celerity;

void Grewe11FeatureSet::eval(Instruction& instruction, int contribution)
{
  string opcode = instruction.getOpcodeName();

  if (INT_ADDSUB.contains_exactely(opcode)) {
    add("int", contribution);
    return;
  }
  if (INT_MUL.contains_exactely(opcode)) {
    add("int", contribution);
    return;
  }
  if (INT_DIV.contains_exactely(opcode)) {
    add("int", contribution);
    return;
  }
  if (BITWISE.contains_exactely(opcode)) {
    add("int", contribution);
    return;
  }
  // float
  if (FLOAT_ADDSUB.contains_exactely(opcode)) {
    add("float", contribution);
    return;
  }
  if (FLOAT_MUL.contains_exactely(opcode)) {
    add("float", contribution);
    return;
  }
  if (FLOAT_DIV.contains_exactely(opcode)) {
    add("float", contribution);
    return;
  }

  // check function calls
  if (const CallInst* ci = dyn_cast<CallInst>(&instruction)) {
    Function* function = ci->getCalledFunction();
    // check intrinsic
    if (ci->getIntrinsicID() != Intrinsic::not_intrinsic) {
      string intrinsic_name = Intrinsic::getName(ci->getIntrinsicID()).str();

      if (MATH_INTRINSIC.substring_of(intrinsic_name))
        add("math", contribution);
      else if (!GENERAL_INTRINSIC.substring_of(intrinsic_name))
        errs() << "WARNING: grewe11: intrinsic " << intrinsic_name << " not recognized\n";

      return;
    }

    // handling function calls
    string function_name = get_demangled_name(*function);
    if (FNAME_SPECIAL.substring_of(function_name)) // math
      add("math", contribution);
    else if (BARRIER.substring_of(function_name)) // barrier
      add("barrier", contribution);
    else if (!(SYCL.substring_of(function_name) || OPENCL.substring_of(function_name)))
      errs() << "WARNING: grewe11: function " << function_name << " not recognized\n";

    return;
  }

  // check load
  if (const LoadInst* li = dyn_cast<LoadInst>(&instruction)) {
    add("mem_acc", contribution);
    unsigned address_space = li->getPointerAddressSpace();
    if (isLocalMemoryAccess(address_space))
      add("mem_loc", contribution);
    return;
  }
  // local mem access
  else if (const StoreInst* si = dyn_cast<StoreInst>(&instruction)) {
    add("mem_acc", contribution);
    unsigned address_space = si->getPointerAddressSpace();
    if (isLocalMemoryAccess(address_space))
      add("mem_loc", contribution);
    return;
  }
  // int4 TODO
  // float4 TODO
}

void Grewe11FeatureSet::normalize(Function& fun)
{
  CoalescedMemAccess ret = getCoalescedMemAccess(fun);
  // assert(ret.mem_access == raw["mem_acc"]);
  if (ret.mem_access == 0)
    feat["mem_coal"] = 0.f;
  else
    feat["mem_coal"] = float(ret.mem_coalesced) / float(ret.mem_access);
}

static celerity::FeatureSet* _static_fs_2_ = new celerity::Grewe11FeatureSet();
static bool _registered_fset_2_ = FSRegistry::registerByKey("grewe11", _static_fs_2_);