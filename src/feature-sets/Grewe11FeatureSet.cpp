
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>

#include "AnalysisUtils.hpp"
#include "FeatureSet.hpp"
#include "InstructionCollection.h"

using namespace std;
using namespace llvm;
using namespace celerity;

void Grewe11FeatureSet::reset()
{
  raw["int"] = 0;
  raw["int4"] = 0;
  raw["float"] = 0;
  raw["float4"] = 0;
  raw["math"] = 0;
  raw["barrier"] = 0;
  raw["mem_acc"] = 0;
  raw["mem_loc"] = 0;
  // raw["per_local_mem"]=0;
  // raw["per_coalesced"]=0;
  raw["mem_coal"] = 0;
  // raw["comp_mem_ratio"]=0;
  // raw["data_transfer"]=0;
  // raw["comp_per_data"]=0;
  // raw["workitems"]=0;
}

void Grewe11FeatureSet::eval(Instruction& inst, int contribution)
{
  string i_name = inst.getOpcodeName();
  unsigned opcode = inst.getOpcode();
  // outs() << "  OPCODE: " << i_name << "\n";
  //  int
  if (INT_ADDSUB.contains(i_name)) {
    add("int", contribution);
    return;
  }
  if (INT_MUL.contains(i_name)) {
    add("int", contribution);
    return;
  }
  if (INT_DIV.contains(i_name)) {
    add("int", contribution);
    return;
  }
  if (BITWISE.contains(i_name)) {
    add("int", contribution);
    return;
  }
  // float
  if (FLOAT_ADDSUB.contains(i_name)) {
    add("float", contribution);
    return;
  }
  if (FLOAT_MUL.contains(i_name)) {
    add("float", contribution);
    return;
  }
  if (FLOAT_DIV.contains(i_name)) {
    add("float", contribution);
    return;
  }

  // check function calls
  if (const CallInst* ci = dyn_cast<CallInst>(&inst)) {
    Function* func = ci->getCalledFunction();
    // check intrinsic
    if (ci->getIntrinsicID() != Intrinsic::not_intrinsic) {
      string intrinsic_name = Intrinsic::getName(ci->getIntrinsicID()).str();
      if (MATH_INTRINSIC.contains(intrinsic_name))
        add("math", contribution);
      else if (!GENERAL_INTRINSIC.contains(intrinsic_name))
        errs() << "WARNING: grewe11: intrinsic " << intrinsic_name << " not recognized\n";
      return;
    }

    // handling function calls
    string fun_name = get_demangled_name(*func);
    if (FNAME_SPECIAL.contains(fun_name)) // math
      add("math", contribution);
    else if (BARRIER.contains(fun_name)) // barrier
      add("barrier", contribution);
    else if (OPENCL.contains(i_name)) // ignore list of OpenCL functions
      ;
    else // all other function calls
      errs() << "WARNING: grewe11: function " << fun_name << "/" << func->getGlobalIdentifier() << " not recognized\n";
    return;
  }
  // check load
  if (const LoadInst* li = dyn_cast<LoadInst>(&inst)) {
    add("mem_acc", contribution);
    unsigned address_space = li->getPointerAddressSpace();
    if (isLocalMemoryAccess(address_space))
      add("mem_loc", contribution);
    return;
  }
  // local mem access
  else if (const StoreInst* si = dyn_cast<StoreInst>(&inst)) {
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