#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>

#include "AnalysisUtils.hpp"
#include "FeatureSet.hpp"
#include "InstructionCollection.h"

using namespace std;
using namespace llvm;
using namespace celerity;

void Fan19FeatureSet::reset()
{
  raw["int_add"] = 0;
  raw["int_mul"] = 0;
  raw["int_div"] = 0;
  raw["int_bw"] = 0;
  raw["flt_add"] = 0;
  raw["flt_mul"] = 0;
  raw["flt_div"] = 0;
  raw["sp_fun"] = 0;
  raw["mem_gl"] = 0;
  raw["mem_loc"] = 0;
}

void Fan19FeatureSet::eval(llvm::Instruction& inst, int contribution)
{
  string i_name = inst.getOpcodeName();
  // outs() << "  OPCODE: " << i_name << "\n";
  if (INT_ADDSUB.contains(i_name)) {
    add("int_add", contribution);
    return;
  }
  if (INT_MUL.contains(i_name)) {
    add("int_mul", contribution);
    return;
  }
  if (INT_DIV.contains(i_name)) {
    add("int_div", contribution);
    return;
  }
  if (BITWISE.contains(i_name)) {
    add("int_bw", contribution);
    return;
  }
  if (FLOAT_ADDSUB.contains(i_name)) {
    add("flt_add", contribution);
    return;
  }
  if (FLOAT_MUL.contains(i_name)) {
    add("flt_mul", contribution);
    return;
  }
  if (FLOAT_DIV.contains(i_name)) {
    add("flt_div", contribution);
    return;
  }

  // check function calls
  if (const CallInst* ci = dyn_cast<CallInst>(&inst)) {
    Function* func = ci->getCalledFunction();
    // check intrinsic
    if (ci->getIntrinsicID() != Intrinsic::not_intrinsic) {
      string intrinsic_name = Intrinsic::getName(ci->getIntrinsicID()).str();
      if (MATH_INTRINSIC.contains(intrinsic_name))
        add("sp_fun", contribution);
      else if (!GENERAL_INTRINSIC.contains(intrinsic_name))
        errs() << "WARNING: fan19: intrinsic " << intrinsic_name << " not recognized\n";
      return;
    }

    // check special functions
    string fun_name = get_demangled_name(*func);
    if (FNAME_SPECIAL.contains(fun_name))
      add("sp_fun", contribution);
    else // all other function calls
      errs() << "WARNING: fan19: function " << fun_name << " not recognized\n";
    return;
  }

  // global & local memory access
  if (const LoadInst* li = dyn_cast<LoadInst>(&inst)) // check load instruction
  {
    const Instruction* previous = li->getPrevNode();
    const AddrSpaceCastInst* cast_inst = dyn_cast<AddrSpaceCastInst>(previous);

    if (cast_inst) {
      cast_inst->getSrcAddressSpace();
    }

    unsigned address_space = li->getPointerAddressSpace();
    std::cout << "load, address_space_type: " << address_space << "\n";
    if (isLocalMemoryAccess(address_space))
      add("mem_loc", contribution);
    if (isGlobalMemoryAccess(address_space))
      add("mem_gl", contribution);
    return;
  } else if (const StoreInst* si = dyn_cast<StoreInst>(&inst)) // check store instuction
  {
    unsigned address_space = si->getPointerAddressSpace();
    std::cout << "store, address_space_type: " << address_space << "\n";
    if (isLocalMemoryAccess(address_space))
      add("mem_loc", contribution);
    if (isGlobalMemoryAccess(address_space))
      add("mem_gl", contribution);
    return;
  }

  // instruction ignored
  bool ignore = CONTROL_FLOW.contains(i_name) || CONVERSION.contains(i_name) || OPENCL.contains(i_name) || VECTOR.contains(i_name) || AGGREGATE.contains(i_name) || IGNORE.contains(i_name);
  if (ignore)
    return;

  // any other instruction
  errs() << "WARNING: fan19: opcode " << i_name << " not recognized\n";
}

static celerity::FeatureSet* _static_fs_1_ = new celerity::Fan19FeatureSet();
static bool _registered_fset_1_ = FSRegistry::registerByKey("fan19", _static_fs_1_);