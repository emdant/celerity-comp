#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>

#include "AnalysisUtils.hpp"
#include "FeatureSet.hpp"
#include "InstructionCollection.h"

using namespace std;
using namespace llvm;
using namespace celerity;

void Fan19FeatureSet::eval(llvm::Instruction& instruction, int contribution)
{
  string opcode = instruction.getOpcodeName();

  if (INT_ADDSUB.contains_exactely(opcode)) {
    add("int_add", contribution);
    return;
  }
  if (INT_MUL.contains_exactely(opcode)) {
    add("int_mul", contribution);
    return;
  }
  if (INT_DIV.contains_exactely(opcode)) {
    add("int_div", contribution);
    return;
  }
  if (BITWISE.contains_exactely(opcode)) {
    add("int_bw", contribution);
    return;
  }
  if (FLOAT_ADDSUB.contains_exactely(opcode)) {
    add("flt_add", contribution);
    return;
  }
  if (FLOAT_MUL.contains_exactely(opcode)) {
    add("flt_mul", contribution);
    return;
  }
  if (FLOAT_DIV.contains_exactely(opcode)) {
    add("flt_div", contribution);
    return;
  }

  // check function calls
  if (const CallInst* ci = dyn_cast<CallInst>(&instruction)) {
    Function* function = ci->getCalledFunction();
    // check intrinsic
    if (ci->getIntrinsicID() != Intrinsic::not_intrinsic) {
      string intrinsic_name = Intrinsic::getName(ci->getIntrinsicID()).str();

      if (MATH_INTRINSIC.substring_of(intrinsic_name))
        add("sp_fun", contribution);
      else if (!GENERAL_INTRINSIC.substring_of(intrinsic_name))
        errs() << "WARNING: fan19: intrinsic " << intrinsic_name << " not recognized\n";

      return;
    }

    // check special functions
    string function_name = get_demangled_name(*function);

    if (FNAME_SPECIAL.substring_of(function_name))
      add("sp_fun", contribution);
    else if (!(SYCL.substring_of(function_name) || OPENCL.substring_of(function_name)))
      errs() << "WARNING: fan19: function " << function_name << " not recognized\n";

    return;
  }

  // global & local memory access
  if (isa<LoadInst>(instruction) || isa<StoreInst>(instruction)) {
    // const llvm::Instruction* previous = instruction.getPrevNode();

    unsigned address_space = 0;
    if (const LoadInst* li = dyn_cast<LoadInst>(&instruction)) {
      const Value* val = li->getPointerOperand();

      for (const Use& use : val->uses()) {
        if (const AddrSpaceCastInst* addrspace_inst = dyn_cast<AddrSpaceCastInst>(use))
          address_space = addrspace_inst->getSrcAddressSpace();
      }
    } else if (const StoreInst* si = dyn_cast<StoreInst>(&instruction)) {
      const Value* val = si->getPointerOperand();

      for (const Use& use : val->uses()) {
        if (const AddrSpaceCastInst* addrspace_inst = dyn_cast<AddrSpaceCastInst>(use))
          address_space = addrspace_inst->getSrcAddressSpace();
      }
    }

    // if (const AddrSpaceCastInst* cast_inst = dyn_cast<AddrSpaceCastInst>(previous))
    //   address_space = cast_inst->getSrcAddressSpace();

    if (isGlobalMemoryAccess(address_space))
      add("mem_gl", contribution);
    if (isLocalMemoryAccess(address_space))
      add("mem_loc", contribution);

    return;
  }

  // instruction ignored
  bool ignore_instruction = CONTROL_FLOW.contains_exactely(opcode) || CONVERSION.contains_exactely(opcode) || OPENCL.substring_of(opcode) ||
                            VECTOR.contains_exactely(opcode) || AGGREGATE.contains_exactely(opcode) || IGNORE.contains_exactely(opcode);
  if (ignore_instruction)
    return;

  // any other instruction
  errs() << "WARNING: fan19: opcode " << opcode << " not recognized\n";
}

static celerity::FeatureSet* _static_fs_1_ = new celerity::Fan19FeatureSet();
static bool _registered_fset_1_ = FSRegistry::registerByKey("fan19", _static_fs_1_);