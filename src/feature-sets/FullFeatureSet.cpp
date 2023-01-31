#include <llvm/IR/Instruction.h>

#include "FeatureSet.hpp"
using namespace celerity;

void FullFeatureSet::eval(llvm::Instruction& inst, int contribution) { add(inst.getOpcodeName(), contribution); }

static celerity::FeatureSet* _static_fs_3_ = new celerity::FullFeatureSet();
static bool _registered_fset_3_ = FSRegistry::registerByKey("full", _static_fs_3_);