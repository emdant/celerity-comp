#include <iostream>
#include <set>
#include <string>

namespace celerity {

class InstructionCollection {
public:
  InstructionCollection() = delete;

  inline InstructionCollection(std::initializer_list<std::string> list) : instructions(list) {}

  inline bool contains_exactely(const std::string& instruction_name) const { return instructions.find(instruction_name) != instructions.end(); }

  inline bool substring_of(const std::string& instruction_name) const
  {
    for (const std::string& s : instructions) {
      if (instruction_name.find(s) != std::string::npos)
        return true;
    }
    return false;
  }

private:
  std::set<std::string> instructions;
};

const InstructionCollection INT_ADDSUB = {"add", "sub"};
const InstructionCollection INT_MUL = {"mul"};
const InstructionCollection INT_DIV = {"udiv", "sdiv", "sdivrem"};
// const InstructionCollection INT_REM    = {"urem","srem"}; // remainder of a division
const InstructionCollection BITWISE = {"shl", "lshr", "ashr", "and", "or", "xor"};

const InstructionCollection FLOAT_ADDSUB = {"fadd", "fsub"};
const InstructionCollection FLOAT_MUL = {"fmul"};
const InstructionCollection FLOAT_DIV = {"fdiv"};
// const InstructionCollection FLOAT_REM  = {"frem"};

const InstructionCollection FNAME_SPECIAL = {"sqrt", "exp", "log", "abs", "fabs", "max", "pow", "floor", "sin", "cos", "tan"};
const InstructionCollection MATH_INTRINSIC = {"llvm.fmuladd", "llvm.canonicalize", "llvm.smul.fix.sat", "llvm.umul.fix", "llvm.smul.fix", "llvm.sqrt", "llvm.powi", "llvm.sin", "llvm.cos", "llvm.pow", "llvm.exp", "llvm.exp2", "llvm.log", "llvm.log10", "llvm.log2", "llvm.fma", "llvm.fabs", "llvm.minnum", "llvm.maxnum", "llvm.minimum", "llvm.maximum", "llvm.copysign", "llvm.floor", "llvm.ceil", "llvm.trunc", "llvm.rint", "llvm.nearbyint", "llvm.round", "llvm.lround", "llvm.llround", "llvm.lrint", "llvm.llrint"};

const InstructionCollection GENERAL_INTRINSIC = {"llvm.assume", "llvm.lifetime.start", "llvm.lifetime.end", "llvm.experimental.noalias.scope.decl"};
const InstructionCollection SYCL = {"GlobalSize", "GlobalOffset", "GlobalInvocationId", "NumWorkgroups", "WorkgroupSize", "WorkgroupId", "LocalInvocationId"};
const InstructionCollection OPENCL = {"get_global_id", "get_local_id", "get_num_groups", "get_group_id", "get_max_sub_group_size", "max", "pow", "floor"};
const InstructionCollection VECTOR = {"extractelement", "insertelement", "shufflevector"};
const InstructionCollection AGGREGATE = {"extractvalue", "insertvalue"};
const InstructionCollection BARRIER = {"barrier", "sub_group_reduce"};
const InstructionCollection CONTROL_FLOW = {"phi", "br", "brcond", "brindirect", "brjt", "select"};
const InstructionCollection CONVERSION = {"uitofp", "fptosi", "sitofp", "bitcast", "addrspacecast"};
const InstructionCollection IGNORE = {"getelementptr", "alloca", "sext", "icmp", "fcmp", "zext", "trunc", "ret", "freeze"};
// const InstructionCollection CALL         = {"call"};

} // namespace celerity
