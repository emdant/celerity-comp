#include <cxxabi.h> //for demangling
#include <fstream>
#include <set>
#include <string>
using namespace std;

#include <llvm/IR/Instruction.h>
using namespace llvm;

#include "FeatureNormalization.hpp"
#include "FeatureSet.hpp"
#include "InstructionCollection.h"
#include "MemAccessFeature.hpp"
using namespace celerity;

const InstructionCollection INT_ADDSUB = {"add", "sub"};
const InstructionCollection INT_MUL = {"mul"};
const InstructionCollection INT_DIV = {"udiv", "sdiv", "sdivrem"};
// const InstructionCollection INT_REM    = {"urem","srem"}; // remainder of a division
const InstructionCollection FLOAT_ADDSUB = {"fadd", "fsub"};
const InstructionCollection FLOAT_MUL = {"fmul"};
const InstructionCollection FLOAT_DIV = {"fdiv"};
// const InstructionCollection FLOAT_REM  = {"frem"};
// const InstructionCollection CALL         = {"call"};
const InstructionCollection FNAME_SPECIAL = {"sqrt", "exp", "log", "abs", "fabs", "max", "pow", "floor", "sin", "cos", "tan"};
const InstructionCollection OPENCL = {"get_global_id", "get_local_id", "get_num_groups", "get_group_id", "get_max_sub_group_size", "max", "pow", "floor"};
const InstructionCollection BITWISE = {"shl", "lshr", "ashr", "and", "or", "xor"};
const InstructionCollection VECTOR = {"extractelement", "insertelement", "shufflevector"};
const InstructionCollection AGGREGATE = {"extractvalue", "insertvalue"};
const InstructionCollection INTRINSIC = {"llvm.fmuladd", "llvm.canonicalize", "llvm.smul.fix.sat", "llvm.umul.fix", "llvm.smul.fix", "llvm.sqrt", "llvm.powi",
    "llvm.sin", "llvm.cos", "llvm.pow", "llvm.exp", "llvm.exp2", "llvm.log", "llvm.log10", "llvm.log2", "llvm.fma", "llvm.fabs", "llvm.minnum", "llvm.maxnum",
    "llvm.minimum", "llvm.maximum", "llvm.copysign", "llvm.floor", "llvm.ceil", "llvm.trunc", "llvm.rint", "llvm.nearbyint", "llvm.round", "llvm.lround",
    "llvm.llround", "llvm.lrint", "llvm.llrint"};
const InstructionCollection BARRIER = {"barrier", "sub_group_reduce"};
const InstructionCollection CONTROL_FLOW = {"phi", "br", "brcond", "brindirect", "brjt"};
const InstructionCollection CONVERSION = {"uitofp", "fptosi", "sitofp", "bitcast"};
const InstructionCollection IGNORE = {"getelementptr", "alloca", "sext", "icmp", "fcmp", "zext", "trunc", "ret"};

const char* demangle_errors[] = {"The demangling operation succeeded", "A memory allocation failure occurred",
    "mangled_name is not a valid name under the C++ ABI mangling rules", "One of the arguments is invalid"};

string celerity::get_demangled_name(const llvm::CallInst& call_inst) {
	const StringRef& fun_name_sr = call_inst.getCalledFunction()->getGlobalIdentifier();
	return fun_name_sr.str();

	// unreachable code -- is string already demangled?
	const char* fun_name = fun_name_sr.str().c_str();
	int status;
	char* demangled_c_str = abi::__cxa_demangle(fun_name, 0, 0, &status);
	string demangled;
	if(demangled_c_str) {
		demangled = demangled_c_str;
		delete demangled_c_str;
	} else {
		demangled = fun_name;
		errs() << " DEMANGLE error for " << fun_name << " status:" << status << "\n"; // demangle_errors[std::abs(status)] << "\n";
	}
	outs() << " DEMANGLE " << fun_name << "->" << demangled << "\n";
	return demangled;
}

void FeatureSet::print(llvm::raw_ostream& out_stream) {
	out_stream << "raw values\n";
	out_stream.changeColor(llvm::raw_null_ostream::Colors::WHITE, true);
	print_feature_names(raw, out_stream);
	out_stream.changeColor(llvm::raw_null_ostream::Colors::WHITE, false);
	print_feature_values(raw, out_stream);
	out_stream << "feature values\n";
	out_stream.changeColor(llvm::raw_null_ostream::Colors::WHITE, true);
	print_feature_names(feat, out_stream);
	out_stream.changeColor(llvm::raw_null_ostream::Colors::WHITE, false);
	print_feature_values(feat, out_stream);
}

void FeatureSet::normalize(llvm::Function& fun) { celerity::normalize(*this); }

void Fan19FeatureSet::reset() {
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

void Fan19FeatureSet::eval(llvm::Instruction& inst, int contribution) {
	string i_name = inst.getOpcodeName();
	// outs() << "  OPCODE: " << i_name << "\n";
	if(INT_ADDSUB.contains(i_name)) {
		add("int_add", contribution);
		return;
	}
	if(INT_MUL.contains(i_name)) {
		add("int_mul", contribution);
		return;
	}
	if(INT_DIV.contains(i_name)) {
		add("int_div", contribution);
		return;
	}
	if(BITWISE.contains(i_name)) {
		add("int_bw", contribution);
		return;
	}
	if(FLOAT_ADDSUB.contains(i_name)) {
		add("flt_add", contribution);
		return;
	}
	if(FLOAT_MUL.contains(i_name)) {
		add("flt_mul", contribution);
		return;
	}
	if(FLOAT_DIV.contains(i_name)) {
		add("flt_div", contribution);
		return;
	}

	// chack function calls
	if(const CallInst* ci = dyn_cast<CallInst>(&inst)) {
		Function* func = ci->getCalledFunction();
		// check intrinsic
		if(ci->getIntrinsicID() != Intrinsic::not_intrinsic) {
			string intrinsic_name = Intrinsic::getName(ci->getIntrinsicID()).str();
			if(INTRINSIC.contains(intrinsic_name))
				add("sp_fun", contribution);
			else
				errs() << "WARNING: fan19: intrinsic " << intrinsic_name << " not recognized\n";
			return;
		}

		// check special functions
		string fun_name = get_demangled_name(*ci);
		if(FNAME_SPECIAL.contains(fun_name))
			add("sp_fun", contribution);
		else // all other function calls
			errs() << "WARNING: fan19: function " << fun_name << " not recognized\n";
		return;
	}

	// global & local memory access
	if(const LoadInst* li = dyn_cast<LoadInst>(&inst)) // check load instruction
	{
		unsigned address_space = li->getPointerAddressSpace();
		if(isLocalMemoryAccess(address_space)) add("mem_loc", contribution);
		if(isGlobalMemoryAccess(address_space)) add("mem_gl", contribution);
		return;
	} else if(const StoreInst* si = dyn_cast<StoreInst>(&inst)) // check store instuction
	{
		unsigned address_space = si->getPointerAddressSpace();
		if(isLocalMemoryAccess(address_space)) add("mem_loc", contribution);
		if(isGlobalMemoryAccess(address_space)) add("mem_gl", contribution);
		return;
	}

	// instruction ignored
	bool ignore = CONTROL_FLOW.contains(i_name) || CONVERSION.contains(i_name) || OPENCL.contains(i_name) || VECTOR.contains(i_name)
	              || AGGREGATE.contains(i_name) || IGNORE.contains(i_name);
	if(ignore) return;

	// any other instruction
	errs() << "WARNING: fan19: opcode " << i_name << " not recognized\n";
}

void Grewe11FeatureSet::reset() {
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

void Grewe11FeatureSet::eval(llvm::Instruction& inst, int contribution) {
	string i_name = inst.getOpcodeName();
	unsigned opcode = inst.getOpcode();
	// outs() << "  OPCODE: " << i_name << "\n";
	//  int
	if(INT_ADDSUB.contains(i_name)) {
		add("int", contribution);
		return;
	}
	if(INT_MUL.contains(i_name)) {
		add("int", contribution);
		return;
	}
	if(INT_DIV.contains(i_name)) {
		add("int", contribution);
		return;
	}
	if(BITWISE.contains(i_name)) {
		add("int", contribution);
		return;
	}
	// float
	if(FLOAT_ADDSUB.contains(i_name)) {
		add("float", contribution);
		return;
	}
	if(FLOAT_MUL.contains(i_name)) {
		add("float", contribution);
		return;
	}
	if(FLOAT_DIV.contains(i_name)) {
		add("float", contribution);
		return;
	}

	// check function calls
	if(const CallInst* ci = dyn_cast<CallInst>(&inst)) {
		Function* func = ci->getCalledFunction();
		// check intrinsic
		if(ci->getIntrinsicID() != Intrinsic::not_intrinsic) {
			string intrinsic_name = Intrinsic::getName(ci->getIntrinsicID()).str();
			if(INTRINSIC.contains(intrinsic_name))
				add("math", contribution);
			else
				errs() << "WARNING: grewe11: intrinsic " << intrinsic_name << " not recognized\n";
			return;
		}

		// handling function calls
		string fun_name = get_demangled_name(*ci);
		if(FNAME_SPECIAL.contains(fun_name)) // math
			add("math", contribution);
		else if(BARRIER.contains(fun_name)) // barrier
			add("barrier", contribution);
		else if(OPENCL.contains(i_name)) // ignore list of OpenCL functions
			;
		else // all other function calls
			errs() << "WARNING: grewe11: function " << fun_name << "/" << func->getGlobalIdentifier() << " not recognized\n";
		return;
	}
	// check load
	if(const LoadInst* li = dyn_cast<LoadInst>(&inst)) {
		add("mem_acc", contribution);
		unsigned address_space = li->getPointerAddressSpace();
		if(isLocalMemoryAccess(address_space)) add("mem_loc", contribution);
		return;
	}
	// local mem access
	else if(const StoreInst* si = dyn_cast<StoreInst>(&inst)) {
		add("mem_acc", contribution);
		unsigned address_space = si->getPointerAddressSpace();
		if(isLocalMemoryAccess(address_space)) add("mem_loc", contribution);
		return;
	}
	// int4 TODO
	// float4 TODO
}

void Grewe11FeatureSet::normalize(llvm::Function& fun) {
	CoalescedMemAccess ret = getCoalescedMemAccess(fun);
	// assert(ret.mem_access == raw["mem_acc"]);
	if(ret.mem_access == 0)
		feat["mem_coal"] = 0.f;
	else
		feat["mem_coal"] = float(ret.mem_coalesced) / float(ret.mem_access);
}

void FullFeatureSet::eval(llvm::Instruction& inst, int contribution) { add(inst.getOpcodeName(), contribution); }

//-----------------------------------------------------------------------------
// Register the available feature sets in the FeatureSet registry
//-----------------------------------------------------------------------------
static celerity::FeatureSet* _static_fs_1_ = new celerity::Fan19FeatureSet();
static bool _registered_fset_1_ = FSRegistry::registerByKey("fan19", _static_fs_1_);
static celerity::FeatureSet* _static_fs_2_ = new celerity::Grewe11FeatureSet();
static bool _registered_fset_2_ = FSRegistry::registerByKey("grewe11", _static_fs_2_);
static celerity::FeatureSet* _static_fs_3_ = new celerity::FullFeatureSet();
static bool _registered_fset_3_ = FSRegistry::registerByKey("full", _static_fs_3_);
