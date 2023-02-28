#include <cxxabi.h>
#include <sstream>
#include <unordered_map>
#include <vector>
using namespace std;

#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
using namespace llvm;

#include "FeatureAnalysis.hpp"
#include "FeaturePrinter.hpp"
#include "KernelInvariant.hpp"
#include "Kofler13Analysis.hpp"
using namespace celerity;

FeatureAnalysis::~FeatureAnalysis() {}

void FeatureAnalysis::extract(BasicBlock& bb)
{
  for (Instruction& i : bb) {
    features->eval(i);
  }
}

void FeatureAnalysis::extract(llvm::Function& fun, llvm::FunctionAnalysisManager& fam)
{
  KernelInvariant ki(fun);
  ki.print(llvm::outs());

  for (llvm::BasicBlock& bb : fun)
    extract(bb);
}

void FeatureAnalysis::finalize(llvm::Function& fun)
{
  // normalize(*features);
  features->normalize(fun);
}

ResultFeatureAnalysis FeatureAnalysis::run(llvm::Function& fun, llvm::FunctionAnalysisManager& fam)
{
  // skip the function if it is only a declaration
  if (fun.isDeclaration())
    return ResultFeatureAnalysis{false, features->getFeatureCounts(), features->getFeatureValues()};

  // skip the function if it is a SYCL kernel wrapper
  if (fun.getName().find("kernel_wrapper") != llvm::StringRef::npos)
    return ResultFeatureAnalysis{false, features->getFeatureCounts(), features->getFeatureValues()};

  if (fun.getName().find("RoundedRangeKernel") != llvm::StringRef::npos)
    return ResultFeatureAnalysis{false, features->getFeatureCounts(), features->getFeatureValues()};

  // nicely printing analysis params
  llvm::raw_ostream& debug = outs();

  if (!debug.is_displayed())
    debug.enable_colors(false);

  debug.changeColor(llvm::raw_null_ostream::Colors::YELLOW, true);
  debug << "IR-function: ";
  debug.changeColor(llvm::raw_null_ostream::Colors::WHITE, false);
  debug << fun.getName().str() << "\n";

  debug.changeColor(llvm::raw_null_ostream::Colors::YELLOW, true);
  debug << "demangled-function: ";
  debug.changeColor(llvm::raw_null_ostream::Colors::WHITE, false);
  debug << get_demangled_name(fun) << "\n";

  debug.changeColor(llvm::raw_null_ostream::Colors::YELLOW, true);
  debug << "feature-set: ";
  debug.changeColor(llvm::raw_null_ostream::Colors::WHITE, false);
  debug << features->getName();

  debug.changeColor(llvm::raw_null_ostream::Colors::YELLOW, true);
  debug << " analysis-name: ";
  debug.changeColor(llvm::raw_null_ostream::Colors::WHITE, false);
  debug << getName() << "\n";

  // reset all feature values
  features->reset();

  // feature extraction
  extract(fun, fam);
  // feature post-processing (e.g., normalization)
  finalize(fun);
  return ResultFeatureAnalysis{true, features->getFeatureCounts(), features->getFeatureValues()};
}
