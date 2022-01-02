#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>

#include "FeatureAnalysis.hpp"
#include "FeatureSet.hpp"

using namespace llvm;

namespace celerity {


/// An LLVM analysis pass to extract features using [Kofler et al., 13] loop heuristics.
/// The heuristic gives more important (x100) to the features inside a loop.
/// It requires the loop analysis pass ("loops") to be executed before of that pass.
struct Kofler13Analysis : public FeatureAnalysis {
 private:
   const int default_loop_contribution = 100;

 public:
    Kofler13Analysis(string feature_set = "fan19") : FeatureAnalysis() { 
      analysis_name="kofler13"; 
      features = FSRegistry::dispatch(feature_set);
    }
    virtual ~Kofler13Analysis(){}

    /// overwrite feature extraction for function
    virtual void extract(llvm::Function &fun, llvm::FunctionAnalysisManager &fam);
    // calculate del loop contribution of a given loop (assume non nesting, which is calculated later)
    int loopContribution(const llvm::Loop &loop, ScalarEvolution &SE);

    static llvm::AnalysisKey Key;
};

} // end namespace celerity
