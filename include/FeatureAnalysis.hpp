#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
using namespace llvm;

#include "FeatureSet.hpp"
#include "Registry.hpp"

namespace celerity {

//using ResultFeatureAnalysis = llvm::StringMap<float>;
struct ResultFeatureAnalysis {
  llvm::StringMap<unsigned> raw;
  llvm::StringMap<float> feat;
};

/// An LLVM analysisfunction pass that extract static code features. 
/// The extraction of features from a single instruction is delegated to a feature set class.
/// In this basic implementation, BB's instruction contributions are summed up.
struct FeatureAnalysis : public llvm::AnalysisInfoMixin<FeatureAnalysis> {

 protected:
    FeatureSet *features;
    string analysis_name;
    // TODO normalization must handled here in the samo way (Normalization)
    
 public:
    FeatureAnalysis(string feature_set = "fan19") : analysis_name("default") {      
      features = FSRegistry::dispatch(feature_set);      
    }
    virtual ~FeatureAnalysis();

    /// this methods allow to change the underlying feature set
    void setFeatureSet(string &feature_set){ features = FSRegistry::dispatch(feature_set); }
    FeatureSet * getFeatureSet(){ return features; }
    string getName(){ return analysis_name; }

    /// runs the analysis on a specific function, returns a StringMap
    using Result = ResultFeatureAnalysis;
    ResultFeatureAnalysis run(llvm::Function &fun, llvm::FunctionAnalysisManager &fam);

    /// feature extraction for basic block
    virtual void extract(llvm::BasicBlock &bb);	
    /// feature extraction for function
    virtual void extract(llvm::Function &fun, llvm::FunctionAnalysisManager &fam);
    /// apply feature postprocessing steps such as normalization
    virtual void finalize();

    static bool isRequired() { return true; }
 
  private:
    static llvm::AnalysisKey Key;
    friend struct llvm::AnalysisInfoMixin<FeatureAnalysis>;   
    //friend struct FeaturePrinterPass; //?

}; // end FeatureAnalysis


struct FeatureAnalysisParam { 
  FeatureSetOptions feature_set; 
  string analysis; 
  string normalization; 
  string filename;
  bool help;
  bool verbose;
};

using FARegistry = Registry<celerity::FeatureAnalysis*>;

} // end namespace celerity
