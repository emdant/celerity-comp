#pragma once

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>

#include "FeatureSet.hpp"
#include "Registry.hpp"

namespace celerity {

/// Results of a feature analysis
struct ResultFeatureAnalysis {
  using counters_type = unsigned;
  using normalization_type = float;

  bool printResult;
  llvm::StringMap<counters_type> features_counters;
  llvm::StringMap<normalization_type> features_normalized;
};

/// Abstract class for analyses that extract static code features.
/// The extraction of features from a single instruction is delegated to a feature set class.
class FeatureAnalysis {
public:
  FeatureAnalysis(std::string feature_set = "fan19") : analysis_name("default") { features = FSRegistry::dispatch(feature_set); }
  virtual ~FeatureAnalysis();

  /// this methods allow to change the underlying feature set
  inline void setFeatureSet(std::string& feature_set) { features = FSRegistry::dispatch(feature_set); }
  inline FeatureSet* getFeatureSet() { return features; }
  inline std::string getName() { return analysis_name; }

  /// runs the analysis on a specific function, returns a StringMap
  using Result = ResultFeatureAnalysis;
  ResultFeatureAnalysis run(llvm::Function& fun, llvm::FunctionAnalysisManager& fam);

  /// feature extraction for basic block
  virtual void extract(llvm::BasicBlock& bb);
  /// feature extraction for function
  virtual void extract(llvm::Function& fun, llvm::FunctionAnalysisManager& fam);
  /// apply feature postprocessing steps such as normalization
  virtual void finalize(llvm::Function& fun);

  inline static bool isRequired() { return true; }

protected:
  FeatureSet* features;
  std::string analysis_name;
}; // end FeatureAnalysis

using FARegistry = Registry<celerity::FeatureAnalysis*>;

} // end namespace celerity
