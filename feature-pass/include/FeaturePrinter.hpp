#pragma once

#include <type_traits>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
using namespace llvm;

#include "AnalysisUtils.hpp"
#include "FeatureAnalysis.hpp"

namespace celerity {

// Pass printing the results of a feature analysis.
template <typename AnalysisType>
struct FeaturePrinterPass : public llvm::PassInfoMixin<celerity::FeaturePrinterPass<AnalysisType>> {
  static_assert(std::is_base_of<FeatureAnalysis, AnalysisType>::value, "AnalysisType must derive from FeatureAnalysis");

public:
  explicit FeaturePrinterPass(llvm::raw_ostream& stream) : out_stream(stream) {}

  llvm::PreservedAnalyses run(llvm::Function& fun, llvm::FunctionAnalysisManager& fam)
  {
    using rfa = ResultFeatureAnalysis;

    ResultFeatureAnalysis& analysis_result = fam.getResult<AnalysisType>(fun);

    if (!analysis_result.printResult)
      return PreservedAnalyses::all();

    if (!out_stream.is_displayed())
      out_stream.enable_colors(false);

    out_stream.changeColor(llvm::raw_null_ostream::Colors::MAGENTA);
    out_stream << "Print features for function: " << fun.getName() << "\n";
    out_stream.changeColor(llvm::raw_null_ostream::Colors::YELLOW);

    out_stream.changeColor(llvm::raw_null_ostream::Colors::WHITE, true);
    print_feature_names<rfa::counters_type>(analysis_result.features_counters, out_stream);
    out_stream.changeColor(llvm::raw_null_ostream::Colors::WHITE, false);
    print_feature_values<rfa::counters_type>(analysis_result.features_counters, out_stream);

    out_stream.changeColor(llvm::raw_null_ostream::Colors::WHITE, true);
    print_feature_names<rfa::normalization_type>(analysis_result.features_normalized, out_stream);
    out_stream.changeColor(llvm::raw_null_ostream::Colors::WHITE, false);
    print_feature_values<rfa::normalization_type>(analysis_result.features_normalized, out_stream);

    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }

private:
  llvm::raw_ostream& out_stream;
};

} // end namespace celerity
