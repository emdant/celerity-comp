#pragma once

#include <map>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>

namespace celerity {

struct ResultCoalescedAnalysis {
  std::map<llvm::Value*, bool> mem_access;
};

struct CoalescedAnalysis : public llvm::AnalysisInfoMixin<CoalescedAnalysis> {
protected:
  std::map<llvm::Value*, bool> mem_access;

public:
  CoalescedAnalysis() {}
  virtual ~CoalescedAnalysis() {}

  using Result = ResultCoalescedAnalysis;
  ResultCoalescedAnalysis run(llvm::Function& fun, llvm::FunctionAnalysisManager& FAM)
  {
    llvm::DominatorTree& DT = FAM.getResult<llvm::DominatorTreeAnalysis>(fun);
    // TODO more accurate coalesced mm=emory access
    return {mem_access};
  }

  friend struct llvm::AnalysisInfoMixin<CoalescedAnalysis>;
  static llvm::AnalysisKey Key;
}; // end DefaultFeatureAnalysis

} // namespace celerity