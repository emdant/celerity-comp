//
// Created by nadjib on 20.06.19.
//
#pragma once

#include <fstream>
#include <iostream>
#include <regex>
#include <string>

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace celerity {

/*
 * An LLVM function pass to extract features.
 * The extraction of features from a single instruction is delegated to a feature set class.
 * In this basic implementation, BB's instruction contributions are summed up.
 */
class CelerityInterfacePass : public llvm::ModulePass {
public:
  static char ID;
  // By default we go for std:cout as the output stream
  std::ostream* outputstreamPtr = &std::cout;

  CelerityInterfacePass() : ModulePass(ID) {}

  ~CelerityInterfacePass() {}

  virtual void getAnalysisUsage(llvm::AnalysisUsage& au) const { au.setPreservesAll(); }

  virtual bool runOnModule(llvm::Module& m);
  virtual bool runOnFunction(llvm::Function& f);

  virtual void printInterfaceHeader();
  virtual void printKernelClass(const std::string& kernelName, llvm::Function& f);

  virtual bool isItaniumEncoding(const std::string& MangledName);
  virtual std::string demangle(const std::string& MangledName);

  // Getter and setter for current output stream
  std::ostream& outputstream() { return *outputstreamPtr; }

  void setOutputStream(std::ostream& outs = std::cout) { outputstreamPtr = &outs; }
};

} // namespace celerity
