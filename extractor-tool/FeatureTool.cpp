#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/InitLLVM.h>

#include "ExtractorCommandLine.hpp"

using namespace std;
using namespace llvm;

/// function to load a module from file
std::unique_ptr<Module> load_module(LLVMContext& context, const std::string& fileName, bool verbose)
{
  SMDiagnostic error;
  if (verbose)
    cout << "loading module from file" << fileName << endl;

  std::unique_ptr<Module> module = llvm::parseIRFile(fileName, error, context);
  if (!module) {
    std::string what = error.getMessage().str();
    std::cerr << "error: " << what;
    exit(1);
  } // end if
  if (verbose) {
    cout << "loading complete" << endl;
    cout << " - name " << module->getName().str() << endl;
    cout << " - number of functions " << module->getFunctionList().size() << endl;
    cout << " - instruction count #" << module->getInstructionCount() << endl;
  }
  return module;
}

// Standalone tool that extracts different features representations out of a LLVM-IR program.
int main(int argc, char* argv[])
{
  InitLLVM X(argc, argv);
  LLVMContext context;

  ExtractorCommandLine ecl(argc, argv, true);
  Expected<FeatureAnalysisParam> param = ecl.parse();
  if (!param) {
    cerr << "params not set\n";
    exit(0);
  }

  // Module loading
  std::unique_ptr<Module> module_ptr = load_module(context, param->ir_filename, param->verbose);

  // Pass management with the new pass pipeline
  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(true, false);
  SI.registerCallbacks(PIC);
  PassBuilder PB(nullptr, llvm::PipelineTuningOptions(), llvm::None, &PIC);

  Expected<PassPlugin> PassPlugin = PassPlugin::Load("./libfeature_pass.so");
  if (!PassPlugin) {
    errs() << "Failed to load passes from libfeature_pass.so plugin\n";
    errs() << "Problem while loading libfeature_pass: " << toString(std::move(PassPlugin.takeError()));
    errs() << "\n";
    return 1;
  }

  PassPlugin->registerPassBuilderCallbacks(PB);

  AAManager AA;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  // Register the AA manager first so that our version is the one used.
  FAM.registerPass([&] { return std::move(AA); });
  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ModulePassManager MPM;

  // Run!
  if (param->verbose)
    cout << "Pass manager run.." << endl;
  MPM.run(*module_ptr, MAM);
  if (param->verbose)
    cout << "Pass manager run completed" << endl;

  return 0;
} // end main
