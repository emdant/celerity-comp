#include <string>

#include "ExtractorCommandLine.hpp"

using namespace std;
using namespace llvm;

ExtractorCommandLine::ExtractorCommandLine(int argc, char** argv, bool printErrors)
{
  ExtractorCommandLine::argc = argc;
  ExtractorCommandLine::argv = argv;
  ExtractorCommandLine::printErrors = printErrors;

  FSet = ("set", cl::desc("Specify the feature set:"), cl::values(clEnumVal(FeatureSetOptions::fan19, "Default feature set for GPU used in [Fan et al. ICPP 19]"), clEnumVal(FeatureSetOptions::grewe13, "Feature set used in [Grewe et al. 13]")));
  // clEnumVal(full, "Feature set mapping all LLVM IR opcode (very large, hard to cover)")

  FAnalysis = ("analysis", cl::desc("Specify the feature analysis algorithm:"), cl::values(clEnumVal(AnalysisOptions::base, "Basic analysis, BB's instruction contributions are summed up"), clEnumVal(AnalysisOptions::kofler13, "Analysis pass to extract features using [Kofler et al., 13] loop heuristics")));

  FNorm = ("norm", cl::desc("Specify the feature normalization algorithm"), cl::value_desc("feature_norm"), cl::init("default"));

  IRFilename = (cl::Positional, cl::desc("<input_bitcode_file>"), cl::Required);

  LibFilename = ("l", cl::desc("Path to the feature-pass library"), cl::init("./libfeature_pass.so"));

  Verbose = ("v", cl::desc("Verbose"), cl::init(false));
}

Expected<FeatureAnalysisParam> ExtractorCommandLine::parse()
{
  cl::ParseCommandLineOptions(argc, argv);

  // if we are using the extractor tool, we need the input file
  FeatureAnalysisParam param = {FeatureSetOptions::fan19, AnalysisOptions::base, "no-norm", "", "", false, false}; // default
  param.feature_set = FSet;
  param.analysis = FAnalysis;
  param.normalization = FNorm;
  param.ir_filename = IRFilename;
  param.lib_filename = LibFilename;
  // param.help = Help;
  param.verbose = Verbose;

  return param;
}