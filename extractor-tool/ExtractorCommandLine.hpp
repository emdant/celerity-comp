#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>

// Supported feature sets
enum class FeatureSetOptions {
  fan19,
  grewe13,
  full
};

enum class AnalysisOptions {
  base,
  kofler13,
  polfeat
};

/* Feature normalization approches
enum class feature_norm {
    NONE,
    SUM,
    MINMAX_LINEAR,
    MINMAX_LOG
};
*/

struct FeatureAnalysisParam {
  FeatureSetOptions feature_set;
  AnalysisOptions analysis;
  std::string normalization;
  std::string ir_filename;
  std::string lib_filename;
  bool help;
  bool verbose;
};

class ExtractorCommandLine {

public:
  ExtractorCommandLine(int argc, char** argv, bool printErrors);
  llvm::Expected<FeatureAnalysisParam> parse();

private:
  int argc;
  char** argv;
  bool printErrors;

  llvm::cl::opt<FeatureSetOptions> FSet;
  llvm::cl::opt<AnalysisOptions> FAnalysis;
  llvm::cl::opt<std::string> FNorm;
  llvm::cl::opt<std::string> IRFilename;
  llvm::cl::opt<std::string> LibFilename;
  llvm::cl::opt<bool> Verbose;
};