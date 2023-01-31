#include <string>

#include <llvm/IR/Instructions.h>

#include "AnalysisUtils.hpp"
#include "FeatureSet.hpp"

using namespace std;
using namespace llvm;

namespace celerity {

void FeatureSet::reset()
{
  for (StringRef rkey : raw.keys())
    raw[rkey] = 0;
  for (StringRef fkey : feat.keys())
    feat[fkey] = 0;
  instruction_num = 0;
  instruction_tot_contrib = 0;
}

void FeatureSet::add(const string& feature_name, int contribution)
{
  int old = raw[feature_name];
  raw[feature_name] = old + contribution;
  instruction_num += 1;
  instruction_tot_contrib += contribution;
}

void FeatureSet::normalize(Function& fun)
{
  float instructionContribution = instruction_tot_contrib == 0 ? 0.f : 1.0f / float(instruction_tot_contrib);

  // for(auto entry : fs.raw){
  // for(StringMap<unsigned>::const_iterator it = fs.raw.begin();
  //     it != fs.raw.end();
  //     it++)
  auto f_names = raw.keys();
  for (auto feature_name : f_names) {
    // for(std::pair<std::string, unsigned> entry : fs.raw){
    // const string feature_name = it->first;
    // float instTypeNum = float(it->second);
    int inst_int_val = raw[feature_name];
    float inst_flt_val = float(inst_int_val);
    feat[feature_name] = inst_flt_val * instructionContribution;
  }
}

void FeatureSet::print(raw_ostream& out_stream)
{
  out_stream << "raw values\n";
  out_stream.changeColor(raw_null_ostream::Colors::WHITE, true);
  print_feature_names(raw, out_stream);

  out_stream.changeColor(raw_null_ostream::Colors::WHITE, false);
  print_feature_values(raw, out_stream);

  out_stream << "feature values\n";
  out_stream.changeColor(raw_null_ostream::Colors::WHITE, true);
  print_feature_names(feat, out_stream);
  out_stream.changeColor(raw_null_ostream::Colors::WHITE, false);
  print_feature_values(feat, out_stream);
}
} // namespace celerity