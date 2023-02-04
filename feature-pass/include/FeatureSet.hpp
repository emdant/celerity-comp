#pragma once

#include <cstdint>
#include <string>
#include <type_traits>

#include <llvm/IR/Instructions.h>

#include "Registry.hpp"

namespace celerity {

/// A set of features, including both raw values and normalized ones.
/// Abstract class, with different subslasses
class FeatureSet {
protected:
  llvm::StringMap<unsigned> raw; // features as counters, before normalization
  llvm::StringMap<float> feat;   // features after normalization, they should not necessarily be the same features as the raw
  int instruction_num;
  int instruction_tot_contrib;
  std::string name;

public:
  FeatureSet() : name("default") {}
  FeatureSet(std::string feature_set_name) : name(feature_set_name) {}
  virtual ~FeatureSet() {}

  llvm::StringMap<unsigned> getFeatureCounts() { return raw; }
  llvm::StringMap<float> getFeatureValues() { return feat; }
  std::string getName() { return name; }

  virtual void reset();
  virtual void add(const std::string& feature_name, int contribution = 1);
  virtual void eval(llvm::Instruction& inst, int contribution = 1) = 0;
  virtual void normalize(llvm::Function& fun);
  virtual void print(llvm::raw_ostream& out_stream);
};

/// Feature set based on Fan's work, specifically designed for GPU architecture.
class Fan19FeatureSet : public FeatureSet {
public:
  Fan19FeatureSet() : FeatureSet("fan19")
  {
    raw["int_add"] = 0;
    raw["int_mul"] = 0;
    raw["int_div"] = 0;
    raw["int_bw"] = 0;
    raw["flt_add"] = 0;
    raw["flt_mul"] = 0;
    raw["flt_div"] = 0;
    raw["sp_fun"] = 0;
    raw["mem_gl"] = 0;
    raw["mem_loc"] = 0;
  }

  virtual ~Fan19FeatureSet() {}
  virtual void eval(llvm::Instruction& inst, int contribution = 1);
};

/// Feature set used by Grewe & O'Boyle. It is very generic and mainly designed to catch mem. vs comp.
class Grewe11FeatureSet : public FeatureSet {
public:
  Grewe11FeatureSet() : FeatureSet("grewe11")
  {
    raw["int"] = 0;
    raw["int4"] = 0;
    raw["float"] = 0;
    raw["float4"] = 0;
    raw["math"] = 0;
    raw["barrier"] = 0;
    raw["mem_acc"] = 0;
    raw["mem_loc"] = 0;
    raw["mem_coal"] = 0;
  }

  virtual ~Grewe11FeatureSet() {}
  virtual void eval(llvm::Instruction& inst, int contribution = 1);
  virtual void normalize(llvm::Function& fun);
};

/// Feature set used by Fan, designed for GPU architecture.
class FullFeatureSet : public FeatureSet {
public:
  FullFeatureSet() : FeatureSet("full") {}
  virtual ~FullFeatureSet() {}
  virtual void eval(llvm::Instruction& inst, int contribution = 1);
};

/// Registry of feature sets
using FSRegistry = Registry<celerity::FeatureSet*>;

} // end namespace celerity
