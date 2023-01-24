#include <iostream>
#include <set>
#include <string>

namespace celerity {

class InstructionCollection {
  public:
	InstructionCollection() = delete;

	inline InstructionCollection(std::initializer_list<std::string> list) : instructions(list) {}

	inline bool contains(const std::string& instruction_name) const { return instructions.find(instruction_name) != instructions.end(); }

  private:
	std::set<std::string> instructions;
};

} // namespace celerity
