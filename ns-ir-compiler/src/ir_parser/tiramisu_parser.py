import json

class TiramisuIRParser:
    """
    Parses TIRAMISU pseudo-polyhedral JSON/AST representations.
    
    Since Tiramisu internally builds a schedule tree and iteration spaces, 
    we expect a structured representation output from the Tiramisu backend.
    """
    def __init__(self):
        pass
        
    def parse_computation(self, tiramisu_json_dump: dict):
        """Parses the computation boundaries."""
        computations = tiramisu_json_dump.get("computations", {})
        results = {}
        for comp_name, comp_data in computations.items():
            results[comp_name] = {
                "expression": comp_data.get("expression"),
                "domain": self.get_iteration_domain(comp_data),
                "schedule": self.extract_schedules(comp_data)
            }
        return results

    def extract_schedules(self, comp_data: dict):
        """Extracts the applied schedule (loop structure)."""
        # Ex: "L0, L1"
        return comp_data.get("schedule", [])
        
    def get_iteration_domain(self, comp_data: dict):
        """Extracts the polyhedral bounds (ISL sets)."""
        # Ex: "{ [i,j] : 0 <= i < 100 and 0 <= j < 100 }"
        return comp_data.get("domain", "")

if __name__ == "__main__":
    parser = TiramisuIRParser()
    mock_dump = {
        "computations": {
            "comp0": {
                "expression": "A(i, j) + B(i, j)",
                "domain": "{ [i,j] : 0 <= i < N and 0 <= j < M }",
                "schedule": [
                    {"type": "loop", "level": 0, "iterator": "i"},
                    {"type": "loop", "level": 1, "iterator": "j"}
                ]
            }
        }
    }
    
    print(json.dumps(parser.parse_computation(mock_dump), indent=2))
