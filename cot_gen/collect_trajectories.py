import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Any

class TrajectoryCollector:
    """Collect and process evaluation trajectories for training data"""
    
    def __init__(self, base_dirs: List[str]):
        self.base_dirs = base_dirs
        self.trajectories = []
    
    def extract_trajectory_data(self, eval_results: List[Any]) -> List[Dict]:
        """Extract structured trajectory data from evaluation results"""
        trajectory_steps = []
        
        for i, step in enumerate(eval_results):
            if isinstance(step, str):
                continue
                
            # Extract decision data
            if "Sub-aspect" in step:  # T2I CompBench or VBench format
                step_data = {
                    "step_number": i,
                    "decision_type": "explore",
                    "sub_aspect": step.get("Sub-aspect", ""),
                    "tool": step.get("Tool", ""),
                    "thought": step.get("Thought", ""),
                    "eval_results": step.get("eval_results", {})
                }
            elif "Analysis" in step:  # Final summary
                step_data = {
                    "step_number": i,
                    "decision_type": "summarize",
                    "thought": step.get("Thought", ""),
                    "analysis": step.get("Analysis", ""),
                    "summary": step.get("Summary", "")
                }
            else:
                continue
                
            trajectory_steps.append(step_data)
        
        return trajectory_steps
    
    def collect_from_directory(self, directory: str) -> List[Dict]:
        """Collect all trajectories from a directory"""
        collected_data = []
        
        # Find all result JSON files
        json_files = glob.glob(os.path.join(directory, "**/*.json"), recursive=True)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract user query (usually first element)
                user_query = data[0] if isinstance(data[0], str) else ""
                
                # Extract trajectory
                trajectory = self.extract_trajectory_data(data)
                
                if trajectory:
                    collected_data.append({
                        "source_file": json_file,
                        "user_query": user_query,
                        "trajectory": trajectory,
                        "total_steps": len(trajectory),
                        "timestamp": os.path.getmtime(json_file)
                    })
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        return collected_data
    
    def collect_all(self) -> None:
        """Collect trajectories from all base directories"""
        for base_dir in self.base_dirs:
            if os.path.exists(base_dir):
                print(f"Collecting from {base_dir}...")
                data = self.collect_from_directory(base_dir)
                self.trajectories.extend(data)
                print(f"  Found {len(data)} trajectories")
    
    def save_dataset(self, output_file: str) -> None:
        """Save collected trajectories to file"""
        with open(output_file, 'w') as f:
            json.dump({
                "collection_date": datetime.now().isoformat(),
                "total_trajectories": len(self.trajectories),
                "trajectories": self.trajectories
            }, f, indent=2)
        print(f"Saved {len(self.trajectories)} trajectories to {output_file}")


# Usage example
if __name__ == "__main__":
    collector = TrajectoryCollector([
        "./eval_t2i_comp_results/",
        "./eval_vbench_results/",
        "./open_domain_results/"
    ])
    
    collector.collect_all()
    collector.save_dataset("collected_trajectories.json")