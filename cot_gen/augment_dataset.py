import json
import random
from typing import List, Dict, Any
import numpy as np

class DatasetAugmenter:
    """Augment and enhance training dataset quality"""
    
    def __init__(self, base_dataset: str, annotations: str = None):
        self.base_data = self.load_json(base_dataset)
        self.annotations = self.load_json(annotations) if annotations else {}
        
    def load_json(self, file_path: str) -> Dict:
        """Load JSON data"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def augment_with_variations(self, trajectory: List[Dict]) -> List[List[Dict]]:
        """Create variations of a trajectory"""
        variations = []
        
        # 1. Early stopping variation
        if len(trajectory) > 3:
            early_stop = trajectory[:random.randint(2, len(trajectory)-1)]
            variations.append(early_stop)
        
        # 2. Different tool selection
        tool_varied = []
        alternative_tools = {
            "Color Binding": ["Color", "Overall Consistency"],
            "Shape Binding": ["Object Class", "Multiple Objects"],
            "Subject Consistency": ["Appearance Style", "Overall Consistency"],
            "Motion Smoothness": ["Dynamic Degree", "Temporal Style"]
        }
        
        for step in trajectory:
            step_copy = step.copy()
            if step.get("tool") in alternative_tools:
                step_copy["tool"] = random.choice(alternative_tools[step["tool"]])
            tool_varied.append(step_copy)
        variations.append(tool_varied)
        
        # 3. Reordered exploration
        if len(trajectory) > 2:
            reordered = trajectory.copy()
            # Swap middle steps
            if len(reordered) > 3:
                idx1, idx2 = random.sample(range(1, len(reordered)-1), 2)
                reordered[idx1], reordered[idx2] = reordered[idx2], reordered[idx1]
            variations.append(reordered)
        
        return variations
    
    def add_negative_examples(self, good_trajectory: List[Dict]) -> List[Dict]:
        """Create negative examples from good trajectories"""
        negative_examples = []
        
        # 1. Redundant exploration (same tool/aspect multiple times)
        if len(good_trajectory) > 2:
            redundant = good_trajectory.copy()
            # Duplicate a random step
            dup_idx = random.randint(0, len(good_trajectory)-2)
            redundant.insert(dup_idx + 1, good_trajectory[dup_idx])
            negative_examples.append({
                "trajectory": redundant,
                "issue": "redundant_exploration",
                "quality": "poor"
            })
        
        # 2. Premature stopping
        if len(good_trajectory) > 3:
            premature = good_trajectory[:2]
            negative_examples.append({
                "trajectory": premature,
                "issue": "premature_stopping",
                "quality": "poor"
            })
        
        # 3. Wrong tool selection
        wrong_tool = []
        for step in good_trajectory:
            step_copy = step.copy()
            if random.random() < 0.3:  # 30% chance of wrong tool
                all_tools = ["Color Binding", "Shape Binding", "Texture Binding", 
                           "Subject Consistency", "Motion Smoothness", "Object Class"]
                step_copy["tool"] = random.choice(all_tools)
            wrong_tool.append(step_copy)
        negative_examples.append({
            "trajectory": wrong_tool,
            "issue": "inappropriate_tools",
            "quality": "poor"
        })
        
        return negative_examples
    
    def enrich_with_reasoning(self, trajectory: List[Dict]) -> List[Dict]:
        """Add detailed reasoning to trajectory steps"""
        enriched = []
        
        reasoning_templates = {
            "explore": [
                "Based on the previous results showing {observation}, we need to explore {aspect} to {goal}",
                "The model performed {performance} on {previous_aspect}, so testing {current_aspect} will help determine {insight}",
                "To fully answer the user's question about {topic}, examining {aspect} with {tool} is essential"
            ],
            "summarize": [
                "After {n} exploration steps, we have sufficient evidence that {conclusion}",
                "The consistent pattern across {aspects} indicates {finding}, providing a complete answer",
                "Further exploration would be redundant as we've established {key_insight}"
            ]
        }
        
        for i, step in enumerate(trajectory):
            step_copy = step.copy()
            
            # Add enriched reasoning
            if step.get("decision_type") == "explore":
                template = random.choice(reasoning_templates["explore"])
                step_copy["enriched_thought"] = template.format(
                    observation="strong performance" if i == 0 else "mixed results",
                    aspect=step.get("sub_aspect", "this aspect"),
                    goal="understand the model's capabilities",
                    performance="well" if random.random() > 0.5 else "poorly",
                    previous_aspect="basic scenarios",
                    current_aspect=step.get("sub_aspect", "complex scenarios"),
                    insight="the model's boundaries",
                    topic="model capabilities",
                    tool=step.get("tool", "evaluation tool")
                )
            else:
                template = random.choice(reasoning_templates["summarize"])
                step_copy["enriched_thought"] = template.format(
                    n=i,
                    conclusion="the model has clear strengths and limitations",
                    aspects="all tested dimensions",
                    finding="consistent behavior patterns",
                    key_insight="the model's capability boundaries"
                )
            
            enriched.append(step_copy)
        
        return enriched
    
    def compute_trajectory_metrics(self, trajectory: List[Dict]) -> Dict:
        """Compute quality metrics for a trajectory"""
        metrics = {
            "length": len(trajectory),
            "tool_diversity": len(set(step.get("tool", "") for step in trajectory)),
            "has_summary": any(step.get("decision_type") == "summarize" for step in trajectory),
            "redundancy_score": 0,
            "completeness_score": 0
        }
        
        # Check for redundancy
        aspects = [step.get("sub_aspect", "") for step in trajectory]
        metrics["redundancy_score"] = len(aspects) - len(set(aspects))
        
        # Estimate completeness (heuristic)
        if metrics["length"] < 3:
            metrics["completeness_score"] = 0.3
        elif metrics["length"] > 8:
            metrics["completeness_score"] = 0.7
        else:
            metrics["completeness_score"] = 0.9
        
        return metrics
    
    def create_augmented_dataset(self, output_file: str):
        """Create fully augmented dataset"""
        augmented_data = {
            "version": "2.0",
            "original_examples": 0,
            "augmented_examples": 0,
            "negative_examples": 0,
            "examples": []
        }
        
        # Process each trajectory
        for item in self.base_data.get("trajectories", []):
            trajectory = item.get("trajectory", [])
            user_query = item.get("user_query", "")
            
            # Original example (enriched)
            enriched = self.enrich_with_reasoning(trajectory)
            metrics = self.compute_trajectory_metrics(enriched)
            
            augmented_data["examples"].append({
                "id": f"original_{augmented_data['original_examples']}",
                "user_query": user_query,
                "trajectory": enriched,
                "quality": "good",
                "metrics": metrics,
                "source": "original"
            })
            augmented_data["original_examples"] += 1
            
            # Variations
            variations = self.augment_with_variations(trajectory)
            for var in variations:
                enriched_var = self.enrich_with_reasoning(var)
                metrics = self.compute_trajectory_metrics(enriched_var)
                
                augmented_data["examples"].append({
                    "id": f"augmented_{augmented_data['augmented_examples']}",
                    "user_query": user_query,
                    "trajectory": enriched_var,
                    "quality": "good",
                    "metrics": metrics,
                    "source": "augmented"
                })
                augmented_data["augmented_examples"] += 1
            
            # Negative examples
            negatives = self.add_negative_examples(trajectory)
            for neg in negatives:
                metrics = self.compute_trajectory_metrics(neg["trajectory"])
                
                augmented_data["examples"].append({
                    "id": f"negative_{augmented_data['negative_examples']}",
                    "user_query": user_query,
                    "trajectory": neg["trajectory"],
                    "quality": neg["quality"],
                    "issue": neg["issue"],
                    "metrics": metrics,
                    "source": "negative"
                })
                augmented_data["negative_examples"] += 1
        
        # Save augmented dataset
        with open(output_file, 'w') as f:
            json.dump(augmented_data, f, indent=2)
        
        print(f"Created augmented dataset with:")
        print(f"  Original examples: {augmented_data['original_examples']}")
        print(f"  Augmented examples: {augmented_data['augmented_examples']}")
        print(f"  Negative examples: {augmented_data['negative_examples']}")
        print(f"  Total examples: {len(augmented_data['examples'])}")


# Usage
if __name__ == "__main__":
    augmenter = DatasetAugmenter("collected_trajectories.json")
    augmenter.create_augmented_dataset("augmented_training_data.json")