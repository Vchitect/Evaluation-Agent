import json
import random
from typing import List, Dict, Any
from base_agent import BaseAgent

class TrainingDataGenerator:
    """Generate diverse training data for plan agent"""
    
    def __init__(self):
        self.query_templates = self.load_query_templates()
        self.tools_t2i = ["Color Binding", "Shape Binding", "Texture Binding", "Non Spatial"]
        self.tools_vbench = [
            "Subject Consistency", "Background Consistency", "Motion Smoothness",
            "Aesthetic Quality", "Imaging Quality", "Object Class", "Human Action",
            "Color", "Spatial Relationship", "Scene"
        ]
    
    def load_query_templates(self) -> Dict[str, List[str]]:
        """Load diverse query templates"""
        return {
            "capability_check": [
                "Can the model generate {object} with {attribute}?",
                "How well does the model handle {scenario}?",
                "Is the model capable of creating {complex_scene}?"
            ],
            "comparison": [
                "How does the model perform on {task1} vs {task2}?",
                "What are the differences in generating {object1} compared to {object2}?"
            ],
            "boundary_finding": [
                "What are the limits of the model's ability to {capability}?",
                "How complex can {scenario} be before the model fails?",
                "What is the maximum number of {elements} the model can handle?"
            ],
            "quality_assessment": [
                "How consistent is the model in generating {aspect}?",
                "What is the quality of {feature} in the generated outputs?",
                "How accurate is the model's {binding_type} binding?"
            ]
        }
    
    def generate_diverse_queries(self, n: int = 100) -> List[str]:
        """Generate diverse evaluation queries"""
        queries = []
        
        # Define substitution values
        substitutions = {
            "object": ["cats", "cars", "buildings", "people", "landscapes"],
            "attribute": ["specific colors", "complex textures", "unusual shapes"],
            "scenario": ["multi-object scenes", "dynamic actions", "abstract concepts"],
            "complex_scene": ["crowded marketplaces", "underwater scenes", "futuristic cities"],
            "task1": ["realistic portraits", "abstract art"],
            "task2": ["photorealistic landscapes", "cartoon characters"],
            "capability": ["generate multiple objects", "maintain consistency", "follow complex prompts"],
            "elements": ["objects", "people", "colors", "textures"],
            "aspect": ["human faces", "animal poses", "architectural details"],
            "feature": ["motion", "lighting", "composition"],
            "binding_type": ["color", "shape", "texture", "spatial"]
        }
        
        for _ in range(n):
            template_type = random.choice(list(self.query_templates.keys()))
            template = random.choice(self.query_templates[template_type])
            
            # Fill in the template
            query = template
            for key, values in substitutions.items():
                if f"{{{key}}}" in query:
                    query = query.replace(f"{{{key}}}", random.choice(values))
            
            queries.append({
                "query": query,
                "type": template_type
            })
        
        return queries
    
    def generate_exploration_sequence(self, query: str, query_type: str) -> List[Dict]:
        """Generate a plausible exploration sequence for a query"""
        sequence = []
        
        # Determine exploration strategy based on query type
        if query_type == "capability_check":
            # Start simple, increase complexity
            complexities = ["simple", "moderate", "complex", "very complex"]
            for i, complexity in enumerate(complexities):
                sequence.append({
                    "step": i + 1,
                    "sub_aspect": f"Testing with {complexity} scenarios",
                    "tool": random.choice(self.tools_t2i + self.tools_vbench),
                    "strategy": "depth-first"
                })
        
        elif query_type == "comparison":
            # Test both aspects separately, then together
            aspects = ["first aspect", "second aspect", "combined comparison"]
            for i, aspect in enumerate(aspects):
                sequence.append({
                    "step": i + 1,
                    "sub_aspect": f"Evaluating {aspect}",
                    "tool": random.choice(self.tools_t2i + self.tools_vbench),
                    "strategy": "breadth-first"
                })
        
        elif query_type == "boundary_finding":
            # Progressive stress testing
            stress_levels = [10, 50, 90, 99]  # percentile of difficulty
            for i, level in enumerate(stress_levels):
                sequence.append({
                    "step": i + 1,
                    "sub_aspect": f"Testing at {level}th percentile difficulty",
                    "tool": random.choice(self.tools_t2i + self.tools_vbench),
                    "strategy": "depth-first"
                })
        
        else:  # quality_assessment
            # Multiple aspects of quality
            quality_aspects = ["consistency", "accuracy", "diversity", "edge cases"]
            for i, aspect in enumerate(quality_aspects):
                sequence.append({
                    "step": i + 1,
                    "sub_aspect": f"Assessing {aspect}",
                    "tool": random.choice(self.tools_t2i + self.tools_vbench),
                    "strategy": "breadth-first"
                })
        
        return sequence
    
    def create_training_example(self, query_data: Dict) -> Dict:
        """Create a complete training example"""
        query = query_data["query"]
        query_type = query_data["type"]
        
        # Generate exploration sequence
        exploration = self.generate_exploration_sequence(query, query_type)
        
        # Create training example
        example = {
            "user_query": query,
            "query_type": query_type,
            "exploration_plan": {
                "strategy": "depth-first" if "boundary" in query_type else "breadth-first",
                "expected_steps": len(exploration),
                "focus_areas": [step["sub_aspect"] for step in exploration]
            },
            "exploration_sequence": exploration,
            "decision_points": []
        }
        
        # Add decision points
        for i, step in enumerate(exploration):
            decision = {
                "step": i + 1,
                "context": {
                    "previous_steps": exploration[:i],
                    "current_observations": f"Simulated results from step {i}"
                },
                "decision": "explore" if i < len(exploration) - 1 else "summarize",
                "reasoning": f"Need to explore {step['sub_aspect']} to fully answer the query"
            }
            example["decision_points"].append(decision)
        
        return example
    
    def generate_dataset(self, n_examples: int = 1000) -> List[Dict]:
        """Generate complete training dataset"""
        queries = self.generate_diverse_queries(n_examples)
        dataset = []
        
        for query_data in queries:
            example = self.create_training_example(query_data)
            dataset.append(example)
        
        return dataset


# Usage
if __name__ == "__main__":
    generator = TrainingDataGenerator()
    
    # Generate training data
    print("Generating training dataset...")
    dataset = generator.generate_dataset(1000)
    
    # Save dataset
    with open("plan_agent_training_data.json", "w") as f:
        json.dump({
            "version": "1.0",
            "total_examples": len(dataset),
            "examples": dataset
        }, f, indent=2)
    
    print(f"Generated {len(dataset)} training examples")