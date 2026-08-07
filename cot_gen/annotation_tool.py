import json
import gradio as gr
from typing import List, Dict, Any
import pandas as pd

class AnnotationInterface:
    """Web interface for annotating plan agent trajectories"""
    
    def __init__(self, data_file: str):
        self.data = self.load_data(data_file)
        self.current_idx = 0
        self.annotations = []
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load trajectories for annotation"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get("trajectories", [])
    
    def get_current_example(self) -> Dict:
        """Get current example for annotation"""
        if 0 <= self.current_idx < len(self.data):
            return self.data[self.current_idx]
        return {}
    
    def format_trajectory(self, trajectory: List[Dict]) -> str:
        """Format trajectory for display"""
        formatted = []
        for step in trajectory:
            if step["decision_type"] == "explore":
                formatted.append(f"Step {step['step_number']}:")
                formatted.append(f"  Sub-aspect: {step['sub_aspect']}")
                formatted.append(f"  Tool: {step['tool']}")
                formatted.append(f"  Thought: {step['thought']}")
            else:
                formatted.append(f"Final Summary:")
                formatted.append(f"  {step.get('summary', '')}")
        return "\n".join(formatted)
    
    def annotate_current(
        self,
        quality_score: int,
        strategy_appropriate: bool,
        exploration_complete: bool,
        optimal_stopping: bool,
        improvements: str,
        alternative_paths: str
    ) -> Dict:
        """Annotate current example"""
        annotation = {
            "example_idx": self.current_idx,
            "user_query": self.get_current_example().get("user_query", ""),
            "quality_score": quality_score,
            "strategy_appropriate": strategy_appropriate,
            "exploration_complete": exploration_complete,
            "optimal_stopping_point": optimal_stopping,
            "suggested_improvements": improvements,
            "alternative_exploration_paths": alternative_paths,
            "trajectory_length": len(self.get_current_example().get("trajectory", []))
        }
        
        self.annotations.append(annotation)
        return annotation
    
    def save_annotations(self, output_file: str = "annotations.json"):
        """Save all annotations"""
        with open(output_file, 'w') as f:
            json.dump({
                "total_annotations": len(self.annotations),
                "annotations": self.annotations
            }, f, indent=2)
        return f"Saved {len(self.annotations)} annotations"
    
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks() as interface:
            gr.Markdown("# Plan Agent Trajectory Annotation Tool")
            
            with gr.Row():
                with gr.Column(scale=2):
                    query_display = gr.Textbox(
                        label="User Query",
                        value=self.get_current_example().get("user_query", ""),
                        interactive=False
                    )
                    trajectory_display = gr.Textbox(
                        label="Exploration Trajectory",
                        value=self.format_trajectory(
                            self.get_current_example().get("trajectory", [])
                        ),
                        lines=20,
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Annotation")
                    quality_score = gr.Slider(
                        1, 5, value=3, step=1,
                        label="Overall Quality (1-5)"
                    )
                    strategy_appropriate = gr.Checkbox(
                        label="Strategy Appropriate for Query?"
                    )
                    exploration_complete = gr.Checkbox(
                        label="Exploration Sufficiently Complete?"
                    )
                    optimal_stopping = gr.Checkbox(
                        label="Stopped at Optimal Point?"
                    )
                    improvements = gr.Textbox(
                        label="Suggested Improvements",
                        lines=3
                    )
                    alternative_paths = gr.Textbox(
                        label="Alternative Exploration Paths",
                        lines=3
                    )
                    
                    with gr.Row():
                        prev_btn = gr.Button("Previous")
                        next_btn = gr.Button("Next")
                        save_btn = gr.Button("Save Annotations")
                    
                    progress = gr.Textbox(
                        label="Progress",
                        value=f"{self.current_idx + 1}/{len(self.data)}"
                    )
            
            # Button actions
            def go_next(q, s, e, o, i, a):
                self.annotate_current(q, s, e, o, i, a)
                self.current_idx = min(self.current_idx + 1, len(self.data) - 1)
                example = self.get_current_example()
                return (
                    example.get("user_query", ""),
                    self.format_trajectory(example.get("trajectory", [])),
                    f"{self.current_idx + 1}/{len(self.data)}"
                )
            
            def go_prev():
                self.current_idx = max(self.current_idx - 1, 0)
                example = self.get_current_example()
                return (
                    example.get("user_query", ""),
                    self.format_trajectory(example.get("trajectory", [])),
                    f"{self.current_idx + 1}/{len(self.data)}"
                )
            
            next_btn.click(
                go_next,
                inputs=[quality_score, strategy_appropriate, exploration_complete,
                       optimal_stopping, improvements, alternative_paths],
                outputs=[query_display, trajectory_display, progress]
            )
            
            prev_btn.click(
                go_prev,
                outputs=[query_display, trajectory_display, progress]
            )
            
            save_btn.click(
                lambda: self.save_annotations(),
                outputs=progress
            )
        
        return interface


# Usage
if __name__ == "__main__":
    # Create annotation interface
    annotator = AnnotationInterface("collected_trajectories.json")
    interface = annotator.create_interface()
    
    # Launch web interface
    interface.launch(share=True)