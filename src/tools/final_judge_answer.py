from src.tools import AsyncTool, ToolResult
from src.registry import TOOL


@TOOL.register_module(name="final_judge_answer_tool", force=True)
class FinalJudgeAnswerTool(AsyncTool):
    name = "final_judge_answer_tool"
    description = (
        "Provides the final evaluation result with structured feedback and binary reward. "
        "This tool should be used to complete the evaluation task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": (
                    "Detailed textual feedback on the task execution. "
                    "Should explain what was done well, what could be improved, "
                    "and any issues identified."
                ),
            },
            "reward": {
                "type": "integer",
                "description": (
                    "Binary reward: 0 for failure/unsatisfactory, 1 for success/satisfactory. "
                    "Consider task completion, correctness, efficiency, and quality."
                ),
                "enum": [0, 1],
            },
        },
        "required": ["feedback", "reward"],
    }
    output_type = "object"

    async def forward(self, feedback: str, reward: int) -> ToolResult:
        """
        Process the evaluation result and return structured output.
        
        Args:
            feedback: Text feedback
            reward: Binary reward (0 or 1)
        
        Returns:
            ToolResult containing the structured evaluation result
        """
        # Validate reward
        if not isinstance(reward, int):
            return ToolResult(
                output=None,
                error=f"Invalid reward type: {type(reward)}. Expected int (0 or 1)."
            )
        
        if reward not in [0, 1]:
            return ToolResult(
                output=None,
                error=f"Invalid reward value: {reward}. Must be 0 or 1."
            )
        
        result = ToolResult(
            output={
                "feedback": feedback,
                "reward": reward
            },
            error=None,
        )
        return result

