from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam

app = FastAPI()

class Completion(BaseModel):
    prompt_messages: List[ChatCompletionMessageParam]
    completion: str
    extra_data: Optional[Dict[str, Any]] = None


async def grading_function(completion: Completion) -> float:
    # Calculate total length of all prompt messages
    prompt_length = 0
    for message in completion.prompt_messages:
        if hasattr(message, "content") and message.content:
            prompt_length += len(str(message.content))
    
    # Calculate target length (20% of prompt length)
    target_length = prompt_length * 0.2
    
    # Get actual completion length
    completion_length = len(completion.completion)
    
    # Calculate how close the completion is to the target length
    # Using a Gaussian-like function that peaks at 1.0 when completion_length = target_length
    # and approaches 0 as the difference increases
    if target_length == 0:  # Edge case: empty prompt
        return 0.0
    
    # Calculate the ratio of actual to target length
    ratio = completion_length / target_length
    
    # Penalize more for being too long than too short
    # Score is 1.0 when ratio = 1.0 (perfect match)
    # Score decreases as ratio moves away from 1.0
    if ratio <= 1.0:
        # For shorter completions: more gradual penalty
        score = ratio
    else:
        # For longer completions: steeper penalty
        score = max(0, 2 - ratio)
    
    return score

@app.post("/grade")
async def grade(completion: Completion):
    reward = await grading_function(completion)
    return {"reward": reward}
