from pydantic import BaseModel, Field
from typing import Optional

class PipelineParams(BaseModel):
    # ClearML Dataset ID
    dataset_id: str
    # Random Seed
    seed: int
    # Percentage of data used for validatng
    val_ratio: float = Field(ge=0, le=0.99)

    epochs: int = Field(ge=0)

    optimizer: str

    workers: int = Field(ge=0)

    device: Optional[int]

    weights : str
    
    mismatch_threshold: float = Field(ge=0, le =0.99)
    dataset_project_name : str
