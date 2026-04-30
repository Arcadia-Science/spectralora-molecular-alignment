from typing import Optional

from pydantic import BaseModel, model_validator


class RamanRequest(BaseModel):
    smiles: Optional[str] = None
    dataset: Optional[str] = None
    molecule_id: Optional[int] = None

    @model_validator(mode="after")
    def validate_input(self):
        has_smiles = self.smiles is not None
        has_db_ref = self.dataset is not None and self.molecule_id is not None
        if not has_smiles and not has_db_ref:
            raise ValueError("Provide either smiles or dataset+molecule_id")
        return self
