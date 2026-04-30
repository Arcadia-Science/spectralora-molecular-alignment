from pydantic import BaseModel


class RamanRequest(BaseModel):
    smiles: str
