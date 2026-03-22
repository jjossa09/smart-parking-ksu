from pydantic import BaseModel
from typing import List, Tuple

class Spot(BaseModel):
    id: str
    label: str
    isOccupied: bool
    polygon: List[Tuple[int, int]]

class ParkingUpdatePayload(BaseModel):
    type: str = "update"
    lotName: str
    frameImage: str
    totalSpots: int
    availableSpots: int
    occupiedSpots: int
    spots: List[Spot]
    timestamp: str
