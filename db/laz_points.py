from sqlalchemy import Column, BigInteger, String, Integer, ARRAY
from geoalchemy2 import Geometry

from .base import Base


class LazPoints(Base):
    __tablename__ = "LidarPointsPly"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, nullable=False)
    file = Column(String, nullable=False)
    points = Column(Geometry("MULTIPOINTZ", dimension=3,
                    spatial_index=False, nullable=False, use_N_D_index=False))
    colors = Column(ARRAY(Integer, zero_indexes=True, dimensions=2))
