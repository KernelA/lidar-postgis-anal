from sqlalchemy import Column, BigInteger, String, Integer, ARRAY
from geoalchemy2 import Geometry

from .base import Base


class LazPoints(Base):
    __tablename__ = "LidarPoints"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, nullable=False, index=True)
    file = Column(String, nullable=False, index=True)
    points = Column(Geometry("MULTIPOINTZ", dimension=3,
                    spatial_index=True, use_N_D_index=True, nullable=False))
    colors = Column(ARRAY(Integer, zero_indexes=True, dimensions=2), nullable=False)
