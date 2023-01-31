from dataclasses import dataclass

import numpy as np

from ..protocol.dataset_proto import Context, Transform
from ..protocol.label_proto import Label, Polygon2dProto


@dataclass
class SimplifiedFrame:
    context: Context
    timestamp_micros: int
    pose: Transform
    laser_labels: [Label]
    no_label_zones: Polygon2dProto
    points: np.ndarray
