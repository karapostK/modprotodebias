from enum import Enum

from algorithms.proto_algs import UProtoMF, UIProtoMF, IProtoMF


class AlgorithmsEnum(Enum):
    uprotomf = UProtoMF
    iprotomf = IProtoMF
    uiprotomf = UIProtoMF
