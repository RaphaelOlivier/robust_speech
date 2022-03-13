from enum import Enum, auto


class Stage(Enum):
    """Completes the sb.Stage enum with an attack stage"""

    ATTACK = auto() # run backward passes through the input
    ADVTARGET = auto() # predict adversarial example against the attack target (on targeted attacks)
    ADVTRUTH = auto() # predict adversarial example against the ground truth
