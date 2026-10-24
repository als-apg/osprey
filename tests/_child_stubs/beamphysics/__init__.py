"""Stand-in for openPMD-beamphysics in connector-host children: lume.variables imports
ParticleGroup eagerly, which loads scipy and matplotlib (~1 s per child), and no child uses it.
Remove once lume imports beamphysics lazily (inside ParticleGroupVariable.validate_value).
"""


class ParticleGroup:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("tests/_child_stubs stand-in: this child needs the real beamphysics")
