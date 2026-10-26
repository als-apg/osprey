"""Constants for the Mock Injector Simulator.

These constants define the behavior of the simulated accelerator environment.
"""

from tests.optimizer.sim.physics.types import VariableMetadata

VARIABLES: dict[str, VariableMetadata] = {
    "LN______MASTPH_AC00": VariableMetadata(
        min_value=0.0,
        max_value=10.0,
    ),
    "LN______AS2_PH_AC00": VariableMetadata(
        min_value=0.0,
        max_value=10.0,
    ),
    "EG______PHASE__AC02": VariableMetadata(
        min_value=-165.0,
        max_value=165.0,
    ),
    "GTL_____SHB1_PHAC00": VariableMetadata(
        min_value=0.0,
        max_value=360.0,
    ),
    "GTL_____SHB2_PHAC00": VariableMetadata(
        min_value=0.0,
        max_value=360.0,
    ),
    "EG______BIAS___AC01": VariableMetadata(
        min_value=40.0,
        max_value=60.0,
    ),
    "EG:LLRF:PulseDelaySP": VariableMetadata(
        min_value=298375.0,
        max_value=298377.0,
    ),
    "S0817:EVR1-DlyGen:0:Delay-SP": VariableMetadata(
        min_value=37215.0,
        max_value=37226.0,
    ),
    "S0817:EVR1-Out:UDC0:Delay-SP": VariableMetadata(
        min_value=0.0,
        max_value=800.0,
    ),
    "BR1_____KI_____AC00": VariableMetadata(
        min_value=95.0,
        max_value=105.0,
    ),
    "TimInjFieldSyncDelaySP": VariableMetadata(
        min_value=14000.0,
        max_value=15000.0,
    ),
    "BR1_____QF_PSGNAC00": VariableMetadata(
        min_value=0.92,
        max_value=1.02,
    ),
    "BR1_____QD_PSGNAC00": VariableMetadata(
        min_value=0.82,
        max_value=0.95,
    ),
    "BR1_____QD_PSOFAC00": VariableMetadata(
        min_value=-6155.0,
        max_value=-5569.0,
    ),
    "BR1_____QF_PSOFAC00": VariableMetadata(
        min_value=-9690.0,
        max_value=-8768.0,
    ),
    "GTL:HCM1:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "GTL:HCM2:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "GTL:HCM3:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "GTL:HCM4:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "GTL:VCM1:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "GTL:VCM2:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "GTL:VCM3:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "GTL:VCM4:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "GTL:BC1:Setpoint": VariableMetadata(
        min_value=-3.1,
        max_value=3.1,
    ),
    "GTL:BC2:Setpoint": VariableMetadata(
        min_value=-3.1,
        max_value=3.1,
    ),
    "LN:HCM1:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "LN:HCM2:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "LN:VCM1:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "LN:VCM2:Setpoint": VariableMetadata(
        min_value=-4.0,
        max_value=4.0,
    ),
    "LN:Q1_1:Setpoint": VariableMetadata(
        min_value=0.0,
        max_value=10.0,
    ),
    "LN:Q1_2:Setpoint": VariableMetadata(
        min_value=0.0,
        max_value=10.0,
    ),
    "LTB:Q1_1:Setpoint": VariableMetadata(
        min_value=-0.5,
        max_value=8.0,
    ),
    "LTB:Q1_2:Setpoint": VariableMetadata(
        min_value=-0.5,
        max_value=8.0,
    ),
    "LTB:Q2:Setpoint": VariableMetadata(
        min_value=-0.5,
        max_value=8.0,
    ),
    "LTB:Q3_1:Setpoint": VariableMetadata(
        min_value=-0.5,
        max_value=8.0,
    ),
    "LTB:Q3_2:Setpoint": VariableMetadata(
        min_value=-0.5,
        max_value=8.0,
    ),
    "LTB:Q4_1:Setpoint": VariableMetadata(
        min_value=-0.5,
        max_value=8.0,
    ),
    "LTB:Q4_2:Setpoint": VariableMetadata(
        min_value=-0.5,
        max_value=8.0,
    ),
    "LTB:Q5:Setpoint": VariableMetadata(
        min_value=-0.5,
        max_value=8.0,
    ),
    "LTB:Q6:Setpoint": VariableMetadata(
        min_value=-0.5,
        max_value=8.0,
    ),
    "LTB:HCM1:Setpoint": VariableMetadata(
        min_value=-3.0,
        max_value=3.0,
    ),
    "LTB:HCM2:Setpoint": VariableMetadata(
        min_value=-3.0,
        max_value=3.0,
    ),
    "LTB:VCM1:Setpoint": VariableMetadata(
        min_value=-3.0,
        max_value=3.0,
    ),
    "LTB:VCM2:Setpoint": VariableMetadata(
        min_value=-3.0,
        max_value=3.0,
    ),
    "LTB:VCM3:Setpoint": VariableMetadata(
        min_value=-3.0,
        max_value=3.0,
    ),
    "LTB:VCM4:Setpoint": VariableMetadata(
        min_value=-3.0,
        max_value=3.0,
    ),
    "LTB:VCM5:Setpoint": VariableMetadata(
        min_value=-3.0,
        max_value=3.0,
    ),
    "LTB:VCM6:Setpoint": VariableMetadata(
        min_value=-3.0,
        max_value=3.0,
    ),
    "LTB:VCM7:Setpoint": VariableMetadata(
        min_value=-3.0,
        max_value=3.0,
    ),
}


GTL_CHARGE_PV: str = "GTL:WCM:Charge"
LTB_CHARGE_PV: str = "LTB:ICT1:Charge"
INJECTION_BUCKET_PV: str = "SR13C___HCM3___AC00"


c = 299792458.0  # Speed of light [m/s]
m_e_eV = 511000.0  # Electron mass [eV]


# Injector hardware parameters

# Frequencies
F_SHB1 = 124.914e6
F_SHB2 = 499.654e6
F_SBAND = 2.997924e9

# Gun
W_GUN = 120e3  # [eV]
PULSE_FWHM = 2e-9  # [s]

# Voltages & Gradients (Max values)
V_SHB_MAX = 45e3  # [V]
E_SBUNCHER_MAX = 3.5e6  # [V/m]
L_SBUNCHER = 0.10  # [m]
E_LINAC_MAX = 13e6  # [V/m]
L_LINAC_SECTION = 2.0  # [m]


# Geometry (Drifts)

# Default drift lengths (configurable)
L_GUN_TO_SHB1 = 1.0  # [m]
L_SHB1_TO_SHB2 = 0.5  # [m]
L_SHB2_TO_BUNCHER = 0.8  # [m]
L_BUNCHER_TO_LINAC = 0.3  # [m]
L_BETWEEN_LINACS = 0.2  # [m]


# Simulation Defaults

N_MACROPARTICLES = 1000
ENERGY_ACCEPTANCE = 0.10  # +/- 10%
PHASE_ACCEPTANCE_DEG = 30.0  # +/- 30 degrees S-band
MIN_BETA = 1e-6  # Minimum relativistic beta to consider particle moving
