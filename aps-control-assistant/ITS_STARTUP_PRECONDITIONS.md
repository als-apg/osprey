# ITS Start-Up Pre-Conditions

This document describes the required pre-conditions and checks that must be verified before starting the ITS (Insertion Test Stand) beamline operations.

## Pre-Start Checklist

### 1. Waveguide Configuration
**Check:** Verify that the waveguide is pointing to ITS

- The waveguide must be properly aligned and directed to the ITS beamline
- Confirm waveguide position and orientation before proceeding

**Verification Command:**
```bash
caget L3:IC1:SWPSB0M L3:IC1:SWPSB1M L3:IC1:SWPSB2M L3:IC1:SWPSB3M
```

**Expected Output (waveguide switched to ITS):**
```
L3:IC1:SWPSB0M                 Enabled
L3:IC1:SWPSB1M                 Enabled
L3:IC1:SWPSB2M                 Disabled
L3:IC1:SWPSB3M                 Disabled
```

**Required State:**
- `L3:IC1:SWPSB0M` must be **Enabled**
- `L3:IC1:SWPSB1M` must be **Enabled**
- `L3:IC1:SWPSB2M` must be **Disabled**
- `L3:IC1:SWPSB3M` must be **Disabled**

These conditions confirm that the waveguide is switched to the ITS beamline.

### 2. L3 Klystron and Gun Parameters
**Check:** Verify L3 klystron forward power and timing settings

Required settings:
- **L3 klystron forward power**: 3.05e6 watts (3.05 MW)
  - PV: `L3:KY:DC2ARF.VAL`
- **L3 gate width**: 1.5 μs (microseconds)
  - PV: `L3:TM:DAGateWidthSetAO`
- **Gun water temperature**: ~110°F
  - PV: `LTS:RTD:temperature0AI` (Gun water temperature)
  - PV: `LTS:RTD:temperature1AI` (Gun water return temperature)

### 3. ITS RF Input Power
**Check:** Verify ITS RF input power matches L3 forward power

- **ITS RF input power**: Should be 3.05e6 watts (3.05 MW)
- This value should match the L3 klystron forward power
  - PV: `LTS:PT3:DC1ARF.VAL`
- Confirms proper RF power transmission from L3 to ITS

### 4. Vacuum Pressure
**Check:** Verify vacuum pressure is within safe operating limits

- **Vacuum pressure**: Must be below 1.0e-8 Torr
- Critical for safe beam operations and equipment protection
- Relevant PVs:
  - `L1:LTS:TW:VP:CO1.VAL` (TW Vacuum CO1)
  - `L1:LTS:GUN:VP:CO1.VAL` (Gun Vacuum CO1)
  - `L1:LTS:GUN:VP:CO2.VAL` (Gun Vacuum CO2)
  - `L1:LTS:BL:VP:CO1.VAL` (BL Vacuum CO1)
  - `L1:LTS:BL:VP:CO2.VAL` (BL Vacuum CO2)
  - `L1:LTS:BL:VP:CO3.VAL` (BL Vacuum CO3)

## Summary Statement

**Before starting ITS operations, verify the following:**

1. The waveguide is correctly pointing to the ITS beamline
2. L3 klystron forward power is set to 3.05e6 watts (3.05 MW)
3. L3 gate width is set to 1.5 μs
4. Gun water temperature is approximately 110°F
5. ITS RF input power reads 3.05e6 watts, matching the L3 forward power
6. Vacuum pressure is below 1.0e-8 Torr at all measurement points

All conditions must be satisfied before proceeding with ITS beam operations.

## Related PVs

| Parameter | PV Address | Expected Value | Units |
|-----------|-----------|----------------|-------|
| L3 Klystron Forward Power | `L3:KY:DC2ARF.VAL` | 3.05e6 | watts |
| L3 Gate Width | `L3:TM:DAGateWidthSetAO` | 1.5 | μs |
| Gun Water Temperature | `LTS:RTD:temperature0AI` | ~110 | °F |
| Gun Water Return Temperature | `LTS:RTD:temperature1AI` | ~110 | °F |
| ITS RF Input Power | `LTS:PT3:DC1ARF.VAL` | 3.05e6 | watts |
| TW Vacuum | `L1:LTS:TW:VP:CO1.VAL` | < 1.0e-8 | Torr |
| Gun Vacuum CO1 | `L1:LTS:GUN:VP:CO1.VAL` | < 1.0e-8 | Torr |
| Gun Vacuum CO2 | `L1:LTS:GUN:VP:CO2.VAL` | < 1.0e-8 | Torr |
| BL Vacuum CO1 | `L1:LTS:BL:VP:CO1.VAL` | < 1.0e-8 | Torr |
| BL Vacuum CO2 | `L1:LTS:BL:VP:CO2.VAL` | < 1.0e-8 | Torr |
| BL Vacuum CO3 | `L1:LTS:BL:VP:CO3.VAL` | < 1.0e-8 | Torr |

## Notes

- These pre-conditions ensure proper RF power delivery, thermal management, and vacuum safety before beam operations
- Deviation from these parameters may result in improper beam delivery or equipment damage
- Vacuum pressure is critical: operation above 1.0e-8 Torr can damage sensitive equipment and compromise beam quality
- Always verify all conditions are met before energizing the ITS beamline
