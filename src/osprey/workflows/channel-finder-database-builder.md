---
workflow: channel-finder-database-builder
category: channel-finder
applies_when: [building_database, adding_descriptions, improving_database]
estimated_time: 1-4 hours (depends on channel count and available documentation)
ai_ready: true
related: [channel-finder-pipeline-selection]
---

# Channel Finder: Database Builder Guide

**Purpose**: Help users build high-quality channel databases with descriptions that enable effective LLM-based channel finding.

**Key Principle**: Descriptions should help the LLM distinguish between channels and understand the user's intent, not just document what exists.

**Target Audience**: Users who have selected a pipeline and need to build or improve their database.

---

## 🤖 AI Quick Start

**Paste this prompt to your AI assistant:**

```
Following @docs/workflows/channel-finder-database-builder.md, help me build my Channel Finder database.

Your role:
1. Ask questions to understand what information I have available
2. Read the database format code and examples to understand schema requirements
3. Help me extract and organize channel information
4. Guide me on writing useful descriptions (not just documenting)
5. Identify where descriptions are most critical
6. Suggest iterative improvement strategy

Start by asking:
- What pipeline am I using? (in-context, hierarchical, middle layer)
- Do I have my-control-assistant initialized? (to see example databases)
- What information sources do I have? (files, docs, domain experts)
- Have I already started a database, or starting from scratch?

Then read the relevant code:
- Database format class: src/osprey/templates/.../databases/{flat,hierarchical,middle_layer}.py
- Example database: my-control-assistant/data/channel_databases/{pipeline}.json
- This helps me understand the schema and see good description patterns

Remember:
- Focus on distinguishing information, not just completeness
- Help me prioritize where descriptions matter most
- Guide me to write descriptions users would actually search for
- Reference example databases to show good description patterns
- Suggest automation where possible (pattern recognition, LLM generation)
```

**Related workflows**: [channel-finder-pipeline-selection.md](channel-finder-pipeline-selection.md)

---

## 📊 Understanding Description Quality

**The goal**: Descriptions should help the LLM match user queries to the right channels.

### What Makes a Good Description?

Look at these examples from real Osprey databases:

**❌ Poor Description** (too vague):
```
"BPM X position"
```
*Problem*: Doesn't explain what BPM is, why X position matters, or how it differs from Y.

**✅ Good Description**:
```
"Horizontal beam position readback in millimeters. Measured using button electrode
signals."
```
*Why better*: Specifies orientation (horizontal), units (mm), and measurement method.

**⭐ Excellent Description**:
```
"Beam Position Monitors: Non-invasive electrostatic pickups that measure beam
position in horizontal (X) and vertical (Y) planes. Critical for orbit correction
and beam stability. 96 BPMs distributed around ring (8 per sector)."
```
*Why best*: Explains what it IS, what it DOES, why it MATTERS, technical context (96 total), and physical distribution.

### The "Five Questions" Framework

Every description should answer as many of these as relevant:

1. **WHAT is it?** → Basic definition in domain terms
2. **WHERE is it?** → Physical location or position in system
3. **WHAT does it measure/control?** → Function and purpose
4. **WHY does it matter?** → Role in operations, safety, performance
5. **HOW does it relate?** → Connections to other components

**Example applying framework:**

```json
{
  "channel": "SR01C:BPM1:X",
  "description": "Horizontal position readback in millimeters (WHAT).
                  Measured using button electrode signals (HOW).
                  Critical for orbit correction and beam stability (WHY)."
}
```

---

## 🎯 Pipeline-Specific Guidance

### For In-Context (Flat) Databases

**Challenge**: Every channel needs a distinguishing description (no hierarchy to help).

**Priority**: Descriptions that help distinguish similar channels.

**Example Pattern**:
```json
{
  "channel": "IP41Pressure",
  "description": "Pressure at IP41 ion pump located at the end of accelerating
                  tube and beginning of beamline1 (beam transport), inside of
                  tank, measured in Torr"
}
```

**What makes this work**:
- ✅ Specific location ("end of accelerating tube")
- ✅ Spatial context ("beginning of beamline1")
- ✅ Physical context ("inside of tank")
- ✅ Units ("Torr")
- ✅ Device identifier ("IP41")

**Description Template for Flat Databases**:
```
"[Measurement/Control] at [Specific Location] [Device Type/ID] located
 [Spatial Context] measured/controlled in [Units]. [Purpose if non-obvious]."
```

**Common Queries to Optimize For**:
- "pressure at the end of the accelerating tube" → needs location keywords
- "ion pump in beamline 1" → needs beamline identifier
- "temperature sensors" → needs sensor type keyword
- "all pressure readings" → needs "pressure" keyword

---

### For Hierarchical Databases

**Challenge**: Descriptions at each branching point must help LLM navigate DOWN the tree, even when the user query mentions concepts at deeper levels.

**Key Insight**: The LLM navigates level-by-level, so each description must preview what's INSIDE to guide correct navigation.

**Priority**: Descriptions at BRANCHING POINTS (where LLM chooses between siblings).

---

#### Navigation Example: Understanding Forward-Looking Descriptions

**User Query**: "Show me BPM horizontal positions"

The query mentions:
- "BPM" (family level - 2 levels deep)
- "horizontal positions" (field level - 4 levels deep)

But LLM navigates step-by-step:

**Step 1 - System Level**: Choose between MAG, VAC, RF, DIAG
```json
{
  "MAG": {
    "_description": "Magnet System: Controls beam trajectory and focusing.
                     Includes dipoles, quadrupoles, sextupoles, and correctors."
  },
  "DIAG": {
    "_description": "Diagnostic System: Beam measurement and monitoring.
                     Includes beam position monitors (BPMs), beam current monitors
                     (DCCT), and radiation monitors."
  }
}
```

**Why this works**:
- ✅ DIAG description mentions "beam position monitors (BPMs)"
- ✅ LLM can match query "BPM" to DIAG even though query doesn't say "diagnostic"
- ❌ Without "BPM" in description, LLM might guess wrong or skip DIAG

**Step 2 - Family Level**: Choose between BPM, DCCT, NEUTRON, GAMMA
```json
{
  "BPM": {
    "_description": "Beam Position Monitors: Measure transverse beam position.
                     Horizontal (X) and vertical (Y) position readbacks. Essential
                     for orbit correction."
  },
  "DCCT": {
    "_description": "DC Current Transformer: Non-intercepting beam current monitor.
                     Measures total circulating current."
  }
}
```

**Why this works**:
- ✅ BPM description mentions "Horizontal (X)"
- ✅ LLM can match query "horizontal positions" to BPM
- ✅ Distinguishes from DCCT (current, not position)

**Step 3 - Device Level**: Choose specific BPM instance (BPM01, BPM02, etc.)
- Usually no descriptions needed - instances are structurally identical

**Step 4 - Field Level**: Choose between POSITION, GOLDEN, OFFSET, STATUS
```json
{
  "POSITION": {
    "_description": "Beam Position: Measured transverse position with X and Y
                     coordinates in millimeters."
  },
  "GOLDEN": {
    "_description": "Golden Orbit Reference: Stored reference positions for
                     orbit correction."
  }
}
```

**Why this works**:
- ✅ POSITION mentions "X and Y coordinates"
- ✅ Distinguishes from GOLDEN (reference, not measurement)

**Step 5 - Subfield Level**: Choose between X, Y
- Self-explanatory - "X" = horizontal, "Y" = vertical

---

#### The Forward-Looking Description Pattern

**Rule**: Each level's description should mention key concepts from CHILD levels.

**System Level** - Preview major families:
```
❌ Bad: "Diagnostic System: Monitors beam parameters."
✅ Good: "Diagnostic System: Beam measurement and monitoring. Includes beam
         position monitors (BPMs), current monitors (DCCT), and radiation
         monitors (neutron, gamma)."
```
*Why*: User queries often mention "BPM" or "current" without saying "diagnostic"

**Family Level** - Preview major fields:
```
❌ Bad: "Beam Position Monitors: Measure beam position."
✅ Good: "Beam Position Monitors: Measure transverse position in horizontal (X)
         and vertical (Y) planes. Provides position readbacks, golden orbit
         references, and position offsets."
```
*Why*: User queries mention "horizontal" or "X position" without saying "BPM" explicitly

**Field Level** - Preview subfields if nested:
```
❌ Bad: "RF Power monitoring."
✅ Good: "RF Power (Kilowatts): Forward power from klystron, reflected power
         from cavity, and net delivered power."
```
*Why*: User might ask for "reflected power" - need to know it's under this field

---

#### Example: Well-Designed Hierarchical Descriptions

From Osprey's hierarchical.json showing forward-looking preview:

```json
{
  "MAG": {
    "_description": "Magnet System (MAG): Controls beam trajectory and focusing
                     in the storage ring. Includes dipole bending magnets,
                     quadrupole focusing magnets, sextupole magnets for
                     chromaticity correction, and corrector magnets for precise
                     beam steering."
  }
}
```
**Analysis**:
- ✅ Lists child families: "dipole", "quadrupole", "sextupole", "corrector"
- ✅ User query "focusing quadrupoles" will match even at system level
- ✅ User query "steering magnets" will match via "corrector magnets"

**Sibling Distinction Example**:

```json
{
  "QF": {
    "_description": "Focusing Quadrupoles (QF): Quadrupole magnets with positive
                     gradient. Focus beam in horizontal plane, defocus in vertical
                     plane. Part of tune correction system. Work together with QD
                     magnets for independent horizontal/vertical tune control."
  },
  "QD": {
    "_description": "Defocusing Quadrupoles (QD): Quadrupole magnets with negative
                     gradient. Defocus beam in horizontal plane, focus in vertical
                     plane. Part of tune correction system. Work together with QF
                     magnets for independent horizontal/vertical tune control."
  }
}
```
**Analysis**:
- ✅ Distinguishing feature up front ("Focusing" vs "Defocusing")
- ✅ Technical detail ("positive gradient" vs "negative gradient")
- ✅ Functional difference (horizontal/vertical focusing behavior differs)
- ✅ Common purpose (both for tune correction)
- ✅ Relationship explained (work together)

---

#### Description Template by Hierarchy Level

**System Level**:
```
"[System Name]: [Primary function]. Includes [list key families with brief
 descriptors]. [Overall scale or context]."
```

Example:
```
"Vacuum System (VAC): Maintains ultra-high vacuum for beam lifetime. Includes
 ion pumps (UHV pumping), pressure gauges (monitoring), and vacuum valves
 (isolation). Operating pressure 10^-9 to 10^-10 Torr."
```

**Family Level**:
```
"[Device Family]: [What they are] that [what they do]. [Key fields available:
 list important child fields]. [Distinguishing features from siblings]."
```

Example:
```
"Beam Position Monitors: Non-invasive measurement of beam position. Provides
 horizontal (X) and vertical (Y) position readbacks, golden orbit references,
 and position offsets. Essential for orbit feedback."
```

**Field Level**:
```
"[Measurement/Control]: [What it represents] in [units]. [Subfields if nested].
 [Purpose or use case]."
```

Example:
```
"RF Power (Kilowatts): Electromagnetic power in RF transmission. Includes
 forward power (from klystron), reflected power (cavity mismatch), and net
 power (actual delivered). Typical: 50-500 kW."
```

---

#### Where Descriptions Matter MOST

**Critical - Always Need Descriptions**:
- ❗ **System level** - User queries rarely mention system names, usually mention families
  - Query: "beam current" → needs DIAG to mention "current monitors"
  - Query: "corrector magnets" → needs MAG to mention "corrector magnets"

- ❗ **Family level** - Distinguish siblings AND preview fields
  - Query: "horizontal correctors" vs "vertical correctors" → distinguish HCM/VCM
  - Query: "current setpoint" → HCM needs to mention setpoint field exists

- ❗ **When siblings have similar names/functions**
  - QF vs QD (both quadrupoles - distinguish focusing/defocusing)
  - HCM vs VCM (both correctors - distinguish horizontal/vertical)
  - FWD vs REV vs NET (all power - distinguish forward/reflected/net)

**Lower Priority - Often Can Skip**:
- ✓ Instance levels (Device expansions like B01, B02, B03...)
  - All instances have identical structure
  - Only describe once at family level

- ✓ When names are self-explanatory
  - "X" vs "Y" position (obvious horizontal/vertical)
  - "SP" vs "RB" (standard setpoint/readback abbreviations)

- ✓ Leaf nodes with unique names
  - If only one option, LLM will select it regardless

---

### For Middle Layer Databases

**Challenge**: Same forward-looking navigation as Hierarchical (see above), but with functional organization.

**Key Difference**: Channel addresses are STORED in the database (not derived from naming patterns), so the hierarchy is purely for logical navigation.

**Navigation applies the same way** - System → Family → Field:

```json
{
  "SR": {
    "_description": "Storage Ring: Main synchrotron light source. Contains beam
                     position monitors (BPM), corrector magnets (HCM, VCM), and
                     RF cavities.",
    "BPM": {
      "_description": "Beam Position Monitors: Measure horizontal (X) and vertical
                       (Y) beam position. Provides Monitor readbacks, Setpoint
                       controls, and error signals.",
      "X": {
        "_description": "Horizontal position readback in millimeters.",
        "ChannelNames": ["SR01C:BPM1:X", "SR01C:BPM2:X", ...]
      }
    }
  }
}
```

**What's the same as Hierarchical**:
- System previews families ("BPM", "HCM", "VCM")
- Family previews fields ("X", "Y", "Monitor", "Setpoint")
- Each level guides navigation to deeper concepts

**What's different from Hierarchical**:
- ✅ Functional grouping (Monitor/Setpoint) not naming-based
- ✅ Actual PV addresses stored at leaf nodes (`ChannelNames`)
- ✅ Often richer metadata (DataType, Units, MemberOf)

**Apply the same forward-looking description principles** from Hierarchical section above

---

## 🔍 Information Gathering Strategy

### Step 1: Inventory Your Sources

Ask yourself (or the user):

**Do you have...**
- [ ] EPICS database files (.db, .template)?
- [ ] Existing documentation (PDFs, wikis, manuals)?
- [ ] Spreadsheets or CSV files with channel lists?
- [ ] Access to domain experts?
- [ ] Existing control system GUIs (with labels/tooltips)?
- [ ] System diagrams or schematics?
- [ ] Naming convention documents?

**Most valuable sources** (in order):
1. ✅ **Middle layer code** (MATLAB Middle Layer, Python equivalents) - contains functional organization
2. ✅ **Control room GUI source code** - shows how operators group and access channels
3. ✅ **Domain expert interviews** - understand operational thinking
4. ✅ **System documentation** with device descriptions and functional diagrams
5. ✅ **EPICS .db files** with DESC fields (often sparse, but quick to extract)
6. ⚠️ **Naming patterns alone** (helpful but not sufficient for descriptions)

### Step 2: Extract What You Can Automatically

**From Middle Layer Code** (MATLAB MML, Python, etc.):
```matlab
% MATLAB Middle Layer example - look for structure definitions
AO.BPM.FamilyName = 'BPM';
AO.BPM.Monitor.Mode = 'Online';
AO.BPM.Monitor.ChannelNames = {'SR01C:BPM1:X', 'SR01C:BPM1:Y', ...};
AO.BPM.Monitor.Description = 'Beam position readback';
```

**What to extract**:
- ✅ System/Family/Field organization (already structured!)
- ✅ Channel groupings and relationships
- ✅ Field descriptions (Monitor, Setpoint, etc.)
- ✅ Comments explaining device purposes
- ✅ Units, modes, and metadata

**From Control Room GUI Source**:
```python
# GUI code shows how operators think about channels
bpm_group = ChannelGroup("Beam Position", channels=[
    "SR01C:BPM1:X",  # Horizontal position sector 1
    "SR01C:BPM1:Y",  # Vertical position sector 1
])

corrector_panel = Panel("Orbit Correction", [
    ("HCM", "Horizontal correctors"),
    ("VCM", "Vertical correctors"),
])
```

**What to extract**:
- ✅ How channels are grouped together
- ✅ User-facing labels and descriptions
- ✅ Functional groupings (what goes on same screen)
- ✅ Operator terminology (what they actually call things)

**From EPICS .db files**:
```bash
# Extract channel names and DESC fields
grep -E "^record\\(|field\\(DESC," file.db | \
  paste - - | \
  sed 's/.*(\([^,]*\),.*/\1/' > channels.txt
```

**From CSV exports**:
- Channel name column → `channel` field
- PV address column → `address` field
- Description/comment column → `description` field (starting point)

**Using LLM for initial descriptions**:
```
Given these channel names:
- SR01C:BPM1:X
- SR01C:BPM1:Y
- SR01C:BPM2:X

Generate initial descriptions that explain:
1. What each channel measures
2. Units (if inferable from name)
3. Distinguishing features

Use this format: "X position..." not "This channel is..."
```

**⚠️ Warning**: LLM-generated descriptions need human review for:
- Technical accuracy
- Domain-specific terminology
- Actual units and ranges
- Functional relationships

### Step 3: Identify Critical Description Points

**Where do descriptions matter most?**

Run this thought experiment:

```
User Query: "Show me beam current"

Which channels should match?
- DCCT:Current ✓
- BPM1:Sum ✓ (sum signal proportional to current)
- RF:Power ✗ (not current, even though related)

Both DCCT and BPM need "current" keyword in descriptions!
```

**Priority Matrix**:

| Situation | Priority | Example |
|-----------|----------|---------|
| Similar names, different functions | 🔴 CRITICAL | HCM vs VCM (horizontal vs vertical) |
| Acronym users won't know | 🔴 CRITICAL | DCCT (explain "DC Current Transformer") |
| Same function, different locations | 🟡 HIGH | IP41 vs IP78 vs IP125 (all ion pumps) |
| Common user queries | 🟡 HIGH | "beam current", "pressure", "temperature" |
| Self-explanatory names | 🟢 LOW | X vs Y position |
| Repeated instances | 🟢 LOW | BPM1 vs BPM2 (describe family once) |

### Step 4: Build Incrementally

**Start Minimal** → **Test** → **Enhance Where Confused**

**Minimal viable database**:
```json
{
  "channels": [
    {
      "channel": "BeamCurrentMonitor",
      "address": "SR:DCCT:Current",
      "description": "Beam current"
    },
    {
      "channel": "BeamPositionMonitor1Horizontal",
      "address": "SR:BPM1:X",
      "description": "BPM horizontal position"
    }
  ]
}
```

**Test with real queries**:
```
Query: "Show me beam current"
→ Did it find BeamCurrentMonitor? ✓
→ Did it also find BPM sum signal? ✗

Enhancement needed: Add "current" keyword to BPM sum signal description
```

**Enhanced database**:
```json
{
  "channels": [
    {
      "channel": "BeamCurrentMonitor",
      "address": "SR:DCCT:Current",
      "description": "DC current transformer beam current readback in milliamperes. Total circulating beam charge."
    },
    {
      "channel": "BeamPositionMonitor1SumSignal",
      "address": "SR:BPM1:Sum",
      "description": "BPM sum signal intensity. Proportional to beam current passing through monitor."
    }
  ]
}
```

**Iteration Strategy**:
1. Start with 10-20 representative channels
2. Test with 5-10 realistic user queries
3. Note where LLM gets confused or misses channels
4. Add descriptions to resolve confusion
5. Expand to full channel set
6. Re-test with expanded queries

---

## ✍️ Writing Effective Descriptions

### Terminology Guidelines

**Use domain terminology that users actually say**:

✅ **Good**:
- "beam position" (what users ask for)
- "horizontal" and "vertical" (clear orientation)
- "readback" and "setpoint" (control system terms)
- "corrector magnet" (what users call them)

❌ **Avoid**:
- "transverse coordinate vector component" (too academic)
- "BPM X-axis signal" (jargon without context)
- Just acronyms without expansion: "DCCT" → explain it!

**Expand acronyms on first use**:
```
"DC Current Transformer (DCCT): Measures beam current..."
"Beam Position Monitor (BPM): Measures horizontal and vertical..."
```

**Include synonyms users might use**:
```
"Beam current" → also mention "circulating charge", "stored current"
"Corrector magnet" → also mention "steering coil", "trim coil"
```

### Technical Detail Guidelines

**Include units when non-obvious**:
```
✅ "Pressure in Torr"
✅ "Current in milliamperes"
✅ "Position in millimeters"
❌ "Pressure reading" (what units?)
```

**Include ranges when helpful**:
```
✅ "Quadrupole current 0-200 Amperes"
✅ "Typical pressure 10^-9 Torr"
✅ "Position range ±10 mm"
```

**Include measurement method when distinguishing**:
```
✅ "Pressure calculated from ion pump current" (vs direct gauge)
✅ "Position measured using button electrodes" (vs stripline)
```

**Include read/write mode**:
```
✅ "Current setpoint (read-write)"
✅ "Current readback (read-only)"
✅ "Golden value (reference only)"
```

### Relationship Guidelines

**Explain how components work together**:
```
"Horizontal corrector magnets (HCM) work with beam position monitors (BPM)
 in feedback loops to maintain desired orbit."
```

**Mention groupings and families**:
```
"One of 96 BPMs distributed around ring (8 per sector)."
"Part of tune correction system along with QD quadrupoles."
```

**Reference related channels**:
```
"Monitor field provides readback; see Setpoint field for control."
"Complementary to VCM (vertical correctors) for 2D orbit correction."
```

---

## 🎨 Description Templates by Device Type

Copy and adapt these templates for common control system devices:

### Position Monitors (BPM, etc.)
```
"[Horizontal/Vertical] position readback in millimeters. Measured using [method].
 [Purpose - e.g., 'Critical for orbit correction']. [Distribution - e.g., '96 BPMs
 total, 8 per sector']."
```

### Magnets (Dipole, Quadrupole, Corrector)
```
"[Magnet type] [function - bending/focusing/steering]. [Technical specs - gradient/
 field]. Current range [X-Y] Amperes. [Purpose in system]."
```

### Vacuum System (Pumps, Gauges, Valves)
```
"[Device type] pressure [measurement/control]. [Location in system]. Measured in
 [units]. [Target/typical value]. [Purpose - e.g., 'Maintains UHV for beam lifetime']."
```

### RF System (Cavities, Klystrons, etc.)
```
"[Parameter - power/voltage/frequency] [measurement/control]. [Technical specs].
 [Purpose - e.g., 'Compensates synchrotron radiation losses']."
```

### Power Supplies
```
"[Output type - current/voltage] [readback/setpoint] for [load device]. [Range].
 [Accuracy if critical]."
```

### Temperature/Environmental
```
"Temperature at [location] measured in Celsius. [Purpose - e.g., 'Monitors cooling
 system performance']. [Interlock info if safety-critical]."
```

### Interlocks/Status
```
"[Status type] indicator. [Values and meanings - e.g., '1=ready, 0=fault'].
 [Consequences if triggered]."
```

---

## 🧪 Testing Your Database

### Query Test Suite

Build a test suite of realistic user queries:

**Coverage Test** (should find the right channels):
```
Query: "beam position monitors"
Expected: All BPM channels (X and Y positions)

Query: "horizontal correctors"
Expected: HCM channels, not VCM

Query: "pressure in the ion pumps"
Expected: Ion pump pressure readbacks, not gauge pressures

Query: "RF cavity voltage"
Expected: Cavity voltage channels, not power or frequency
```

**Precision Test** (should NOT find wrong channels):
```
Query: "beam current"
Should find: DCCT current, BPM sum signals
Should NOT find: Magnet currents, pump currents

Query: "setpoints"
Should find: Writable setpoint channels
Should NOT find: Readback channels
```

**Terminology Test** (handle different phrasings):
```
Query: "beam position" = "orbit" = "BPM readings"
Query: "steering magnets" = "correctors" = "trim coils"
Query: "vacuum pressure" = "pressure gauges" = "vacuum readback"
```

### Extracting from Documentation

When reading system docs:

**Look for**:
- ✅ Function descriptions ("The DCCT measures...")
- ✅ Specifications tables (ranges, units, quantities)
- ✅ System diagrams (relationships, locations)
- ✅ Operational procedures (what gets monitored/controlled)

**Extract**:
- Device definitions
- Technical specs
- Operational context
- Relationships between systems

**Transform to descriptions**:
```
Documentation: "The storage ring has 96 BPMs arranged in 12 sectors with
                8 BPMs per sector..."

Description: "Beam Position Monitor. One of 96 BPMs distributed around ring
              (8 per sector). Measures beam position in X and Y..."
```

---

## 💡 Common Pitfalls to Avoid

### Over-Documentation
❌ **Don't**: Write encyclopedia entries for every channel
```json
{
  "description": "The Beam Position Monitor is a non-invasive diagnostic device
                  that uses the principle of electromagnetic induction to detect
                  the position of a charged particle beam. The monitor consists
                  of four button electrodes arranged symmetrically around the
                  beam pipe. Each electrode capacitively couples to the beam's
                  image charge..." [500 more words]
}
```

✅ **Do**: Write concise, query-optimized descriptions
```json
{
  "description": "Horizontal beam position in millimeters measured by button
                  electrodes. One of 96 BPMs for orbit measurement and correction."
}
```

### Under-Documentation
❌ **Don't**: Leave descriptions empty or too minimal
```json
{
  "description": "BPM X"
}
```

✅ **Do**: Include distinguishing and contextual information
```json
{
  "description": "Horizontal position readback from beam position monitor.
                  Critical for orbit correction."
}
```

### Copy-Paste Without Customization
❌ **Don't**: Reuse identical descriptions for different channels
```json
{
  "BPM1:X": {"description": "Beam position monitor"},
  "BPM1:Y": {"description": "Beam position monitor"},  // Same!
  "BPM2:X": {"description": "Beam position monitor"}   // Same!
}
```

✅ **Do**: Customize with distinguishing details
```json
{
  "BPM1:X": {"description": "Horizontal position from BPM 1"},
  "BPM1:Y": {"description": "Vertical position from BPM 1"},
  "BPM2:X": {"description": "Horizontal position from BPM 2"}
}
```

### Jargon Without Context
❌ **Don't**: Use unexplained acronyms or technical terms
```json
{
  "description": "DCCT RB for I_beam via ACCT method"
}
```

✅ **Do**: Expand acronyms and explain terms
```json
{
  "description": "DC Current Transformer (DCCT) beam current readback in
                  milliamperes. Non-invasive measurement of total circulating charge."
}
```

### Inconsistent Terminology
❌ **Don't**: Use different words for same concept
```json
{
  "HCM1": {"description": "Horizontal steering magnet"},
  "HCM2": {"description": "X-axis corrector coil"},
  "HCM3": {"description": "Horizontal trim dipole"}
}
```

✅ **Do**: Use consistent terminology (but include synonyms)
```json
{
  "HCM1": {"description": "Horizontal corrector magnet (steering coil) for orbit correction"},
  "HCM2": {"description": "Horizontal corrector magnet (steering coil) for orbit correction"},
  "HCM3": {"description": "Horizontal corrector magnet (steering coil) for orbit correction"}
}
```

---

## 🔍 Using Code as Reference

**AI Assistants**: Always ask to read database format code and examples before giving advice. These contain authoritative schema requirements and excellent description patterns.

### Essential Files to Read

**Database Format Classes** (READ FIRST):
```
In-Context:     src/osprey/templates/.../databases/flat.py
Hierarchical:   src/osprey/templates/.../databases/hierarchical.py
Middle Layer:   src/osprey/templates/.../databases/middle_layer.py
```

**What these files contain:**
- Complete schema definitions in class docstrings
- Required vs optional fields with validation logic
- Design rationale explaining why the format works this way
- Example snippets demonstrating key patterns
- Error messages and what triggers them

**Example Databases** (if user has my-control-assistant initialized):
```
my-control-assistant/data/channel_databases/in_context.json       (255 channels, UCSB FEL)
my-control-assistant/data/channel_databases/hierarchical.json     (1,050 channels, accelerator)
my-control-assistant/data/channel_databases/middle_layer.json     (300+ channels, MML-style)
```

**What to learn from examples:**
- ✅ **Description patterns**: Study how existing descriptions are written
- ✅ **Forward-looking navigation**: See how hierarchical descriptions preview child levels
- ✅ **Template usage**: Understand standalone vs template channel patterns
- ✅ **Metadata organization**: See what optional fields are actually useful
- ✅ **Consistent terminology**: Observe naming conventions that work

### How AI Should Use These Resources

**Before giving schema advice:**
1. Ask: "Do you have my-control-assistant initialized?"
2. Read the database format class for user's pipeline
3. Read the corresponding example database
4. Quote from authoritative sources, not memory

**When helping write descriptions:**
1. Read example database to see description patterns
2. Point out specific good examples: "Like this description from hierarchical.json..."
3. Explain WHY the example works (forward-looking, distinguishing, etc.)
4. Adapt patterns to user's specific channels

**When troubleshooting:**
1. Read validation logic in database class
2. Understand what the error actually checks
3. Show user the relevant code section
4. Explain how to fix based on schema requirements

**Example AI workflow:**
```
User: "Help me write descriptions for my hierarchical database's system level"

AI: "Let me read the hierarchical database code and examples to see the schema
     and good description patterns."

[Reads hierarchical.py class docstring and hierarchical.json example]

AI: "Looking at the example database, I can see that system-level descriptions
     use a forward-looking pattern. For instance, the DIAG system description
     (line 234 in hierarchical.json) says:

     'Diagnostic System: Beam measurement and monitoring. Includes beam position
      monitors (BPMs), current monitors (DCCT), and radiation monitors.'

     Notice how it mentions 'BPMs' and 'DCCT' even though these are family-level
     concepts. This helps the LLM navigate correctly when users query 'BPM'
     without saying 'diagnostic system'.

     For your MAG system, following this pattern, I'd suggest..."
```

### Code-Informed Best Practices

**For In-Context (Flat) Databases:**
- Read `flat.py` to understand template expansion rules
- See `in_context.json` for examples of template vs standalone channels
- Note how channel names vs addresses are handled
- Understand chunking strategy from class implementation

**For Hierarchical Databases:**
- Read `hierarchical.py` to understand level types (tree vs instances)
- See `hierarchical.json` for forward-looking description examples
- Study instance expansion patterns (`_expansion` definitions)
- Understand naming pattern assembly logic

**For Middle Layer Databases:**
- Read `middle_layer.py` to understand System/Family/Field structure
- See `middle_layer.json` for functional organization patterns
- Note optional metadata fields and their purposes
- Understand how ChannelNames lists are accessed

### Quick Reference: Where to Find Answers

| Question | Where to Look |
|----------|---------------|
| "What fields are required?" | Database class `__init__` and validation methods |
| "How should I write descriptions?" | Example database JSON files |
| "What does this error mean?" | Database class validation logic |
| "Can I use this optional field?" | Database class docstring and example JSON |
| "How do templates work?" | `flat.py` or `template.py` class implementation |
| "How does hierarchy navigation work?" | `hierarchical.py` `_navigate_recursive` method |
| "What metadata is useful?" | Example database `_metadata` sections |

**Remember**: The code is the authoritative source. When AI and human memory disagree with the code, trust the code.

---

## 📚 Additional Resources

### Related Workflows

- **Pipeline Selection** (`channel-finder-pipeline-selection.md`): How to choose the right pipeline for your system
- **Testing Workflow** (`testing-workflow.md`): How to test your database with real queries
- **Pre-Merge Cleanup** (`pre-merge-cleanup.md`): Validate database before committing

### Related Documentation

- **Control Assistant Tutorial Part 2**: Complete guide to Channel Finder pipelines with detailed format documentation
- **API Reference**: Database class API documentation (auto-generated from docstrings)

---
