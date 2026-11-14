# Question templates

One question per group of findings. Each has the same four parts: the situation in
plain language, a small picture, the evidence, and the options with the recommended one
first and what each would write. The question that follows is one line.

Rules that hold for every template:

- **No kind names in the opening sentence.** "Two sources disagree about which supply
  feeds QF1" reads; "relation-contested on powered_by" does not. Quote the build's line
  under the evidence for a kind this skill has no notes on, or when the person asks.
- **Real ids, real counts.** Two or three example records taken from the findings, and
  the count of the whole group. Never an invented example.
- **Every option says what gets written**, in which file.
- **Recommend only with a checkable reason.** Otherwise list the options unranked and
  say the evidence does not favour one.
- **Always offer "rule on some of them one by one" for a group, and "leave it" for a
  warning.**
- Pictures are a few lines of ASCII, drawn for this finding. They show the disagreement,
  not the facility.

The names and values in the templates are placeholders.

## Two documents disagree, with a pattern

For claims contested between the same two documents in the same way.

```
The rack layout and the cable schedule disagree on which output feeds 6 correctors.

   CM-PS-03
   ┌─────────┐
   │ OUT1 ───┼──▶ HCM1     rack layout (D-114, drawing)
   │ OUT2 ───┼──▶ HCM1     cable schedule (S-22, spec)
   └─────────┘

 6 of 6 correctors on this supply: the layout says OUT<n>, the schedule OUT<n+1>.
 The schedule has no row for OUT1. Examples: HCM1, HCM2, VCM1.

 1. The rack layout wins for power (recommended: the schedule is off by one
    from its first row). Writes powered_by: [drawing, spec] to policy.yaml.
    This also settles 2 other contested feeds, both in favour of the layout.
 2. The cable schedule wins. Writes powered_by: [spec, drawing].
 3. Rule on some of them one by one.
 4. Leave contested.
```

## One claim against one claim

For a single contested key, or an exception pulled out of a group.

```
Two sources name different racks for the BPM electronics of sector 4.

   BPM-S04-E  ──▶  RACK-11 / U07    rack layout (D-114)
   BPM-S04-E  ──▶  RACK-12 / U07    cable schedule (S-22, row 88)

 Every other BPM chassis of sector 4 is in RACK-11 in both documents.

 1. RACK-11 is right (recommended: all neighbours agree). Writes a drop fix
    for the schedule's claim, with your reason.
 2. RACK-12 is right. Writes a drop fix for the layout's claim.
 3. Leave contested.
```

## Same name, maybe the same thing

For shared names, stubs that resemble known devices, joined records a layer states
apart, and places of one name under several parents. Never recommend "the same" on the
strength of the name; recommend only when other fields decide it.

```
Two records carry the name QF1PS in the cable-schedule naming.

   SR/S01/QF1-PS     class PowerSupply    place SR/S01/RACK-03
   BR/S01/QF1-PS     class PowerSupply    place BR/S01/RACK-01

 Different rings, different racks. The names differ only by ring prefix,
 which the cable schedule does not write.

 1. Two devices; the schedule's name is not unique (recommended: the places
    differ). Writes nothing; the shared name stays reported.
 2. Two devices, and the schedule's rows should say which ring. A correction
    in the schedule, then a re-import.
 3. One device described twice. Tell me which record is the real one.
```

## Two layers disagree on a field

For field conflicts between imports.

```
The MML export and the channel list disagree on the role of 48 channels.

   SR:QF1:SP      MML export: readback      channel list: setpoint
   SR:QF1:RB      MML export: readback      channel list: readback

 All 48 end in :SP. No channel ending in :RB is affected.

 1. The channel list is right: these are setpoints (recommended: the suffix
    and the list agree). Writes 48 set fixes to fixes.yaml, one per channel,
    each recording what both sources say today.
 2. The MML export is right. Writes 48 set fixes the other way.
 3. Rule on some of them one by one.
```

A stop has no "leave it": the build produces nothing until it is ruled.

## Something is missing

For references to records that do not exist, and placeholder devices.

```
12 cable-schedule rows name devices no source describes.

   schedule row 40:   QF1  ──cable C-0040──▶  JB-S01-A   ?
   schedule row 41:   QD1  ──cable C-0041──▶  JB-S01-A   ?

 All 12 unknown names start with JB- (junction boxes?). 4 distinct names.

 1. They are real devices not yet described. Tell me their class and I add
    4 devices by fix; or keep them as placeholders for now.
 2. They are described elsewhere under another name. Tell me which.
 3. The rows should not be in the facility. Writes drop fixes for 12 claims.
```

There is no recommendation here because nothing in the tree says what a `JB-` is. Asking
what something is comes before asking what to do about it.

## A ruling went stale

For a fix whose recorded source values no longer match, after an import.

```
An earlier decision no longer matches its source.

   SR:QF:SP  role
   when ruled:   MML export said readback   ──▶ fix: setpoint
   today:        MML export says setpoint

 Recorded reason: "The export lists the setpoint as a monitor."

 1. The export was corrected; the fix is no longer needed (recommended).
    Removes the fix from fixes.yaml.
 2. Keep the fix. Updates what it records about the source.
```

## Approving a diff

For authored files. Show the diff, then one line.

```
 records/devices.yaml

 -  - id: SR/S01/QF1
 -    class: Quad
 +  - id: SR/S01/QF1
 +    class: Quadrupole

 Apply this change? yes / change it / no
```
