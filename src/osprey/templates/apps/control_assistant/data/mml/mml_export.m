function files = mml_export(outdir)
%MML_EXPORT Write the current sub-machine's Middle Layer as OSPREY's input set.
%
%   MML_EXPORT writes five files into the current folder for the sub-machine
%   the Middle Layer is set up for:
%
%       <machine>.<submachine>.lattice.mat   the model ring (THERING)
%       <machine>.<submachine>.ao.json       the Accelerator Objects (getao)
%       <machine>.<submachine>.ad.json       the Accelerator Data    (getad)
%       <machine>.<submachine>.va.json       per-family calibration and nominals
%       <machine>.<submachine>.response.json the orbit response matrix
%
%   MML_EXPORT(OUTDIR) writes them into OUTDIR instead.
%
%   FILES = MML_EXPORT(...) returns the five paths in the order AO, AD,
%   lattice, VA, response.
%
%   Run it once per sub-machine, after the MML setpath for that sub-machine,
%   then import the AO file with
%
%       osprey mml import <machine>.<submachine>.ao.json
%
%   The importer reads the four siblings beside it on its own.
%
%   The order the files are written in
%   ----------------------------------
%   THERING is saved before the export samples anything. getpvmodel reaches a
%   reading through the ring's own closed orbit and turns radiation off or the
%   cavity on to get one, and leaves the ring in whichever state it needed, so
%   a lattice saved after the sampling is no longer the lattice the rest of
%   the export describes. measbpmresp steps its correctors on a copy of the
%   ring, so what a model measurement leaves behind is what its own readings
%   needed rather than the columns it built.
%
%   A family whose conversion or readback functions refuse a sample is
%   recorded under "refused" with the MATLAB message and costs the export
%   nothing but its own block; the same holds for the response matrix.
%
%   What is written
%   ---------------
%   Every JSON file starts with an "_export" block naming the exporter
%   version, the MATLAB version, the machine, the sub-machine and a
%   timestamp. Every value is rewritten into the spelling OSPREY's importer
%   expects:
%
%     * function handles become {"$fn": func2str(h), "file": <path or "">};
%     * char matrices become one deblanked string per row;
%     * text that begins at the Middle Layer root - a handle's file, a data
%       directory, a response-file name - is recorded relative to that root,
%       so it names the same file in every checkout of the tree rather than
%       the account the export happened to run under;
%     * Inf, -Inf and NaN become the strings "Inf", "-Inf" and "NaN", so no
%       non-finite number is ever encoded (jsonencode is called with
%       'ConvertInfAndNaN', false, so nothing is written as null);
%     * logicals become 0/1, string objects become char, sparse matrices
%       become full, and a "Handles" field (graphics handles) is dropped.
%
%   Matrix shape is kept: a 1-row value is a flat array, an N-row value an
%   array of rows.
%
%   What va.json holds
%   ------------------
%   These keys are the whole file, and they are what OSPREY's importer reads:
%
%     lattice             the ring every other number here belongs to
%       elements          numel(THERING), a RingParam element included, so
%                         the element indices the Middle Layer uses hold
%       famname_sha256    sha256 over the UTF-8 bytes of the ring's FamName
%                         list in ring order, each name as the ring carries
%                         it, joined with newlines, none trailing
%       energy_gev        the model energy every sample below was taken at
%       ringparam_indices where the ring's parameter elements sit, if any
%     families            one block per family of the Accelerator Objects
%       device_list       the devices every per-device row below is in the
%                         order of
%       fields            every field the family answers to, by name;
%                         Setpoint and Monitor are the two that are sampled
%       nominals          what the family is set to, per field name:
%                         values, units, at_type, at_index, synthetic
%       Setpoint          calibration, energy_scaling, energy_deviation
%       Monitor           calibration, monitor_inverse, readout
%       energy_candidate  whether the ring's energy is read from this family
%       energy_table      device_row, grid, values, finite_span, I_nom,
%                         energy_at_nominal
%       refused           every refusal the block met, in the order the
%                         steps ran, joined with "; "
%
%   A calibration - and a monitor_inverse, which is one sampled the other way
%   round - is either {kind: "linear", gain, offset} or {kind: "table", grid,
%   values, finite_span}, with grid_source beside it, anchor for a grid laid
%   around a hardware setting, and fcn, the conversion function's own name. A
%   linear one carries no grid: its line is the conversion. A monitor_inverse
%   carries grid_source and fcn but no anchor.
%
%   A readout is what the Middle Layer corrects the family's own readings by,
%   one value per device under each key the family states:
%
%     gain, offset      the scale and the shift a reading is corrected by,
%                       the offset in the family's own HARDWARE units - the
%                       units its readback answers in, millimetres on most
%                       beam monitors, and not the physics units of the
%                       conversion beside it; both are already inside that
%                       conversion
%     roll, crunch      the rotation in radians and the shear that carry the
%                       two planes a beam monitor reads in into the model's,
%                       and are in no conversion at all
%
%   Every family whose Monitor conversion was sampled may carry one, not the
%   beam monitors alone: a facility that calibrates its magnets and correctors
%   states their gains and rolls the same way and under the same names, and
%   the numbers are as true of those families as of a monitor. What a magnet's
%   roll turns is the magnet in the ring rather than a pair of reading planes,
%   so a consumer reads roll against the family it sits on. A key the family
%   states nothing for is absent rather than a default.
%
%   A key whose numbers came from the facility's physics data rather than from
%   the Accelerator Objects can hold "NaN" for single devices: the physics data
%   is stored against its own device list, and a device that list does not
%   cover comes back as no number, named on the console as it happens. A "NaN"
%   entry therefore means the facility states nothing for that one device, and
%   a consumer reads it as an absent number rather than as a value.
%
%   The words a reader switches on are these and no others:
%
%     kind              "linear" | "table"
%     grid_source       "range" | "setpoint" | "fallback"
%     anchor            "nominal" | "range_midpoint" | "zero"
%     energy_scaling    "brho" | "none"
%
%   A key whose fact was never read is absent rather than empty: a family
%   with no Setpoint carries no Setpoint key and one that refused nothing
%   carries no refused key, while a block that carries a fact carries it
%   whatever else refused. device_list, fields and energy_candidate are in
%   every block, except a family the export could not read at all, whose
%   block is refused alone; nominals, Setpoint, Monitor, energy_table and
%   refused are there when there was something to write. lattice is always
%   there too, carrying either its four facts or the reason it has none.
%
%   Reading the shapes
%   ------------------
%   Every per-device value is one row per device in DeviceList order, and an
%   N-row matrix is written as an array of rows. One row is written flat and
%   one number as a number: a one-device family's device list, gain column or
%   table row is a flat array rather than an array holding one row, and a
%   ringparam_indices of one index is that index. A reader takes a flat array
%   as the single row it is, and a bare number as the one-entry list it is.
%
%   Requires a MATLAB with jsonencode and its Java runtime - the lattice
%   digest is taken through it - an initialised Middle Layer and the
%   sub-machine's simulator model loaded.

EXPORTER_VERSION = 'mml_export 2.0.0';

if nargin < 1 || isempty(outdir)
    outdir = pwd;
end
if ~exist(outdir, 'dir')
    error('mml_export:outdir', 'Output folder does not exist: %s', outdir);
end
if exist('getao', 'file') == 0 || exist('getad', 'file') == 0
    error('mml_export:mml', ...
        'getao/getad not found. Run the Middle Layer setpath for the sub-machine first.');
end

AO = getao;
AD = getad;
if isempty(AO) || ~isstruct(AO)
    error('mml_export:ao', 'getao returned no Accelerator Objects. Is the Middle Layer initialised?');
end
if isempty(AD) || ~isstruct(AD)
    error('mml_export:ad', 'getad returned no Accelerator Data. Is the Middle Layer initialised?');
end

machine = local_text(local_field(AD, 'Machine'));
submachine = local_text(local_field(AD, 'SubMachine'));
if isempty(machine) || isempty(submachine)
    error('mml_export:names', 'AD.Machine and AD.SubMachine must both be set.');
end

export = struct( ...
    'exporter', EXPORTER_VERSION, ...
    'matlab', version, ...
    'machine', machine, ...
    'submachine', submachine, ...
    'timestamp', datestr(now, 'yyyy-mm-ddTHH:MM:SS'));

stem = [local_filename(machine) '.' local_filename(submachine)];
aoFile = fullfile(outdir, [stem '.ao.json']);
adFile = fullfile(outdir, [stem '.ad.json']);
latticeFile = fullfile(outdir, [stem '.lattice.mat']);
vaFile = fullfile(outdir, [stem '.va.json']);
responseFile = fullfile(outdir, [stem '.response.json']);

local_save_lattice(latticeFile);

local_write(aoFile, local_document(export, local_normalize(AO)));
local_write(adFile, local_document(export, local_normalize(AD)));
local_va_export(export, AO, AD, vaFile, responseFile);

fprintf('Wrote %s\n', latticeFile);
fprintf('Wrote %s\n', aoFile);
fprintf('Wrote %s\n', adFile);
fprintf('Wrote %s\n', vaFile);
fprintf('Wrote %s\n', responseFile);

if nargout > 0
    files = {aoFile, adFile, latticeFile, vaFile, responseFile};
end
end


function local_save_lattice(path)
% Save the model ring under its own name, before the export samples anything.
%
% pyAT reads the file back with at.load_mat(use='THERING'), and -v7 is the
% newest format SciPy's MAT reader accepts.
global THERING
if isempty(THERING)
    error('mml_export:lattice', ...
        ['THERING is empty. Load the sub-machine''s simulator model before ' ...
         'exporting: the lattice is what the rest of the export describes.']);
end
save(path, 'THERING', '-v7');
end


function local_va_export(export, AO, AD, vaFile, responseFile)
% Write the two virtual-accelerator siblings: what each family's Setpoint and
% Monitor fields do in hardware units, and the orbit response matrix.
%
% Every family is sampled on its own, and so is the response matrix, because
% a facility's conversion functions are its own code and any one of them may
% refuse. A refusal is a "refused" message in that family's block, never an
% error out of the export.
%
% The ring is fingerprinted first, and every family is sampled at the one
% model energy read here: a calibration sampled at one energy beside a
% fingerprint naming another describes no machine, and the importer holds the
% two against each other.
energy = [];
try
    energy = getenergymodel;
catch
    % A model energy the Middle Layer cannot answer for masks nothing here.
    % It reaches the fingerprint, which refuses it by name and makes the
    % document unusable to the importer, and every family, whose
    % energy_scaling and energy table refuse it by name while a calibration
    % is sampled at the energy the Middle Layer substitutes. That is where
    % the reason belongs; raised here it would cost the export both of the
    % files this function writes.
end

try
    va = struct('lattice', local_lattice_fingerprint(energy));
catch err
    va = struct('lattice', struct('refused', err.message));
end
va.families = struct();

names = fieldnames(AO);
for k = 1:numel(names)
    family = names{k};
    try
        va.families.(family) = local_va_family(family, AO, energy);
    catch err
        % The last resort. A refusal the sampling itself saw is recorded
        % beside the facts already in the block; only a failure outside any
        % of them - an unreadable family, a broken AO entry - lands here,
        % and then there is nothing left to keep.
        va.families.(family) = struct('refused', err.message);
    end
end

try
    response = local_response(AD);
catch err
    response = struct('refused', err.message);
end

local_write(vaFile, local_document(export, local_normalize(va)));
local_write(responseFile, local_document(export, local_normalize(response)));
end


function fingerprint = local_lattice_fingerprint(energy)
% The ring the rest of this file describes, in four facts a consumer can
% recompute from the saved lattice.
%
% The lattice travels as a MATLAB file beside this one, and nothing stops the
% two from being separated, re-saved or paired with another sub-machine's
% export. The fingerprint is what a consumer pairs them by: it recomputes
% these four from the lattice it holds and refuses the pair when they
% disagree, rather than binding a calibration to elements it was never
% sampled against.
%
% The element count includes the ring's parameter element where it carries
% one. It is an element of THERING, the Middle Layer indexes past it, and a
% count that left it out would place every other element one off.
%
% The digest is over the family names in ring order and nothing else: the
% names are what a consumer binds an element by, and a lattice whose element
% order or naming moved is not the one these calibrations were sampled over.
% Positions and strengths are deliberately outside it - a ring re-solved to
% another working point is still the same ring for that purpose.
global THERING
local_require_lattice_energy(energy);

fingerprint = struct();
fingerprint.elements = numel(THERING);
fingerprint.famname_sha256 = local_sha256(strjoin(local_famnames(THERING), newline));
fingerprint.energy_gev = energy;
fingerprint.ringparam_indices = local_ringparam_indices(THERING);
end


function names = local_famnames(ring)
% The family name of every element, in ring order, as the ring carries it.
%
% An element with no name of its own is refused: the digest exists to say
% which elements these calibrations were sampled over, and an element a
% consumer cannot name is one it cannot bind to either.
%
% The name goes into the digest verbatim - neither trimmed nor cut to a row.
% The digest is the pairing contract, and the consumer recomputes it over the
% name the lattice file holds, so a rule applied on this side alone refuses a
% pair that is in fact the same ring. A char of more than one row is no one
% name, and is refused by name rather than quietly reduced to its first.
names = cell(1, numel(ring));
for k = 1:numel(ring)
    name = '';
    if isstruct(ring{k})
        name = local_field(ring{k}, 'FamName');
        if isa(name, 'string')
            name = char(name);
        end
    end
    if ~ischar(name) || isempty(name)
        error('mml_export:lattice', ...
            'Element %d of THERING carries no FamName, so the ring has no fingerprint.', k);
    end
    if size(name, 1) > 1
        error('mml_export:lattice', ...
            ['Element %d of THERING carries a FamName of %d rows, which is no one ' ...
            'name, so the ring has no fingerprint.'], k, size(name, 1));
    end
    names{k} = name;
end
end


function local_require_lattice_energy(energy)
% Refuse a model energy that is not one positive number, before the ring is
% fingerprinted at it.
%
% The energy is the one every conversion in this file was sampled at, and the
% Middle Layer reads it off the ring itself, so the importer holds it against
% the energy of the lattice it loads. An energy that is not one number is
% refused rather than written: a fingerprint a consumer cannot compare is one
% that pairs any lattice with any export.
if ~isscalar(energy) || ~isfinite(energy) || energy <= 0
    error('mml_export:lattice', ...
        ['getenergymodel gave %s for the model energy, which is not one positive ' ...
         'number of GeV. Load the sub-machine''s model before exporting.'], ...
        mat2str(energy));
end
end


function indices = local_ringparam_indices(ring)
% Where the ring's parameter elements sit, one-based, in ring order.
%
% A parameter element carries the ring's own properties rather than a piece of
% beam line, and a reader may keep it, drop it or tag it. Its place is
% recorded so a consumer that drops one knows which index every element after
% it moved by, and a consumer that keeps one knows which element is not a
% magnet.
indices = [];
for k = 1:numel(ring)
    if isstruct(ring{k}) && strcmp(local_text(local_field(ring{k}, 'Class')), 'RingParam')
        indices(end+1) = k; %#ok<AGROW>
    end
end
end


function digest = local_sha256(text)
% One string's SHA-256, as lower-case hexadecimal.
%
% The digest is taken through the Java runtime MATLAB ships with, which needs
% no toolbox and no function from outside the installation.
%
% The bytes are the string's UTF-8 encoding, named rather than left to the
% default: the default is the encoding of the machine the export ran on, and
% a digest that changes with the machine pairs nothing. They are handed over
% as signed bytes because a Java byte is signed, and a byte above 127 handed
% over unsigned is not the byte it stands for.
bytes = typecast(unicode2native(text, 'UTF-8'), 'int8');
sha = java.security.MessageDigest.getInstance('SHA-256');
raw = typecast(sha.digest(bytes), 'uint8');
digest = lower(reshape(dec2hex(raw, 2)', 1, []));
end


function block = local_va_family(family, AO, energy)
% One family's VA block: the Setpoint calibration, the Monitor inverse, the
% energy facts and the per-device nominal, sampled through the facility's own
% conversion functions.
%
% The block is filled fact by fact, so that a family whose conversion refuses
% partway keeps the ones already in hand beside its "refused" message.
%
% The nominal is read first, because it is the setting every grid below is
% laid around: read after them, each grid would be sized by zero instead and
% the calibration would be sampled where the family never sits. The one
% column of hardware nominals it hands back is then the same seam the
% calibrations and the energy table are sized by, so the three describe one
% operating point rather than three.
%
% What the block carries of a sampled field is the calibration, not the
% samples: a line reproduces its own grid from two numbers, and a table
% carries the grid it was sampled over inside itself. The samples stay here
% for the energy table to be sampled over and for the second energy to be
% compared against.
%
% Of the steps that can refuse, every one that refused leaves its own message.
% A refusal does not stop the steps after it: a nominal the Middle Layer could
% not give leaves the grids to be sized by Range or zero, a refused sample
% leaves the energy table to refuse on its own account, and a readout the
% family states in the wrong shape costs the family that and nothing else.
% Which of the steps ran at all is visible by the keys they left.
block = struct();
DeviceList = local_device_list(AO, family);
block.device_list = DeviceList;
block.fields = local_field_names(AO, family);

[recorded, nominals, refusedNominal] = local_nominals(family, AO, DeviceList);
if ~isempty(fieldnames(recorded))
    block.nominals = recorded;
end

[sampled, refusedSample] = local_sample_fields(family, AO, nominals, energy);
if isfield(sampled, 'Setpoint')
    setpoint = struct('calibration', sampled.Setpoint.calibration);
    if isfield(sampled.Setpoint, 'energy_scaling')
        setpoint.energy_scaling = sampled.Setpoint.energy_scaling;
        setpoint.energy_deviation = sampled.Setpoint.energy_deviation;
    end
    block.Setpoint = setpoint;
end
[readout, refusedReadout] = local_readout(family, AO, DeviceList);
if isfield(sampled, 'Monitor')
    monitor = struct('calibration', sampled.Monitor.calibration);
    if isfield(sampled.Monitor, 'monitor_inverse')
        monitor.monitor_inverse = sampled.Monitor.monitor_inverse;
    end
    if ~isempty(fieldnames(readout))
        monitor.readout = readout;
    end
    block.Monitor = monitor;
end

refusedTable = '';
block.energy_candidate = local_energy_candidate(AO, family);
if block.energy_candidate
    [energyTable, refusedTable] = ...
        local_energy_table(family, DeviceList, sampled, nominals, energy);
    if ~isempty(fieldnames(energyTable))
        block.energy_table = energyTable;
    end
end

refused = local_refusals({refusedNominal, refusedSample, refusedReadout, refusedTable});
if ~isempty(refused)
    block.refused = refused;
end
end


function [readout, refused] = local_readout(family, AO, DeviceList)
% What the Middle Layer corrects a family's own reading by, per device.
%
% Four numbers the Middle Layer keeps beside a family describe how its
% readings differ from the model's: a gain and an offset scaling and shifting
% each reading, and a roll and a crunch rotating and shearing the pair of
% planes a beam monitor reads in. The first two are already inside this
% family's sampled conversion, which is the gain and offset conversion applied
% before the physics one; the second two are not in any conversion at all,
% because a conversion carries one family's readings and a rotation between
% planes carries two families' together.
%
% Any family may state them, not the beam monitors alone. A facility that
% calibrates its magnets and correctors keeps their gains and rolls under the
% same four names, and a magnet's roll turns the magnet rather than a pair of
% reading planes. The export writes what the family states and leaves the
% reading of it to the family it sits on.
%
% They are written whole rather than folded in, because what they describe is
% the reading's calibration and not its conversion: a readout error is seeded
% about them and a conversion cannot carry a rotation between two families.
% Writing them beside the conversion leaves a consumer free to apply them or
% not; folding them in would leave it no way to tell they were there.
%
% Each is looked up in the three places the Middle Layer's own readers look,
% in their order: under the family's Monitor field, then on the family itself,
% which is where a facility keeps the numbers its monitors share, then in the
% facility's physics data, which is where a facility that fits its monitors
% from an orbit measurement keeps them until the operating mode copies them up
% into the Accelerator Objects. A number none of the three carries is absent
% rather than a default, so what is written is what the facility states. A
% number in a shape that is not one per device is refused by name, and costs
% the family that one number and nothing else - the other three are written.
readout = struct();
refusals = {};
keys = {'Gain', 'gain'; 'Offset', 'offset'; 'Roll', 'roll'; 'Crunch', 'crunch'};
nDev = size(DeviceList, 1);
stored = struct();
if isfield(AO, family)
    stored = AO.(family);
end
for k = 1:size(keys, 1)
    name = keys{k, 1};
    where = sprintf('%s.Monitor.%s', family, name);
    value = local_subfield(AO, family, 'Monitor', name);
    if isempty(value)
        where = sprintf('%s.%s', family, name);
        value = local_field(stored, name);
    end
    if isempty(value)
        where = sprintf('the physics data''s %s %s', family, name);
        value = local_physdata(family, name, DeviceList);
    end
    if isempty(value)
        continue
    end
    try
        readout.(keys{k, 2}) = local_readout_column(where, value, nDev);
    catch err
        refusals{end+1} = err.message; %#ok<AGROW>
    end
end
refused = local_refusals(refusals);
end


function value = local_physdata(family, name, DeviceList)
% One readout number as the facility's physics data holds it, or nothing.
%
% The third place the Middle Layer looks is the physics data file the
% Accelerator Data names, which it reads and never writes. A facility that
% keeps no such file, or keeps one stating nothing for this family, answers
% the read with an error, and an error here is the same answer as an empty
% field: this family states no such number. Reaching it is worth the read
% because a facility can sit in exactly that state - fitted numbers on file,
% not yet copied into the Accelerator Objects - and a reading corrected by
% numbers the export did not write would be corrected by nothing.
value = [];
try
    value = getphysdata(family, name, DeviceList);
catch
    value = [];
end
end


function column = local_readout_column(where, value, nDev)
% One readout number for each of the family's devices.
%
% A facility may state one number for a whole family rather than one per
% device, and the Middle Layer hands that one number back for every device in
% the list. The export writes what the Middle Layer would answer, so a single
% number is written out per device here too, and a consumer never has to know
% which of the two shapes the facility wrote it in.
%
% A column already holding one number per device is taken as it stands.
% Anything else cannot be one number per device: a row of several is handed to
% every device whole, more than one number per device is more than this key
% means, and a value that is not a number is not a correction at all. Each is
% refused by name, naming the shape found and the place it was found in, so
% the refusal points at the field to look at rather than at a guess.
if ~isnumeric(value)
    error('mml_export:readout', ...
        '%s holds %s values rather than numbers.', where, class(value));
end
if isscalar(value)
    column = ones(nDev, 1) * double(value);
    return
end
rows = size(value, 1);
if rows == nDev && numel(value) == nDev
    column = double(value(:));
    return
end
error('mml_export:readout', ...
    '%s holds %d by %d values for %d devices.', where, rows, numel(value) / rows, nDev);
end


function names = local_field_names(AO, family)
% Every field the family answers to, by name, in the order the Accelerator
% Objects hold them.
%
% A field of a family is a struct of its own carrying a Mode - the word the
% Middle Layer reads that field through, from the machine or from the model.
% That is what separates a field from the family's other structs: the AT block
% describes a lattice element rather than a way of reading the family, and it
% carries no Mode.
%
% Setpoint and Monitor are the two this export samples. The rest are named
% here and nothing more, so a consumer can see what else the family answers
% to without this file claiming to have converted it.
names = {};
body = AO.(family);
held = fieldnames(body);
for k = 1:numel(held)
    value = body.(held{k});
    if isstruct(value) && isscalar(value) && isfield(value, 'Mode')
        names{end+1} = held{k}; %#ok<AGROW>
    end
end
end


function refused = local_refusals(messages)
% Every refusal a sequence of steps met, in the order they ran, or nothing
% when none refused.
%
% One message is not enough: a step that refused does not stop the steps
% after it, so a block whose nominal and whose sampling both refused has two
% reasons, and carrying only the first leaves the key the second explains
% missing with nothing said about it.
%
% One reason met twice is carried once, because the same sentence repeated
% reads as two faults.
kept = {};
for k = 1:numel(messages)
    message = messages{k};
    if isempty(message) || any(strcmp(kept, message))
        continue
    end
    kept{end+1} = message; %#ok<AGROW>
end
refused = strjoin(kept, '; ');
end


function [sampled, refused] = local_sample_fields(family, AO, nominals, energy)
% The Setpoint and Monitor calibrations of one family, whether its Setpoint
% conversion carries the beam's rigidity, and the message of the call that
% stopped them.
%
% NOMINALS carries one column of hardware nominals per field name, the
% settings the grids are sized by; a field whose nominal the Middle Layer
% could not give a number for is sized by its own Range instead, and by zero
% only when there is no band either. ENERGY is the model energy the conversion
% runs at, in GeV, the same energy the Middle Layer's own write path converts
% at.
%
% Each step is taken inside its own try: a family that refuses one conversion
% keeps every field sampled before it, and the refusal is the message of the
% first call that failed. Sampling stops there - the family is refused
% already, and a second message says no more than the first.
sampled = struct();
refused = '';
DeviceList = local_device_list(AO, family);

names = {'Setpoint', 'Monitor'};
for k = 1:numel(names)
    name = names{k};
    if ~isfield(AO.(family), name) || ~isstruct(AO.(family).(name))
        continue
    end
    try
        [calibration, grid, values] = local_sample_calibration( ...
            family, name, DeviceList, AO, local_nominal_of(nominals, name), energy);
        sampled.(name) = struct('calibration', calibration, 'grid', grid, 'values', values);
    catch err
        refused = err.message;
        return
    end
end

if isfield(sampled, 'Setpoint')
    try
        [sampled.Setpoint.energy_scaling, sampled.Setpoint.energy_deviation] = ...
            local_energy_scaling(family, 'Setpoint', DeviceList, AO, ...
                sampled.Setpoint.grid, sampled.Setpoint.values, energy);
    catch err
        refused = err.message;
        return
    end
end

if isfield(sampled, 'Monitor')
    try
        sampled.Monitor.monitor_inverse = ...
            local_sample_inverse(family, DeviceList, AO, sampled, energy);
    catch err
        refused = err.message;
    end
end
end


function [calibration, grid, values] = local_sample_calibration(family, field, DeviceList, AO, nominal, energy)
% One field's hardware-to-physics calibration, sampled through the Middle
% Layer's own conversion.
%
% The conversion is taken by running hw2physics over a grid of hardware
% values - the call the Middle Layer's write path makes itself - and never by
% reading the stored conversion parameters: the same function name is a
% different algorithm at every facility, and a ramp table or an excitation
% curve is not in the Accelerator Objects at all. Because hw2physics converts
% to corrected hardware units first, the per-device gain and offset of the
% calibration are already inside the sampled numbers.
%
% A conversion that is a straight line in the hardware value is written as
% that line's gain and offset, taken from two of its own samples, so the
% consumer needs no table; anything else is written as the grid and its
% image, which the consumer reads by interpolation and continues linearly
% beyond the ends.
%
% GRID and VALUES are returned beside the calibration because they are what a
% second sampling at another energy is compared against, which the compact
% linear form no longer holds.
if isempty(DeviceList)
    error('mml_export:devices', 'Family %s lists no devices to sample.', family);
end

nDev = size(DeviceList, 1);
[grid, source, anchor] = local_hardware_grid(local_range(AO, family, field, nDev), nominal, nDev);
values = local_sample_hw2physics(family, field, DeviceList, grid, energy);
local_require_finite(family, field, AO, 'HW2PhysicsFcn', grid, values);

calibration = local_calibration(grid, values);
calibration.grid_source = source;
calibration.anchor = anchor;
calibration.fcn = local_fcn_name(local_subfield(AO, family, field, 'HW2PhysicsFcn'));
end


function inverse = local_sample_inverse(family, DeviceList, AO, sampled, energy)
% The Monitor field's physics-to-hardware conversion, sampled the way the
% Middle Layer's simulator reads a monitor back: physics2hw over what the
% model holds.
%
% physics2hw is not the inverse of hw2physics by construction. The two
% parameter sets are independent data, and the gain and offset are applied
% after the conversion on the way out where they are applied before it on the
% way in, so the inverse is sampled rather than derived from the calibration.
%
% Its grid is in physics units and is the physics image of the family's own
% hardware grid: the Setpoint grid where the family has a setpoint, which is
% exactly the span a write can reach, and for a monitor-only family the
% image of its Monitor Range, stretched where it must be to hold the
% anchor. A monitor-only family whose Monitor carries no Range is sampled
% over a beam position span instead.
if isfield(sampled, 'Setpoint')
    grid = sampled.Setpoint.values;
    source = 'setpoint';
elseif isfield(sampled, 'Monitor') && strcmp(sampled.Monitor.calibration.grid_source, 'range')
    grid = sampled.Monitor.values;
    source = 'range';
else
    grid = local_monitor_grid(size(DeviceList, 1));
    source = 'fallback';
end

values = local_sample_physics2hw(family, 'Monitor', DeviceList, grid, energy);
local_require_finite(family, 'Monitor', AO, 'Physics2HWFcn', grid, values);

inverse = local_calibration(grid, values);
inverse.grid_source = source;
inverse.fcn = local_fcn_name(local_subfield(AO, family, 'Monitor', 'Physics2HWFcn'));
end


function values = local_sample_hw2physics(family, field, DeviceList, grid, energy)
% The physics image of a hardware grid, one column of device values per call.
%
% A facility's conversion function is written for one value per device.
% Handing it the whole grid at once is what makes it refuse, so the grid is
% walked column by column however long it is.
values = zeros(size(grid));
for k = 1:size(grid, 2)
    column = hw2physics(family, field, grid(:, k), DeviceList, energy);
    values(:, k) = local_column(family, field, 'hw2physics', column, size(grid, 1));
end
end


function values = local_sample_physics2hw(family, field, DeviceList, grid, energy)
% The hardware image of a physics grid, one column of device values per call.
values = zeros(size(grid));
for k = 1:size(grid, 2)
    column = physics2hw(family, field, grid(:, k), DeviceList, energy);
    values(:, k) = local_column(family, field, 'physics2hw', column, size(grid, 1));
end
end


function column = local_column(family, field, fcn, column, nDev)
% One conversion call's answer, as the column of device values it was asked
% for.
%
% The conversion is the facility's own code, and the export takes its shape on
% trust nowhere: an empty answer is a null assignment in MATLAB and would
% delete the grid point it was sampled at, and a row is accepted silently and
% then read back along the wrong dimension. Both are refused by name instead.
if numel(column) ~= nDev
    error('mml_export:shape', '%s.%s: %s answered with %d values for %d devices.', ...
        family, field, fcn, numel(column), nDev);
end
column = column(:);
end


function [scaling, deviation] = local_energy_scaling(family, field, DeviceList, AO, grid, values, energy)
% Whether a family's conversion carries the beam's rigidity, measured rather
% than assumed.
%
% The same hardware grid is converted a second time at an energy two per cent
% above the deck's, and each sample is weighed with the rigidity of the energy
% it was taken at. A conversion that divides by the rigidity - a measured ramp
% table or an excitation polynomial on its way to a normalised strength -
% leaves k*Brho unchanged, and the family follows the energy knob. A
% conversion that ignores the energy it is handed, which is what the Middle
% Layer's own gain-and-offset branch does, leaves k unchanged instead, so
% k*Brho moves by the whole ratio of the two rigidities, and that family
% reads as none - its strength does not follow the beam energy here, because
% it does not follow it in the Middle Layer either.
%
% Brho is the Middle Layer's own getbrho, rest mass included, and nothing here
% re-derives it. The massless E/c form cannot decide this: it misses the
% rigidity itself by 1.7e-4 at 3 GeV, and - what the comparison actually rests
% on - it moves the RATIO of the two rigidities by 3.3e-6 over the step,
% several times the tolerance, so every scaled family would read as none.
%
% DEVIATION is the largest relative move measured, and is a fact about the
% family either way: near machine precision for a family that scales, near the
% rigidity ratio for one that does not.
local_require_energy(family, field, energy);

step = local_energy_step();
shifted = local_sample_hw2physics(family, field, DeviceList, grid, energy * step);
local_require_finite(family, field, AO, 'HW2PhysicsFcn', grid, shifted);

reference = values * getbrho(energy);
moved = shifted * getbrho(energy * step);

% A grid point beyond the end of a facility's conversion table converts to a
% value that is not a number, and a point only one of the two energies reached
% compares nothing. Either way it is left out of the measurement rather than
% deciding it, and a field with no comparable point at all is refused.
both = isfinite(reference) & isfinite(moved);
if ~any(both(:))
    error('mml_export:energy', ...
        '%s.%s: no grid point converted at both %g and %g GeV.', ...
        family, field, energy, energy * step);
end

scale = max(abs(reference), abs(moved));
deviations = abs(moved - reference) ./ scale;
deviations(scale == 0 | ~both) = 0;
deviation = max(deviations(:));

if deviation <= local_energy_tolerance()
    scaling = 'brho';
else
    scaling = 'none';
end
end


function step = local_energy_step()
% How far the energy is moved to see whether a conversion follows it. Far
% enough that a conversion which does not scale misses by four orders of
% magnitude more than the tolerance below, and near enough that a facility's
% conversion still answers over the same hardware grid.
step = 1.02;
end


function tol = local_energy_tolerance()
% How far k*Brho may move between the two energies before the conversion is
% read as not carrying the rigidity. A conversion that divides by Brho returns
% the same product to machine precision; one that does not moves by the whole
% ratio of the two rigidities, which is the size of the step itself.
tol = 1e-6;
end


function local_require_energy(family, field, energy)
% Refuse a model energy that is not one positive number, before any
% conversion is run at it.
%
% getenergymodel answers with nothing when the deck carries no Energy field
% and the Middle Layer's global is unset, and every conversion below takes
% that silently: hw2physics substitutes the machine energy for an empty one
% and getbrho does the same, so two samples meant to be taken at different
% energies are taken at the same one and every family reads as carrying the
% rigidity. An energy per element is refused for the same reason - it converts
% to one rigidity per element and compares nothing.
if ~isscalar(energy) || ~isfinite(energy) || energy <= 0
    error('mml_export:energy', ...
        ['%s.%s: getenergymodel gave %s for the model energy, which is not one ' ...
         'positive number of GeV. Load the sub-machine''s model before exporting.'], ...
        family, field, mat2str(energy));
end
end


function candidate = local_energy_candidate(AO, family)
% Whether the ring's energy is read from this family.
%
% Two spellings say a family is a main bend, and a facility uses one or the
% other: the lattice type the Middle Layer's own write path dispatches on, and
% the family's membership list. NSLS-II's BEND family carries the lattice type
% of a sextupole and is a bend by its membership alone, so the type on its own
% would miss the one family whose current sets that ring's energy.
%
% A family that is also a corrector trims the bend rather than setting the
% energy - SPEAR3 lists a bend trim and a corrector dipole among its
% correctors, both with the bend lattice type - and the energy is never read
% from a trim.
memberOf = local_members(AO, family);
candidate = (strcmpi(local_text(local_subfield(AO, family, 'AT', 'ATType')), 'BEND') ...
    || local_member(memberOf, 'BEND')) ...
    && ~local_member(memberOf, 'COR');
end


function [energyTable, refused] = local_energy_table(family, DeviceList, sampled, nominals, energy)
% What the facility's own conversion says the ring's energy is over this
% family's hardware grid, and where the deck sits on it.
%
% The table is sampled on one device row, the row the conversion converts by
% default - the family's first - because a facility's bend2gev answers for one
% device at a time and branches on its sector: it reads the ramp coefficients
% of the device it is handed, and two sectors of the same family can carry
% different ones. The row is written beside the table so a consumer knows
% whose ramp it holds.
%
% One grid point per call. Handed a column of currents with one device row,
% the conversion loops over the row rather than the currents and answers for
% the first point alone, which would be written as a table of one number
% repeated and read as a ring whose energy does not move.
%
% The table is written exactly as it was sampled. A conversion that answers
% with the same energy at every current - a facility whose bend2gev hands back
% the deck energy - is a fact about that facility, and what a ring with no
% energy knob means is the consumer's call, not this script's.
%
% A conversion built on a measured ramp ends where its ramp does, and beyond
% it the table carries entries that are not numbers. They are kept where they
% are, and the span the table does answer over is recorded beside them, the
% same marker a calibration table carries.
%
% Every step is inside one try: the conversion is the facility's own code, and
% a family whose table sampled but whose nominal current refused keeps the
% table beside its refusal.
field = 'Setpoint';
energyTable = struct();
refused = '';

try
    local_require_energy(family, field, energy);
    if isempty(DeviceList) || ~isfield(sampled, field)
        error('mml_export:energy_table', ...
            '%s: the energy table is sampled over the %s grid, which this family has none of.', ...
            family, field);
    end

    row = DeviceList(1, :);
    grid = sampled.(field).grid(1, :);
    values = zeros(1, numel(grid));
    for k = 1:numel(grid)
        answer = bend2gev(family, field, grid(k), row, 'Hardware');
        values(k) = local_column(family, field, 'bend2gev', answer, 1);
    end
    if ~any(isfinite(values))
        error('mml_export:energy_table', ...
            '%s: bend2gev answered with no number over the grid %g to %g.', ...
            family, min(grid), max(grid));
    end

    energyTable.device_row = row;
    energyTable.grid = grid;
    energyTable.values = values;
    energyTable.finite_span = local_finite_span(grid, values);

    I_nom = local_nominal_current(family, field, DeviceList, nominals, energy);
    energyTable.I_nom = I_nom;
    answer = bend2gev(family, field, I_nom, row, 'Hardware');
    energyAtNominal = local_column(family, field, 'bend2gev', answer, 1);

    % A finite nominal current the conversion answers no number for means the
    % facility's ramp does not cover the setting the deck sits at, which is the
    % one fact a consumer deciding whether this family carries the ring's
    % energy most needs stated. It is refused by name rather than written as a
    % number that is not one, beside the table and the current it was asked at,
    % and the refusal comes before the answer reaches the table: a scalar
    % spelled "NaN" in the one slot every consumer of this table reads a number
    % from reads like a number of this machine.
    if ~isfinite(energyAtNominal)
        error('mml_export:energy_table', ...
            '%s.%s: bend2gev is %s at the nominal current %g.', ...
            family, field, mat2str(energyAtNominal), I_nom);
    end
    energyTable.energy_at_nominal = energyAtNominal;
catch err
    refused = err.message;
end
end


function current = local_nominal_current(family, field, DeviceList, nominals, energy)
% The current the deck's energy sits at, on the device row the table was
% sampled over.
%
% The facility's own inverse is asked first: gev2bend turns the deck energy
% into the current that produces it, which is the setting the energy knob
% moves away from. Not every facility ships one - NSLS-II's storage ring has a
% bend-to-energy conversion and no inverse - and then the nominal this export
% already read from the Middle Layer stands in, taken from the same seam every
% other grid is sized by rather than read again here.
%
% The units are spelled out because a facility's copy of either conversion
% defaults them from the family's own units and would then convert its answer
% into physics units, where the grid and every nominal in this export are
% hardware.
row = DeviceList(1, :);
if exist('gev2bend', 'file') == 0
    nominal = local_nominal_of(nominals, field);
    if numel(nominal) ~= size(DeviceList, 1)
        nominal = NaN;
    end
    current = nominal(1);
    origin = 'the nominal the Middle Layer gave';
else
    answer = gev2bend(family, field, energy, row, 'Hardware');
    current = local_column(family, field, 'gev2bend', answer, 1);
    origin = 'gev2bend';
end

if ~isfinite(current)
    error('mml_export:energy_table', ...
        '%s.%s: %s is %s for the deck energy of %g GeV, so the table has no nominal.', ...
        family, field, origin, mat2str(current), energy);
end
end


function [recorded, nominals, refused] = local_nominals(family, AO, DeviceList)
% What the Middle Layer says one family is set to, per device, in hardware
% units.
%
% The nominal is the Middle Layer's own model read, the same call its
% channel-name path makes, so a facility's conversion, gains and rolls are
% already inside the number and nothing here re-derives them. It is read
% after the lattice is saved, because the read mutates the ring to reach a
% solvable state.
%
% RECORDED is what the family's block carries: the numbers, the units the read
% says they are in, the AT block it went through and whether it read the model
% at all. NOMINALS is the same numbers as one column per field, the shape
% every grid in this export is sized by, and it carries a field only when the
% read answered in hardware units: a hardware grid laid around a physics
% number is a grid the facility's conversion was never asked over, and the
% export sizes that field by its own Range instead.
%
% Every step is inside one try: a read the facility's own code refuses costs
% that family its nominal and nothing else, and the message stands where the
% number would have been.
recorded = struct();
nominals = struct();
refused = '';

field = local_nominal_field(AO, family);
if isempty(field)
    return
end

try
    nDev = size(DeviceList, 1);
    if nDev == 0
        error('mml_export:devices', ...
            'Family %s lists no devices to read a nominal for.', family);
    end

    at = local_at_of(AO, family, field);
    [values, units] = local_sample_nominal(family, field, DeviceList, nDev);

    record = struct();
    record.values = values;
    record.units = units;
    record.at_type = local_text(local_field(at, 'ATType'));
    record.at_index = local_field(at, 'ATIndex');
    record.synthetic = local_synthetic(family, at, values);
    recorded.(field) = record;

    if ~isempty(units) && ~strcmpi(units, 'Hardware')
        error('mml_export:nominals', ...
            ['%s.%s: getpvmodel answered the nominal in %s units, not the ' ...
             'hardware units it was asked in.'], family, field, units);
    end
    nominals.(field) = values;
catch err
    refused = err.message;
end
end


function field = local_nominal_field(AO, family)
% The one field a family's nominal is read from: its Setpoint, and its Monitor
% when it has no Setpoint.
%
% A setpoint is the setting itself, whatever else the family reads back. A
% family that only reads - a beam monitor, a photon monitor - has no setting
% other than its reading, and that reading is its nominal. A family with
% neither carries no nominal at all.
names = {'Setpoint', 'Monitor'};
field = '';
for k = 1:numel(names)
    if isfield(AO.(family), names{k}) && isstruct(AO.(family).(names{k}))
        field = names{k};
        return
    end
end
end


function [values, units] = local_sample_nominal(family, field, DeviceList, nDev)
% One field's nominal as the Middle Layer's model read answers it, and the
% units the read says the answer is in.
%
% The read is asked for hardware units, because a nominal in physics units
% sizes no hardware grid, and for the struct output, because the numbers alone
% do not say which units they came back in: the read converts into hardware
% units for a family whose conversion it knows and hands back physics units
% otherwise, and the Units field of that struct is where it says which it did.
%
% The struct is taken where it arrives. The Middle Layer's online read builds
% the same struct with struct('Data', ...), the form MATLAB accepts, while its
% model read assigns the field onto the numeric answer instead, which MATLAB
% refuses, and no caller inside the Middle Layer asks it for one. So the read
% is asked again for the numbers alone, and then there is no statement to
% record: the units it was asked in are the only ones there are.
%
% The second read masks nothing. Both forms are the same read to the same
% point - only the tail that shapes the output differs - so a model the read
% refuses refuses it again, and that message is the one the family records as
% its refusal.
%
% The whole device list goes in one call - the model read is written for a
% list of devices, unlike the per-device energy conversions - and no fourth
% number is passed: the read converts at the model's own energy, and a fourth
% argument is read as a time.
try
    answer = getpvmodel(family, field, DeviceList, 'Hardware', 'Struct');
catch
    answer = getpvmodel(family, field, DeviceList, 'Hardware', 'Numeric');
end

units = '';
if isstruct(answer)
    units = local_text(local_field(answer, 'Units'));
    answer = local_field(answer, 'Data');
end
values = local_column(family, field, 'getpvmodel', answer, nDev);
end


function at = local_at_of(AO, family, field)
% The AT block the model read goes through.
%
% The read takes the field's own AT block where it has one and the family's
% otherwise. The family's block is the one every other Middle Layer function
% reads; a field-level block is an override honoured inside the simulator
% alone, so the read resolves it in that order and so does this.
%
% Its index is recorded as it stands. A device the Middle Layer indexes with
% fewer elements than its siblings is padded with entries that are not
% numbers, and they are written in the spelling every other non-finite value
% of the export is written in rather than dropped: which device the padding
% belongs to is what tells a consumer how many elements each device has.
at = local_subfield(AO, family, field, 'AT');
if isempty(at)
    at = local_field(AO.(family), 'AT');
end
end


function synthetic = local_synthetic(family, at, values)
% Whether the model read answers this family with a number it makes up rather
% than one it reads off the ring.
%
% Five answers of the read are not measurements of the model: a family with no
% AT block and no element of its own name is answered from a stored field or
% from zeros, with no contact with the ring at all; the two do-nothing lattice
% types answer 0 and NaN; the photon-monitor stub answers ones; the
% beam-current family answers a fixed milliamp figure decaying on the wall
% clock; and a lattice type no branch of the read knows - a facility's own
% spelling, or the field name the read substitutes for a family with no AT
% block - answers a column of values that are not numbers. The first four are
% the read's own dispatch names, the same at every facility, not any one
% facility's families; the fifth is read off the answer, because the spellings
% that reach it are the facility's own.
%
% The number is recorded either way - it is what the Middle Layer would hand
% any consumer asking for this family - and the flag is what says not to read
% it as a setting of this machine.
global THERING
synthetic = strcmpi(family, 'DCCT') ...
    || any(strcmpi(local_text(local_field(at, 'ATType')), {'Septum', 'null', 'Photon BPM'})) ...
    || (isempty(at) && isempty(findcells(THERING, 'FamName', family))) ...
    || all(~isfinite(values));
end


function local_require_finite(family, field, AO, fcnField, grid, values)
% Refuse a field whose conversion answered with nothing a consumer can use.
%
% A device row with fewer than two finite samples is not a calibration: there
% is no line to take a gain from, and written out it becomes a table of "NaN"
% strings that reads like data all the way to the first value a model is asked
% for. A conversion built on a measured table answers outside that table with
% a value that is not a number, so a grid that misses the table - a unipolar
% magnet sampled about zero, a Range that does not hold the anchor - lands
% here rather than in the export, named with the conversion that answered and
% the span it was asked over, and keeps whatever was sampled before it.
%
% A row that is finite over part of its grid is a conversion with an end: it
% is kept, and local_finite_span records where its numbers stop.
bad = find(sum(isfinite(values), 2) < 2);
if ~isempty(bad)
    error('mml_export:samples', ...
        '%s.%s: %s answered with nothing usable over the grid %g to %g for device row(s) %s.', ...
        family, field, local_fcn_name(local_subfield(AO, family, field, fcnField)), ...
        min(min(grid(bad, :))), max(max(grid(bad, :))), mat2str(bad(:)'));
end
end


function calibration = local_calibration(grid, values)
% A sampled conversion as the least the consumer needs to reproduce it.
%
% The line is taken through the two outermost samples of each device, the
% widest lever arm the grid offers, and then held against every sample on
% that device's row. One device off the line makes the whole field a table:
% the consumer converts a family in one shape, not device by device.
%
% A table carries the span each of its rows actually answers over, because a
% sampled conversion may end before its grid does; a line carries none, since
% a sample that is not a number is never on one.
points = size(grid, 2);
gain = (values(:, points) - values(:, 1)) ./ (grid(:, points) - grid(:, 1));
offset = values(:, 1) - gain .* grid(:, 1);

scale = max(abs(values), [], 2);
scale(~isfinite(scale) | scale == 0) = 1;
residual = abs(values - (gain .* grid + offset));

if all(isfinite(gain)) && all(isfinite(offset)) ...
        && all(all(residual <= local_linear_tolerance() * scale))
    calibration = struct('kind', 'linear', 'gain', gain, 'offset', offset);
else
    calibration = struct('kind', 'table', 'grid', grid, 'values', values, ...
        'finite_span', local_finite_span(grid, values));
end
end


function [grid, source, anchor] = local_hardware_grid(range, nominal, nDev)
% The hardware values a field is sampled at, one row per device.
%
% A device is sampled over its own Range, stretched just far enough to hold
% its anchor where the band the facility states does not reach it. A
% conversion is written for the span the facility runs the device over and
% turns over outside it, so the narrowest grid that still holds the anchor is
% the one that stays a conversion the whole way across. A device with no band
% at all has nothing to stretch and spans +-max(2|anchor|, 1) about zero,
% which holds the anchor and still leaves a lever arm when the anchor is zero.
%
% The word names the weakest grid the field used, the way the anchor word
% names the weakest anchor: a field is on its Range when every device had a
% band to lay a grid over, whether or not a band had to be stretched, and
% falls back as soon as one device had none. A table states the rows it was
% sampled at, and the grid behind a line is the Range and the nominals the
% export states beside it.
points = local_grid_points();
[nominal, anchor] = local_anchor(nominal, range, nDev);

banded = local_finite_band(range, nDev);
low = -max(2 * abs(nominal), 1);
high = -low;
if any(banded)
    % Only a device with a band has one to stretch, and MATLAB checks the
    % column subscript of an empty range even when no row is selected.
    low(banded) = min(range(banded, 1), nominal(banded));
    high(banded) = max(range(banded, 2), nominal(banded));
end

if all(banded)
    source = 'range';
else
    source = 'fallback';
end

grid = low + (high - low) .* (0:(points - 1)) / (points - 1);
end


function grid = local_monitor_grid(nDev)
% The physics grid a monitor-only family is sampled over when its Monitor
% carries no Range: 10 mm of beam position either side of the axis, in
% metres. Wider than any orbit the model holds, so the consumer reads the
% inverse by interpolation rather than beyond its ends.
span = 0.010;
grid = ones(nDev, 1) * linspace(-span, span, local_grid_points());
end


function n = local_grid_points()
% How many points every calibration is sampled at. Odd, so the middle of a
% symmetric grid is a sample rather than a gap between two.
n = 33;
end


function tol = local_linear_tolerance()
% How far a sample may sit off the line through two of its own before the
% conversion is written as a table, relative to the largest value that device
% was sampled at. A linear conversion reproduces its line to machine
% precision; a polynomial or a measured curve misses it by orders of
% magnitude more than this.
tol = 1e-9;
end


function range = local_range(AO, family, field, nDev)
% A field's Range as one row per device. The Middle Layer stores either one
% band for the whole family or one band per device, in the DeviceList's
% order; anything else is no band at all.
range = local_subfield(AO, family, field, 'Range');
if isempty(range) || ~isnumeric(range) || size(range, 2) ~= 2
    range = [];
elseif size(range, 1) == 1
    range = ones(nDev, 1) * range;
elseif size(range, 1) ~= nDev
    range = [];
end
end


function [anchored, anchor] = local_anchor(nominal, range, nDev)
% The nominal a grid is sized by, as one finite number per device, and the
% word for where that number came from.
%
% A nominal the Middle Layer could not give a number for is not spread over
% the row as a NaN, and not passed off as a setting either: a device with a
% finite band is sized by the middle of its band, which keeps the samples
% inside the span the facility's conversion covers, and a device with no band
% falls back to zero. The word is the weakest of the three the field used, so
% a consumer can tell a grid built on a setting from one built around a magnet
% whose setting was never read.
anchored = nominal(:);
if numel(anchored) ~= nDev
    anchored = NaN(nDev, 1);
end
anchor = 'nominal';

missing = ~isfinite(anchored);
if ~any(missing)
    return
end

usable = missing & local_finite_band(range, nDev);
% A field with no band hands over an empty range, and MATLAB checks the column
% subscript of an empty array even when no row is selected.
if any(usable)
    anchored(usable) = (range(usable, 1) + range(usable, 2)) / 2;
end
anchored(missing & ~usable) = 0;
if any(missing & ~usable)
    anchor = 'zero';
else
    anchor = 'range_midpoint';
end
end


function span = local_finite_span(grid, values)
% The hardware span each device's table actually answers over: the first and
% last grid point that converted to a number.
%
% A sampled conversion built on a measured table has an end, and beyond it the
% table is written with those entries spelled out rather than dropped. The
% span is what tells a consumer which part of the row is a conversion; an
% entry that is not a number inside the span is a gap in the facility's own
% table and stays spelled where it is.
% A row with no finite sample is not a table at all, and it is refused by the
% same name a row of unusable samples is refused by: the span of a row that
% converted nowhere would otherwise be read off an empty index list, and the
% family would record MATLAB's index message instead of what went wrong.
span = zeros(size(grid, 1), 2);
for r = 1:size(grid, 1)
    finite = find(isfinite(values(r, :)));
    if isempty(finite)
        error('mml_export:samples', ...
            'Device row %d converted to no number anywhere on its grid.', r);
    end
    span(r, :) = [grid(r, finite(1)) grid(r, finite(end))];
end
end


function band = local_finite_band(range, nDev)
% Which devices carry a band a grid could be laid over: two finite ends the
% right way round, one row per device.
band = false(nDev, 1);
if ~isempty(range) && size(range, 1) == nDev
    band = all(isfinite(range), 2) & range(:, 1) < range(:, 2);
end
end


function nominal = local_nominal_of(nominals, field)
% The hardware nominals of one field, or nothing when there are none.
nominal = [];
if isstruct(nominals) && isfield(nominals, field)
    nominal = nominals.(field);
end
end


function DeviceList = local_device_list(AO, family)
% The devices a family's values are indexed by, in the order every per-device
% column of the export is written in.
DeviceList = [];
if isfield(AO.(family), 'DeviceList')
    DeviceList = AO.(family).DeviceList;
end
end


function members = local_members(AO, family)
% The groups a family says it belongs to, as one cell of names. The Middle
% Layer stores either one name or a cell of them, and a family may carry none.
members = {};
if isfield(AO, family) && isfield(AO.(family), 'MemberOf')
    members = AO.(family).MemberOf;
end
if ischar(members) || isa(members, 'string')
    members = {local_text(members)};
elseif ~iscell(members)
    members = {};
end
end


function found = local_member(members, name)
% Whether a membership list names one group, in whatever case it spells it.
found = false;
for k = 1:numel(members)
    if strcmpi(local_text(members{k}), name)
        found = true;
        return
    end
end
end


function value = local_subfield(AO, family, field, name)
% One key of one field of one family, or nothing where the family does not
% carry it.
%
% The field has to be one struct: a struct array there answers with one value
% per element, which is no value to hand back, and a facility that holds a
% field that way states nothing this export can read.
value = [];
if isfield(AO, family) && isfield(AO.(family), field) && isstruct(AO.(family).(field)) ...
        && isscalar(AO.(family).(field)) && isfield(AO.(family).(field), name)
    value = AO.(family).(field).(name);
end
end


function name = local_fcn_name(value)
% A conversion function by its name. The name is what a consumer recognises
% the conversion by; the handle's file path names a folder on the machine the
% export ran on and means nothing anywhere else.
if isa(value, 'function_handle')
    name = func2str(value);
elseif isa(value, 'string')
    name = char(value);
elseif ischar(value)
    name = strtrim(value);
else
    name = '';
end
end


function response = local_response(AD)
% The facility's orbit response matrix in physics units with no energy
% scaling, together with the operating point it was taken at.
%
% The matrix is the Middle Layer's own. The read resolves the file the
% Accelerator Data names, takes the four orbit families the facility itself
% names as its monitors and its correctors, and hands back one block per
% monitor and corrector pair; nothing here re-indexes it or names a family.
%
% A facility that names a file is the case this is worth the most in: a file
% is where a measurement of the machine is kept, and a measured matrix is what
% the machine did rather than what its conversions say it would do. A facility
% that names none has the model measured instead, and a model matrix says
% nothing about any machine. Which of the two a block holds is not the same
% question as whether a file answered, because a facility may keep a matrix it
% computed from a model in a file like any other; the mode each side was read
% in settles it, and that is what each block's origin states.
%
% Physics units, because a matrix in hardware units is a matrix per
% conversion and the conversions are written separately. No energy scaling,
% because the energy the matrix was measured at is recorded beside it: a
% matrix scaled to the deck's energy on the way out no longer belongs to the
% operating point its own correctors sat at, which is the point a consumer
% converts it about.
%
% FILE is the file the read answered from, relative to the Middle Layer root,
% and is empty when no file answered and the model was measured instead. It is
% the only record of which of those two happened. A block's origin answers the
% other question, whether the numbers were measured on a machine or computed
% from a model, and a file can hold either.
if isempty(local_response_files(AD))
    [S, file] = local_response_of_model();
else
    [S, file] = getbpmresp('Struct', 'NoEnergyScaling', 'Physics');
end
if isempty(S) || ~isstruct(S)
    error('mml_export:response', ...
        'The read answered with no response matrix for this sub-machine''s orbit families.');
end

response = struct();
response.file = local_mml_path(local_text(file));
response.blocks = local_response_blocks(S);
end


function [S, file] = local_response_of_model()
% The orbit response matrix measured on the deck, for a sub-machine whose
% Accelerator Data names no response matrix file at all.
%
% The Middle Layer resolves the file through AD.OpsData.RespFiles, and that is
% the one answer it treats as a question for the operator: with no name to
% resolve it opens a file-chooser dialog. An export runs unattended, often
% with no display to draw a window on, so the read is never asked in that
% state. What it does instead is what every other way of having no file
% already ends in - a name that resolves to nothing, a file holding no matrix
% for these families - which is the Middle Layer's own measurement of the
% model, taken with no dialog and no display.
%
% Blanks are not names and drop out before the question is asked, so a list
% holding nothing else is measured here as well. A name the facility does have
% is left to the read, whether or not anything is there.
%
% Energy scaling is not asked for because there is none to refuse: the model
% measurement builds its matrix in physics units directly, and the word the
% file read takes for it is not one this measurement knows - passed on it
% would be read as the name of a family. Archiving and display are off: an
% export writes its own files and says so itself.
%
% FILE is empty, the way the read leaves it when no file answered. Each block
% records the mode of the two sides it was measured between, and both are the
% simulator's, so every block's own origin says the matrix is the model's.
S = measbpmresp('Model', 'Struct', 'Physics', 'NoArchive', 'NoDisplay');
file = '';
end


function files = local_response_files(AD)
% The response matrix files the Accelerator Data names, as one cell of names.
%
% The Middle Layer stores either one name or a cell of them, and a blank name
% is no name at all: it matches no file, and its own resolution merely steps
% over it. The answer that resolution reads as the request for a dialog box is
% the bare empty value - '' or [] rather than a cell holding a blank - which
% is what an Accelerator Data with no such field answers with.
files = {};
ops = local_field(AD, 'OpsData');
if ~isstruct(ops)
    return
end

names = local_field(ops, 'RespFiles');
if ischar(names) || isa(names, 'string')
    names = {char(names)};
end
if ~iscell(names)
    return
end

for k = 1:numel(names)
    name = local_text(names{k});
    if ~isempty(name)
        files{end+1} = name; %#ok<AGROW>
    end
end
end


function blocks = local_response_blocks(S)
% One block per monitor and corrector pair the response matrix holds, in the
% order the read lays them out.
%
% A block is named by the two families it was measured between rather than by
% its place in that grid. The grid is one row per monitor plane and one column
% per corrector plane, but which plane a row is belongs to the facility's own
% list of orbit families, and a facility's own copy of the read may list them
% in another order. The families are what the rest of this export describes
% and what a consumer aligns the matrix by.
blocks = cell(1, numel(S));
n = 0;
for m = 1:size(S, 1)
    for a = 1:size(S, 2)
        n = n + 1;
        blocks{n} = local_response_block(S(m, a));
    end
end
end


function block = local_response_block(s)
% One block of the response matrix: whose numbers they are, what they say,
% and the operating point they say it at.
%
% Every measured number is written with six significant digits, and the units
% the matrix is in are written beside it: the read converts a stored matrix
% into the units it was asked for, so what the file held and what is written
% here need not be the same units.
block = struct();
block.monitor = local_response_side(s, 'Monitor');
block.actuator = local_response_side(s, 'Actuator');
block.origin = local_response_origin(block.monitor.mode, block.actuator.mode);
block.timestamp = local_response_timestamp(local_field(s, 'TimeStamp'));
block.gev = local_response_rounded(local_response_number(s, 'GeV'));
block.units = local_text(local_field(s, 'Units'));
block.units_string = local_text(local_field(s, 'UnitsString'));
block.modulation_method = local_text(local_field(s, 'ModulationMethod'));
block.actuator_delta = local_response_rounded(local_response_number(s, 'ActuatorDelta'));
block.data = local_response_rounded(local_response_matrix(s, ...
    size(block.monitor.device_list, 1), size(block.actuator.device_list, 1)));
end


function side = local_response_side(s, name)
% One side of a block: the family the matrix was measured on, the devices its
% rows or its columns are in the order of, the mode that side was read in,
% which of its devices the measurement counts as good, and what that side was
% at while the matrix was taken.
%
% The last of those is what the matrix is a secant about - the setting each
% corrector sat at, the orbit each monitor read - so a consumer converting the
% matrix into other units converts it about the point it was measured at
% rather than about zero.
%
% A side with no device list is refused: the list is what the matrix is
% aligned by, and rows belonging to no device are rows a consumer can only
% read against the wrong ones. The operating point is held to that same list,
% as the matrix and the flags are: a column of another length is read against
% the wrong devices, and the whole point of recording it is that a consumer
% converts each device's row about that device's own setting.
body = local_field(s, name);
if ~isstruct(body)
    error('mml_export:response', 'The response matrix carries no %s of its own.', name);
end

devices = local_field(body, 'DeviceList');
if isempty(devices) || ~isnumeric(devices)
    error('mml_export:response', ...
        'The response matrix %s carries no device list to align it by.', name);
end

nDev = size(devices, 1);
side = struct();
side.family = local_text(local_field(body, 'FamilyName'));
side.device_list = devices;
side.mode = local_text(local_field(body, 'Mode'));
side.status = local_response_status(body, name, nDev);
side.data = local_response_rounded(local_response_point(body, name, nDev));
end


function point = local_response_point(body, name, nDev)
% What one side was at while the matrix was taken, one value per device.
%
% A side that carries no operating point at all answers with the one value
% that is not a number, the way every other absent number of this block does;
% a point of another length is refused, because read device by device it would
% convert each row about another device's setting.
point = local_response_number(body, 'Data');
if isscalar(point) && isnan(point)
    return
end
if numel(point) ~= nDev
    error('mml_export:response', ...
        'The response matrix %s carries %d operating-point values for %d devices.', ...
        name, numel(point), nDev);
end
end


function status = local_response_status(body, name, nDev)
% Which devices of one side the measurement counts as good, one flag per
% device.
%
% A side that carries no status at all is every device good. The Middle Layer
% keeps the flag per family and fills the flag of a device its file does not
% hold with a zero, so a side with no flags is a measurement written before
% the flag was kept rather than one saying no device is good, and a consumer
% filtering on it would otherwise drop the whole matrix. One flag for the
% whole side is spread over its devices; the Middle Layer's own read leaves
% such a flag as it stands, so the spread is this export's rule and not a
% reading of its behaviour.
%
% A flag kept as true and false is a flag, and is taken as one: it is written
% out as 0 and 1 like every other logical of this export. Read as no flags at
% all it would turn a side's bad devices into good ones, which is the wrong
% direction for a value a consumer drops devices by.
%
% A list of another length is refused: read device by device it would name the
% wrong devices.
status = local_field(body, 'Status');
if isempty(status) || ~(isnumeric(status) || islogical(status))
    status = ones(nDev, 1);
elseif isscalar(status)
    status = status * ones(nDev, 1);
end

status = double(status(:));
if numel(status) ~= nDev
    error('mml_export:response', ...
        'The response matrix %s carries %d status flags for %d devices.', ...
        name, numel(status), nDev);
end
end


function data = local_response_matrix(s, nMonitor, nActuator)
% The matrix itself, one row per monitor and one column per corrector.
%
% Its shape is held against the two device lists rather than taken on trust.
% The read fills a device its file does not hold with a row or a column of
% values that are not numbers and then indexes the matrix down to the devices
% it was asked for; a matrix that no longer lines up with the lists written
% beside it is read against the wrong devices, and every number of the
% comparison it exists for comes out wrong by a device.
data = local_field(s, 'Data');
if ~isnumeric(data) || size(data, 1) ~= nMonitor || size(data, 2) ~= nActuator
    error('mml_export:response', ...
        'The response matrix is %s for %d monitors and %d correctors.', ...
        mat2str(size(data)), nMonitor, nActuator);
end
end


function origin = local_response_origin(monitorMode, actuatorMode)
% Whether the matrix is a measurement of the machine or of its model.
%
% Each side records the mode the Middle Layer read it in, and the two words
% the simulator answers in are the only ones that say a number never came off
% the machine. Every other mode reads as measured, an absent mode included:
% the mode the Middle Layer gives a field it holds none for is online. Both
% words are recorded beside this one, so a consumer is never left with the
% summary alone.
%
% It is the fact a comparison rests on: a model matrix compared against a
% model compares the model with itself.
origin = 'measured';
if any(strcmpi(monitorMode, {'Simulator', 'Model'})) ...
        || any(strcmpi(actuatorMode, {'Simulator', 'Model'}))
    origin = 'model';
end
end


function text = local_response_timestamp(value)
% When the matrix was measured, in the spelling this export writes its own
% timestamp in.
%
% The Middle Layer stores it as the clock vector of the measurement or as a
% serial date number, and both are what datenum reads. A stamp that is
% neither is written as no stamp rather than costing the matrix its block: the
% stamp says how old the measurement is, and the measurement is what the file
% is for.
text = '';
if isempty(value)
    return
end

try
    text = datestr(datenum(value), 'yyyy-mm-ddTHH:MM:SS');
catch
    text = '';
end
end


function value = local_response_number(s, name)
% One numeric fact of the measurement, or a value that is not a number when
% the file carries none.
%
% Nothing here is refused for being non-finite. The read fills the delta and
% the operating point of a device its file does not hold with values that are
% not numbers, and they are the file's own answer for that device: they are
% written in the spelling every other non-finite value of this export is
% written in, beside the status flag that says the device is not to be used.
value = local_field(s, name);
if ~isnumeric(value) || isempty(value)
    value = NaN;
end
end


function out = local_response_rounded(x)
% One measured value with the digits a measurement carries.
%
% The last digits of a response matrix are noise, and a facility's file holds
% it at the full width of a double: megabytes of text per export that a
% consumer compares nothing against. Device lists and status flags are not
% rounded - they are identities, not measurements.
%
% The value is taken as a double first: a file that stored a measurement in
% another numeric class is rounded rather than refused, and rounding to a
% number of digits is a double's operation in MATLAB.
out = round(double(x), local_response_digits(), 'significant');
end


function n = local_response_digits()
% How many significant digits of the response matrix are written. Wider than
% any comparison made against it - a consumer reads agreement with a model in
% per cent - and narrow enough that the file a facility ships stays a file a
% repository can hold.
n = 6;
end


function text = local_document(export, body)
% One JSON object: the "_export" block first, then the body's own keys.
% "_export" is not a legal MATLAB field name, so the block is spliced in as text.
%
% The splice takes the body's own opening brace off, so a body that encoded as
% anything but an object would be joined into text that is not JSON. It
% is refused by name here rather than written out.
head = ['{"_export":' local_encode(export)];
encoded = local_encode(body);
if isempty(encoded) || encoded(1) ~= '{'
    error('mml_export:document', ...
        'The body of a JSON file encoded as %s rather than as an object.', ...
        mat2str(encoded(1:min(numel(encoded), 20))));
end
if strcmp(encoded, '{}')
    text = [head '}'];
else
    text = [head ',' encoded(2:end)];
end
end


function text = local_encode(value)
text = jsonencode(value, 'ConvertInfAndNaN', false);
end


function out = local_normalize(value)
% Rewrite one MATLAB value into the spelling the importer expects.
if isa(value, 'function_handle')
    out = local_handle(value);
elseif isstruct(value)
    if isscalar(value)
        out = local_struct(value);
    else
        out = local_rows(value);
    end
elseif isa(value, 'string')
    out = local_normalize(char(value));
elseif ischar(value)
    out = local_char(value);
elseif iscell(value)
    out = local_rows(value);
elseif islogical(value)
    out = local_normalize(double(value));
elseif isnumeric(value)
    out = local_numeric(value);
else
    warning('mml_export:type', 'Writing a value of class %s as its class name.', class(value));
    out = ['<' class(value) '>'];
end
end


function out = local_struct(s)
out = struct();
names = fieldnames(s);
for k = 1:numel(names)
    name = names{k};
    if strcmp(name, 'Handles')
        continue
    end
    out.(name) = local_normalize(s.(name));
end
end


function out = local_handle(h)
info = functions(h);
file = '';
if isfield(info, 'file')
    file = info.file;
end
out = containers.Map({'$fn', 'file'}, {func2str(h), local_mml_path(local_text(file))});
end


function out = local_char(c)
if size(c, 1) <= 1
    out = local_mml_path(deblank(c));
    return
end
out = cell(1, size(c, 1));
for r = 1:size(c, 1)
    out{r} = local_mml_path(deblank(c(r, :)));
end
end


function out = local_numeric(x)
if issparse(x)
    x = full(x);
end
if ~isreal(x)
    warning('mml_export:complex', 'Writing a complex value as text.');
    out = mat2str(x);
    return
end
if ndims(x) > 2
    x = reshape(x, size(x, 1), []);
end
if all(isfinite(x(:)))
    out = x;
    return
end
if isscalar(x)
    out = local_nonfinite(x);
elseif size(x, 1) == 1 || size(x, 2) == 1
    out = local_row(x(:)');
else
    out = cell(1, size(x, 1));
    for r = 1:size(x, 1)
        out{r} = local_row(x(r, :));
    end
end
end


function out = local_row(x)
% A numeric vector as a cell, non-finite entries spelled as strings.
out = num2cell(double(x));
for k = 1:numel(x)
    if ~isfinite(x(k))
        out{k} = local_nonfinite(x(k));
    end
end
end


function out = local_nonfinite(x)
if isnan(x)
    out = 'NaN';
elseif x > 0
    out = 'Inf';
else
    out = '-Inf';
end
end


function out = local_rows(value)
% A cell or struct array, element by element, keeping 1-row vs N-row shape.
if isempty(value)
    out = {};
    return
end
if ndims(value) > 2
    value = reshape(value, size(value, 1), []);
end
if size(value, 1) == 1 || size(value, 2) == 1
    out = local_elements(value(:)');
    return
end
out = cell(1, size(value, 1));
for r = 1:size(value, 1)
    out{r} = local_elements(value(r, :));
end
end


function out = local_elements(value)
out = cell(1, numel(value));
for k = 1:numel(value)
    if iscell(value)
        out{k} = local_normalize(value{k});
    else
        out{k} = local_normalize(value(k));
    end
end
end


function value = local_field(s, name)
if isfield(s, name)
    value = s.(name);
else
    value = '';
end
end


function text = local_text(value)
% One string as text: the first row of a char matrix, trimmed.
%
% A value carrying no text is no text rather than an error. An absent field
% arrives here as the empty string, whose first row is an index past the end,
% and every label this export reads - the units a matrix is in, the mode a
% side was read in, the lattice type of an AT block - is a field a facility's
% own data need not carry. A missing word is never a reason to refuse the
% fact beside it.
text = '';
if isa(value, 'string')
    value = char(value);
end
if ischar(value) && ~isempty(value)
    text = strtrim(value(1, :));
end
end


function out = local_mml_path(text)
% One path, relative to the Middle Layer root.
%
% An absolute path is the exporting machine's own: an account name, a home
% directory and a checkout nobody else has. What survives the trip is the
% part under the Middle Layer root - machine/Spear3/StorageRing/amp2k.m
% names the same file wherever that tree is unpacked. Text that does not
% begin at the root is not a path into it and stands as it is.
out = text;
root = local_mml_root();
if isempty(root) || isempty(out) || ~ischar(out)
    return
end
if strncmp(out, root, numel(root))
    out = out(numel(root) + 1:end);
end
end


function root = local_mml_root()
% The Middle Layer root, ending in a separator, read once per session.
%
% The read costs a getfamilydata lookup and every text value in the export
% asks for it, so the answer is held. One session exports from one install.
persistent cached
if isempty(cached)
    cached = '';
    if exist('getmmlroot', 'file') ~= 0
        cached = getmmlroot;
    end
    if ~isempty(cached) && cached(end) ~= filesep
        cached = [cached, filesep];
    end
end
root = cached;
end


function name = local_filename(text)
name = lower(regexprep(text, '[^A-Za-z0-9_-]+', '_'));
end


function local_write(path, text)
fid = fopen(path, 'w', 'n', 'UTF-8');
if fid < 0
    error('mml_export:write', 'Cannot write %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', text);
end
