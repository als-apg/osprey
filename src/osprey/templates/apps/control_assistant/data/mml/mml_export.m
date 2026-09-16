function files = mml_export(outdir)
%MML_EXPORT Write the current sub-machine's AO and AD as JSON for OSPREY.
%
%   MML_EXPORT writes two files into the current folder for the sub-machine
%   the Middle Layer is set up for:
%
%       <machine>.<submachine>.ao.json   the Accelerator Objects (getao)
%       <machine>.<submachine>.ad.json   the Accelerator Data    (getad)
%
%   MML_EXPORT(OUTDIR) writes them into OUTDIR instead.
%
%   FILES = MML_EXPORT(...) returns the two paths, AO first.
%
%   Run it once per sub-machine, after the MML setpath for that sub-machine,
%   then import the AO file with
%
%       osprey mml import <machine>.<submachine>.ao.json
%
%   The importer reads the AD file beside it on its own.
%
%   What is written
%   ---------------
%   Both files start with an "_export" block naming the exporter version,
%   the MATLAB version, the machine, the sub-machine and a timestamp. Every
%   value is rewritten into the spelling OSPREY's importer expects:
%
%     * function handles become {"$fn": func2str(h), "file": <path or "">};
%     * char matrices become one deblanked string per row;
%     * Inf, -Inf and NaN become the strings "Inf", "-Inf" and "NaN", so no
%       non-finite number is ever encoded (jsonencode is called with
%       'ConvertInfAndNaN', false, so nothing is written as null);
%     * logicals become 0/1, string objects become char, sparse matrices
%       become full, and a "Handles" field (graphics handles) is dropped.
%
%   Matrix shape is kept: a 1-row value is a flat array, an N-row value an
%   array of rows.
%
%   Requires a MATLAB with jsonencode and an initialised Middle Layer.

EXPORTER_VERSION = 'mml_export 1.0.0';

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

local_write(aoFile, local_document(export, local_normalize(AO)));
local_write(adFile, local_document(export, local_normalize(AD)));

fprintf('Wrote %s\n', aoFile);
fprintf('Wrote %s\n', adFile);

if nargout > 0
    files = {aoFile, adFile};
end
end


function text = local_document(export, body)
% One JSON object: the "_export" block first, then the body's own keys.
% "_export" is not a legal MATLAB field name, so the block is spliced in as text.
head = ['{"_export":' local_encode(export)];
encoded = local_encode(body);
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
out = containers.Map({'$fn', 'file'}, {func2str(h), file});
end


function out = local_char(c)
if size(c, 1) <= 1
    out = deblank(c);
    return
end
out = cell(1, size(c, 1));
for r = 1:size(c, 1)
    out{r} = deblank(c(r, :));
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
if isa(value, 'string')
    value = char(value);
end
if ischar(value)
    text = strtrim(value(1, :));
else
    text = '';
end
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
fprintf(fid, '%s', text);
end
