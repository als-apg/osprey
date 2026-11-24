% Excerpt of MML machine/ALS/Booster/boosterlattice.m: every statement up to the end of
% the lattice array BOOSTER, verbatim, without blank and comment-only lines.
% scripts/facility_demo/_mml.py parses it when MML is absent; the tests check it
% against the MML original when MML is present.

function varargout = boosterlattice(varargin)
global THERING
if nargin >= 1
    Energy = varargin{1};
else
    Energy = 1.9e9;
end
Brho = getbrho(Energy/1e9);
AP = aperture('AP',  [-10, 10, -10, 10], 'AperturePass');
START = marker('START', 'IdentityPass');
BPM   = marker('BPM',   'IdentityPass');
TV    = marker('TV',    'IdentityPass');
END   = marker('END',   'IdentityPass');
L0 = 75;              % Design length [m]
C0 = 299792458;       % Speed of light [m/s]
HarmNumber = 125;     % Base  RF on storage ring
CAV = rfcavity('CAV', 0, 1.2e+6, HarmNumber*C0/L0, HarmNumber ,'CavityPass');
LL1 = 0.546875;
LL2 = 0.496875;
LL3 = 2.093750;
L1  = drift('L1', LL1, 'DriftPass');
L2  = drift('L2', LL2, 'DriftPass');
L3  = drift('L3', LL3, 'DriftPass');
L_SF = 0.0;
L_SD = 0.0;
L1_SD  = drift('L1_SD', LL1-L_SD, 'DriftPass');
L2_SF  = drift('L2_SF', LL2-L_SF, 'DriftPass');
Lcm = 0;
HCM = corrector('HCM', Lcm, [0 0], 'CorrectorPass');
VCM = corrector('VCM', Lcm, [0 0], 'CorrectorPass');
QF = quadrupole('QF', .15,  2.7682214*0.96, 'StrMPoleSymplectic4RadPass');
QD = quadrupole('QD', .10, -2.5401249*0.99, 'StrMPoleSymplectic4RadPass');
Ksf =  0;  %
Ksd = -0;  %
SF = sextupole('SF', L_SF, Ksf, 'StrMPoleSymplectic4RadPass');
SD = sextupole('SD', L_SD, Ksd, 'StrMPoleSymplectic4RadPass');
if 0
    BU = rbend('BEND', 1.05,   15.0*pi/180,  7.5*pi/180,  7.5*pi/180, 0, 'BndMPoleSymplectic4RadPass');
    BD = marker('BD', 'IdentityPass');
elseif 0
    BU = rbend('BEND', 1.05/2,  7.5*pi/180,  7.5*pi/180,           0, 0, 'BndMPoleSymplectic4RadPass');
    BD = rbend('BEND', 1.05/2,  7.5*pi/180,           0,  7.5*pi/180, 0, 'BndMPoleSymplectic4RadPass');
else
    BU = rbend('BEND', 1.05,   15.0*pi/180,  0,  0, 0, 'BndMPoleSymplectic4RadPass');
    BD = marker('BD', 'IdentityPass');
end
CELL1 = [L3 QF QF BPM HCM TV L2 BU BD L1 VCM QD QD BPM SD L1_SD BU BD L2 HCM QF QF BPM SF L2_SF BU BD L1_SD SD QD QD BPM VCM L1 BU BD L2_SF SF BPM QF QF HCM TV L2 BU BD L1_SD SD QD QD BPM VCM L1 BU BD L2 HCM BPM QF QF VCM L3 BPM QD QD  ];
CELL2 = [L3 QF QF BPM HCM    L2 BU BD L1 VCM QD QD BPM SD L1_SD BU BD L2 HCM QF QF BPM SF L2_SF BU BD L1_SD SD QD QD BPM VCM L1 BU BD L2_SF SF BPM QF QF HCM    L2 BU BD L1_SD SD QD QD BPM VCM L1 BU BD L2 HCM BPM QF QF VCM L3     QD QD BPM];
CELL3 = [L3 QF QF BPM HCM    L2 BU BD L1 VCM QD QD BPM SD L1_SD BU BD L2 HCM QF QF BPM SF L2_SF BU BD L1_SD SD QD QD BPM VCM L1 BU BD L2_SF SF BPM QF QF HCM    L2 BU BD L1_SD SD QD QD BPM VCM L1 BU BD L2 HCM BPM QF QF VCM L3     QD QD BPM];
CELL4 = [L3 QF QF BPM HCM    L2 BU BD L1 VCM QD QD BPM SD L1_SD BU BD L2 HCM QF QF BPM SF L2_SF BU BD L1_SD SD QD QD BPM VCM L1 BU BD L2_SF SF BPM QF QF HCM TV L2 BU BD L1_SD SD QD QD BPM VCM L1 BU BD L2 HCM BPM QF QF VCM L3     QD QD BPM];
BOOSTER = [ START AP CAV CELL1 CELL2 CELL3 CELL4 END];
