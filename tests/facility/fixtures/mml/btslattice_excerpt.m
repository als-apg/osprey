% Excerpt of MML machine/ALS/BTS/btslattice.m: every statement up to the end of the
% lattice arrays BTS, verbatim, without blank and comment-only lines.
% scripts/facility_demo/_mml.py parses it and tests/facility/test_element_table.py
% scans it when MML is absent; both check it against the MML original when MML
% is present.

function varargout = btslattice(varargin)
global THERING
if nargin >=1
    Energy = varargin{1};
else
    Energy = 1.9e9;
end
AP =  aperture('AP',  [-10, 10, -10, 10], 'AperturePass');
START = marker('START', 'IdentityPass');
BPM   = marker('BPM', 'IdentityPass');
Lcm = 6*.0254;  % 6 inches is the physical length of the iron core
HCM = corrector('HCM', Lcm, [0 0], 'CorrectorPass');
VCM = corrector('VCM', Lcm, [0 0], 'CorrectorPass');
QD  = quadrupole('QD', 1.0E-001,-2.3256700, 'StrMPoleSymplectic4RadPass');
L1  = drift('L1', 2.0048900E-001, 'DriftPass');
SEN = rbend('SEN', .6, -2.0*pi/180, -1.0*pi/180, -1.0*pi/180, 0, 'BndMPoleSymplectic4RadPass');
L2  = drift('L2', 2.0E-001, 'DriftPass');
SEK = rbend('SEK', .9, -10.0*pi/180, -5.0*pi/180, -5.0*pi/180, 0, 'BndMPoleSymplectic4RadPass');
L3  = drift('L3', .798989-Lcm/2, 'DriftPass');
L4  = drift('L4', .648535-Lcm/2, 'DriftPass');
L5  = drift('L5', .382917-Lcm/2, 'DriftPass');
L6  = drift('L6', .300000-Lcm/2, 'DriftPass');
TV1 = marker('TV', 'IdentityPass');
L7  = drift('L7', .2, 'DriftPass');
Q1  = quadrupole('Q', .3, 1.3921, 'StrMPoleSymplectic4RadPass');
L8  = drift('L8', .20-Lcm/2, 'DriftPass');
L9  = drift('L9', .25-Lcm, 'DriftPass');
L10 = drift('L10', 3.4095580-Lcm/2, 'DriftPass');
TV2 = marker('TV', 'IdentityPass');
L11 = drift('L11', 2.0E-001, 'DriftPass');
Q21 = quadrupole('Q', 3.0E-001,-6.1290000E-001, 'StrMPoleSymplectic4RadPass');
L12 = drift('L12', 3.0E-001, 'DriftPass');
Q22 = quadrupole('Q', 3.0E-001,-2.9980000E-001, 'StrMPoleSymplectic4RadPass');
L13 = drift('L13', .16-Lcm/2, 'DriftPass');
L14 = drift('L14', .175-Lcm, 'DriftPass');
L15 = drift('L15', .452-Lcm/2, 'DriftPass');
B1  = rbend('BEND', 1.4, 20*pi/180, 10*pi/180, 10*pi/180, 0, 'BndMPoleSymplectic4RadPass');
L16 = drift('L16', 1.6685430, 'DriftPass');
L17 = drift('L17', 3.5642900E-001, 'DriftPass');
Q31 = quadrupole('Q', 3.0E-001,-3.9970000E-001, 'StrMPoleSymplectic4RadPass');
L18 = drift('L18', 3.0E-001, 'DriftPass');
Q32 = quadrupole('Q', 3.0E-001, 1.6189000, 'StrMPoleSymplectic4RadPass');
L19 = drift('L19', .15-Lcm/2, 'DriftPass');
L20 = drift('L20', .175-Lcm, 'DriftPass');
L21 = drift('L21', .20-Lcm/2, 'DriftPass');
TV3 = marker('TV', 'IdentityPass');
L22 = drift('L22', 1.1710280, 'DriftPass');
B2  = rbend('BEND', 1.4, 20*pi/180, 10*pi/180, 10*pi/180, 0.0, 'BndMPoleSymplectic4RadPass');
L23 = drift('L23', 6.0719400E-001, 'DriftPass');
Q4  = quadrupole('Q', 3.0E-001, 2.2984000, 'StrMPoleSymplectic4RadPass');
L24 = drift('L24', 3.0359400E-001, 'DriftPass');
TV4 = marker('TV', 'IdentityPass');
L25 = drift('L25', 1.1918470-Lcm/2, 'DriftPass');
L26 = drift('L26', .175-Lcm, 'DriftPass');
L27 = drift('L27', .15-Lcm/2, 'DriftPass');
Q51 = quadrupole('Q', 3.0E-001,-2.9246000, 'StrMPoleSymplectic4RadPass');
L28 = drift('L28', 3.0E-001, 'DriftPass');
Q52 = quadrupole('Q', 3.0E-001, 2.7048000, 'StrMPoleSymplectic4RadPass');
L29 = drift('L29', 2.0426500E-001, 'DriftPass');
L30 = drift('L30', 9.1510000E-001, 'DriftPass');
B3  = rbend('BEND', 1.4, 20*pi/180, 10*pi/180, 10*pi/180, 0, 'BndMPoleSymplectic4RadPass');
L31 = drift('L31', .579682-Lcm/2, 'DriftPass');
L32 = drift('L32', .175-Lcm, 'DriftPass');
L33 = drift('L33', 2.3067730-Lcm/2, 'DriftPass');
L34 = drift('L34', 3.5634900E-001, 'DriftPass');
Q61 = quadrupole('Q', 3.0E-001, 2.1585000, 'StrMPoleSymplectic4RadPass');
L35 = drift('L35', 3.0E-001, 'DriftPass');
Q62 = quadrupole('Q', 3.0E-001,-2.0719000, 'StrMPoleSymplectic4RadPass');
L36 = drift('L36', .149446-Lcm/2, 'DriftPass');
L37 = drift('L37', .175-Lcm, 'DriftPass');
L38 = drift('L38', .87103-Lcm/2, 'DriftPass');
B4  = rbend('BEND', 1.3989, 18.4*pi/180, 9.2*pi/180, 9.2*pi/180, 0.0, 'BndMPoleSymplectic4RadPass');
L39 = drift('L39', 6.8926600E-001, 'DriftPass');
TV5 = marker('TV', 'IdentityPass');
L40 = drift('L40', 1.3422340, 'DriftPass');
L41 = drift('L41', 3.5634900E-001, 'DriftPass');
Q7  = quadrupole('Q', 3.0E-001, 1.7455000, 'StrMPoleSymplectic4RadPass');
L42 = drift('L42', .20-Lcm/2, 'DriftPass');
L43 = drift('L43', .25-Lcm, 'DriftPass');
L44 = drift('L44', 1.3856790-Lcm/2, 'DriftPass');
L45 = drift('L45', .955427-Lcm/2, 'DriftPass');
L46 = drift('L46', .24727900-Lcm, 'DriftPass');
L47 = drift('L47', .392873-Lcm/2, 'DriftPass');
TV6 = marker('TV', 'IdentityPass');
L48 = drift('L48', 8.0712800E-001, 'DriftPass');
SIK = rbend('SIK', 9.0E-1, 10.07*pi/180, 10.07*pi/180, 0.0*pi/180, 0.0, 'BndMPoleSymplectic4RadPass');
L49 = drift('L49', 2.7700000E-001, 'DriftPass');
SIN = rbend('SIN', .6, 2.0*pi/180, 2.0*pi/180, 0.0*pi/180, 0.0, 'BndMPoleSymplectic4RadPass');
L50 = drift('L50', 2.2743000E-001, 'DriftPass');
SRBMP3 = rbend('SRBMP3', .7, 0, 0, 0, 0, 'BndMPoleSymplectic4RadPass');
SRBMP4 = rbend('SRBMP4', .7, 0, 0, 0, 0, 'BndMPoleSymplectic4RadPass');
L51 = drift('L51', 3.0E-001, 'DriftPass');
CTR_SRQF = marker('CTR_SRQF', 'IdentityPass');
SRTV1 = marker('TV', 'IdentityPass');
BTS = [START AP L1 SEN L2 SEK L3 HCM L4 BPM L5 VCM L6 TV1 L7 Q1 L8 HCM L9 VCM ...
L10 TV2 L11 Q21 L12 Q22 L13 HCM L14 VCM L15 B1 L16 BPM L17 Q31 L18 Q32 ...
L19 HCM L20 VCM L21 TV3 L22 B2 L23 Q4 L24 TV4 L25 HCM L26 VCM L27 Q51 L28 Q52 ...
L29 BPM L30 B3 L31 HCM L32 VCM L33 BPM L34 Q61 L35 Q62 L36 HCM L37 VCM L38 B4 ...
L39 TV5 L40 BPM L41 Q7 L42 HCM L43 VCM L44 BPM L45 HCM L46 VCM L47 TV6 L48 SIK L49 SIN ...
L50 SRBMP3 L50 SRBMP4 L51 CTR_SRQF SRTV1];
