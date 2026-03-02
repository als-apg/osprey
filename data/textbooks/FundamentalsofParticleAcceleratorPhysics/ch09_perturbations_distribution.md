Perturbations to the regular 6-D particle distribution in linacs and in storage rings are discussed in this Chapter. "Regular" refers here to the charge density distribution function at the injection point in a linac, defined in terms of normalized emittances, and to the equilibrium distribution in a storage ring. Perturbations include (i) longitudinal-transverse coupling induced by variation of the particle's longitudinal momentum in a dispersive region (_synchro-betatron excitation_), (ii) particles diffusion by Coulomb scattering internal to the bunch (_intrabeam scattering_), and (iii) a variety of single-bunch and multi-bunch instabilities excited by the interaction of the beam particles with e.m. fields either internal to the beam, or associated to image currents on the vacuum chamber's surface (_collective effects_). A rigorous quantitative treatment of all these effects is beyond the aim of this book. Yet, a qualitative treatment and a classification of collective effects is provided. The main physical quantities are introduced in a single treatment for both linear and circular accelerators.

### 11 Synchro-Betatron Excitation

Owing to the linear superposition of betatron and dispersive motion in the complete solution of Hill's equation, the abrupt change of the particle's momentum in a dispersive region leads to a variation of the betatron amplitude: \(\Delta x_{\beta}=-\Delta x_{D}=-D_{x}\,\Delta\,p/p\). This implies a local change of the C-S invariant which, extended to the beam particle distribution, can result into emittance growth.

This synchro-betatron excitation was observed to be at the origin of the equilibrium horizontal emittance in a synchrotron (see Eq. 17). Not surprisingly, \(\epsilon_{x,eq}\) is proportional to the average dispersion function through \(H_{x}\) (see Eq. 22). In that case, an equilibrium can be reached because the shift of the radiating particle from an off-energy orbit, due to the change of its _total_ momentum, is compensated (on average over a turn) by the increase of the only _longitudinal_ momentum in RF cavities, thus by a reduction of the particle's angular divergence (\(x^{\prime}=\Delta\,p_{x}/p_{z}\)), hence of the beam emittance.

These considerations suggest that, in general, RF cavities and magnetic insertion devices with high field should not be installed in dispersive regions because, similarly to emission of synchrotron radiation, the change in longitudinal momentum could induce emittance growth, and therefore to a modification of the equilibrium emittance defined in Eq. 8.22.

As previously discussed, the emission of radiation due to longitudinal acceleration in linacs is negligible. Nevertheless, coherent synchrotron radiation (CSR) can be emitted by relatively short electron bunches in dipole magnets of magnetic compressors and switchyard lines. In spite of the single-pass dynamics, the CSR intensity and the consequent change of particle's momentum can be large enough to induce emittance growth in the plane of non-zero dispersion.

The emittance growth induced by synchrotron radiation is an incoherent effect, i.e., the total radiated field is the linear superposition of the field radiated by each particle, and the change of particle's momentum is not correlated with the particle's longitudinal position internal to the bunch. On the contrary, the change of momentum by CSR is correlated with \(z\). This implies a change of the particles' distribution in the transverse phase space, which is the resultant of a mismatch of individual bunch slices, each slice being associated to a different \(z\)-coordinate.

### 10.2 Intrabeam Scattering

Intrabeam scattering is the multiple small-angle Coulomb scattering of charged particles [1]. Unlike Touschek scattering, which is a single-scattering effect from the horizontal to the longitudinal direction of motion, leading to particle loss, intrabeam scattering is a diffusion process in all three dimensions. As such, it drives a growth with time of the transverse and the longitudinal beam emittance. In the following, the emittance growth rate will be reported without derivation, but a physical interpretation of its functional dependence from beam parameters is given, to discriminate intrabeam scatterng in linacs and in storage rings.

A common approach to derive intrabeam scattering growth rates starts from the assumption that the effect in an accelerated beam can be modelled as scattering of gas molecules in a closed box, where the box plays the role of magnetic and RF focusing in the accelerator, to keep the particles together. If we assume that the particles' three velocity components are independent, the scattering of particles leads to a Gaussian distribution in the 3-D momentum space.

Unlike in a closed box, however, the orbit curvature in a storage ring produces dispersion. Because of it, a sudden energy change translates into a change of the betatron amplitudes, thus to coupling of betatron and synchrotron motion, as discussed above. Moreover, the curvature leads to the so-called negative mass behaviour, which implies that an equilibrium condition above transition cannot exist. Finally, the derivation assumes that the particle's velocities are non-relativistic in the center-of-mass frame, which is strictly true for linacs only in specific cases. As of today, intrabeam scattering plays a major role in enlarging, for example, the bunch duration of proton beams in circular colliders, the transverse emittance of electron beams in multi-bend storage ring light sources, and the energy spread in highly dense electron beams in linacs for free-electron lasers.

#### Storage Rings

The intrabeam scattering growth rates are defined as the relative time-variation of beam's rms emittances [1]. Since in each plane the bunch dimensions are proportional to the square root of the emittance, a relative change of the bunch dimensions is half of the relative change of the emittance. In particular, we assume a round beam, no transverse coupling, and horizontal dispersion only. For the longitudinal plane we assume a bunched beam in the presence of synchrotron oscillations and constant mean energy. In case of small-amplitude oscillations (see Eq. 4.31), and with notation as in Eq. 8.44, the longitudinal emittance is proportional to the Hamiltonian, \(\epsilon_{z}=\frac{\langle J_{z}\rangle}{\kappa_{z}}=\frac{\langle H\rangle}{ \kappa_{z}}\). Then, according to Piwinski:

\[\left\{\begin{array}{l}\frac{1}{r_{x}^{1bs}}=\frac{1}{\epsilon_{x}}\frac{d \epsilon_{x}}{dt}=\frac{1}{2\langle x^{2}\rangle}\frac{d\langle x^{2}\rangle }{dt}\propto A_{S}f_{x}\left(\sigma_{x,y}^{\beta},\sigma_{x^{\prime},y^{ \prime}}^{\beta},\,D_{x},\sigma_{\delta}\right)\\ \frac{1}{r_{y}^{1bs}}=\frac{1}{\epsilon_{y}}\frac{d\epsilon_{y}}{dt}=\frac{1} {2\langle y^{2}\rangle}\frac{d\langle y^{2}\rangle}{dt}\propto A_{S}f_{y} \left(\sigma_{x,y}^{\beta},\sigma_{x^{\prime},y^{\prime}}^{\beta},\,D_{x}, \sigma_{\delta}\right)\\ \frac{1}{r_{z}^{1bs}}=\frac{1}{\epsilon_{z}}\frac{d\epsilon_{z}}{dt}=\frac{1} {2\langle H\rangle}\frac{d\langle H\rangle}{dt}\propto A_{S}f_{z}\left( \sigma_{x,y}^{\beta},\sigma_{x^{\prime},y^{\prime}}^{\beta},\,D_{x},\sigma_{ \delta}\right)\\ \\ A_{S}=\frac{r_{0}^{2}N_{b}[Clog]}{64\pi^{2}\sigma_{z}\sigma_{\delta}x,\beta \epsilon_{x},\beta\epsilon_{y},\beta^{3}\gamma^{4}} \tag{9.2}\]

The physics of intrabeam scattering is in the coefficient \(A_{S}\). There, \(r_{0}\) is the classical particle's radius, \(N_{b}\) the number of particles in a bunch, and all other symbols are self-explanatory. The term \([Clog]\) is said "Coulomb logarithm", and its argument is the ratio of maximum and minimum scattering angle relevant to the process, according to \(\tan\left(\frac{\theta_{min,max}}{2}\right)=\frac{2r_{0}}{b_{max,min}\nu^{2} \sigma_{x^{\prime}}^{2}}\), with \(b\) the scattering impact parameter.

In spite of some arbitrariness in the definition of \(\theta_{min}\), it is common to calculate it for \(b_{max}\approx\sigma_{x}\). By definition \(\theta_{max}<\pi\), but its upper limit is commonly restricted to \(\sim 10\theta_{min}\). This approach discards single scattering events in the tails of the charge distribution, which may heavily bias the calculation of intrabeam scattering in the bunch core. The logarithmic dependence makes intrabeam scattering weakly dependent from the argument of \([Clog]\). For few-GeV energy electron storage rings, \([Clog]\sim 10\). Equation 9.2 suggests that intrabeam scattering is more effective for high charge density beams.

A manipulation of the growth rates in the three planes of motion leads to the following condition (see also Eq. 8.39):

\[\begin{array}{l}\frac{d}{dt}\left[\left\langle\delta H\right\rangle\left(\frac{ 1}{\gamma^{2}}-\frac{D_{x}^{2}}{\beta_{x}^{2}}\right)+\frac{\left\langle\delta \epsilon_{x}\right\rangle}{\beta_{x}}+\frac{\left\langle\delta\epsilon_{y} \right\rangle}{\beta_{y}}\right]=0\\ \\ \Rightarrow\left\langle H\right\rangle\left(\frac{1}{\gamma^{2}}-\alpha_{c} \right)+\frac{\left\langle\epsilon_{x}\right\rangle}{\beta_{x}}+\frac{\left \langle\epsilon_{y}\right\rangle}{\beta_{y}}\approx const.\end{array} \tag{9.3}\]

and we used \(\left\langle\delta q\right\rangle\approx\left\langle q\right\rangle dt/T_{0}\). Below transition energy, the sum of the three invariants is limited, namely, particles can exchange their oscillation energy among the three planes of motion. The beam behaves as gas molecules in a closed box or, in other words, an equilibrium distribution can exist, in which intrabeam scattering does not change the beam's dimensions any longer. Above transition energy, instead, the negative coefficient of \(\left\langle H\right\rangle\) allows the oscillation energy to increase potentially in an indefinite way. In this case, an equilibrium distribution cannot exist.

#### Linacs

The advent of high brightness electron linacs for short wavelength free-electron lasers has only recently led to the detection of a noticeable effect of intrabeam scattering on the beam energy distribution [2]. The reason for this is that, if the beam is not stored in the accelerator for an extremely long time, a very high charge density would be required to make intrabeam scattering apparent. The relatively large transverse size of electron beams (from tens to few hundreds of microns) and the low growth rate of transverse emittance has so far allowed intrabeam scattering to be neglected in the transverse planes for any practical purpose. For this reason, the impact of intrabeam scattering on the beam energy spread only is treated in the following.

Equation 9.1 is re-written for the energy spread growth rate in a linac, in the approximation of ultra-relativistic, round beam (\(\beta_{x}\approx\beta_{y}\), \(\epsilon_{x}\approx\epsilon_{y}\)) passing through a straight non-dispersive section, but in the presence of acceleration, \(\gamma=\gamma_{0}+G\,\Delta s\):

\[\begin{array}{l}\frac{1}{\sigma_{\delta}}\frac{d\sigma_{\delta}}{ds}=\frac{G }{\sigma_{\delta}}\frac{d\sigma_{\delta}}{d\gamma}=\frac{G}{\tau_{acc}}+\frac {1}{\tau_{ibs}}=\frac{G}{\tau_{acc}}+\frac{A_{L}}{\gamma^{3/2}\sigma_{\delta }^{2}},\\ \\ A_{L}=\frac{r_{0}^{2}N_{b}[Clog]}{8\epsilon_{n}^{3/2}\beta_{x}^{1/2}\sigma_{z}} \end{array} \tag{9.4}\]

\(G\) is the accelerating gradient, and in the following we neglect the weak dependence of \([Clog]\) from the beam energy. \(\tau_{acc}\) is found by solving the equation for no intrabeam scattering (\(\tau_{ibs}\rightarrow\infty\)), and by imposing that the _absolute_ energy spread does not change during acceleration (see e.g. Eq. 4.7 and discussion there):\[\begin{array}{l}\frac{d\sigma_{\gamma}}{ds}=\frac{d}{ds}(\gamma\sigma_{\delta})= \frac{d\gamma}{ds}\sigma_{\delta}+\gamma\frac{d\sigma_{\delta}}{ds}\equiv 0;\\ \frac{1}{\sigma_{\delta}}\frac{d\sigma_{\delta}}{ds}=-\frac{1}{\gamma}\frac{d \gamma}{ds};\\ \frac{1}{\sigma_{\delta}}\frac{d\sigma_{\delta}}{d\gamma}=-\frac{1}{\gamma}= \frac{1}{\tau_{acc}}\\ \Rightarrow\sigma_{\delta}=\sigma_{\delta_{0}}\frac{\gamma_{0}}{\gamma}\\ \end{array} \tag{9.5}\]

With the prescription of Eq. 9.5 for \(\tau_{acc}\), Eq. 9.4 can be solved for \(\sigma_{\delta}(\gamma)\):

\[\begin{array}{l}\frac{d\sigma_{\delta}^{2}}{d\gamma}+\frac{2\sigma_{\delta} ^{2}}{\gamma}-\frac{2A_{L}}{G}\frac{1}{\gamma^{3/2}}=0\\ \Rightarrow\sigma_{\delta}^{2}=\frac{c_{0}}{\gamma^{2}}+\frac{4}{3}\frac{A_{ L}}{G}\frac{1}{\sqrt{\gamma}}\\ \end{array} \tag{9.6}\]

The coefficient \(c_{0}\) is found by imposing \(\sigma_{\delta}(0)\equiv\sigma_{\delta_{0}}\), and it results \(c_{0}=\gamma_{0}^{2}\sigma_{\delta,0}^{2}-\frac{4A_{L}}{3G}\gamma_{0}^{3/2}\). This is substituted into Eq. 9.6 to get the solution:

\[\sigma_{\delta}^{2}(\gamma)=\sigma_{\delta_{0}}^{2}\frac{\gamma_{0}^{2}}{ \gamma^{2}}+\frac{4}{3}\frac{A_{L}}{G}\frac{1}{\sqrt{\gamma}}\left[1-\left( \frac{\gamma_{0}}{\gamma}\right)^{3/2}\right]\equiv\sigma_{\delta_{0}}^{2} \frac{\gamma_{0}^{2}}{\gamma^{2}}+\sigma_{\delta,IBS}^{2}(\gamma) \tag{9.7}\]

The simpler scenario of no acceleration, such as intrabeam scattering in a drift at constant energy \(\gamma=\gamma_{0}\), is described by Eq. 9.4 for \(\tau_{acc}\rightarrow\infty\):

\[\begin{array}{l}\frac{d\sigma_{\delta}^{2}}{ds}=\frac{2A}{\gamma_{0}^{3/2}} =\frac{r_{0}^{2}N_{b}[Clog]}{4\gamma_{0}^{3/2}\epsilon_{n}^{3/2}\beta_{x}(s)^ {1/2}\sigma_{z}}\\ \\ \Rightarrow\sigma_{\delta}^{2}=\sigma_{\delta,0}^{2}+\frac{r_{0}^{2}N_{b}[Clog ]}{4\gamma_{0}^{3/2}\epsilon_{n}^{3/2}\sigma_{z}}\int_{0}^{L}\frac{ds}{\sqrt {\beta_{x}(s)}}\approx\sigma_{\delta,0}^{2}+\frac{r_{0}^{2}N_{b}[Clog]}{4 \gamma_{0}^{2}\epsilon_{n}\langle\sigma_{x}\rangle\sigma_{z}}L\\ \end{array} \tag{9.8}\]

The Coulomb logarithm in single-pass accelerators can be estimated in analogy to storage rings, with the prescription that its argument be proportional to the time \(\tau\) the particles take to travel along the beamline:

\[\left\{\begin{array}{l}[Clog]=\ln\left(\frac{q_{max}\epsilon_{n}}{2\sqrt{2 }r_{0}}\right)\\ \\ q_{max}\approx\frac{c\tau\,N_{b}r_{0}^{2}}{2\gamma^{3/2}\epsilon_{n}^{3/2} \langle\beta_{x}\rangle^{1/2}\sigma_{z}}+o\left(\xi\right)\\ \end{array}\right. \tag{9.9}\]

The approximated expression for \(q_{max}\) holds as long as \(\xi=\sigma_{\delta}\sqrt{\frac{\langle\beta_{x}\rangle}{\gamma\epsilon_{n}}}\ll 1\), which in fact makes \(q_{max}\) independent from \(\sigma_{\delta}\).

Equation 9.7 points out that when \(\gamma\gg\gamma_{0}\), and for any given \(G\), (i) \(\sigma_{\delta,IBS}\sim\gamma^{-1/4}\), i.e., the effect of intrabeam scattering on the _relative_ energy spread depends weakly from the beam mean energy (the IBS-induced absolute energy spread goes like \(\gamma^{5/4}\) instead), and (ii) the growth of _relative_ energy spread evaluated at the end of acceleration is largely independent from the initial beam energy.

### 9.3 Collective Effects

Collective effects refer to all those phenomena in which the beam, being a collection of charges, acts back on itself via either direct particle-particle interaction, or the environment in which it travels [3]. They are current-dependent effects, and can either establish an instability of the particles' motion, or determine a new equilibrium distribution.

Collective effects due to direct particle-particle interaction comprise _direct space charge force_ and _coherent synchrotron radiation instability_. In storage rings, they can be mediated by the interaction of the beam with the surrounding vacuum chamber, which makes this classification not that rigid.

The second class of interactions originates in the production of image charges on the wall of the vacuum chamber and of RF cavities. The finite conductivity of the metallic surroundings, as well as the abrupt change of the vacuum chamber profile, determines a causality principle according to which leading particles act back on trailing particles through the e.m. field associated to the image currents. The head-tail interaction is named _wake field_, if expressed in the time domain. Its counterpart in the frequency domain is called _impedance_. The classification remains ambiguous since an impedance can also be used to model a direct tail-head interaction such as CSR in single-pass accelerators.

All collective effects are potential sources of instability if the interaction of the beam with the e.m. fields establishes a positive feedback. This can lead to the deformation of the transverse, longitudinal and energy distribution of the accelerated beam. Damping mechanisms include Landau damping and negative feedback loops.

Landau damping originates in the spread of betatron tune, synchrotron tune and energy of the beam particles, in turn excited by nonlinearities in the transverse (higher order multipoles, field errors) and longitudinal motion (RF curvature, large synchrotron oscillation amplitudes). Landau damping is therefore intrinsic to the beam dynamics. It can be made even more effective via octupole magnets and higher harmonic RF cavities in storage rings, via off-crest acceleration and interaction with an external laser (to enlarge the beam energy spread) in linacs. The stochastic and possibly chaotic motion associated to Landau damping can lead to beam lifetime reduction in storage rings, which has therefore to be traded off with beam stability. Negative feedback loops rely on pick-ups (sensors and actuators), which sample the beam's motion and impose an e.m. kick, either transverse or longitudinal, to damp beam's coherent oscillations.

#### 9.3.1 Wakefields

Let us consider an ultra-relativistic bunch of total charge \(Q\), at longitudinal position \(s\) along the accelerator. For simplicity, the surrounding is assumed to have cylindrical symmetry. A "test" particle of charge \(q\) travels behind it at relative position \((z,r)\) in cylindrical coordinates. In the presence of finite conductivity or discontinuitiesof the vacuum chamber (including e.g. RF cavities), the bunch charge \(Q\) acts as a source of e.m. field, which catches up with the trailing test charge \(q\)[3].

The longitudinal and transverse component of the Lorentz's force of interaction of the two charges, normalized to the source and to the test charge, provides a purely geometric function, called _wake function_ or wake field, in that it only depends from the conductivity or the geometry of the elements surrounding the particles:

\[\left\{\begin{array}{l}w_{\parallel}(s,z)=-\frac{1}{q\,Q}\int_{0}^{s}ds^{ \prime}F_{\parallel}(s^{\prime},z)=-\frac{1}{Q}\int_{0}^{s}ds^{\prime}E_{z} \ \ \left[\frac{V}{C}\right]\\ \\ w_{\perp}(s,z)=-\frac{1}{q\,Q}\frac{d}{dr}\int_{0}^{s}ds^{\prime}F_{\perp}(s^{ \prime},z,r)=-\frac{1}{Q}\frac{d}{dr}\int_{0}^{s}ds^{\prime}(E_{x,y}\pm v_{z} \times B_{y,x})\ \ \left[\frac{V}{C\cdot m}\right]\end{array}\right. \tag{9.10}\]

The introduction of the e.m. field gives the alternative definition of wake function as the space-integrated field per unit of source charge. The integration is intended over the trajectory of the _test_ particle, i.e., at the location where the e.m. field is sampled. The sign convention is such that a test particle at \(z<0\) is behind the beam. Sometimes the wake function is given per unit length of travel of the test particle, so that \(w_{\parallel}^{\prime}(s,z)=dw_{\parallel}/ds\) is in units of \([V/(C\cdot m)]\) and \(w_{\perp}^{\prime}(s,z,r)=dw_{\perp}/ds\) is in units of \([V/(C\cdot m^{2})]\).

The Wake functions in Eq. 9.10 are truncated at first order in the lateral coordinate \(r\); for this reason they are said to be "monopole" (\(w_{\parallel}\)) and "dipole" approximation (\(w_{\perp}\)) of the actual wakefield. It then emerges that \(w_{\parallel}\) is non-zero independently from the radial position of the source. On the contrary, \(w_{\perp}\) is non-zero only if the source is off the symmetry axis of the vacuum chamber or of the RF cavity (\(r\neq 0\)). Higher order terms of the wake field become important, for example, when the beam size is comparable to the vacuum chamber radius.

The wake function is a Green's function because it describes the response of the system (the vacuum chamber or the RF cavity) to a point-like and unitary stimulus. The convolution of the wake function with the source charge distribution gives the path-integrated field, namely, the e.m. potential generated by \(Q\) and sampled at the location occupied by \(q\), during the time \(q\) has traveled a distance \(s\). Such quantity is named _wake potential_:

\[\left\{\begin{array}{l}V_{\parallel}(s,z)=\iint dx^{\prime}dy^{\prime}\int_ {-\infty}^{z}dz^{\prime}\rho(x^{\prime},\,y^{\prime},z^{\prime})w_{\parallel }(s,z-z^{\prime})=\\ \\ =\int_{-\infty}^{z}dz^{\prime}\lambda(z^{\prime})w_{\parallel}(s,z-z^{\prime}) \ \ \left[V\right]\\ \\ V_{\perp}(s,z)=\iint dx^{\prime}dy^{\prime}\int_{-\infty}^{z}dz^{\prime}\rho(x ^{\prime},\,y^{\prime},z^{\prime})w_{\perp}(s,z-z^{\prime})\ \ =\\ \\ =\int_{-\infty}^{z}dz^{\prime}\lambda(z^{\prime})w_{\perp}(s,z-z^{\prime}) \ \ \left[\frac{V}{m}\right]\end{array}\right. \tag{9.11}\]

where we defined \(\iint dr^{3}\rho(\widetilde{r})=\int dz\lambda(z)=Q\), with \(\lambda(z)\) the longitudinal charge distribution function, or current profile.

The two-particle mathematical treatment adopted so far is identically valid if, for example, the test charge belongs to the source bunch. Then, \(s\) is the bunch coordinatealong the accelerator, \(z\) the coordinate internal to the bunch. In reality, all bunch particles can play both roles of "source" and "test" particle, because they produce wakefield and are, in turn, affected by the wakefield generated by all other leading particles. By energy conservation, the e.m. energy stored in the cavity (or cavity-like insertion) in the form of wake field the total energy loss accumulated by the bunch through the element. For an element long \(L\), the energy loss is the convolution of the wake potential with the charge distribution function. For the longitudinal wakefield in monopole approximation:

\[\begin{array}{l}\Delta E(L)=\int_{-\infty}^{+\infty}dz\lambda(z)V_{\parallel }(L,z)\ \ \ [J]\\ \\ \Rightarrow k_{\parallel}=-\frac{\Delta E}{Q^{2}}\ \ \left[\frac{V}{C}\right] \end{array} \tag{9.12}\]

\(k_{\parallel}\) is called _loss factor_, it is always positive and commonly of the order of few to several \(\sim V/pC\) in single or multi-cell RF cavities. By definition, the loss factor depends only from the geometry of the RF cavity.

The analogous quantity for the transverse plane is called _transverse kick factor_. It measures the total change of transverse momentum accumulated by the bunch passing through, e.g., an RF cavity long \(L\), per unit of lateral distance from the cavity electric axis:

\[\begin{array}{l}c\frac{d}{dr}\Delta p_{\perp}(L)=\int_{-\infty}^{+\infty}dz \lambda(z)V_{\perp}(L,z)\ \ \left[\frac{J}{m}\right]\\ \\ \Rightarrow k_{\perp}=-\frac{c}{Q^{2}}\frac{dp_{\perp}}{dr}\ \ \left[\frac{V}{C\cdot m}\right] \end{array} \tag{9.13}\]

\(k_{\perp}\) is typically of the order of few to several \(\sim V/(pC\cdot mm)\) in single or multi-cell RF cavities.

#### Impedances

The Fourier transform (FT) of the wake function is called _impedance_[3]:

\[\left\{\begin{array}{l}Z_{\parallel}(\omega)=\frac{1}{c}\int_{-\infty}^{+ \infty}dze^{-i\omega z/c}w_{\parallel}(s,z)\ \ [\Omega]\\ \\ Z_{\perp}(\omega)=\frac{i}{c}\int_{-\infty}^{+\infty}dze^{-i\omega z/c}w_{ \perp}(s,z)\ \ \left[\frac{\Omega}{m}\right]\end{array}\right. \tag{9.14}\]

The impedance is still a geometric property of the environment of the accelerated beam. Its use largely simplifies the calculation of collective effects in recirculating accelerators, (though being used also in single pass systems) by virtue of the following relationship:

\[V(\omega)=-I(\omega)Z(\omega) \tag{9.15}\]

with \(V(\omega)\) and \(I(\omega)\) the Fourier transform of the wakefield-induced potential and of the bunch current, respectively. Impedances show the following characteristics.

1. Eq. 9.10 states that wakefields are real functions. This implies \(Z_{\parallel}(\omega)^{*}=Z_{\parallel}(-\omega)\) and \(Z_{\perp}(\omega)^{*}=Z_{\perp}(-\omega)\).
2. The causality principle \(w(z)=0\) for \(z>0\) implies \(Z(\omega)\to 0\) for \(\omega\to\pm\infty\).
3. The Panofsky-Wenzel theorem establishes a one-to-one correspondence between the longitudinal and the transverse component of an impedance. The relation is exact for a smooth, infinitely long, cylindrical vacuum chamber of radius \(a\), it is approximately valid for any chamber discontinuity provided its characteristic size is small compared to \(a\), and valid only on average for large objects like RF cavities: \[Z_{\perp}(\omega)\approx\frac{2c}{a^{2}}\frac{Z_{\parallel}(\omega)}{\omega} \approx\frac{2R}{a^{2}}\frac{Z_{\parallel}(\omega)}{n},\quad n=\frac{\omega} {\omega_{0}}\] (9.16) The second equality of Eq. 9.16 applies to storage rings only, with \(R\) the equivalent storage ring radius, and \(\omega_{0}\) the revolution frequency hereafter.

In a cavity-like structure, the total wakefield can be described as the superposition of quasi-monochromatic waves, or "modes". Modes of characteristic wavelength smaller than the cavity apertures cannot be trapped. In other words, there exists a natural _cutoff frequency_\(\omega_{c}\approx c/a\) above which there are no resonant modes in the cavity (modes beyond cutoff exist but they are associated to emission of synchrotron radiation, which is not considered here). Translating this picture into the frequency domain, we infer that the total longitudinal impedance of an accelerator can be modeled as the superposition of _single-resonator impedances_:

\[\left\{\begin{array}{ll}Z_{\parallel}(\omega)=\sum_{k}Z_{\parallel,k}(\omega ),&Z_{\parallel,k}(\omega)=\frac{R_{s,k}}{1+i\,Q_{k}\left(\frac{\omega_{k}}{ \omega}-\frac{\omega}{\omega_{k}}\right)}\\ \\ Z_{\perp}(\omega)=\sum_{k}Z_{\perp,k}(\omega),&Z_{\perp,k}(\omega)\approx \frac{2}{a}\frac{\omega_{c}}{\omega}Z_{\parallel,k}(\omega)\end{array}\right. \tag{9.17}\]

Each resonator or mode is characterized by its own quality factor (\(Q_{k}\)), shunt impedance (\(R_{s,k}\), in units of \(\Omega\)) and resonant angular frequency \(\omega_{k}\). Depending on the characteristic decay time scale of the wakefield, i.e., on the quality factor of the cavity-like element which allows the e.m. field to be stored, two classes of interactions can be discriminated.

Wakefields acting on the same source bunch are called _short range_. They correspond in the frequency domain to a _broad-band impedance_ (\(Z_{bb}\)) or the superposition of low-Q resonators. For example, they are generated by the finite conductivity of the vacuum chamber (_resistive wall_ wakefield), by discontinuities of the vacuum chamber (bellows, joints, etc.) and low-Q parasitic modes in RF cavities (_geometric or diffraction_ wakefield). The total broad-band impedance is conventionally modeled through a broad-band resonator with \(Q=1\), \(\omega_{k}=\omega_{c}\). The broad-band shunt impedance \(R_{s,bb}\) is consequently retrieved from fit to experimental data or simulations:

\[\left|\frac{Z_{\parallel}}{n}\right|_{bb}\equiv\lim_{\omega\to 0}\left|\frac{Z_{ \parallel}}{n}\right|=\frac{R_{s}}{Q}\frac{\omega_{0}}{\omega_{k}}\equiv R_{ s,bb}\frac{\omega_{0}}{\omega_{c}} \tag{9.18}\]

The Longitudinal broad-band impedance of modern storage rings is typically in the range 0.1-1 \(\Omega\).

Wakefields lasting enough time to affect trailing bunches are called _long range_. They correspond in the frequency domain to a narrow-band impedance (\(Z_{nb}\)), or sum of high-Q resonators. They are for example stored in RF cavities or cavity-like insertions of the vacuum chamber. For this reason, they are commonly depicted as _Same_ (of the fundamental) and _High Order Modes_ (SOM and HOMs), or "parasitic RF modes" in general.

Since measurable effects of the longitudinal broad-band impedance, such as current-dependent bunch lengthening and energy spread growth, depend from the convolution of the accelerator impedance (\(Z_{bb,\parallel}(\omega)\)) and the bunch power spectrum (\(|I(\omega)|^{2}\)), it is convenient to introduce the _effective impedance_:

\[\left|\frac{Z_{\parallel}}{n}\right|_{eff}=\frac{\int_{-\infty}^{+\infty}\frac {Z_{\parallel}(\omega)}{n}|I(\omega)|^{2}d\omega}{\int_{-\infty}^{+\infty}|I( \omega)|^{2}d\omega} \tag{9.19}\]

#### Classification

Table 9.1 summarizes some common sources of collective effects in linacs and storage rings. The associated instability, either single bunch (SB) or multi bunch (MB), is depicted via established nomenclature in the literature A short description of each effect is given below.

_Space Charge Force (SC)_

The interaction can be intended as inelastic Coulomb scattering internal to the bunch. The strength of the average force depends from the 3-D spatial charge distribution. The electric and magnetic component of the space charge force cancel at very high energy. In fact, the transverse force evaluated in the laboratory reference frame, \(F_{sc}\), is the Lorentz's force, with the electric field proportional to the total

\begin{table}
\begin{tabular}{l|l|l|l|l} \hline Source & Linac & \multicolumn{2}{l}{Storage ring} \\ \hline  & SB & MB & SB & MB \\ \hline SC & \(\epsilon_{6D}\)-dilution at injection & & Low \(I_{b}\) at injection & \\ \cline{2-4} CSR & \(\epsilon_{x}\)-growth, MBI & & MWI, Power loss & \\ \hline RW & Energy chirp, energy loss & & & \\ \hline \(Z_{bb,\parallel}\) & & & & \\ \hline \(Z_{bb,\perp}\) & BBU & & TMCI & \\ \hline \(Z_{nb,\parallel}\) & & Transient beam loading & Power loss & Transient beam loading, LCBI \\ \hline \(Z_{nb,\perp}\) & & BBU & & TCBI \\ \hline \end{tabular}
\end{table}
Table 9.1: Classification of collective effects of single bunch (SB, or short range wakefield) and multi-bunch (MB, or long range wakefield)charge, \(E_{sc}\sim N_{b}\), and the magnetic field generated by the beam current \(B_{sc}\sim I_{b}\sim v_{z}N_{b}\sim v_{z}E_{sc}\). Hence, \(F_{sc,\perp}=q(E_{sc}-v_{z}\times B_{sc})\sim(1-v_{z}^{2})N_{b}\sim N_{b}/\gamma^{2}\). An exact derivation starting from the beam's rest frame is given in Eqs. 63 and 64.

In high charge density linacs, the direct space charge force dilutes the beam 6-D emittance. In electron linacs, its effect is counteracted by very high RF gradients in the first stage of acceleration (RF "Gun" cavities), to boost the beam to ultra-relativistic energies in relatively short distances.

In storage rings, the direct space charge force generates an incoherent tune shift which, coupled to synchrotron oscillations of the bunched beams, leads to a tune spread [4]. The effect goes like \(\sim\gamma^{-3}\), and it can be harmful for the beam lifetime if strong resonances are crossed. An indirect space charge force is produced by the interaction of the beam self-field with the vacuum chamber, in the presence of coherent betatron oscillations (i.e., of the bunch as a whole). In this case, a coherent-so-called _Laslet_--tune shift is observed, whose energy dependence is \(\sim\gamma^{-1}\). The space charge force is particularly harmful in low energy proton storage rings. Being it proportional to the bunch charge, it limits the maximum current injected at low energy.

#### Coherent Synchrotron Radiation (CSR)

Long wavelength synchrotron radiation is emitted by trailing particles in a bunch on a curved path, and it catches up with leading particles of the same bunch. The intensity of radiation is enhanced in electron accelerators because of the lighter rest mass with respect to hadron beams, at the same total energy. Also, the effect is amplified by high charge, short bunches, because the intensity of the emitted radiation scales as \(\sim N_{b}^{2}\), and the spectrum covers the range of wavelengths comparable to, or shorter than, the bunch length, i.e., commonly IR-THz frequencies.

In electron linacs, such tail-head interaction causes a net total energy loss, variation of longitudinal momentum along the bunch, and emittance growth in the bending plane, because the change of longitudinal momentum happens in a dispersive region. It also participates to the so-called _microbunching instability_ (MBI), in synergy to the space charge force, i.e., the development of broad-band micron-scale longitudinal modulations.

In storage rings, such instability is mediated by the vacuum chamber and it is named _microwave instability_ (MWI, also "turbulent bunch lengthening"). The intermittent emission of intense bursts of synchrotron radiation reflect the chaotic particles' motion in the longitudinal phase space, where energy and density modulations develop at wavelengths in the IR-THz region. The bursts alternate to relaxation of the charge distribution.

#### Resistive Wall Wakefield (RW)

The resistive wall wake field is associated to the "skin effect" of the vacuum chamber. The corresponding longitudinal broad-band impedance is proportional to the skin depth \(\delta(\omega)\sim c/\sqrt{\sigma|\omega|}\) of the chamber (whose conductivity is \(\sigma\)) and inversely proportional to the chamber radius.

In linacs adopting small gap chambers (e.g., long collimators or low gap magnetic devices), the wakefield generates a nonlinear correlation of the particle's energy along the bunch, and it can contribute substantially to additional mean energy loss. In storage rings, it contributes to the overall broad-band impedance of the accelerator, thus supporting the MWI.

Short Range Longitudinal Geometric Wakefield (\(Z_{bb,\parallel}\))The corresponding impedance collects the broad-band contribution of the vacuum chamber discontinuities (bellows, joints, etc.) and low-Q parasitic RF modes.

In linacs, it adds to the RW to deform the beam energy distribution, i.e., to lower the beam mean energy, and to induce a linear and, for high charge bunches in small iris cavities, nonlinear energy-chirp. The beam total energy spread is increased accordingly. The linear chirp can partly be compensated by off-crest acceleration, although at the expense of a lower total accelerating voltage.

In storage rings, the longitudinal wakefield superimposes to the RF potential. The _potential well distortion_ can lead to either bunch lengthening or shortening, depending from the characteristic wavelength of the wakefield and the natural bunch length (i.e., in the limit of zero-current). The longitudinal charge distribution is deformed as well, and it can substantially deviate from a Gaussian.

The wakefield also adds to the RW in driving the MWI. The so-called _Keil-Schnell criterion_, extended by Boussard to short wavelength modulations in bunched beams, estimates the upper limit of the single bunch _peak_ current to avoid that the instability build up:

\[\hat{I}_{b,th}\approx 2\pi\,\frac{|\eta|(E/e)\sigma_{\delta}^{2}}{|Z_{ \parallel}/n|_{bb}} \tag{9.20}\]

The _average_ current threshold for a Gaussian bunch is \(\langle I_{b,th}\rangle=\hat{I}_{b,th}\frac{\alpha_{0}\sigma_{t}}{\sqrt{2\pi}}\), with \(\sigma_{\delta}\) and \(\sigma_{t}=\frac{|\alpha_{c}|}{\Omega_{s}}\sigma_{\delta}\) (see Eq. 8.12) the unperturbed rms relative energy spread and bunch duration.

Short Range Transverse Geometric Wakefield (\(Z_{bb,\perp}\))This is the transverse counterpart of \(Z_{bb,\parallel}\) and it is proportional to the relative misalignment of the beam and the symmetry axis of the chamber or RF cavity. Leading particles in a bunch drive betatron oscillations of trailing particles. In other words, the wakefield determines a misalignment of bunch slices in the transverse phase space, or deformation of the transverse charge distribution.

In linacs, the misalignment of the bunch tail with respect to the head translates into an effective growth of the transverse emittance. If the wake potential is large enough, it can lead to beam loss. Since the single bunch _beam break-up instability_ (BBU) is the result of a deterministic sum of transverse kicks collected along the accelerator (on top of trajectory jitter), it can be counteracted by trajectory bumps to make the kicks to cancel each other.

In storage rings, it is customary to analyse the transverse oscillations in terms of normal modes, referred to as "head-tail" modes. The instability is called _Transverse Mode-Coupling Instability_ (TMCI) or _fast head-tail instability_. The \(m\)-th mode has \(m\) nodes along the bunch. For example, for mode \(m=0\) all particles have the same betatron phase (rigid dipole motion), whereas for \(m=\pm 1\) the head and tail have opposite phases, etc. The angular frequency of each mode is modulated by the constant exchange of head and tail particles via synchrotron oscillations, i.e., \(\omega_{m}=\omega_{\beta}+m\Omega_{s}\).

The threshold for the single bunch _average_ current in order to avoid that the TMCI builds up is usually higher than for the MWI:

\[\langle I_{b,th}\rangle\approx\sqrt{2\pi}\,\tfrac{|\eta|(E/e)\sigma_{\delta}}{ \langle\beta_{x,y}|Z_{\perp}|_{hb}\rangle_{R}}F(\sigma_{z}) \tag{9.21}\]

The form factor \(F(\sigma_{z})\) is basically the ratio of the total machine impedance to the effective impedance for the transverse plane (see Eq. 9.19). It is \(F\approx 1\) for short bunches, and it grows linearly with the bunch length for longer bunches.

If the chromaticity is non-zero, the beam energy spread leads to the modulation of the betatron tune at the synchrotron tune, thus a phase shift between head and tail of the bunch and, eventually, to damping of the head-tail modes. Above transition energy, positive chromaticity damps the mode \(m=0\), but it excites the modes \(m=\pm 1\). However, the characteristic time of these modes is usually longer than the damping time, which allows most electron storage rings to operate stably with slightly positive chromaticity.

Long Range Geometric Wakefields \((Z_{nb,\parallel},\,Z_{nb,\perp})\)

Among multi-bunch instabilities, _dipole coupled-bunch oscillations_ usually dominate the beam dynamics. In linacs, \(Z_{nb,\parallel}\) translates into the so-called _transient beam loading_, or differential (i.e., bunch-to-bunch) mean energy loss and energy chirp along the bunch train. In the transverse plane, \(Z_{nb,\perp}\) generates the _beam break up_ instability (BBU), i.e., trailing bunches in a train are pushed off-axis from the wakefield generated by leading bunches. Both instabilities are common in high-Q RF cavities traversed by many high charge bunches, such as in superconducting linacs.

In storage rings, the transient beam loading generates additionally a bunch-to-bunch synchrotron tune spread, whose range depends also from the kind of bunch fill pattern in the accelerator.

Dipole coupled-bunch oscillations translate into the motion of the bunches about their nominal centers as if they were rigid macroparticles. Such oscillations are called _Longitudinal Coupled-Bunch Instability_ (LCBI) and _Transverse Coupled-Bunch Instability_ (TCBI). For \(M\) bunches equally spaced, each multi-bunch mode, either longitudinal or transverse, is characterized by a bunch-to-bunch phase difference \(\Delta\phi=2\pi l/M\), where the "mode number" \(l\) can only take the values \(l=0,\,1,\,2,\,...,\,M-1\). The net phase advance of each mode is constrained by the periodic motion to be \(2\pi\). Each multi-bunch mode has a characteristic set of angularfrequencies: \(\omega_{p}=[\,p\,M\,\pm\,(l+v)]\omega_{0}\), with \(p\) integer and \(v\) either the synchrotron or the betatron tune, depending if the mode is longitudinal or transverse, respectively.

LCBI can affect the brightness of light sources because energy oscillations at dispersive locations determine an effective growth of the beam size, as well as a spectral broadening of undulator emission. Similar effect arises from TCBI because of the enlargement of the effective size and angular divergence of the radiation source. Reduction of CBI is usually accomplished by means of a combination of Landau damping, negative multi-bunch feedback, and suppression of parasitic RF modes. The latter action can be taken through a suitable design of the RF cavity (e.g., including multiple ports or antennas to absorb undesired modes), and/or a mechanical deformation of the cavity geometry (via mechanical stress, positioning of tuning rods or temperature control, to induce de-tuning of the HOMs). Landau damping is favoured in the longitudinal plane by a lower voltage from the main RF and the adoption of higher harmonic cavities (leading to longer bunches, thus smaller synchrotron frequency), and by octupole magnets in the transverse plane (inducing a larger spread of the betatron tune).

##### Parasitic Power Loss

Although not directly related to beam instabilities, parasitic power loss due to the dissipation of energy via image currents can limit the stored beam current, because of excessive heating of the vacuum chamber. This is a common challenge in electron storage rings. For \(M\) identical bunches in a train and single bunch average current \(\langle I_{b}\rangle\), the parasitic power loss is:

\[P=M\langle I_{b}\rangle^{2}Z_{loss} \tag{9.22}\]

where \(Z_{loss}\) is the real part of the effective impedance causing the energy loss. It is usually contributed by \(Z_{bb,\parallel}\) for \(\frac{\omega_{r}}{Q_{r}}\gg M\omega_{0}\), and by \(Z_{nb,\parallel}\) for \(\omega_{r}\approx nM\omega_{0}\), \(n\in\mathbb{N}\) (namely, for the resonant frequency of the HOM close to a harmonic of the bunch frequency). The contribution of RW to Ohmic losses is usually negligible with respect to the one by \(Z_{bb,\parallel}\).

#### Robinson's Instability

To minimize the effect of geometric wakefields, the vacuum chamber internal geometry has to be carefully designed to avoid trapping of resonant modes. RF cavities constitute one important exception to this, since they are built exactly to resonate at the fundamental mode of the accelerating field [5]. This means they are also perfect narrow-band resonators for the parasitic mode excited by the beam at the resonant fundamental frequency of the cavity, \(\omega\approx\omega_{RF}=h\omega_{0}\). Even in case of a beam perfectly aligned onto the cavity electric axis, the LCBI is expected to be driven. How could one get rid of, or at least alleviate, the instability?

We have seen that the presence of \(Z_{nb,\parallel}(\omega)\) drives longitudinal oscillations of the bunch as a whole. They are called coherent synchrotron oscillations because they happen both in phase (arrival time at the RF cavity) and energy (mean value over the bunch particles). As a practical case, let us consider an electron storage ring above transition energy, i.e., \(\eta<0\) (see Eq. 4.19). The bunch revolution frequency is \(\omega_{riv}=\omega_{0}(1+\eta\delta)\), namely, \(\omega_{riv}<\omega_{0}\) for a relative energy deviation (with respect to the synchronous bunch) \(\delta>0\).

The resistive part of \(Z_{nb,\parallel}(\omega)\) or \(\mathbb{R}e(Z_{nb,\parallel}(\omega))\) implies mean energy loss. In order to damp coherent synchrotron oscillations, i.e., to reduce the energy deviation turn-by-turn, we would like to have the impedance larger for those bunches which have \(\delta>0\) or \(\omega_{riv}<\omega_{0}\), and smaller for those with \(\delta<0\) or \(\omega_{riv}>\omega_{0}\).

In other words, \(\mathbb{R}e(Z_{nb,\parallel}(\omega))\) should be centered--i.e., the cavity tuned--at a frequency slightly smaller than the nominal RF frequency, \(\omega_{k}<h\omega_{0}\), as shown in Fig. 1-left plot. In particular, since the mode of oscillation induced by the impedance is modulated by the synchrotron frequency \(\omega_{s}\), we require \(\mathbb{R}e(Z_{nb,\parallel}(\omega))\) to be smaller at \(h\omega_{0}+\omega_{s}\), and larger at \(h\omega_{0}-\omega_{s}\). The opposite happens for beams below transition. Such procedure is called _Robinson damping_ or "Robinson's stability criterion".

## References

* [1] A. Piwinski, Intra-Beam scattering, in _Proceedings of Joint US-CERN School on Particle Accelerators at South Padre Island_, Texas, USA, ed. by M. Month, S. Turner (Published by Springer-Verlag LNP 296, 1986), p. 297
* [2] S. Di Mitri et al., Experimental evidence of intrabeam scattering in a free-electron laser driver. New J. Phys. **22**, 083053 (2020); G. Perosa, S. Di Mitri, Matrix model for collective phenomena in electron beam's longitudinal phase space. Sci. Rep. **11**, 7895 (2021)
* [3] M. Furman, J. Byrd, S. Chattopadhyay, Beam instabilities, in _Synchrotron Radiation Sources--A Primer_, ed. by H. Winick (Published by World Scientific, Singapore, 1995), pp. 306-342. ISBN: 9810218567

Figure 1: Real part of the geometric impedance \(Z_{nb,\parallel}(\omega)\) resonant at the fundamental frequency \(\omega_{k}\) of the RF cavity, where \(\Delta\omega=h\omega_{0}-\omega_{k}\), \(\omega_{0}\) is the revolution frequency of the synchronous particle, and \(\omega_{s}\) the synchrotron frequency. On the left, the RF cavity is tuned (the impedance is peaked) at a frequency slightly smaller than \(h\omega_{0}\), which implies Robinson damping above transition energy. The opposite is on the right, which implies damping (excitation) below (above) transition energy

* [4] A. Hofmann, Tune shifts from self-field images, in _Proceedings of CERN Accelerator School: 5th General Accelerators Physical Course_, Geneva, Switzerland. CERN 94-01, vol. I, ed. by S. Turner (1994), pp. 329-348
* [5] A. Chao, Beam dynamics of collective instabilities in high-energy accelerators, in _Proceedings of CERN Accelerator School: Intensity Limitations in Particle Beams_, Geneva, Switzerland, vol. 3/2017, ed. by W. Herr (2017)

