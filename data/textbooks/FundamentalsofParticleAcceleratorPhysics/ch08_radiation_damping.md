In single-pass or few turns-only recirculating accelerators, the particle beam distribution is largely determined by its configuration at injection, eventually modified at a later stage by RF and magnetic elements. The emission of radiation due to longitudinal acceleration is typically negligible in the particle's energy budget.

On the contrary, the relevant and continuous emission of synchrotron radiation in a storage ring, together with the replenishment of kinetic energy provided by RF cavities, determine a change of the injected 6-D particle distribution over a time scale much longer than the single turn. At the end, the distribution will tend to a Gaussian in each sub-phase space ("equilibrium"), independently from the initial conditions.

This process is commonly described as the sum of two distinct contributions to the particles' invariants, _radiation damping_ and _quantum excitation_[1]. The former origins in the _linear_ expansion of the radiated energy about the particle's energy, and it results in a damping of the amplitudes of oscillation. The latter emerges when the beam's energy spread due to radiation emission (_second order_ momentum) is taken into account. We first consider the impact of these effects on synchrotron oscillations, then we extend the results to the transverse planes.

### 8.1 Radiation Damping and Quantum Excitation

The Longitudinal motion of a single particle in a storage ring is described by Eq. 4.31, in the assumption that the accelerator behaves as a linear conservative system. This assumption is here revisited. In order to move to homogeneous variables in a normalized longitudinal phase space, Eq. 4.43 is recalled to find:\[\begin{array}{l}\phi=\omega_{RF}\Delta t=h\omega_{s}\Delta t\sim\frac{\Delta E}{E_{ 0}}\frac{h\alpha_{c}}{Q_{s}}=h\omega_{s}\frac{\Delta E}{E_{0}}\frac{\alpha_{c}} {\Omega_{s}}\\ \Rightarrow\Delta t\sim\Delta E\frac{\alpha_{c}}{E_{0}\Omega_{s}}\end{array} \tag{8.1}\]

In the following, we will use the notation \(\tau\approxeq\Delta z/c\), \(\epsilon=\Delta E\) for the particle's coordinates relative to the synchronous particle.

Let \(A_{\epsilon}\) be the single particle's invariant for the longitudinal motion in a conservative system. Its variation with time due to radiation emission is considered. The variation is assumed to be adiabatic, so that the motion can be described as a pure harmonic oscillator on a single turn basis:

\[\left\{\begin{array}{l}\epsilon\left(t\right)=A_{\epsilon}(t)\cos(\Omega_{s} t+\phi_{0})\equiv A_{\epsilon}(t)\cos\phi\\ \tau\left(t\right)=-\left(\frac{\alpha_{c}}{E_{0}\Omega_{s}}\right)A_{\epsilon }(t)\sin\phi\end{array}\right.\qquad\qquad\Rightarrow\left\{\begin{array}{l }A_{\epsilon}^{2}=\epsilon^{2}+\tau^{2}\left(\frac{E_{0}\Omega_{s}}{\alpha_{c} }\right)^{2}\\ \langle\epsilon^{2}(t)\rangle_{\phi}=\frac{A_{\epsilon}^{2}(t)}{2}\end{array}\right. \tag{8.2}\]

\(\langle\epsilon^{2}(t)\rangle_{\phi}\) is intended to be averaged over all phases of the bunch particles at a given time \(t\). In this sense, it is just the beam's rms energy spread, and \(A_{\epsilon}\) becomes the normalized _rms_ amplitude of oscillation.

The Top equation on the r.h.s. of Eq. 8.2 is used to calculate the variation of the squared amplitude, \(dA_{\epsilon}^{2}\), averaged over all the synchrotron phases. We assume that the instantaneous emission of photons does change the particle's energy by the amount \(d\epsilon=-u\), but _not_ its phase, or \(d\tau=0\). Then, the radiated energy \(u(\epsilon)\) is expanded to first order in the particle's energy:

\[\begin{array}{l}\langle dA_{\epsilon}^{2}\rangle_{\phi}=\langle d\epsilon^{2 }\rangle+\langle d\tau^{2}\rangle\left(\frac{E_{0}\Omega_{s}}{\alpha_{c}} \right)^{2}=2\langle\epsilon d\epsilon\rangle+\frac{1}{2}\langle(2d\epsilon)d \epsilon\rangle=-2\langle\epsilon u\rangle+\langle u^{2}\rangle\approx\\ \approxeq-2\langle\epsilon\frac{du}{d\epsilon}\epsilon\rangle+\langle u^{2} \rangle=-A_{\epsilon}^{2}\frac{du}{d\epsilon}+\langle u^{2}\rangle\end{array} \tag{8.3}\]

where the derivative \(du/d\epsilon\) is a function of \(\epsilon\), characteristic of the photon distribution of synchrotron radiation.

The average growth rate in a turn is additionally averaged over the closed orbit of equivalent radius \(R\):

\[\begin{array}{l}\langle\frac{d}{dt}\langle dA_{\epsilon}^{2}\rangle_{\phi} \rangle_{R}=\langle\frac{d}{dt}\left(\langle A_{\epsilon}^{2}(t)\rangle_{\phi }-\langle A_{\epsilon}^{2}(0)\rangle_{\phi}\rangle\right)_{R}=\langle\frac{d }{dt}A_{\epsilon}^{2}\rangle_{R}=\\ =-\langle A_{\epsilon}^{2}\rangle_{R}\langle\frac{d}{dt}\frac{du}{d\epsilon} \rangle_{R}+\langle\frac{d}{dt}\langle u^{2}\rangle_{\phi}\rangle_{R}\end{array} \tag{8.4}\]

The physical meaning of the two terms on the r.h.s. of Eq. 8.4 is elucidated below.

#### Longitudinal Motion

If only the first term on the r.h.s of Eq. 8.4 were taken, we would get:

\[\langle\frac{dA_{\epsilon}^{2}}{dt}\rangle_{R,damp}=-\langle A_{\epsilon}^{2} \rangle_{R}\langle\frac{d}{dt}\frac{du}{d\epsilon}\rangle_{R} \tag{8.5}\]We remind that \(\frac{du}{dt}=P\propto E^{4}/R^{2}\propto B_{y}^{2}E^{2}\) is the instantaneous synchrotron radiation power introduced in Eq. 7.12. \(P(B_{y},\,E)\) is calculated below at first order in the particle's energy (\(\epsilon\)), for the off-energy particle travelling on a dispersive orbit in a dipole magnet, i.e., \(x\approxeq D_{x}\epsilon/E_{0}\). This is justified by the fact that, although in general \(x=x_{\beta}+x_{D}\), we are interested here in the dependence of the radiated power from the off-energy coordinate. Moreover, we assume \(\langle x_{\beta}\rangle=0\), which allows us to neglect the betatron motion at this stage. Still, to be more general, a quadrupole gradient embedded in the dipole is considered ("combined" dipole magnet).

Because of the energy deviation \(\epsilon\), a given bending angle corresponds to a different path length and bending radius for the generic and the synchronous particle, respectively, such that (see Fig. 8.1):

\[\theta=\tfrac{ds}{R}=\tfrac{dl}{(R+x)}\Rightarrow dl=\left(1+\tfrac{x}{R} \right)ds \tag{8.6}\]

For ultra-relativistic particles (\(C=v_{z}T_{0}\approxeq cT_{0}\)) we find:

\[u =\oint\tfrac{dl}{c}\,P(B_{y},\,E)=\oint\tfrac{ds}{c}\left(1+\tfrac {x}{R}\right)\left(P_{0}+\tfrac{\partial P}{\partial E}\epsilon+\tfrac{ \partial P}{\partial B_{y}}\tfrac{\partial B_{y}}{\partial x}\tfrac{\partial x }{\partial E}\epsilon\right)=\] \[=\oint\tfrac{ds}{c}\left(1+\tfrac{D_{x}}{R}\,\tfrac{\epsilon}{E_{ 0}}\right)\left(P_{0}+\tfrac{2P_{0}}{E_{0}}\epsilon+\tfrac{2P_{0}}{B_{y}}g\,D _{x}\,\tfrac{\epsilon}{E_{0}}\right)=\] \[\approx\oint\tfrac{ds}{c}\left(P_{0}+P_{0}\tfrac{D_{x}}{R}\, \tfrac{\epsilon}{E_{0}}+\tfrac{2P_{0}}{E_{0}}\epsilon+2P_{0}kD_{x}\,R\,\tfrac{ \epsilon}{E_{0}}\right);\]

\[\tfrac{du}{d\epsilon}=\oint\tfrac{ds}{c}\left(\tfrac{2P_{0}}{E_{0}}+\tfrac{P_ {0}}{E_{0}}\tfrac{D_{x}}{R}+2\tfrac{P_{0}}{E_{0}}kD_{x}\,R\right);\]

\[\tfrac{d}{dt}\tfrac{du}{d\epsilon}=\tfrac{d}{dt}\left(\tfrac{2U_{0}}{E_{0}} \right)+\tfrac{d}{dt}\oint\tfrac{ds}{c}\,\tfrac{P_{0}}{E_{0}}D_{x}\,R\left( \tfrac{1}{R^{2}}+2k\right); \tag{8.7}\]

\[\langle\tfrac{d}{dt}\tfrac{du}{d\epsilon}\rangle_{R} =\tfrac{1}{C}\oint\tfrac{ds}{dt}\tfrac{2U_{0}}{E_{0}}+\tfrac{1}{C} \oint\tfrac{ds}{dt}\oint\tfrac{ds}{c}\,\tfrac{P_{0}}{E_{0}}D_{x}\,R\left( \tfrac{1}{R^{2}}+2k\right)=\] \[=\tfrac{2U_{0}}{T_{0}E_{0}}+\tfrac{1}{T_{0}E_{0}}\oint\tfrac{ds}{ c}\,P_{0}D_{x}\,R\left(\tfrac{1}{R^{2}}+2k\right)=\] \[=\tfrac{U_{0}}{T_{0}E_{0}}\left[2+\oint ds\left(\tfrac{P_{0}R^{2} }{U_{0}c}\right)\tfrac{D_{x}}{R}\left(\tfrac{1}{R^{2}}+2k\right)\right]=\] \[=\tfrac{U_{0}}{T_{0}E_{0}}\left[2+\tfrac{\oint ds\,\tfrac{D_{x}}{R }\left(\tfrac{1}{R^{2}}+2k\right)}{\oint\tfrac{ds}{R^{2}}}\right]\equiv \tfrac{U_{0}}{T_{0}E_{0}}(2+\mathbb{D})\]

The coefficient \(\mathbb{D}\) introduced above depends only from the linear optics of the accelerator. Its physical meaning is elucidated below in the case of, for example, an isomagnetic lattice (\(R(s)=R\)) and separate function dipole magnets (\(k=0\)):

\[\mathbb{D}\rightarrow\left(iso\right)\ \tfrac{1}{C}\oint ds\,D_{x}\,R\left( \tfrac{1}{R^{2}}+2k\right)\rightarrow\left(sep\right)\ \tfrac{1}{C}\oint\tfrac{D_{x}}{R}ds=\alpha_{c} \tag{8.8}\]By substituting the result of Eq. 8.7 into Eq. 8.5 we find:

\[\begin{array}{l}\langle\frac{dA_{\epsilon}^{2}}{dt}\rangle_{R,damp}=-\langle A_ {\epsilon}^{2}\rangle_{R}\frac{U_{0}}{T_{0}E_{0}}(2+\mathbb{D})\equiv-\frac{2}{ \tau_{\epsilon}}\langle A_{\epsilon}^{2}\rangle_{R}\\ \\ \Rightarrow\left\{\begin{array}{l}\langle A_{\epsilon}^{2}(t)\rangle_{R}= \langle A_{\epsilon}^{2}(0)\rangle_{R}e^{-\frac{2t}{\tau_{\epsilon}}}\\ \\ \alpha_{\epsilon}=1/\tau_{\epsilon}=\frac{U_{0}}{2T_{0}E_{0}}(2+\mathbb{D}) \equiv\frac{J_{\epsilon}U_{0}}{2T_{0}E_{0}}\end{array}\right.\end{array} \tag{8.9}\]

Synchrotron oscillations are damped only if \(\mathbb{D}>-2\). \(\alpha_{\epsilon}\) is called _damping coefficient_ for the longitudinal plane, the characteristic decay time \(\tau_{\epsilon}=1/\alpha_{\epsilon}\) is the _longitudinal damping time_, and \(J_{\epsilon}=2+\mathbb{D}\) is the _longitudinal partition number_.

Roughly speaking, the damping time corresponds to the number of turns (\(1/T_{0}\)) that the particle would require to completely exhaust its energy via radiation emission (\(U_{0}/E_{0}\)).

If radiation damping were the only process in action, it would shrink the oscillation amplitude, and therefore the beam's longitudinal emittance, to arbitrarily small values over a sufficiently long time. In other words, all representative points in the phase space would collapse to one point, which is in contrast to the experimental observation. This paradox is explained by the presence of the second term on the r.h.s. of Eq. 8.4.

Since each particle emits independently from the others, but photons all belong to the same distribution, averaging over the particles' phase in Eq. 8.3 can be replaced by averaging over the photon energies (Campbell's theorem) [2]. Be \(n(u)\) the photon energy distribution of synchrotron radiation introduced in Eq. 7.28 normalized to the total number of emitted photons, \(\int_{0}^{\infty}n(u)du=N\). Making use of Eqs. 7.29 and 7.32 we find:

\[\begin{array}{l}\langle\frac{dA_{\epsilon}^{2}}{dt}\rangle_{R,exc}=\langle \frac{d}{dt}\int_{0}^{\infty}u^{2}n(u)du\rangle_{R}=\langle\frac{d}{dt}\int_{ 0}^{\infty}n(u)du\frac{\int_{0}^{\infty}u^{2}n(u)du}{\int_{0}^{\infty}n(u)du} \rangle_{R}=\\ \\ =\langle\frac{dN}{dt}\langle u^{2}\rangle_{R}\rangle=\langle\frac{15\sqrt{3}} {8}\frac{P_{0}}{u_{c}}\frac{11}{2T}u_{c}^{2}\rangle_{R}=\frac{55}{24\sqrt{3}} \langle P_{0}u_{c}\rangle_{R}\end{array} \tag{8.10}\]

Figure 8.1: Top view of reference (\(ds\)) and distorted orbit (\(dl\)) in a dipole magnet

The balance of quantum excitation in Eq. 8.10 with radiation damping in Eq. 8.9 leads to the existence of an equilibrium value for the oscillation amplitude:

\[\begin{array}{l}\langle\frac{dA_{e}^{2}}{dt}\rangle_{R}=\langle\frac{dA_{e}^{2 }}{dt}\rangle_{R,damp}+\langle\frac{dA_{e}^{2}}{dt}\rangle_{R,exc}=-\frac{2}{ \tau_{e}}\langle A_{e}^{2}\rangle_{R}+\frac{55}{24\sqrt{3}}\langle P_{0}u_{c} \rangle_{R}\equiv 0;\\ \\ \Rightarrow\frac{\langle A_{e}^{2}\rangle_{R}}{2E_{0}^{2}}=\left(\frac{\epsilon _{eq}}{E_{0}}\right)^{2}=\sigma_{\delta,eq}^{2}=\frac{1}{E_{0}^{2}}\frac{55}{ 96\sqrt{3}}\tau_{e}\langle P_{0}u_{c}\rangle_{R}=\\ \\ =\frac{1}{E_{0}^{2}}\frac{55}{96\sqrt{3}}\frac{2E_{0}}{J_{e}}\frac{3}{2}\hbar c \gamma^{3}\frac{\langle 1/|R^{3}|\rangle}{\langle 1/R^{2}\rangle}=\frac{55}{32 \sqrt{3}}\frac{\hbar}{moc}\frac{\gamma^{2}}{J_{e}}\frac{\langle 1/|R^{3}| \rangle}{\langle 1/R^{2}\rangle}\\ \\ \Rightarrow\sigma_{\delta,eq}^{iso}=\gamma\sqrt{\frac{C_{e}}{J_{e}|R|}}, \hskip 14.226378ptC_{e}=3.836\cdot 10^{-13}m\end{array} \tag{8.11}\]

The dependence of \(\sigma_{\delta,eq}^{2}\) from the average radius squared and cubic in Eq. 8.11 comes, respectively, from \(\langle P_{0}u_{c}\rangle\sim\langle\frac{1}{R^{2}}\cdot\frac{1}{R}\rangle\), and \(\tau_{e}\sim\frac{T_{0}}{U_{0}}=\frac{1}{\langle P_{0}\rangle}\sim\frac{1}{ \langle 1/R^{2}\rangle}\cdot\sigma_{\delta,eq}^{iso}\) is the rms relative energy spread at equilibrium of an electron beam in an isomagnetic lattice. Since \(R\sim E\), one has \(\sigma_{\delta,eq}\propto\sqrt{\gamma}\). Such weak dependence explains the observed limited range \(\sim 0.06-0.12\%\) of the relative energy spread at storage ring light sources spanning \(\sim 1-8\) GeV beam energies.

The bunch duration at equilibrium is obtained through the usual conversion factor (see Eq. 8.1):

\[\sigma_{t,eq}\,=\,\frac{|\alpha_{c}|}{\Omega_{s}}\sigma_{\delta,eq} \tag{8.12}\]

To make this explicit, the expression of the synchrotron frequency in Eq. 4.32 is simplified by introducing the peak accelerating gradient \(\dot{V_{0}}\) in [V/m], and by considering ultra-relativistic electrons (or positrons) above transition energy:

\[\left\{\begin{array}{l}V_{0}\sin\psi_{s}=\frac{dV}{d\psi}=\frac{dV}{cdt} \frac{cdt}{d\psi}=\frac{c\dot{V}_{0}(s)}{\omega_{RF}}\\ \eta\approx-\alpha_{c}\\ p_{z,s}\approx\frac{E_{0}}{c}\end{array}\right.\Rightarrow\Omega_{s}\approx \sqrt{\frac{ec\dot{V}_{0}\alpha_{c}}{T_{0}E_{0}}} \tag{8.13}\]

Replacing this into Eq. 8.12, we obtain for an isomagnetic lattice:

\[c\sigma_{t,eq}\approx\alpha_{c}\gamma\sqrt{\left|\frac{2\pi\,R_{s}\gamma m_{ e}c^{2}}{e\dot{V}_{0}\alpha_{c}}\frac{C_{e}}{J_{e}R}\right|}\approx\sqrt{2\pi\,C_{e}} \sqrt{\left|\frac{\alpha_{c}}{J_{e}(e\dot{V}_{0}/m_{e}c^{2})}\right|}\gamma^{ 3/2} \tag{8.14}\]

Typical values of the rms natural bunch duration (e.g., in the presence of a single frequency of the RF system) are of few tens' of picoseconds. Equation 8.14 illustrates in a rigorous manner the dependence of the bunch duration \(\sigma_{t}\propto\sqrt{\left|\alpha_{c}/\dot{V}_{0}\right|}\), previously inferred in Eq. 4.44.

#### Horizontal Motion

The equilibrium emittance in the horizontal plane is retrieved by analysing the variation of the Floquet's normalized amplitude of oscillation (Eq. 4.109) in the presence of radiation emission. The amplitude variation is due to both betatron and dispersive motion, i.e., \(x=x_{\beta}+x_{D}\). By introducing the notation \(H_{x}=\gamma_{x}\,D_{x}^{2}+2\alpha_{x}\,D_{x}\,D_{x}^{\prime}+\beta_{x}\,D_{x }^{\prime 2}\) we find:

\[A_{x}^{2}=\gamma_{x}x^{2}+2\alpha_{x}xx^{\prime}+\beta_{x}x^{ \prime 2}=A_{x,\,\beta}^{2}+H_{x}\left(\tfrac{\epsilon}{E_{0}}\right)^{2}+\] \[\qquad+2\left[\gamma_{x}x_{\beta}D_{x}\tfrac{\epsilon}{E_{0}}+2 \alpha_{x}\left(x_{\beta}D_{x}\tfrac{\epsilon}{E_{0}}\right)\left(x_{\beta}^{ \prime}D_{x}^{\prime}\tfrac{\epsilon}{E_{0}}\right)+\beta_{x}\left(x_{\beta}^ {\prime}D_{x}^{\prime}\tfrac{\epsilon}{E_{0}}\right)\right];\] \[dA_{x}^{2}=dA_{x,\,\beta}^{2}+\tfrac{H_{x}}{E_{0}^{2}}d\epsilon^ {2}+f\left(x_{\beta},x_{\beta}^{\prime}\right);\] \[\langle dA_{x}^{2}\rangle_{\phi}=\langle dA_{x,\,\beta}^{2} \rangle_{\phi}+\tfrac{H_{x}}{E_{0}^{2}}\langle d\epsilon^{2}\rangle_{\phi}+f \left(\langle x_{\beta}\rangle_{\phi},\langle x_{\beta}^{\prime}\rangle_{\phi }\right)=\langle dA_{x,\,\beta}^{2}\rangle_{\phi}+\tfrac{H_{x}}{E_{0}^{2}}N_{ph }\langle u^{2}\rangle_{n} \tag{8.15}\]

We note that, so as \(J\) is a definite positive quantity, also \(H_{x}\) is. The very last equality made use of Eq. 8.10 for the average of the photon energies, and \(\langle x_{\beta}\rangle_{\phi}=\langle x_{\beta}^{\prime}\rangle_{\phi}=0\). As already for the longitudinal motion, also in this case we can discriminate the two contributions of radiation damping (\(dA_{x,\,\beta}^{2}\)) and quantum excitation (\(\langle u^{2}\rangle_{n}\)).

According to Eq. 4.109, the betatron normalized amplitude \(dA_{x,\,\beta}^{2}\) is:

\[dA_{x,\,\beta}^{2}=dw^{2}+dw^{2}=2wdw+2w^{\prime}dw^{\prime} \tag{8.16}\]

Since photon emission is assumed to be instantaneous, the variation of the particle's position is null, only the angular divergence is affected. Because of the superposition of betatron and dispersive motion, we find for \(dw\):

\[dx=dx_{\beta}+dx_{D}\equiv 0\Rightarrow dx_{\beta}=-dx_{D}=-D_{x}\tfrac{d \epsilon}{E_{0}}\Rightarrow dw=\tfrac{D_{x}}{\sqrt{\beta_{x}}}\tfrac{u}{E_{0}} \tag{8.17}\]

The variation of the normalized angular divergence \(dw^{\prime}\) is retrieved from the definition of \(x^{\prime}\):

\[\left\{\begin{array}{l}x^{\prime}=\tfrac{p_{x}}{p_{z}}\approx\tfrac{p_{x}}{ p}\\ x^{\prime}=\tfrac{dx}{ds}=\tfrac{dx}{dw}\tfrac{dw}{d\phi}\tfrac{d\phi}{ds}= \sqrt{\beta_{x}}w^{\prime}\tfrac{1}{\beta_{x}}=\tfrac{w^{\prime}}{\sqrt{\beta _{x}}}\end{array}\right. \tag{8.18}\]

\[\Rightarrow\left\{\begin{array}{l}\tfrac{dx^{\prime}}{x^{\prime}}=\tfrac{ dp}{p}=\tfrac{d\epsilon}{E_{0}}\\ \tfrac{dx^{\prime}}{x^{\prime}}=\tfrac{dw^{\prime}}{\sqrt{\beta_{x}}}\tfrac{1} {x^{\prime}}=\tfrac{dw^{\prime}}{w^{\prime}}\end{array}\right.\Rightarrow \tfrac{dw^{\prime}}{w^{\prime}}=-\tfrac{u}{E_{0}} \tag{8.19}\]

[MISSING_PAGE_FAIL:183]

The limit taken for an isomagnetic lattice puts in evidence the action of the -function. The beam's emittance is generated by the spread of particles' betatron orbits in the transverse phase space, driven by the radiating process. Since the local bump of the betatron amplitude (at the instant of emission) is equal (in absolute value) to the dispersive bump, the average dispersion function and its derivative eventually contribute to determining the particle distribution at equilibrium.

#### Vertical Motion

The absence of vertical dispersion would suggest null quantum excitation and therefore null vertical emittance for sufficiently long time, by virtue of the only surviving term of radiation damping in Eq. 8.15. In reality, a non-zero vertical beam size is always observed at equilibrium because of the change of the particle's vertical momentum in response of photons emitted off the bending plane, within the characteristic angular divergence of - of synchrotron radiation.

Owing to the absence of vertical dispersion in ideal configurations,. This is expanded up to second order in the radiated energy, in the assumption that the instantaneous photon emission does not change the particle's position, but only its angular divergence:

(8.23)

By using Eqs. 8.18 and 8.19:

(8.24)

The variation of the squared amplitude is first averaged over all the betatron phases, then the rate of its growth is averaged over the orbit, and eventually forced to zero at equilibrium:

(8.25)The limit is taken for an isomagnetic lattice. The damping coefficient for the vertical plane is, and is the _vertical damping time_. In the absence of vertical dispersion, the vertical partition number is.

Typical values of the average betatron function and bending radius in electron synchrotrons would lead to pm. In reality, the vertical emittance at equilibrium is contributed by betatron coupling and spurious vertical dispersion (), for example due to misaligned magnetic elements, skew quadrupole magnets, etc. Both betatron coupling and spurious vertical dispersion depend from the horizontal motion, and therefore the vertical emittance is proportional to the horizontal one via the so-called coupling coefficient,.

#### Robinson's Theorem

Robinson's theorem [1, 2] states that _the sum of the 3 partition numbers is constant_, whatever the magnetic lattice and the RF parameters are.

To demonstrate the theorem, we consider the transfer matrix of a synchrotron,, which applies to the vector,,. By virtue of Floquet's theorem (see e.g. Eq. 4.99), the eigenvalues of the matrix can be written in the form:

(8.26)

where for the three plans of motion, and the very last equality is for, which will be verified a posteriori.

The matrix describes the instantaneous emission of radiation through an infinitesimal element of the orbit:

(8.27)

For infinitesimal instantaneous perturbation to the particle's motion via emission of photon energy, the change in angular divergence is the one calculated in Eq. 8.18, which does not depend upon other coordinates:

(8.28)

The quantity is the amount of energy restored by RF cavities to keep the particle's energy constant on average in one turn. Its absolute value is equal to the radiated energy, and the negative sign indicates that the divergence lowers as the particle is accelerated () by the RF field.

Since the rate of emission is quadratic in the particle's total energy (see e.g. Eq. 7.12), the change in relative energy deviation is:

\[\tfrac{d\delta}{\delta}=\tfrac{d\delta}{dt}\tfrac{dt}{\delta}=\tfrac{d(dE)/dt}{ dE/dt}=\tfrac{dP}{P}=-2\tfrac{dE}{E_{0}}=-2\tfrac{u}{E_{0}}\quad\Rightarrow m_{6}=-2 \tfrac{u}{E_{0}} \tag{8.29}\]

Equations 8.28, 8.29 demonstrate that \(M_{ds}\) is diagonal. Moreover, since the photon emission does not change the particle's position in the 3 planes of motion, \(du_{1}=du_{3}=du_{5}=0\Rightarrow m_{1}=m_{3}=m_{5}=0\), and \(m_{2},m_{4},m_{6}\) are the only non-zero terms.

Finally, \(\det M\) is calculated with the help of Eqs. 8.28 and 8.29 at the first order in \(u\):

\[\det M_{ds}\approxeq\prod_{i=1}^{6}(1+m_{i})\approxeq 1+m_{2}+m_{4}+m_{6}+o(m_ {i}m_{k})=1-2\tfrac{\delta\epsilon_{RF}}{E_{0}}-2\tfrac{u}{E_{0}};\] \[\det M=\det\left(\prod_{ds}M_{ds}\right)=\prod_{ds}\left(\det M_{ ds}\right)=\prod_{ds}\left[1-\tfrac{2}{E_{0}}(\epsilon_{RF}+u)\right]=1- \tfrac{4U_{0}}{E_{0}} \tag{8.30}\]

The equality of \(\det M\) in Eqs. 8.26 and 8.30 demonstrates that

\[\sum_{j=1}^{3}\alpha_{j}=\sum_{j=1}^{3}J_{j}\tfrac{U_{0}}{2T_{0}E_{0}}=\tfrac{ 2U_{0}}{T_{0}E_{0}}\quad\Rightarrow\quad\sum_{j=1}^{3}J_{j}=4 \tag{8.31}\]

We draw the following considerations.

* The initial assumption \(|\alpha_{j}T_{0}|\sim U_{0}/E_{0}\ll 1\) is met in any practical case.
* The theorem holds as long as the external fields are known a priori, i.e., no beam-induced fields are considered.
* The theorem still applies to the case of linear coupling between horizontal and vertical plane (in this case, the normal mode emittances reach equilibrium), as well as to vertical dipole magnets (vertical dispersion).
* Only the choice \(J_{i}>0\) leads to stable motion. The characteristic time scale to reach equilibrium is the damping time. In particular, by virtue of Eqs. 8.9, 8.21 and 8.25, and when horizontal dispersion only is considered, the beam distribution reaches an equilibrium in all the 3 planes of motion only if \(-2<\mathbb{D}<1\).

Robinson's theorem can be put in a Lorentz's invariant form by noticing that, according to Eqs. 8.9, 8.21 and 8.25, \(2\alpha_{tot}=\sum_{i}2\alpha_{i}\) is the rate of reduction of the 6-D phase space volume. Being it the inverse of a characteristic time, its Lorentz's invariant form is found by passing from the time interval in the laboratory frame to the proper time interval:

\[2\alpha_{\tau}=\tfrac{dt}{d\tau}2\alpha_{tot}=2\gamma\,\sum_{i=1}^{3}\alpha_{ i}=\tfrac{2E_{0}}{m_{0}c^{2}}\tfrac{2P_{0}}{E_{0}}=\tfrac{4P_{0}}{m_{0}c^{2}} \tag{8.32}\]

This says that the characteristic rate of phase space reduction due to radiation damping is 4 times the ratio of the power radiated in one turn and the particle's rest energy. Not surprisingly, \(\alpha_{\tau}\) is the ratio of two Lorentz's invariants.

#### Radiation Integrals

The partition numbers can be cast in the form of integrals of functions--so-called _radiation integrals_--which depend only from the accelerator optics. In case of combined dipole magnets, the radiation integrals are:

\[\begin{array}{l}I_{2}=\oint\frac{ds}{R^{2}},\quad I_{3}=\oint\frac{ds}{|R^{3}|},\quad I_{4}=\oint ds\frac{D_{x}}{R}\left(\frac{1}{R^{2}}+2k\right),\quad I_{5}= \oint ds\frac{H_{x}}{|R^{3}|}\\ \\ \Rightarrow J_{x}=1-\frac{I_{4}}{I_{2}},\quad J_{y}=1,\quad J_{\epsilon}=2+ \frac{I_{4}}{I_{2}}\end{array} \tag{8.33}\]

The damping times become:

\[\tau_{x}=\frac{3T_{0}}{R\gamma^{3}}\frac{1}{I_{2}-I_{4}},\quad\tau_{y}=\frac{3T _{0}}{R\gamma^{3}}\frac{1}{I_{2}},\quad\tau_{\epsilon}=\frac{3T_{0}}{R\gamma^{ 3}}\frac{1}{2I_{2}+I_{4}} \tag{8.34}\]

For completeness, we report some other relevant quantities at equilibrium, for ultra-relativistic electrons:

\[\epsilon_{x}=C_{e}\gamma^{2}\frac{I_{5}}{I_{2}-I_{4}}=C_{e}\frac{\gamma^{2}}{ J_{x}}\frac{I_{5}}{I_{2}}\]

\[\sigma_{\delta}^{2}=C_{e}\gamma^{2}\frac{I_{3}}{2I_{2}+I_{4}}=C_{e}\frac{\gamma ^{2}}{J_{\epsilon}}\frac{I_{3}}{I_{2}} \tag{8.35}\]

\[U_{0}=\frac{\epsilon_{c}^{2}}{6\pi\epsilon_{0}}\gamma^{4}I_{2}\]

Synchrotrons based on separate function magnets (\(k=0\)), or with small quadrupole gradients compared to the dipoles' weak focusing \(k\ll R^{-2}\), show \(J_{x}\approx 1\), \(J_{y}=1\) and \(J_{\epsilon}\approx 2\). Therefore, they naturally provide damping in all the 3 planes of motion.

In general, the partition numbers can be tuned through diverse techniques, among which the most common are recalled below.

* Gradient ("Robinson") wiggler magnet. This is a few-poles wiggler magnet aiming at reducing \(J_{\epsilon}\to 2\) while increasing \(J_{x}\to 1\). Equation 8.33 shows that this can be obtained by increasing \(I_{2}\). But, at the same time, \(I_{4}\) has to be kept small, which implies \(2D_{x}k/R<0\) or equivalently \(\frac{D_{x}}{B_{y}}\frac{dB_{y}}{dx}<0\). Namely, in each wiggler pole, the dipolar field and the field gradient have to show opposite sign. A symmetric distribution of the magnetic components allows the dispersion to be closed at the end, and the initial direction of motion to be preserved.
* Dipole ("damping") wiggler magnet. One or multiple wigglers with pure dipolar field are installed in the ring to stimulate additional emission of radiation. This basically shortens the damping times in all planes, but depending upon the wiggler field, if installed in a dispersive region or not, it can either enlarge or reduce the emittances at equilibrium.

* Variation of the RF frequency. A small variation of the main RF above transition energy determines a variation of the orbit length according to \(\frac{df}{f}=-\frac{dL}{L}=-\alpha_{c}\,\frac{dE}{E_{0}}\). The orbit shift inside quadrupole magnets can be such that the emission of synchrotron radiation in those magnets is enhanced (feed-down dipole effect), contributing to additional damping.

#### 8.1.5.1 Discussion:To Be or Not to Be at Equilibrium?

What is the typical damping time of an electron beam in a 3 GeV, 0.6 km-long storage ring, in the presence of synchrotron radiation emission from 0.6 T dipole magnets? What is the damping time of a proton beam at the same total energy, assuming the same dipole magnet curvature radius? Estimate the damping time of the proton beam stored in the 27 km-long circumference of LHC, at 7 TeV total energy and assuming a dipole field of 20 T.

Since the damping times are of the order of the time needed for the particle to exhaust its total energy by radiation emission, their approximate value is \(T_{0}E_{0}/U_{0}\), where for the electrons \(T_{0}=0.6\) km/c = 2 \(\mu\)s and \(U_{0}=429\) keV from Eq. 7.14. We find \(\tau_{e}\approx 14\) ms.

Since \(\tau\approx T_{0}E_{0}/U_{0}\approx E_{0}/P_{0}\), Eq. 7.12 allows us to write \(\tau\sim\frac{E_{0}R^{2}}{\beta^{4}\gamma^{4}}\). For same total energy and curvature radius of the dipole magnets, the ratio of proton and electron beam damping time is:

\[\frac{\tau_{p}}{\tau_{e}}=\frac{\beta_{c}^{4}\gamma_{e}^{4}}{\beta_{p}^{4} \gamma_{p}^{4}}=\left(\frac{\gamma_{e}^{2}-1}{\gamma_{p}^{2}-1}\right)^{2} \approx 10^{13}\ \ \Rightarrow\ \ \tau_{p}=10^{13}\tau_{e}\approx 10^{11}s\sim 6000\ y \tag{8.36}\]

Hence, proton beams in a storage ring at multi-GeV total energy do not reach an equilibrium distribution in the sense of Eq. 8.4 because the emission of synchrotron radiation is suppressed by the large proton's rest mass.

The amount of energy radiated in the dipole magnets of LHC is calculated by means of Eq. 7.13, where \(R_{p}=E_{p}[GeV]/(0.3\cdot B_{y}[T])=1167\) m, and protons are in the ultra-relativistic limit (\(\beta\approx 1\)). We find \(U_{p}\approx 16\) keV and \(\tau_{p}\approx(27km/c)\cdot 7TeV/16keV=39375\) s \(\approx 11\) h.

#### 8.1.5.2 Discussion:Equilibrium Emittance of Multi-bend Lattices

How does the equilibrium emittance depend from the number of dipoles? What is the relationship between equilibrium emittance, momentum compaction, and betatron tune? Consider the equilibrium horizontal emittance in Eq. 8.22 in the approximation of small bending angle (\(\theta_{b}=l_{b}/R\ll 1\)), beam waist (\(\alpha_{x}\approx 0\)), and constant effective betatron function inside the dipoles of an isomagnetric lattice.

The dispersion function and its first derivative are both proportional to the bending angle (see Eq. 4.74, although their specific value depends on the magnetic focusing in between dipole magnets):\[\frac{\langle H_{x}\rangle_{B}}{R}\approx\frac{1}{R}\left(\frac{1}{\beta_{x}} \langle D_{x}^{2}\rangle+\beta_{x}\langle D_{x}^{2}\rangle\right)\approx\frac{ \theta_{b}}{l_{d}}\left[\frac{l_{d}^{2}\theta_{b}^{2}}{4\beta_{x}}+\beta_{x} \theta_{b}^{2}\right]\approx\theta_{b}^{3}\left(\frac{l_{d}}{\beta_{x}}+ \frac{\beta_{x}}{l_{d}}\right)\approx\left(\frac{2\pi}{N_{d}}\right)^{3}\]

\[\Rightarrow\epsilon_{x,eq}=F\frac{C_{e}}{J_{x}}\frac{\gamma^{2}}{N_{d}^{3}} \tag{8.37}\]

where commonly the average betatron function in dipoles \(\beta_{x}\approx l_{d}\), \(\langle D_{x}\rangle\approx\theta_{b}l_{b}\), \(N_{d}\) is the number of dipoles in the ring, and \(F\) a constant of the order of unity which depends from the specific lattice design. In short, a large number of dipole magnets keeps the individual bending angle small, thus a small dispersion function is generated and, eventually, a small equilibrium emittance.

Equation 8.37 allows us to write:

\[\frac{\langle H_{x}\rangle}{R}\approx\theta_{b}^{3}\approx\frac{\langle D_{x} \rangle^{3}}{l_{d}^{3}}\approx\frac{\langle D_{x}\rangle^{3}}{\beta_{x}^{3}} \approx\frac{\langle D_{x}^{2}\rangle}{\beta_{x}\,R}\ \ \Rightarrow\ \ R\approx\frac{\beta_{x}^{2}}{\langle D_{x}\rangle} \tag{8.38}\]

The definition of betatron tune and momentum compaction is recalled, and the result of Eq. 8.38 is used to find:

\[\begin{array}{l}Q_{x}=\frac{1}{2\pi}\oint\frac{ds}{\beta_{x}}\approx\frac{R }{\langle\beta_{x}\rangle}\approx\frac{\langle\beta_{x}\rangle}{\langle D_{x }\rangle}\\ \\ \alpha_{c}=\frac{1}{C}\oint\frac{D_{x}}{R}ds\approx\frac{\langle D_{x}\rangle}{R }\approx\frac{\langle D_{x}\rangle^{2}}{\beta_{x}^{2}}\approx\frac{1}{Q_{x}^{ 2}}\end{array} \tag{8.39}\]

Equation 8.39 is more and more accurate for \(C\approx 2\pi\,R\) (like in multi-bend lattices with respect to double or triple-bend achromatic cells), \(\langle D_{x}^{2}\rangle\approx\langle D_{x}\rangle^{2}\) (i.e., the standard deviation of the dispersion function is small), and the average betatron function along the lattice is comparable to the average one in the dipoles, \(\langle\beta_{x}\rangle\approx\beta_{x}\). The peculiarity of small \(\langle D_{x}\rangle\) in multi-bend lattices implies \(\alpha_{c}\) typically one order of magnitude smaller than in double-bend lattices, and consequently \(\sim 3\)-times larger horizontal betatron tune.

#### Vlasov-Fokker-Planck Equation

Vlasov's equation, in Eq. 5.37, describes a Hamiltonian system. The Vlasov-Fokker-Planck's (VFP) equation is the extension of Vlasov's equation to the presence of dissipative and random perturbative forces [2].

In a synchrotron light source, radiation damping plays the role of a dissipative force that, in the absence of quantum excitation, would lead to the collapse of the phase space volume. According to it, the beam phase space density distribution function increases with time. For example, in the longitudinal plane \(\frac{d\psi}{dt}=2\alpha_{e}\,\psi\). Quantum excitation relies on the random emission of photons, and as such it leads to particles diffusion in the phase space, or \(\frac{d\psi}{dt}=D\frac{d^{2}\psi}{dE^{2}}\), with \(D\) a diffusion coefficient.

The VFP equation for the longitudinal plane in the presence of these two perturbations to the single particle's Hamiltonian motion becomes:

\[\frac{d\psi}{dt}=\frac{\partial\psi}{\partial t}+\frac{\partial\psi}{\partial q} \dot{q}+\frac{\partial\psi}{\partial p}\,\dot{p}=\frac{\partial\psi}{\partial t }+\frac{\partial\psi}{\partial\phi}\dot{\phi}+\frac{\partial\psi}{\partial E} \dot{\epsilon}=2\alpha_{\epsilon}\psi\,+\,D\frac{d^{2}\psi}{dE^{2}} \tag{8.40}\]

where we introduced the reduced variables (\(\phi\), \(\epsilon\)).

The generic particle's energy deviation is the sum of the RF energy gain and the energy loss by synchrotron radiation emission. It is expanded at first order in the particle's energy, and its time-derivative taken on average in a turn:

\[\dot{\epsilon}\approx\frac{1}{T_{0}}\left(q\,V(\phi)-\frac{dU}{dE}\epsilon \right)\equiv\dot{\epsilon}_{0}-2\alpha_{\epsilon}\epsilon \tag{8.41}\]

By replacing Eq. 8.41 into Eq. 8.40:

\[\frac{\partial\psi}{\partial t}+\frac{\partial\psi}{\partial\phi}\dot{\phi}+ \frac{\partial\psi}{\partial E}\dot{\epsilon}_{0}=2\alpha_{\epsilon}\psi\,+ \,\frac{\partial\psi}{\partial E}2\alpha_{\epsilon}\epsilon\,+\,D\frac{d^{2} \psi}{dE^{2}} \tag{8.42}\]

The l.h.s. of Eq. 8.42 describes the Hamiltonian flow of the distribution function in the absence of radiation damping and quantum excitation, thus it must vanish by virtue of Eq. 5.37. The r.h.s. can be written as the first partial derivative with respect to the particle's energy, whose argument therefore has to be independent from energy:

\[\begin{array}{l}2\alpha_{\epsilon}\frac{\partial}{\partial E}\left(\psi\, \epsilon\,+\,\frac{D}{2\alpha_{\epsilon}}\frac{\partial\psi}{\partial E} \right)=0;\\ \psi\epsilon\,+\,\frac{D}{2\alpha_{\epsilon}}\frac{\partial\psi}{\partial E}= f(\phi);\\ \Rightarrow\psi\left(\phi\,,\epsilon\right)=F(\phi)e^{-\frac{1}{2}\frac{ \epsilon^{2}}{\left(D/2\alpha_{\epsilon}\right)^{2}}}\equiv F(\phi)e^{-\frac{ 1}{2}\left(\frac{\epsilon}{\alpha_{E}}\right)^{2}}\end{array} \tag{8.43}\]

We have found that the stationary solution of Eq. 8.40, the so-called "equilibrium" distribution, is Gaussian in the energy coordinate.

The explicit form of \(F(\phi)\) is obtained below by recurring to the Hamiltonian for the longitudinal motion. At equilibrium, this has to correspond to oscillations in the longitudinal phase space (\(z\), \(\epsilon\)), see Eq. 8.2. We re-write the equations of motion for the relative energy deviation and the longitudinal coordinate internal to the bunch, for the independent variable \(s\) along the accelerator:

\[\left\{\begin{array}{l}z=-\frac{1}{\kappa_{z}}\sqrt{2J_{z}}\cos(\kappa_{z} s)\\ \delta=\sqrt{2J_{z}}\sin(\kappa_{z}s)\end{array}\right.,\;\;\kappa_{z}:=\frac{ \Omega_{s}}{\alpha_{c}c} \tag{8.44}\]

It is straightforward to verify that these equations obey the Hamiltonian \(H=J_{z}=\left(\frac{\kappa_{z}^{2}z^{2}+\delta^{2}}{2}\right)\). The longitudinal rms emittance results \(\epsilon_{z}=\sigma_{z}\sigma_{\delta}=\frac{\left\langle J_{z}\right\rangle }{\kappa_{z}}\).

Since the system at equilibrium behaves as a Hamiltonian system, then according to the theorem in Eq. 5.39 the phase space distribution function has to be function of the Hamiltonian only, or \(\psi\left(z,\delta\right)=\psi\left(J_{z}\right)\). Owing to the fact that it must be Gaussian in the energy coordinate as previously found in Eq. 8.43, it follows:

\[\psi\left(z,\delta\right)=\psi\left(J_{z}\right)=\psi\left(0,0\right)e^{-\frac{ 1}{2}\left(\frac{x_{z}^{2}\cdot z^{2}}{x_{z}^{2}\cdot\sigma_{z}^{2}}+\frac{ \beta^{2}}{\sigma_{\delta}^{2}}\right)} \tag{8.45}\] \[\Rightarrow\psi\left(J_{z}\right)=\frac{1}{2\pi\sigma_{z}\sigma _{\delta}}e^{-\frac{1}{2}\left(\frac{x_{z}^{2}}{\sigma_{z}^{2}}+\frac{\delta^{ 2}}{\sigma_{\delta}^{2}}\right)}\] \[\iint_{-\infty}^{+\infty}\psi\left(J_{z}\right)dzd\delta\equiv 1\]

Since radiation damping and quantum excitation behave similarly in the transverse planes, the same kind of stationary distribution function is expected in the transverse phase spaces:

\[\psi\left(u,u^{\prime}\right)=\frac{1}{2\pi\sigma_{u}\sigma_{u^{\prime}}}e^{- \frac{1}{2}\left(\frac{u_{u}^{2}}{\sigma_{u}^{2}}+\frac{u^{\prime 2}}{\sigma_{u^{ \prime}}^{2}}\right)},\ \ \ u=x,\ y \tag{8.46}\]

Alternatively, a Gaussian distribution function for the stationary state can be predicted by recurring to the Central Limit theorem, because emission of synchrotron radiation can be intended as an incoherent perturbation to the large population of particles in a bunch, as long as collective effects like particles interaction via self-induced fields are ignored (see later), and the linear approximation \(U=U(\epsilon)\) allows one to neglect deformations of the Gaussian tails by nonlinear diffusive processes.

### Lifetime

#### Quantum Lifetime

The 2-D phase space distribution function in a synchrotron tends to a Gaussian over a time scale of few damping times. In the most general case of non-zero correlation between particles' position and angle (\(u,u^{\prime}\)), it results [4]:

\[\rho\left(u,u^{\prime}\right)=\frac{1}{2\pi\sigma_{u}\sigma_{u^{\prime}}}e^{- \left(\frac{u^{2}}{2\sigma_{u}^{2}}+\frac{uu^{\prime}}{2\sigma_{u}u^{\prime}} +\frac{u^{\prime 2}}{2\sigma_{u^{\prime}}^{2}}\right)}=\frac{1}{2\pi\epsilon}e^{- \frac{1}{2\epsilon}\left(\gamma u^{2}+2\alpha uu^{\prime}+\beta u^{\prime 2} \right)}=\frac{1}{\langle W\rangle}e^{-\frac{W}{\langle W\rangle}} \tag{8.47}\]

All variables above are intended in the \(u\)-phase space, and the suffix is suppressed for brevity of notation. \(W=2J\) is the single particle invariant defined by \(2J=\pi\left(\gamma u^{2}+2\alpha uu^{\prime}+\beta u^{\prime 2}\right)\), \(\langle J\rangle=\pi\,\epsilon\), and \(\epsilon=\sigma_{u}\sigma_{u^{\prime}}\) is the beam rms emittance (see also Eq. 4.132). The distribution function is normalized to unity, \(\int_{0}^{\infty}\rho\left(W\right)dW=1\).

A Gaussian distribution has infinitely long tails. If no limitation is imposed, a stationary situation exists where the number of particles crossing an arbitrary boundary due to quantum excitation (rate Q) equals the number of particles entering due to radiation damping (rate D). If a limitation is present in correspondence of the oscillation amplitude \(W_{c}\), instead, the rate of lost particles crossing the border determines the beam "lifetime" (rate L). Because of the cut to the Gaussian tails, the distribution will be modified to some extent. But, if the restriction is far enough from the beam core (\(W_{c}\gg\epsilon\)), the rate can be calculated by assuming that the number of lost particles per turn is small, and therefore the distribution is still approximately Gaussian. Consequently, we can assume that the number of particles crossing \(W_{c}\) and being lost, is very nearly the same as if there were no limitations. In particular, the particle loss rate can be estimated as that due to radiation damping (rate \(\mathrm{L}\approxeq\) rate \(\mathrm{Q}=\mathrm{rate}\)\(\mathrm{D}\)).

To calculate the loss rate at \(W=W_{c}\), we first consider the fraction of particles in the differential amplitude range \(dW\), i.e., \(dN(W)=N\rho dW\). The variation of the amplitude due to radiation damping in a characteristic damping time \(\tau\) is \(W(t)=\hat{W}e^{-\frac{2t}{\tau}}\). By virtue of Eq. 8.47 we have:

\[\left\{\begin{array}{l}\frac{dN}{dW}=\frac{N}{\left\langle W\right\rangle}e ^{-\frac{W}{\left\langle W\right\rangle}}\\ \\ \frac{dW}{dt}=-\frac{2}{\tau}W\end{array}\right. \tag{8.48}\]

\[\left(\frac{dN}{dt}\right)_{W_{c}}=\left(\frac{dN}{dW}\frac{dW}{dt}\right)_{ W_{c}}=-\frac{2N}{\tau}\frac{W_{c}}{\left\langle W\right\rangle}e^{-\frac{W_{c}}{ \left\langle W\right\rangle}}\Rightarrow\left\{\begin{array}{l}N\left(t \right)=N_{0}e^{-\frac{t}{t_{q}}}\\ \\ \tau_{q}=\frac{\tau}{2}\frac{\left\langle W\right\rangle}{W_{c}}e^{\frac{W_{ c}}{\left\langle W\right\rangle}}=\frac{\tau}{2}\frac{e^{b}}{\xi}\end{array}\right. \tag{8.49}\]

The stored current decays exponentially with time. The characteristic constant of decay, \(\tau_{q}\), is called _quantum lifetime_ and it is equal to the damping time multiplied by a large factor, in proportion to \(\xi_{u}:=\frac{W_{c}}{\left\langle W\right\rangle}=\frac{\pi u_{max}^{2}/\beta _{u}}{2\pi\epsilon_{u}}=\frac{u_{max}^{2}}{2\sigma_{u}^{2}}\). This brings to the golden rule for the ratio of accelerator acceptance and beam size \(\frac{u_{max}}{2\sigma_{u}}\geq 6.5\) to guarantee \(\tau_{q}\geq 100\) h. By virtue of the description of particle's longitudinal motion through C-S parameters (see Eq. 4.141), a quantum lifetime can be similarly defined for the motion in the longitudinal phase space in the presence of a dynamical boundary.

In summary, quantum lifetime origins in quantum fluctuations of the particles' energy due to photon emission. Owing to the extension of the Gaussian tails of the charge distribution in phase space, particles can exceed the transverse and/or the longitudinal acceptance of the accelerator, thus reducing the stored current with time. The two distinct cases of transverse and longitudinal limitation to the particles' motion are treated below.

#### Dynamic Aperture

In the transverse planes, let \(u_{max}\) be the physical aperture of the vacuum chamber and \(\xi_{x,y}\) the smallest ratio of aperture and beam size along the accelerator. In common situations \(\xi_{x,y}>5\), hence \(\tau_{q}\) is very large for any practical purpose. However, nonlinearities in the particles' motion can substantially reduce the beam's lifetime.

This can be understood by recalling the definition of _dynamic aperture_ (DA) as the region in the configuration space (\(x\), \(y\)) within which particle's motion remains stable for a sufficiently long time. In other words, the DA (in each plane of motion,respectively) is the initial amplitude (\(A_{i}\)) in correspondence of which the particle's motion is unbounded (\(A_{f}\rightarrow\infty\)) after a large number of turns, as shown in Fig. 8.2.

The DA can either exceed or stay within the vacuum chamber. Particle dynamics is said to be linear when, despite the presence of nonlinear magnetic elements and machine errors, the correlation of final and initial amplitude of oscillation is approximately linear, or \(A_{f}\propto A_{i}\). In this case \(\tau_{q}\) is determined by physical restrictions. If the motion is nonlinear, instead, \(A_{f}\propto(A_{i})^{n}\), \(n>1\). Particles will always be lost on the chamber, but losses will now concern particles at smaller initial amplitudes (roughly speaking, particles closer to the bunch core). In this case \(\tau_{q}\) is dominated by the DA. The smallest limitation among physical restriction and DA constitutes the _transverse acceptance_ of the accelerator.

#### Overvoltage

In the longitudinal plane, \(\xi_{\epsilon}=\frac{\delta_{acc}^{2}}{2\sigma_{\delta}^{2}}\). Since \(\sigma_{\delta}\) is basically determined by the beam's energy and the dipole's bending radius (see Eq. 8.11), \(\tau_{q}\) can be made larger by a larger RF energy acceptance [3]. According to Eq. 4.37, \(\delta_{acc}\) can be made larger in turn by a larger peak RF voltage \(e\hat{V}\), well in excess of the energy loss per turn \(U_{0}\). The ratio \(q:=\frac{e\hat{V}}{U_{0}}>1\) is denominated _overvoltage_ factor.

At first, we express \(\delta_{acc}(\psi)\) as function of \(q\) for the synchronous phase, where the energy gain per turn provided by the RF to balance the energy loss is \(|\Delta\,E|=|e\hat{V}\cos\psi_{s}|\equiv|U_{0}|\). We also recall the peak accelerating gradient as the time derivative of the peak accelerating voltage, see Eq. 8.13. Then we have:

\[\left\{\begin{array}{l}\psi_{s}=\arccos(\frac{U_{0}}{e\hat{V}})=\arccos(1/ q)\\ \\ e\hat{V}=-\omega_{RF}e\hat{V}\sin\psi_{s}\end{array}\right. \tag{8.50}\]

Figure 8.2: Correspondence of initial (\(A_{i}\)) and final oscillation amplitude (\(A_{f}\)) in the presence of linear (blue) and nonlinear motion (red)

From the second equation:

\[\frac{(e\dot{V})^{2}}{\omega_{RF}^{2}}=(e\dot{V})^{2}(1-\cos^{2}\psi_{s})=(e\dot{ V})^{2}-\Delta E^{2}=U_{0}^{2}\left(q^{2}-1\right) \tag{8.51}\]

The RF energy acceptance (squared) defined in Eq. 4.37 is re-written:

\[\begin{split}\delta_{acc}^{2}&\approx 2\frac{e\dot{V}} {\pi h\alpha_{c}E_{0}}\left[(\psi_{s}-\pi)\cos\psi_{s}-\sin\psi_{s}\right]= \frac{2}{\pi h\alpha_{c}E_{0}}\left(-\psi_{s}\Delta E+e\dot{V}/\omega_{RF} \right)=\\ &=\frac{U_{0}}{\pi h\alpha_{c}E_{0}}2\left[\sqrt{q^{2}-1}-\arccos (\frac{1}{q})\right]\equiv\frac{U_{0}}{\pi h\alpha_{c}E_{0}}F(q)\end{split} \tag{8.52}\]

If we substitute \(U_{0}\) from dipole magnets only (see Eq. 7.13) into Eq. 8.52, we obtain the following expression for electrons in an isomagnetic lattice:

\[\begin{split}\xi_{\epsilon}&=\frac{\delta_{acc}^{2} }{2\sigma_{\delta}^{2}}=\left|\frac{J_{e}R}{2C_{e}\gamma^{2}}\frac{U_{0}}{\pi h \alpha_{c}E_{0}}F(q)\right|=\frac{64}{55\sqrt{3}}\frac{r_{e}}{\hbar c}\left| \frac{J_{e}E_{0}}{\hbar\alpha_{c}}F(q)\right|\approx\left|\frac{J_{e}}{\hbar \alpha_{c}}\right|F(q)E_{0}[GeV]\end{split} \tag{8.53}\]

In most practical cases, \(\left|\frac{J_{e}}{\hbar\alpha_{c}}\right|\approx 1-10\).

Figure 8.3 shows \(F(q)\) and \(\delta_{acc}\) as function of \(q\) (Eq. 8.52) for typical parameters of a medium energy low-emittance electron storage ring. The presence of nonlinearities in the longitudinal dynamics, e.g. due to higher order momentum compaction, can easily reduce the RF energy acceptance predicted at first order by a factor 2-3. For this reason, the total RF peak voltage is commonly sized to have \(q>4\), which still guarantees \(\delta_{acc}/\sigma_{\delta}>10\) for usual values of the energy spread \(\sigma_{\delta}\approx 0.1\%\).

Because of linear and nonlinear chromaticity, the betatron motion of off-energy particles can be distorted so that, at the end, the off-energy DA is smaller than the on-energy DA. If the off-energy DA contributes to particles loss more than the RF bucket height does, the quantum lifetime is determined by an effective "DA energy acceptance", usually named _momentum acceptance_.

According to Eq. 8.52, the contribution to \(U_{0}\) by magnetic elements other than dipoles, such as wigglers and undulators (discussed in the following section), enlarges the RF energy acceptance, thus the quantum lifetime. The same result is obtainedwith a small \(|\alpha_{c}|\). This, however, tends to shorten the acceptance in phase, i.e., the bunch length at equilibrium. From Eq. 8.14:

\[\begin{array}{l}\sigma_{t,eq}^{2}=2\pi\frac{55}{32\sqrt{3}}\frac{\hbar}{m_{c}c }\gamma^{3}\frac{\alpha_{c}m_{c}c^{2}}{J_{\epsilon}\epsilon V_{0}}=2\pi\frac{55 }{32\sqrt{3}}\frac{\hbar}{m_{c}c}\gamma^{3}\frac{\alpha_{c}m_{c}c^{2}}{J_{ \epsilon}\sqrt{q^{2}-1}}\frac{1}{\omega_{RF}V_{0}}=\\ =2\pi\frac{55\sqrt{3}}{64}\frac{\hbar}{r_{e}m_{c}c^{2}}\frac{2\pi R}{2\pi cJ} \frac{\alpha_{c}}{J_{\epsilon}\omega_{RF}\sqrt{q^{2}-1}}=\left(\frac{55\sqrt{3 }}{64}\right)\frac{\hbar c}{2\pi r_{e}}\frac{T_{0}}{E_{0}}\frac{\alpha_{c}}{J_ {\epsilon}\omega_{RF}\sqrt{q^{2}-1}}\end{array} \tag{8.54}\]

Since \(\sigma_{t,eq}\propto\sqrt{\frac{\alpha_{c}}{q}}\), a trade-off for the value of \(q\) is usually found to maximize the RF bucket _area_.

#### 8.2.3.1 Discussion: Transverse and Longitudinal Acceptance of a Light Source

Which of the three planes of motion first limits the quantum lifetime, if the on-energy dynamic aperture is larger than the vacuum chamber, and we neglect the contribution from the off-energy dynamic aperture? For a quantitative discussion, let us consider typical parameters of a medium-energy storage ring light source, such as 2.5 GeV beam energy, 0.2 nm rad horizontal emittance. The ellipsoidal vacuum chamber has size 12 \(\times\) 4 mm, \(\langle\beta_{x}\rangle\approx\langle\beta_{y}\rangle\approx\) 10 m, the coupling factor is \(\sim\) 1%, \(h\alpha_{c}\approx\) 0.05, and \(\sigma_{\delta}=0.1\)%. We also assume a dipole bending radius of 10 m, and \(J_{\epsilon}\approx\) 2. The total peak RF voltage is 2 MV.

In the transverse planes:

\[\begin{array}{l}\xi_{y}=\frac{1}{2}\left(\frac{v_{max}}{\sigma_{y}}\right)^{ 2}\approx\frac{1}{2}\left(\frac{x_{max}/3}{\sigma_{x}/10}\right)^{2}\approx 9 \xi_{x},\\ \\ \xi_{x}=\frac{1}{2}\left(\frac{12\,mm}{\sqrt{\epsilon_{x}p_{x}}}\right)^{2} \approx 3.6\cdot 10^{4}\end{array} \tag{8.55}\]

In the longitudinal plane, \(\xi_{\epsilon}=\frac{2\cdot 2\cdot 5}{0.05}F(q)\). The overvoltage function is evaluated for \(U_{0}\approx 88.45\frac{E^{4}[GeV]}{R[m]}=346\) keV, therefore \(q=5.8\). According to Fig. 8.3, \(F(q)\approx\) 8, thus \(\xi_{\epsilon}\approx 800\). The quantum lifetime results dominated by the longitudinal dynamics.

Let us now consider the off-energy DA and assume that, for example, it is internal to the vacuum chamber for relative energy deviation \(|\Delta|\geq 2\)%. Since it results \(\delta_{acc}=\sqrt{\left|\frac{U_{0}F(q)}{\pi\,h\alpha_{c}E}\right|}\approx 8\)%, we expect the quantum lifetime to be dominated by momentum acceptance (still true in case of 3-fold reduction of \(\delta_{acc}\) by nonlinearities in the longitudinal phase space).

#### 8.2.4 Residual Gas Interactions

The beam lifetime can be degraded by scattering of stored particles on residual gas in the vacuum chamber. Such interaction is comprehensive of the following effects [4].

* _Large angle elastic (Coulomb) scattering_, which causes particles loss if the scattered particles hit a transverse physical aperture, or they are pushed outside the dynamic aperture. The interaction is described by the Rutherford's cross section: \[\sigma\approx\frac{r_{e}^{2}Z^{2}}{(\gamma^{2}\theta^{2})},\] (8.56) with \(Z\) the atomic number of the gas ion and \(\theta\) the scattering angle. The latter can be expressed, with analogy to Eq. 6.2, as function of the maximum lateral displacement set by the vacuum chamber (\(A\)), the betatron function at the position of the aperture limitation, and the average betatron function along the ring (assuming a distributed interaction): \[\sigma_{el}=\frac{2\pi r_{e}^{2}Z^{2}}{\gamma^{2}}\frac{\langle\beta\rangle \beta_{A}}{A^{2}}\] (8.57)
* _Inelastic scattering_, for electrons only, comprehensive of Bremsstrahlung (the electron is scattered by the atomic nucleus and emits a photon, but the atom is left unexcited) and inelastic atomic scattering (scattering by an atomic electron, the atom is left excited). Both these processes generate large particle's energy loss, possibly exceeding the RF or the momentum acceptance. The cross section is \(\sigma\propto 4r_{e}^{2}Z^{2}\alpha\) (\(\alpha=1/137\) the fine structure constant), weakly dependent from the particle's energy.
* _Ion trapping_, i.e., the production of ions from scattering of electrons on residual gas. Ions are accumulated in specific regions of the vacuum chamber by focusing imposed by the circulating beam, until the repulsive space charge force of the ions starts limiting the ions concentration. The focusing strength imposed to the ions can be calculated with a formalism analogue to that used in colliders for evaluating the beam-beam tune shift (see later). We anticipate that, if \(\Delta u^{\prime}=a_{x,\,y}u\) is the angular divergence acquired by the ion as function of its (small) lateral distance \(u\) from the stored beam's axis, and \(a_{k}\) is the (linearized) focusing strength, we have: \[a_{x,\,y}=\frac{Z}{A}\frac{2r_{p}N_{e}}{\sigma_{x,\,y}(\sigma_{x}+\sigma_{y})}\] (8.58) where \(r_{p}\), \(Z\), \(A\), \(N_{e}\) are, respectively, the classical proton radius, the ion atomic number, the ion mass number, and the number of electrons in a bunch of transverse sizes \(\sigma_{x,\,y}\) at the interaction point. The ions are accumulated if the succession of focusing kicks due to the periodic spacing \(s_{b}\) of the stored electron bunches is such to guarantee a periodic motion. That is, the transfer matrix of the ions' motion has to have \(|Tr(M)|<2\). This implies a critical ion atomic mass, so that only heavier species are trapped: \[\begin{array}{l}M=\begin{pmatrix}1&s_{b}\\ 0&1\end{pmatrix}\begin{pmatrix}1&0\\ -a_{x,\,y}&1\end{pmatrix}=\begin{pmatrix}1-a_{x,\,y}s_{b}&s_{b}\\ -a_{x,\,y}&1\end{pmatrix}\\ \\ \Rightarrow\left(\frac{A}{Z}\right)_{trap}>\left(\frac{A}{Z}\right)_{c}=\frac{2 r_{p}N_{e}s_{b}}{\sigma_{y}(\sigma_{x}+\sigma_{y})}\end{array}\] (8.59)where we have assumed \(\sigma_{x}\gg\sigma_{y}\). Electron storage rings often implement "dark gaps" (several consecutive empty RF buckets) in the filling pattern to avoid stable resonances of the ions' motion.
* _Inelastic nuclear scattering_, for protons only, generates beam loss through nuclear reactions.

Gas scattering is counteracted with pumping systems aimed at obtaining low vacuum pressures (these can be as low as \(10^{-10}\) bar in a storage ring). This is accompanied by a suitable preparation of the vacuum chamber to minimize desorption of gas molecules from the surface as possibly induced by synchrotron radiation.

If \(\sigma\) is the cross section of the interaction of ultra-relativistic beam particles with residual gas, and \(n_{g}\) is the number of gas atoms per unit volume, the number of particles traversing the unit volume is:

\[dN=-N\sigma n_{g}cdt\quad\Rightarrow\quad N(t)=N_{0}e^{-\frac{t}{c\sigma n_{g }}}\equiv N_{0}e^{-\frac{t}{t_{g}}} \tag{8.60}\]

If the gas concentration contains \(n_{i}\) molecules of type \(i\), each molecule of type \(i\) made of \(k_{i,\,j}\) atoms, then the total beam-gas scattering lifetime associated to the interaction \(\sigma\) becomes:

\[\tfrac{1}{\tau_{g}}=c\sigma n_{g}=c\sigma\,\sum_{i,\,j}k_{i,\,j}n_{i}=c\sigma \,\sum_{i,\,j}k_{i,\,j}\,\tfrac{p_{i}}{KT}=\tfrac{c\sigma}{KT}\sum_{i,\,j}k_ {i,\,j}\,p_{i} \tag{8.61}\]

#### Touschek Lifetime

Touschek scattering, first explained by B. Touschek after the observation of current loss in the ADA storage ring in Frascati (Italy), describes the scattering of charged particles in the same bunch [4].

Collisions internal to the bunch happen all the time in all directions. However, if observed in the reference frame of the beam center of mass (c.m.), the particles' motion appears almost exclusively in the transverse planes (the longitudinal velocity relative to the c.m. being almost zero). In other words, the transverse momenta are much larger than the longitudinal one. _Touschek scattering_ is momentum transfer from the transverse to the longitudinal plane in occasional large angle elastic scattering, which can therefore push particles off the RF or the momentum acceptance. Momentum transfer from the longitudinal to the transverse planes is also present (intrabeam scattering), but it is usually not harmful for the beam lifetime, as discussed in the following Section.

If two particles collide in the c.m. frame transferring their transverse momentum \(\vec{p}^{\,\prime}_{i}=(p^{\,\prime}_{x},0)\) to longitudinal momentum \(\vec{p}^{\,\prime}_{f}=(0,\,p^{\,\prime}_{z})=(0,\,p^{\,\prime}_{x})\), then the variation of longitudinal momentum in the laboratory frame is (see Fig. 8.4):\[\Delta p_{z}=p_{z,\,f}-p_{z,i}=\gamma\,(p^{\prime}_{z,\,f}+\frac{\beta}{c}E^{ \prime}_{f})-\gamma\,(p^{\prime}_{z,\,i}+\frac{\beta}{c}E^{\prime}_{i})=\gamma\, \Delta p^{\prime}_{z}+\gamma\,\frac{\beta}{c}\Delta E^{\prime}=\]

\[=\gamma\,(p^{\prime}_{z,\,f}-p^{\prime}_{z,\,i})=\gamma\,p^{\prime}_{x}=\gamma \,p_{x}=\gamma\,p_{z}\sigma_{u^{\prime}}\]

\[\Rightarrow\frac{\Delta p_{z}}{p_{z}}\approx\gamma\sqrt{\frac{\epsilon_{u}}{ \beta_{u}}},\ \ u=x,\,y \tag{8.62}\]

where we used Eqs. 1.17, 1.18 for the momentum transformation, \(\Delta E^{\prime}=0\) for elastic scattering, and the definition of angular divergence. The momentum transfer is boosted by a factor \(\gamma\) in the laboratory frame. If the normalized divergence \(\gamma\sigma_{u^{\prime}}\) exceeds the RF or the momentum acceptance, then the particle gets lost.

The cross section of Touschek scattering calculated in the c.m. frame is the Moller cross section in the non-relativistic approximation, \(\sigma_{T}\propto 2\pi\,r_{0}^{2}/\beta^{4}\). This suggests that the process is far more harmful for electron beams (lighter rest mass) with respect to proton beams, and that it is counteracted by a higher beam rigidity (\(\beta\to 1\)).

The Touschek lifetime is introduced by calculating the fraction \(dN_{loss}\) of particles lost with respect to the total number of particles \(dN_{V}\) contained in an element of volume \(\sigma_{T}dl\), whose charge density is \(\rho=dN_{V}/dV\):

\[\begin{array}{l}\frac{dN_{loss}}{dN_{V}}=\int\rho\sigma dl=\int\rho\sigma\, v_{z}dt\\ \\ \frac{dN_{loss}}{dt}=\frac{dN_{loss}}{dN_{V}}\frac{dN_{V}}{dt}=\int\rho\sigma \,v_{z}dN_{V}\approx\langle\sigma\,v_{z}\rangle\int\rho^{2}dV=\langle\sigma\, v_{z}\rangle\frac{N^{2}}{8\pi^{3/2}\sigma_{x}\sigma_{y}\sigma_{z}}\end{array} \tag{8.63}\]

The integration is done assuming a 3-D Gaussian distribution normalized to \(N\). The average \(\langle\sigma\,v_{z}\rangle\), which is assumed to be independent from the particles' coordinates, has to contain the information on the relative momentum deviation induced by the scattering or, equivalently, the normalized angular divergence as in Eq. 8.62. The additional dependence of the cross section from the beam's parameters can be inferred from the qualitative similarity with the scattering described by the Rutherford's cross section in Eq. 8.56. Here, \(Z=1\) and the scattering angle is replaced

Figure 8.4: Scattering of two particles transferring all their transverse momentum to longitudinal momentum, in the center of mass reference frame (left) and in the laboratory frame (right). The particle beam moves with a Lorentz factor \(\gamma\) in the laboratory

by the longitudinal acceptance \(\delta_{acc}\). We infer for the _Touschek lifetime_ (assuming scattering in one plane only):

\[\tfrac{1}{\tau}=\tfrac{1}{N}\,\tfrac{dN}{dt}\sim\tfrac{1}{\gamma\sigma_{x}^{ \prime}}\,\frac{cr_{0}^{2}}{\gamma^{2}\delta_{acc}^{2}}\,\frac{N}{V} \tag{8.64}\]

An exact derivation for flat and round beam, respectively, provides:

\[\left\{\begin{array}{l}\tfrac{1}{\tau_{fl}}\,\approx\,\frac{r_{0}^{2}c}{8\pi \gamma^{3}\sigma_{x}\delta_{acc}^{2}}\,\frac{N}{\sigma_{x}\sigma_{y}\sigma_{z} }\\ \\ \tfrac{1}{\tau_{ro}}\,\approx\,\frac{r_{0}^{2}c}{4\sqrt{\pi}\gamma^{4}\sigma _{x}^{\prime}\sigma_{y}\delta_{acc}}\,\frac{N}{\sigma_{x}\sigma_{y}\sigma_{z}} \end{array}\right. \tag{8.65}\]

By virtue of Eq. 8.60, the _total lifetime_ due to diverse beam-gas interactions and any other interaction of constant cross section, satisfies:

\[\tfrac{1}{\tau_{tot}}=\sum_{m}\,\tfrac{1}{\tau_{m}} \tag{8.66}\]

## References

* [1] R.P. Walker, Radiation damping, Quantum excitation and equilibrium beam properties, in _Proceedings of CERN Accelerator School: 5th General Accelerator Physics Course_, CERN 94-01, vol. I, ed. by S. Turner (Geneva, Switzerland, 1994), pp. 461-498
* [2] J.M. Jowett, Introductory statistical mechanics for electron storage rings, in _Lectures Given at the U.S. Summer School on Physics of High Energy Particle Accelerators_ (Stanford, CA, USA, 1985). Also SLAC-PUB-4033 (1986)
* [3] D.J. Thompson, D.M. Dykes, R.F. Systems, in _Synchrotron Radiation Sources--A Primer_, ed. by H. Winick (Published by World Scientific, Singapore, 1995), pp. 87-97. ISBN: 9810218567
* [4] C. Bocchetta, Lifetime and beam quality, in _Proceedings of CERN Accelerator School: Synchrotron Radiation and Free Electron Lasers_, CERN 98-04, ed. by S. Turner (Geneva, Switzerland, 1998), pp. 221-253

