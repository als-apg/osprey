The development of resonant accelerators during the first half of the XX century allowed particles to reach kinetic energies one order magnitude higher than in electrostatic devices. Nowadays, kinetic energies approaching and exceeding the GeV-level are obtained by means of Alvarez-type RF structures, or _RF cavities_.

The metallic boundary of the cavity allows the electric field to have a longitudinal component for acceleration. The internal geometry of the structure, often of cylindrical symmetry, is shaped according to the range of velocity exploited by the accelerated particles, and optimized to maximize the accelerating gradient per input e.m. power, while minimizing the probability of discharges ("breakdown rate"). RF structures installed in series and synchronized by an external common master clock constitute a linear accelerator or _RF linac_. Synchronization of RF field and particles' arrival time implies a pulsed time pattern of the particle beam, i.e., a series of bunches.

Linacs adopt RF structures made of many tens' of internal gaps, called _cells_. RF cavities made of single or few cells are used in high energy circular accelerators, where particles' energy is increased through the cavity over a large number of turns, thus a small amount of energy gain per turn is admitted.

In spite of the rotational electric field in RF structures, the process of energy gain can still be described in terms of an effective electric voltage \(\Delta V\), resulting from the path-integral of the time-varying electric field synchronized to the particle's arrival time, i.e., \(\Delta T=-q\,\Delta V\) as already in Eq.2. The total energy of a charged particle initially in a field-free region, accelerated in a RF structure, and eventually extracted from it, is:

\[E_{f}=E_{i}-q\,\Delta V=m_{0}c^{2}+T_{i}+\Delta T=m_{0}c^{2}+T_{f} \tag{3.1}\]

### Principles of Acceleration

#### Theorem of E.M. Acceleration

Acceleration of charged particles by an external rotational electric field requires boundary conditions which, as we will see below, can be satisfied by metallic surfaces. To demonstrate this [1], we first consider an e.m. wave in vacuum. The wave can also be thought as the _far field_ component of an e.m. field derived from the Lienard-Wiechert retarded potentials. Maxwell's equations imply the wave equation:

\[\nabla^{2}\vec{E}=-\frac{\omega^{2}}{c^{2}}\vec{E} \tag{3.2}\]

whose solutions can be expressed as sum of waves of the form:

\[\vec{E}(\vec{r},t)=\vec{E}_{0}e^{i(\vec{k}\vec{r}-\omega t)} \tag{3.3}\]

If \(\vec{k}\) is real, Eq. 3.3 describes plane waves travelling in the direction of \(\vec{k}\) at the velocity \(c\). By applying Maxwell's equation \(\vec{\nabla}\vec{E}=0\), we obtain \(\vec{k}\cdot\vec{E}=0\), that is, the electric field is perpendicular to the direction of propagation (transversely polarized wave). In summary, e.m. waves remain in phase only with particles travelling at ultra-relativistic velocities. If the particles are not travelling parallel to \(\vec{k}\), any interaction will be periodic and will add to nothing. If the particles move along \(\vec{k}\), there is not net acceleration because \(\vec{E}\) is orthogonal to the particle's velocity. This demonstrates the following theorem:

_no combination of far fields in free vacuum can produce net linear acceleration_.

A second demonstration, which recurs to energy conservation in Quantum Mechanics, is presented to point out that the theorem is not a consequence of a "classical" interpretation of Electromagnetism; rather, it concerns an intrinsic property of e.m. fields.

Far fields can be represented as a sum of photons. Acceleration of a charged massive particle by an e.m. field can be represented by the absorption of photons with no simultaneous emission. But, such an interaction is forbidden by energy and momentum conservation. This is evident if we calculate the Lorentz's invariant \(p^{\mu}p_{\mu}\) of the particle before and after photon absorption, in the reference frame in which the particle is initially at rest. The particle's 4-momentum after photon absorption would be \(p^{\mu}=(m_{0}c^{2}+E_{\gamma},\,\vec{p}_{\gamma})\), and thereby the particle's invariant results:

\[\begin{split} m_{0}^{2}c^{4}&=\left(m_{0}c^{2}+E_{ \gamma}\right)^{2}-p_{\gamma}^{2}c^{2};\\ p_{\gamma}^{2}c^{2}&=E_{\gamma}^{2}+2m_{0}c^{2}E_ {\gamma}\end{split} \tag{3.4}\]

Since for a photon \(p_{\gamma}c=E_{\gamma}\), Eq. 3.4 holds only for \(E_{\gamma}=0\). Thus, there cannot be such acceleration by far field.

The above theorem concerns "net, linear" acceleration. What do these terms mean?

* "Net" means that acceleration should be experienced by a particle moving from and to a region of space free of e.m. field. In other words, local acceleration internal to a region occupied by e.m. field is still possible. But, when the motion is across a field-free region, the average accelerating effect is null.
* "Linear" means that the particle's energy change is linear with the amplitude of the accelerating field. There can be second order effects (e.g., Compton scattering) in which the acceleration is proportional to the square of the field amplitude. These cases, however, require extremely intense fields which are not part of conventional RF accelerators.

#### 3.1.2 Pill-Box

Modern RF structures [2, 3] are made of several cells, delimited by "disks" with central irises to allow particles to pass through. The disks allow the e.m. wave flowing through the cavity to constructively interfere, so generating a longitudinal component of the electric field, parallel to the particle's direction of motion (waveguide effect). Acceleration is therefore provided in respect of the theorem of e.m. acceleration.

The most simplified geometry of a cell, denominated _pill-box_, is shown in Fig. 3.1. Typical cell sizes are of the order of cm's in length and diameter; the iris radius usually ranges from few to 10 mm or so. The spatial distribution of electric and magnetic field lines inside a pill-box is derived below. In spite of more sophisticated cell geometries in real accelerators, the expression for the main component of the on-axis accelerating field remains quantitatively valid.

Figure 3.1: Left: pill-box geometry and field lines. Right: periodic RF structure

Let us assume a cell of infinite electrical conductivity. We impose four boundary conditions:

1. The longitudinal electric field is required to be null at the cell's surface and maximum on-axis, where the beam is assumed to travel \(\Rightarrow\)\(E_{z}(r=R)=0\).
2. The time-varying \(E_{z}\) implies the presence of a similarly time-varying azimuthal magnetic field \(B_{\phi}\). However, we would like to have no magnetic field on-axis, because it would deflect particles via Lorentz's force \(\Rightarrow\)\(B_{\phi}(r=0)=0\).
3. By cylindrical symmetry of the cell, the radial component of the electric field does not depend from \(z\), \(\frac{\partial E_{z}}{\partial z}=0\).
4. By cylindrical symmetry of the cell, the radial component of the magnetic field does not depend from \(\phi\), \(\frac{\partial B_{r}}{\partial\phi}=0\).

Maxwell's differential equations for \(B_{\phi}\) and \(E_{z}\) are:

\[\left\{\begin{array}{l}\left(\vec{\nabla}\times\vec{E}\right)_{\phi}=-\frac {\partial B_{\phi}}{\partial t}\\ \left(\vec{\nabla}\times\vec{B}\right)_{z}=-\frac{1}{c^{2}}\frac{\partial E_{z }}{\partial t}\end{array}\right.\Rightarrow\left\{\begin{array}{l}\frac{ \partial E_{r}}{\partial z}-\frac{\partial E_{z}}{\partial r}=-\frac{\partial B _{\phi}}{\partial t}\\ \frac{1}{r}\left[\frac{\partial(r\,B_{\phi})}{\partial r}-\frac{\partial B_{r }}{\partial\phi}\right]=-\frac{1}{c^{2}}\frac{\partial E_{z}}{\partial t}\end{array}\right. \tag{3.5}\]

We put to zero the derivatives as prescribed by points 3. and 4. above, further differentiate the top equation with respect to \(r\) and the bottom equation with respect to \(t\), and finally substitute the second equation into the first one, to get a second order partial differential equation for \(E_{z}\):

\[\left\{\begin{array}{l}\frac{\partial^{2}E_{z}}{\partial r^{2}}=\frac{ \partial^{2}B_{\phi}}{\partial r\partial t}\\ \\ \frac{1}{r}\frac{\partial B_{\phi}}{\partial t}+\frac{\partial^{2}B_{\phi}}{ \partial r\partial t}=\frac{1}{c^{2}}\frac{\partial^{2}E_{z}}{\partial t^{2}} \end{array}\right.\Rightarrow\frac{\partial^{2}E_{z}}{\partial r^{2}}+\frac{1}{ r}\frac{\partial E_{z}}{\partial r}=\frac{1}{c^{2}}\frac{\partial^{2}E_{z}}{ \partial t^{2}} \tag{3.6}\]

Equation 3.6 is a wave equation and it can be solved by separation of variables. We search a solution \(E_{z}(r,t)=A(r)e^{i(\omega t+\phi_{0})}\). By substituting this into the wave equation, we obtain an analogous equation for \(A(r)\), whose solution is:

\[A(r)=a_{0}\,J_{0}\left(\frac{\omega r}{c}\right) \tag{3.7}\]

with \(J_{0}\) Bessel function of the first kind and 0-th order. The condition 1. above is satisfied by the first zero of \(J_{0}\), which happens to be for the argument

\[\frac{\omega R}{c}=2.405\Rightarrow\omega\propto\frac{1}{R} \tag{3.8}\]

Hence, the higher the frequency of the accelerating field in an RF cavity is, the smaller the cell's outer radius has to be, with direct consequences on the mechanical accuracy required for machining the cell geometry. As an example, \(f_{RF}=1.5\,\mathrm{GHz}\) (L-band RF) implies \(R=7.7\,\mathrm{cm}\), \(f_{RF}=12\,\mathrm{GHz}\) (X-band RF) leads to \(R=1\,\mathrm{cm}\).

The azimuthal magnetic field can be determined with analogous procedure (partial derivative \(\partial_{t}\) applied to the top equation of Eq. 3.5, \(\partial_{r}\) applied to the bottom equation, then substitution of one equation into the other). The solution of the wave equation is searched in the form \(B_{\phi}(r,\,t)=C(r)e^{i(\omega t+\phi_{0})}\). The separation of variables leads to an equation for the amplitude whose solution is \(C(r)=c_{0}J_{1}(\frac{\omega r}{c})\equiv B_{\phi,0}\). \(J_{1}\) is the Bessel function of first kind and 1-st order. In general, since the electric and magnetic fields belong to the same wave, it results \(B_{\phi,0}\propto E_{z,0}\).

The Bessel functions \(J_{0}(x),\,J_{1}(x)\) are shown in Fig. 3.2. They represent the radial distribution of the peak values of \(E_{z}\) and \(B_{\phi}\) in a cell. As expected, \(E_{z}(r)\) is maximum on-axis and zero at the cell's surface. Since particle beams are commonly aligned to the cells' electrical axis (\(r\approx 0\)), we limit our attention to the on-axis accelerating field:

\[J_{0}(r\approx 0)\approx 1\ \ \Rightarrow\ \ E_{z}\approx E_{z,0}\cos(\omega t+\phi_{0}) \tag{3.9}\]

On the contrary, \(B_{\phi}(r)\) is zero on-axis, and maximum in proximity of the surface. The variation with time of the magnetic field induces currents on the metallic surface, which lead to power dissipation via thermal load.

\(E_{z}\), \(B_{\phi}\) are called _fundamental modes_ because they correspond to the lowest frequency of oscillation among all those allowed by the cavity geometry. For the purpose of longitudinal acceleration, the sum of electric field amplitudes corresponding to different wave vectors ("higher order modes") is made negligible with respect to the fundamental mode, which therefore represents the largest part of e.m. energy stored in the cavity.

In general, Spatial configurations of the e.m field in a cell or a waveguide are classified as _Transverse Magnetic_ (\(T\,M_{ijk}\)) or _Transverse Electric_ (\(T\,E_{ijk}\)). \(T\,M\) (\(TE\)) stays for magnetic (electric) field transverse to the direction of propagation of the e.m. wave. The three numerical indexes classify the number of zeros of the magnetic (electric) field in the radial (\(0<r<R\)), azimuthal (\(0<\phi<2\pi\)), and longitudinal coordinate (\(0<z<L\)). The third index is often suppressed when \(k=0\). TM modes with a longitudinal component of the electric field are accelerating modes (see Fig. 3.1-left plot). TE modes are used, for example, to transversely deflect charged particle beams; such RF structures are often referred to as _transverse deflecting cavities_.

Figure 3.2: Bessel functions of the first kind, of order 0 (blue) and 1 (orange), proportional to the amplitude of \(E_{z}\) and \(B_{\phi}\) in a cell, respectively. Half cell is delimited by the dotted lines. The vertical line represents the lateral surface of the cell, where \(J_{0}\) has its first zero; the cell axis is along z=0

### 3.2 Periodic Structures

#### Travelling Wave

Since an accelerating structure with many cells can be approximated to a long periodic system, the _Floquet's theorem_ (whose demonstration is postponed) can be applied. It results that the field \(E_{z}\) at two cross sections separated by one period (cell) differs at most by a complex number. Namely, \(E_{z}\) is a periodic function of the longitudinal coordinate \(s\) along the structure, and therefore it can be expanded in Fourier series:

\[E_{z}(r,t)=\sum_{n=-\infty}^{+\infty}a_{n}\,J_{0}(k_{n},r)\cos(\omega t-k_{n}s+ \phi_{0}),\hskip 14.226378ptk_{n}=k_{0}+\frac{2\pi\,n}{d} \tag{3.10}\]

with \(d\) the distance between two consecutive irises, or cell length. Equation 3.10 describes the linear superposition of \(n\) "modes" of an e.m. _travelling wave_ (TW), all modes oscillating at the same frequency \(\omega=2\pi\,f_{RF}\) but with a different wave number \(k_{n}\). Consequently the phase velocity \(v_{ph}^{n}=\omega/k_{n}\) is different for each mode.

Usually, the RF structure's inner geometry is built in a way that the fundamental mode's amplitude \(a_{0}\) is larger than the sum of amplitudes \(a_{n}\) of all other modes. Moreover, \(v_{ph}^{0}\approxeq c\) for application to ultra-relativistic particles. This way, particles are synchronous to the wavefront of the on-axis fundamental mode:

\[E_{z}^{TW}(r\approx 0,n=0)=E_{z,0}^{TW}\cos(\omega t-ks+\phi_{0}) \tag{3.11}\]

Synchronism between charged particles and fundamental accelerating mode means that the field phase seen by the particle is the same at all the successive cells, at the time the particle traverses them. For the synchronous particle we find:

\[\begin{array}{l}\omega t_{m}+\phi_{m}(s)=\omega t_{m+1}+\phi_{m+1}(s)=\omega \left(t_{m}+\frac{d}{\beta_{z}c}\right)+\phi_{m+1}(s)\\ \\ \Rightarrow\Delta\phi_{m}=\phi_{m}-\phi_{m+1}=\frac{2\pi\,f_{RF}}{\beta_{z}c} d=\frac{2\pi}{\beta_{z}}\frac{d}{\lambda_{RF}}\end{array} \tag{3.12}\]

\(\Delta\phi_{m}\) is called _cell phase advance_. Since the field periodicity imposes \(\frac{d}{\lambda_{RF}}\in\hat{\mathbb{Q}}\), \(\Delta\phi_{m}\) assumes values like \(\pi/3,2\pi/3,5\pi/4\), etc. when the structure is tuned for acceleration of ultra-relativistic particles (\(\beta_{z}\approx 1\)).

In practice, some RF power generated by an external source (klystron or solid state amplifier) is injected into the structure, where resonant modes build up. The constructive interference of e.m. waves established by a suitable geometry of irises privileges the fundamental accelerating mode. The e.m. energy flows through the structure with a group velocity \(v_{g}<c\), and it is extracted at the end of the structure by a waveguide, to be eventually dissipated on a load.

#### 3.2.2 Standing Wave

If the structure's end is closed by a reflective medium (e.g., a Cu plate), a stationary wave, also called _standing wave_ (SW), can build up as the resultant of constructive interference of the forward and backward (reflected) travelling wave. With reference to Fig. 3.3-left plot and recalling Eq. 3.11, we express the accelerating field in a SW as the superposition of the forward-propagating wave and the reflected wave:

\[E_{z}^{SW}(r\approx 0,n=0)=E_{z,0}^{TW}\cos(\omega t-ks+\phi_{0})+E_{z,0}^{TW} \cos(\omega t-k(2l-s)+\phi_{0}) \tag{3.13}\]

and hereafter we put \(\phi_{0}\equiv 0\) for simplicity. Equation 3.13 can be written in Euler's notation, where we retain the \(\mathbb{R}e\) part only of the oscillatory term. Hereafter, we continue assuming ultra-relativistic particles. Since the phase advance of the fundamental mode through the whole structure of length \(l\) is \(\Delta\phi_{l}=2\pi m\) (\(m\) integer), and assuming \(p\in\mathbb{N}\) cells of length \(d\), Eq. 3.12 allows us to write \(l=pd=p\Delta\phi_{l}/k=2\pi\,p^{\prime}/k\), \(p^{\prime}\in\mathbb{N}\). Then we have:

\[\begin{split}& e^{i(\omega t-ks)}+e^{i[\omega t-k(2l-s)]}=e^{i( \omega t-ks)}+e^{i(\omega t+ks)}e^{-i2kl}=\\ &=e^{i(\omega t-ks)}+e^{i(\omega t+ks)}e^{-ip^{\prime}4\pi}=2 \cos(\omega t)\cos(ks)\end{split} \tag{3.14}\]

The SW peak field is re-defined as twice the peak field of each TW propagating through the structure, so that:

\[E_{z}^{SW}(r\approx 0,n=0)=E_{z,0}^{SW}\cos(\omega t+\phi_{0})\cos(ks) \tag{3.15}\]

#### Synchronous Phase

Since a bunch is made of a very large number of particles, it has practical sense to identify a reference or _synchronous particle_ (either virtual or real, such as the bunch center of mass). The coordinates of all other particles are referred to it. If \(t_{s}\) is the

Figure 3.3: Left: superposition of a forward (1) and backward (2) e.m. wave travelling in a periodic structure with closed end. The wavefront of the standing wave generated by the constructive interference of the two travelling waves in correspondence of the coordinate \(s\) is formed at the time the initial wave takes to travel a path long \(l+(l-s)=2l-s\). Right: bunch of particles with internal coordinate \(z\); the bunch head is at positive \(z\)arrival time of the synchronous particle at a certain \(s\) in the RF structure and \(z\) is the distance, internal to the bunch, of the generic particle from the synchronous particle, then we can write for the generic particle:

\[\omega t=\omega(t_{s}+\Delta t)=\omega t_{s}+\omega\frac{z}{v_{z}}=\omega t_{s}+kz \tag{3.16}\]

By definition, \(z=0\) is the internal coordinate of the synchronous particle, see Fig. 3.3-right plot. In such case, Eqs. 3.11 and 3.15 can be re-written in the following standard notation:

\[\begin{split}& E_{z}^{TW}=E_{z,0}^{TW}\cos(\omega t_{s}-ks+\phi_{0}+ kz)\equiv E_{z,0}^{TW}\cos(\phi_{RF}^{TW}+kz)\\ & E_{z}^{SW}=\tilde{E}_{z,0}^{SW}\cos(\omega t_{s}+\phi_{0}+kz) \cos(ks)\approxeq E_{z,0}^{SW}\cos(\phi_{RF}^{SW}+kz)\end{split} \tag{3.17}\]

In the SW, \(\cos(ks)\approxeq const\). was assumed. Namely, at any time \(t\), \(E_{z}^{SW}\) is approximated to a constant amplitude along the cell.

The phase \(\phi_{RF}\) is called _synchronous phase_. In jargon, _on-crest_ acceleration refers to the choice of \(\phi_{RF}\) which maximizes the energy gain (e.g., \(\phi_{RF}=0\) for \(E_{z,0}\sim\cos\phi\)). Any phase different from that, determines _off-crest_ acceleration. Deceleration is, of course, possible as well.

In summary, the accelerating electric field inside periodic RF structures is usually reduced to the on-axis longitudinal fundamental mode \(TM_{010}\). The expression of \(E_{z}(t,s)\) for a TW and a SW differs for its dependence from the s-coordinate along the structure (compare Eqs. 3.11 and 3.15). In a SW, such a dependence is often neglected. In both cases \(E_{z}\) can be written as an amplitude times an oscillatory term, whose phase is the sum of the synchronous phase and the relative phase of the generic particle (see Eq. 3.17). It is intuitive to identify \(\phi_{RF}\) as the arrival "time" of the bunch as a whole, while \(kz\) determines the spread in energy gain internal to the bunch, due to the slightly different arrival times of the individual particles at a specific \(s\) inside the structure.

#### Transit Time Factor

The energy gain of a particle in an SW structure is calculated below. The individual cell has longitudinal coordinates \([-g/2,\,g/2]\). By virtue of the synchronization between the particle's phase and the phase of the electric field, the total energy gain is the gain in a cell times the number of cells. The energy gain in a single cell is:\[\begin{split}&\Delta E^{SW}(g,t)=q\int_{-g/2}^{g/2}E_{z}^{SW}ds=q\, \tilde{E}_{z,0}^{SW}\int_{-g/2}^{g/2}ds\cos(ot+kz+\phi_{0})\cos(ks)=\\ &\approx q\,E_{z,0}^{SW}\int_{-g/2}^{g/2}ds\left[\cos(\frac{\alpha s }{\beta_{z}c})\cos(kz+\phi_{0})-\sin(\frac{\alpha s}{\beta_{z}c})\sin(kz+\phi_ {0})\right]=\\ &=q\,E_{z,0}^{SW}\frac{1}{\frac{\omega s}{\beta_{z}c}}\,\sin( \frac{\alpha s}{\beta_{z}c})|_{-g/2}^{g/2}\cos(kz+\phi_{0})=q\,E_{z,0}^{SW}g \frac{\sin x}{x}\cos(kz+\phi_{0})=\\ &=q\,\Delta V_{0}^{SW}(g)T_{tr}\,\cos(kz+\phi_{0})\end{split} \tag{3.18}\]

We defined \(x=\frac{\omega s}{2\beta_{z}c}\), and made use of the approximation of slowly varying field amplitude internally to the cell, \(\cos(ks)\approxeq const\). The integral of odd sin-like functions over an even path is zero.

The dimensionless _transit time factor_\(T_{tr}=\sin x/x<1\)\(\forall x\) describes the reduction of the nominal energy gain \(q\,\Delta V_{0}^{SW}\) due to the time interval \(\Delta t=g/(\beta_{z}c)\) the particle takes to travel through the cell, and during which the electric field amplitude \(E_{z}(t)\) reduces compared to the peak value. \(T_{tr}\) depends from the ratio \(g/\lambda_{RF}\), and it is commonly in the range 0.85-0.95.

The calculation is repeated below for a TW structure (see the electric field in Eq. 3.11):

\[\begin{split}&\Delta E^{TW}(g,t)=q\,\int_{-g/2}^{g/2}E_{z}^{TW}ds=q\,E_{z,0}^{TW} \int_{-g/2}^{g/2}ds\cos(\omega t-ks+kz+\phi_{0})\approxeq\\ &\approxeq q\,E_{z,0}^{TW}\,\int_{-g/2}^{g/2}ds\cos(kz+\phi_{0}) =q\,E_{z,0}^{TW}g\cos(kz+\phi_{0})=\\ &=q\,\Delta V_{0}^{TW}(g)\cos(kz+\phi_{0})\end{split} \tag{3.19}\]

where we have taken the limit \(\omega t-ks\to 0\)\(\forall s\), \(t\) when \(\beta_{z}\to 1\). In conclusion, in a TW structure \(T_{tr}=1\) because the accelerating field is synchronous with the particle at any point along the structure (\(v_{ph}\approxeq v_{z}\approxeq c\)). In other words, the particle is always "surfing" the wavefront of the travelling wave.

### RLC Circuit Model

In this chapter, the electric field amplitude or, equivalently, the effective peak voltage of a RF cavity is quantified in terms of macroscopic parameters related to the RF power injected into the cavity, and to the structure's geometry. To do this, the pill-box is modelled as a resonant RLC circuit [2, 3].

#### 3.3.1 Standing Wave

Let us consider a metallic pill-box, as shown in Fig. 3.1. An external RF source injects RF power into it, so generating an effective time-varying electric voltage of amplitude \(\Delta V_{0}\). The disks at the cell edges constitute a capacity, \(C\). The cylindrical surface introduces an inductance, \(L\). The cavity material is characterized by a resistive _shunt impedance_, \(R_{s}\). The cell can therefore be analysed as a resonant RLC circuit supplied by an oscillator.

The circuit is said to be resonant because there exists a specific frequency at which the transmisssion of e.m. energy through the circuit is maximized. The frequency depends from the reactive impedances, \(\omega=1/\sqrt{LC}\). In other words, the geometry of the cell selects the frequency of the largest amplitude component of the e.m. field, i.e. the fundamental mode, consistently with Eq. 3.8.

The analysis of the RLC circuit is first done for the case of RF energy stored in the cell, i.e., the RF cavity has closed ends and the accelerating field behaves as in a standing wave. The energy stored in the circuit is \(U_{0}=\frac{1}{2}C\,\Delta\,V_{0}^{2}\). The shunt impedance is defined as the resistive part of the cell through which the power averaged over one RF cycle is dissipated according to Ohm's law:

\[R_{s}=\frac{\Delta\,V_{0}^{2}}{\langle P_{d}\rangle} \tag{3.20}\]

Let us introduce a figure of merit, the so-called _quality factor_\(Q\) of the cell, as the ratio of the power stored in the cavity in a RF cycle and the time-averaged dissipated power:

\[Q:=\frac{U_{0}\omega}{\langle P_{d}\rangle}=\frac{C\,\Delta\,V_{0}^{2}}{2} \frac{\omega R_{s}}{\Delta\,V_{0}^{2}}=\frac{R_{s}}{2\sqrt{L/C}} \tag{3.21}\]

When evaluated in the absence of charged beam traversing the cavity, \(Q\) is said "unloaded", and often noted as \(Q_{0}\).

To extend the characterization of a single cell to a periodic structure, the aforementioned quantities are defined per unit length. Equation 3.21 becomes:

\[Q=\frac{\frac{dU_{0}}{ds}\omega}{\frac{d\langle P_{d}\rangle}{ds}}=\frac{u \omega}{\frac{d\langle P_{d}\rangle}{ds}} \tag{3.22}\]

and from Eq. 3.20:

\[\Delta\,V_{0}=\sqrt{\langle P_{d}\rangle r_{s}l} \tag{3.23}\]

with \(r_{s}=dR_{s}/ds\) and \(l\) the cavity total length.

Since the e.m. energy is first injected into the cavity and then dissipated through \(r_{s}\), we expect a decrement with time from the initial value \(U_{0}\), until a new RF pulse fills the cavity again. From the definition of instantaneous dissipated power, and from Eq. 3.21 evaluated at the generic time \(t\), we get:

\[P_{d}=-\frac{dU(t)}{dt}=\frac{\omega U}{Q}\;\;\;\Rightarrow\;\;U(t)=U_{0}e^{- \frac{\omega}{Q}t} \tag{3.24}\]

The ratio:

\[t_{f}=\frac{Q}{\omega} \tag{3.25}\]is denominated _filling time_. Equation 3.24 shows that, specifically for a SW structure, \(Q\) is proportional to the number of RF cycles during which the stored energy is kept close to its initial, maximum value (the higher \(Q\) is, the longer is the time interval during which the RF field resonates in the cavity at large amplitude).

The peak value of the accelerating field is derived from Eq. 3.23:

\[\left(E_{z,0}^{SW}\right)^{2}=\left(\frac{d\,\Delta V_{0}(s)}{ds}\right)^{2} \approx\left(\sqrt{r_{s}\,\frac{d\left(P_{d}\right)}{ds}\,\frac{ds}{ds}} \right)^{2}=r_{s}\,\frac{d\left\langle P_{d}\right\rangle}{ds} \tag{3.26}\]

Let us now pay attention to the ratio \(r_{s}/Q\), which is made explicit by means of Eqs. 3.26 and 3.22:

\[\frac{r_{s}}{Q}=\frac{\left(E_{z,0}^{SW}\right)^{2}}{\frac{d\left\langle P_{d} \right\rangle}{ds}}\,\frac{\frac{d\left(P_{d}\right)}{ds}}{u\omega}=\frac{ \left(E_{z,0}^{SW}\right)^{2}}{u\omega} \tag{3.27}\]

The ratio is proportional to the accelerating field amplitude squared, per averaged stored power. It quantifies the capability of the cavity of transforming a certain amount of e.m. energy into acceleration and, for this reason, it is intended to quantify the _efficiency of acceleration_.

Equation 3.21 shows that \(r_{s}/Q\propto\sqrt{L/C}\). Since the cylindrical geometry of the cell suggests \(L\propto R\propto 1/\omega\), it turns out that \(C\propto 1/\omega\) and therefore \(r_{s}/Q\) is independent from \(\omega\), as well as from the material of the structure. Indeed, that ratio only depends from the _geometry_ of the cavity. Consequently, \(Q\sim R_{s}/G\), where \(G\) stays for a numerical factor only dependent from the cavity geometry.

#### Travelling Wave Constant Impedance

The RLC description of a SW structure can be identically applied to the case of a TW by replacing \(\left\langle P_{d}\right\rangle\rightarrow-P\), where \(P\) is now the RF power travelling along the structure and finally absorbed by a load. The opposite sign means that \(P\) is not absorbed from the structure, but it actively contributes to acceleration by flowing through it. Equation 3.24 can be re-written as follows:

\[P=\frac{dU(t)}{dt}=\frac{dU}{ds}\frac{ds}{dt}=u\cdot v_{g} \tag{3.28}\]

where \(v_{g}<c\) is the group velocity of the fundamental mode (i.e., the velocity at which the e.m. energy propagates through the structrue). In this case, the filling time is simply:

\[t=\frac{l}{v_{g}} \tag{3.29}\]Commonly, \(v_{g}\approx c\left(\frac{r_{in}}{r_{out}}\right)^{4}\approx(0.01-0.1)c\), with \(r_{in}\) and \(r_{out}\) the inner and outer radius of the cell iris, respectively.

If the TW structure is perfectly periodic, i.e., all cells are identical, then \(r_{s}=\frac{dR_{g}}{ds}=\frac{R_{g}}{l}\) and the structure is named _TW constant impedance_ (TW-CI). As \(P(s)\) is flowing through the structure, \(R_{s}\) reduces its initial value \(P(s=0)=P_{0}\) by unavoidable dissipation. To explicit the variation of \(P\) along the structure, we recall Eqs. 3.22 and 3.28:

\[\left\{\begin{array}{l}Q=-\frac{u\omega}{\frac{dP}{ds}}\\ P=u\cdot v_{g}\end{array}\right.\quad\Rightarrow\frac{dP}{ds}=-\frac{\omega}{ Q}\frac{P}{v_{g}}\quad\quad\Rightarrow P(s)=P_{0}e^{-\frac{\omega}{\partial v_{g}}s} \tag{3.30}\]

Since the longitudinal gradient of the power flowing through the structure is negative, \(Q\) is still correctly defined as a positive quantity.

An analogous dependence for the electric field amplitude is found by relating it to the power, see Eq. 3.27:

\[\left\{\begin{array}{l}\frac{r_{s}}{Q}=\frac{E_{z,0}^{2}}{u\omega}\\ P=u\cdot v_{g}\end{array}\right.\Rightarrow E_{z,0}^{2}=\frac{\omega r_{s}}{ Qv_{g}}P\quad\quad\Rightarrow\left(E_{z,0}^{CI}\right)^{2}=\frac{\omega r_{s}}{ Qv_{g}}P_{0}e^{-\frac{\omega}{\partial v_{g}}s} \tag{3.31}\]

In conclusion, the peak field in a TW-CI is _not_ constant through the structure, but exponentially damped. For a normal-conducting TW-CI with typical parameters \(f_{RF}=3\,\mathrm{GHz}\), \(Q\approx 10^{4}\) and \(v_{g}=0.1c\), the field amplitude at the end of a \(3\,\mathrm{m}\)-long structure is approximately \(80\%\) of its initial value.

The reduction of the field amplitude in a TW-CI is commonly expressed as function of a dimensionless constant denominated _attenuation factor_, which is basically determined by the structure's geometry:

\[\tau:=\frac{\omega l}{2Qv_{g}}\Rightarrow\left\{\begin{array}{l}E_{z,0}^{ CI}(s)=\sqrt{2\tau\frac{P_{0}r_{s}}{l}}e^{-\tau\frac{s}{l}}\\ \\ \left|\Delta V_{0}^{CI}(s)\right|=\left|-\int_{0}^{l}E_{z,0}^{CI}(s)ds\right|= \sqrt{2\tau\,P_{0}r_{s}l}\left(\frac{1-e^{-\tau}}{\tau}\right)\end{array}\right. \tag{3.32}\]

#### 3.3.2.1 Discussion: Optimum Attenuation Factor

Equation 3.32 shows that the accelerating voltage can be maximized by a proper choice of \(\tau\) or, in practice, by a suitable combination of the structure's parameters that define it. So, what is the optimum value of \(\tau\)?

We need to find the maximum of \(\Delta V_{0}^{CI}\) as function of \(\tau\), i.e., \(\frac{d\Delta V_{0}}{d\tau}\equiv 0\). Doing so in Eq. 3.32, we find the following relation for \(\tau\):

\[\tau=\frac{1}{2}\left(e^{\tau}-1\right)\quad\Rightarrow\tau_{opt}=1.26\quad \Rightarrow\left(\Delta V_{0}^{CI}\right)_{max}\approxeq 0.9\sqrt{P_{0}r_{s}l} \tag{3.33}\]In conclusion, a proper choice of \(\tau\) allows the effective accelerating voltage in a TW-CI structure to be only 10% smaller than the nominal peak voltage in a SW (compare with Eq. 3.23). Moreover, taking into account the typical value of \(T_{tr}\), the two peak voltages are essentially at the same level for identical input power and shunt impedance. However, since the TW-CI shows lower peak electric field on average along the structure than in a SW, the former tends to be less subject to discharges. Especially when high accelerating gradients are requested.

#### Travelling Wave Constant Gradient

A periodic RF structure in which the peak electric field is constant along the structure is said _TW Constant Gradient_ (TW-CG). As we will see, this requirement translates into a specific cells' geometry. Let us express the constancy of the accelerating field by recalling Eq. 3.26, now re-written for a TW structure:

\[\left(E_{z,0}^{CG}\right)^{2}=-r_{s}\,\frac{dP}{ds}\equiv const.\Rightarrow P _{l}=P_{0}+Cl \tag{3.34}\]

The last equality defines \(C=\frac{P(l)-P_{0}}{l}\). Expressed in terms of the generic coordinate \(s\) along the structure, it becomes:

\[P(s)=P_{0}+Cs=P_{0}+\left(\frac{P(l)-P_{0}}{l}\right)s \tag{3.35}\]

\(P(l)\) in Eq. 3.34 can be expressed in terms of an attenuation factor, in analogy to the power in a TW-CI (see Eq. 3.30 in the light of \(\tau\) defined in Eq. 3.32): \(P_{l}\equiv P_{0}e^{-2\tau}\). We will show below that such factor is indeed the same factor introduced in Eq. 3.32. By substituting \(P_{l}\) in Eq. 3.35 we get:

\[P(s)=P_{0}+\left(P_{0}e^{-2\tau}-P_{0}\right)\frac{s}{l}=P_{0}\left[1-\left(1 -e^{-2\tau}\right)\frac{s}{l}\right] \tag{3.36}\]

The group velocity can be calculated from Eq. 3.28:

\[v_{g}(s)=\frac{P(s)}{u}=P(s)\frac{\omega}{Q\,\frac{dP}{ds}}=\frac{\omega}{Q} \frac{P_{0}\left[1-\left(1-e^{-2\tau}\right)\frac{s}{l}\right]}{\frac{P_{0}}{ l}\left(1-e^{-2\tau}\right)}=\frac{\omega l}{Q}\left[\left(\frac{1}{1-e^{-2\tau}} \right)-\frac{s}{l}\right] \tag{3.37}\]

In conclusion, the constraint of a constant field amplitude translates into a group velocity varying with \(s\). Since \(v_{g}\) is determined by the cell geometry, Eq. 3.37 implies that the cell iris radius slightly changes along the structure or, in other words, the structure is quasi-periodic. In practice, as \(P(s)\) decreases along the structure because of the dissipation of the e.m. energy via resistive impedance, \(v_{g}\) decreases at an identical rate. This means that the iris inner radius gradually shrinks along the structure. The iris variation along a structure of \(\sim\)100 cells and initial iris radius of \(\sim\)10 mm,can be smaller than \(1\,\mathrm{mm}\). One can also think of the gradual reduction of the iris radius as a way to keep the density of e.m. energy, hence the electric field amplitude, constant as the power decreases.

The electric field amplitude can be evaluated by recalling Eq. 3.34 and by replacing there the power per unit length (in turn evaluated from \(P(s)\) in Eq. 3.35):

\[\begin{split}& E_{z,0}^{CG}=\sqrt{r_{s}\frac{dP}{ds}}=\sqrt{\frac{ \omega r_{s}}{\mathcal{Q}v_{g}}P(s)}=\left\{\begin{array}{l}\frac{\omega r_ {s}}{\mathcal{Q}}\frac{P_{0}\left[1-\left(1-e^{-2\tau}\right)\frac{s}{2} \right]}{\frac{\omega r}{\mathcal{Q}}\frac{\left[1-\left(1-e^{-2\tau}\right) \frac{s}{2}\right]}{\left(1-e^{-2\tau}\right)}}\\ \frac{\omega r_{s}}{\mathcal{Q}}\frac{\left[1-\left(1-e^{-2\tau}\right)\frac{s }{2}\right]}{\left(1-e^{-2\tau}\right)}\end{array}\right\}^{1/2}=\\ &=\sqrt{\left(1-e^{-2\tau}\right)\frac{P_{0}r_{s}}{l}}\end{split} \tag{3.38}\]

which is constant in \(s\), as expected. The effective accelerating voltage is:

\[\left|\Delta V_{0}^{CG}\right|=\left|-\int_{0}^{l}E_{z,0}^{CG}(s)ds\right|= \sqrt{\left(1-e^{-2\tau}\right)P_{0}r_{s}l} \tag{3.39}\]

Since \(v_{g}\) is not constant with \(s\), the filling time in a TW-CG has to be calculated more carefully:

\[t_{f}=\int_{0}^{l}\frac{ds}{v_{g}(s)}=\cdots=2\tau\frac{Q}{\omega} \tag{3.40}\]

where the integral is evaluated for the expression of \(v_{g}(s)\) in Eq. 3.37. By comparing Eq. 3.40 with Eq. 3.29, we end up with a definition of \(\tau=\frac{\omega l}{2Qv_{g}}\), which is the attenuation factor introduced for a TW-CI in Eq. 3.32.

#### Comparison

The three types of periodic multi-cells RF structures introduced so far, i.e., SW, TW-CI and TW-CG, show different values of the peak accelerating field and voltage, as well as a different dependence of those quantities from the \(s\)-coordinate internal to the structure. If a large accelerating voltage is often required to maximize the particles' energy over a given length, a moderate peak field is generally desired to avoid discharges in the structure.

Table 3.1 compares the peak voltage (see Eqs. 3.23, 3.32 and 3.39) and the peak field (see Eqs. 3.26, 3.32 and 3.38) of SW, TW-CI and TW-CG structures, assuming identical shunt impedance, structure's length and input RF power, normalized to the quantity of a SW. Figure 3.4 plots those quantities versus \(\tau\). In reality, the SW peak voltage should be rescaled by the transit time factor (e.g., \(T_{tr}\approx 0.9\)). The peak field reported for the TW-CI is at the entrance of the structure (\(s=0\)); its average average value along the structure results into the same scaling factor of the peak voltage.

It is worth reporting that the filling time is commonly much longer in a SW than in TWs, with the consequence that longer RF pulses, at typical repetition rates of 10-100 Hz, put the structure under mechanical and thermal stress, which can be source of shot-to-shot fluctuations of the RF performance. Such limited reliability is often compensated by making SW structures not longer than 1-2 m, and therefore limiting the total peak voltage. A TW-CI may be an attractive choice because of some easiness in the fabrication of perfectly identical cells. In the last decade, however, machining accuracy has reached the \(\mu\)m level, so reducing the fabrication cost of slightly different cells in a TW-CG.

#### Time Scales in RF Structures

The operation of RF structures covers \(\sim\)4 orders of magnitude in space, from the structure's length to typical bunch lengths, and \(\sim\)12 orders of magnitude in time, from the RF pulse repetition period to the bunch duration. They are recalled below and sketched in Fig. 5.

1. Power generators operate in several frequency ranges, such as L-band (1.3 or 1.5 GHz), S-band (2.998 or 2.856 GHz), C-band (around 6 GHz) and X-band (11.4 or 12 GHz). For the sake of discussion, let us consider the S-band range, whose _RF period_ corresponds to approximately \(T_{RF}=1/f_{RF}\approxeq 333\) ps, or \(\lambda_{RF}=c/f_{RF}=0.1\) m.

Figure 4: Accelerating peak voltage (left) and field (right) for a SW, TW-CI and TW-CG structure, normalized to the value of a SW, as function of the attenuation factor. The transit time factor is not considered for the SW. See also Table 1

\begin{table}
\begin{tabular}{p{113.8pt}|p{56.9pt}|p{56.9pt}|p{56.9pt}} \hline  & \(SW\) & \(T\,W-CI\) & \(T\,W-CG\) \\ \hline \(\frac{\Delta V_{0}}{\sqrt{P_{rf}}l}\) & 1 & \(\sqrt{\frac{2}{\tau}}\left(1-e^{-\tau}\right)\) & \(\sqrt{1-e^{-2\tau}}\) \\ \hline \(\frac{E_{z,0}}{\sqrt{P_{rf}}l}\) & 1 & \(\sqrt{2\tau}\) & \(\sqrt{1-e^{-2\tau}}\) \\ \hline \end{tabular}
\end{table}
Table 1: Normalized peak accelerating voltage and peak field in SW, TW-CI and TW-CG structures. \(P\) is intended to be \((P_{d})\) in SW, \(P_{0}\) in TW structures2. The Typical operation of TW structures is at phase advance of \(2\pi/3\) or so. Namely, one RF period is covered by few cells, which makes the length of each cell a fraction of the RF wavelength; for example, \(L_{cell}\approx\lambda_{RF}/3\approx 30\,\)mm. The accelerating voltage is amplified by a large number of cells, say 100 or so, so that the typical _RF structure length_ is around \(L_{str}\approx 100L_{cell}\approx 3\) m.
3. In order for the accelerated beam particles to see almost the same RF field amplitude and phase, the _bunch duration_ should be much shorter than the RF period, e.g., \(\Delta t_{b}\leq T_{RF}/10\approx 30\) ps. As a matter of fact, electron bunches as short as tens' of ps can be further manipulated in linacs to reach the \(\sim\) fs time scale.
4. The time interval corresponding to \(1^{\circ}\) S-band RF phase is \(\Delta\phi_{s}=(2\pi\,f_{RF}\Delta t)\cdot(180/\pi)\equiv 1\Rightarrow\Delta t (1^{\circ})\approx 1\) ps. The accuracy in _RF phase control_ can similarly

Figure 3.5: Schematic (not to scale) of length and time scales associated to the operation of a NC TW S-band structure. From top to bottom: RF structure with \(2\pi/3\)-phase advance accelerating mode, RF pulse repetition rate, RF pulse duration synchronized to a particle bunch, bunch at the stable phase within an RF period, single bunch duration

be quantified in terms of a time interval (much) smaller than the bunch duration, so that the effective RF phase of all particles in a bunch is known with sufficient precision. For bunch duration at the ps-time scale, \(\Delta\phi_{s}(\Delta t\leq 0.1ps)\leq 0.1^{\circ}\). Nowadays state-of-the-art low-level RF controllers can guarantee RF phase stability down to 0.001-0.01\({}^{\circ}\) S-band.
5. We have so far considered one bunch only injected into the accelerator. However, if the duration of the accelerating \(RF\)_pulse_ is long enough, then multiple bunches can be injected in series to increase the effective beam repetition rate. The maximum duration of the RF pulse is commonly inversely proportional to the uniformity of the RF field along the whole pulse. The _minimum_ duration is determined by the time interval required by the RF power to propagate through the structure and to allow the accelerating mode to resonate. This is just the _filling time_. For a TW structure, one may reasonably have \(\Delta t_{RF}\geq t_{f}=L_{str}/v_{g}\approx 3m/0.1c=0.1\)\(\mu\)s. Indeed, RF pulse durations are commonly in the range 0.1-1 \(\mu\)s.

One could then wonder how often a new single bunch (or train of bunches) can be injected into the RF structure or, in other words, what is the timing pattern of the RF pulse. This largely depends on the capability of the cooling system to absorb the RF average power dissipated on the cavity walls (low surface resistance), and that of keeping the accelerating field high over the duration of the pulse train (high quality factor). Technological choices group RF structures in two families.

Superconducting (SC) RF structures, commonly designed as SW in the L-band frequency range, are commonly made of either Nb-alloy or Cu-bulk covered by a film of Nb. The cavity is immersed in liquid He at 2-4 K, where Nb behaves as a superconductor. Such cavities tolerate CW RF power pulsed up to the MHz repetition rate. By virtue of the high quality factor \(Q\sim 10^{7}-10^{11}\), the accelerating field amplitude remains approximately constant for at least \(Q/\omega_{RF}\approx 10\) ms. The maximum accelerating gradient at the maximum repetition rate can be as high as 20-40 MV/m.

Normal-conducting (NC) RF structures, made of Cu, are characterized by \(Q\sim 10^{3}-10^{5}\). They are water-cooled, and cannot normally operate above 1 kHz repetition rate or so. The accelerating gradient is in the range \(\sim 10-100\) MV/m, inversely proportional to the RF repetition rate. These cavities are said to run in _pulsed mode_ because the accelerating mode is present in the cavity only for relatively short time intervals, during which the beam is injected into the cavity.

It emerges that SC linacs are suitable for the acceleration of bunch trains (from 100 to 1000s bunches per train, with internal separation multiple of the RF period), and therefore useful for high beam repetition rates. This is at the expense of the maximum accelerating gradient, which can be \(\sim\)3 times lower than in NC structures. The latter ones are usually limited to 2-bunches operation in a single RF pulse.

## References

* [1] R. Palmer, _Acceleration Theorems, Proceedings of the 6th Workshop on Advanced Accelerator Concepts_, BNL-61317 or CAP 1112-94C, Lake Geneva, WI, 1994 (1992)
* [2] M. Puglisi, _Conventional RF Cavity Design, Proceedings of CERN Accelerator School on RF Engineering for Particle Accelerators_, CERN 92-03, Geneva, Switzerland, vol. I (1992)
* [3] J. Le Duff, _Dynamics and Acceleration in Linear Structures, Proceedings of CERN Accelerator School_, CERN 94-01, Geneva, Switzerland, vol. I (1994), pp. 253-277

