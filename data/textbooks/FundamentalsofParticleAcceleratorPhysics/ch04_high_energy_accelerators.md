## High Energy Accelerators

High energy accelerators driven by RF cavities can provide kinetic energies up to 5 orders of magnitude higher than DTLs, betatrons and cyclotrons. They can be grouped into three families: _synchrotrons_ (circular geometry), _RF linacs_ (single-pass, linear geometry), and _energy-recovery linacs_ (race-track geometry). The use of radiofrequency implies that they are all resonant accelerators, synchronized to bunched beams. They are called _light sources_ if devoted to emission of radiation, _colliders_ if designed for collisions of accelerated beams, hence used for particle physics experiments.

To date, synchrotrons are the only accelerators capable of storing up to \(\sim\) TeV-energy particle beams for a relatively long time (from a fraction of hour to days). This is made possible by three main advancements with respect to betatrons and cyclotrons.

1. Distinct accelerator components, installed along the reference closed orbit exploit distinct functions, thus decoupling the electric and the magnetic action: acceleration by RF cavities, beam guiding and focusing by magnets.
2. The synchronism established every turn between the RF electric field and the beam arrival time permits a small energy increase per turn, thereby a small orbit excursion (in combination with a properly ramped magnetic guiding when the beam energy is increased turn-by-turn). Consequently, the magnets can be installed in a series, aligned to the reference orbit.
3. The split of magnetic guiding along the orbit breaks the azimuthal symmetry exploited in cyclotrons and betatrons. This makes the maximum kinetic energy scaling only _linearly_ with the orbit radius. Moreover, the smaller the orbit excursion and the beam transverse size is, the more compact the magnets can be, the stronger is the magnetic focusing they can provide.

Unlike single or few pass accelerators, synchrotrons pose to the beam the additional constraint of periodic motion and long-term stability. The former modifies the boundary conditions of the equations of motion with respect to e.g., linacs. The latter implies persistent emission of radiation, which in turn affects the 6-D particle distribution. In the following, the longitudinal dynamics in a low energy linac is treated first, extended to ultra-relativistic motion then. The periodic longitudinal motion in a synchrotron is finally discussed. The second part of the Chapter treats single particle linear and nonlinear transverse dynamics. Nonlinear terms in the equations of motion are often neglected in linacs, while they are relevant for the determination of stable motion in synchrotrons.

### General Features

The synchrotron was conceived by V. Veksler (1944) and E. MacMillan (1945), following the betatron and, as this one, with the aim of overcoming energy limitations of the Lawrence's cyclotron. Synchrotrons are circular accelerators in which the beam's energy is increased turn-by-turn ("energy ramp"), until the target energy is reached. A synchrotron is specified to be a _storage ring_ when the accelerator is supplied by a "full-energy" injection system, and the beam energy is constant in the main ring.

In its simplest configuration, the injection stage to a storage ring is made of a short linac followed by a smaller synchrotron called _booster ring_, as illustrated in Fig. 4.1. In most recent storage ring light sources, the electron booster ring is installed in the same tunnel of the main ring. In few cases, beam injection is made directly into the storage ring from a high energy linac. Storage ring light sources produce radiation from IR to hard x-rays. The typical beam energy range is 2-8 GeV, their circumferences span \(\sim\)0.2-2 km.

A circular collider complex can involve different particle species (leptons, hadrons, ions), manipulated at different energy levels and in different collisional configurations. The stages of beam production, acceleration and injection into the main storage ring can therefore be many and diverse. The Large Hadron Collider (LHC) is the largest circular collider in the world, with a circumference of approximately 27 km and beams' kinetic energies approaching the TeV scale. The Future Circular Collider (FCC) project is a storage ring targeting \(\sim\)10 TeV invariant mass energy, over a circumference of \(\sim\)100 km.

By virtue of the small orbit excursion and small transverse beam sizes, the beam stored in a synchrotron propagates into a _vacuum chamber_ of diameter few to tens' millimeters wide. The chamber is interrupted by, or better it incorporates, RF cavities, diagnostic elements, valves and vacuum pumps to keep the inner pressure very low, so to minimize scattering of the stored particles with residual ions.

The chamber is surrounded by permanent magnets and electromagnets. The latter are commonly made of Cu coils surrounding Fe yokes, to maximize the magnetic field flux in proximity of the beam's orbit. The Lorentz's force exploited in _dipole magnets_ bends the beam in the horizontal plane, so defining a reference closed orbit.

Multipole magnets_, and in particular quadrupoles, control the beam's transverse sizes by forcing off-axis particles to travel in proximity of the reference orbit ("focusing").

Straight sections connecting consecutive dipole magnets make the synchrotron closed orbit a polygonal. The straight sections can host components very specific to the accelerator's scope. For example, an arrangement of particularly strong quadrupole magnets, called _final focusing_, is adopted in colliders in proximity of the interaction point to squeeze the beam sizes and therefore increase the charge density locally. Special dipolar arrays, named _Insertion Devices_ (IDs), are installed in the straight sections of light sources to produce radiation with specific spectral features.

The split of functionalities that contributed to the success of synchrotrons, applies similarly to high energy RF linacs, where few meters-long RF structures alternate to quadrupole magnets. The maximum kinetic energy reached so far in a RF linac is \(\sim\)20 GeV over a length of \(\sim\)3 km, at the RF repetition rate of 120 Hz. Repetition rates as high as 1 MHz can only be reached at SC RF linacs, typically at the expense of peak accelerating gradient.

Energy-recovery linacs (ERLs) are made of parallel and relatively long straight sections, hosting one or two linacs, connected by magnetic arcs. Unlike single-pass linacs, the race-track geometry allows the beam to be recirculated, and therefore accelerated, once or in a few passes. Before the beam is dumped, it can be decelerated, so that a large fraction of the beam's e.m. power can be recovered and stored in the linac, and there used for acceleration of a newly injected beam.

In the last decade, the test of very high gradient (\(\sim\)GeV/m) particle accelerators has been carried out worldwide. The electric field providing longitudinal acceleration

Figure 1: Accelerator complex of the Elettra2.0 storage ring light source upgrade project, in Italy. The main ring circumference is 260 m long (on scale). Electrons are generated in a thermo-ionic gun, accelerated in two RF structures to 100 MeV, then ramped in energy by a booster ring over multiple turns. They are finally injected into the main ring at the energy of 2.4 GeV. (Original picture courtesy of S. Lizzit)

("wake field") is generated by the ionization of a neutral plasma in a small capillary traversed by a laser or a charged particle beam. A probe bunch is then injected into the plasma to surf the wake field ("plasma-wake field accelerators"). An analogous process has been proposed for a dielectric channel in place of the plasma capillary. In this case, acceleration is provided by the interaction of the probe bunch with the e.m. field associated to the image currents excited by the leading bunch on the dielectric surface ("dielectric-wake field accelerator"). All these non-conventional accelerators are not treated in this book, and the Reader is referred to the additional bibliography for an introduction to the subject.

### Longitudinal Dynamics

#### Phase Stability in a Linac

If a particle moves in a linac in weakly relativistic regime, its velocity increases with the kinetic energy (see Fig.1.4). In this case, the beam's injection phase is important because it determines the relative motion of particles in a bunch, by keeping the particles ensemble bounded or unbounded, depending from their relative spread in velocity. The proper choice of injection phases which allow stable motion constitutes the so-called _phase stability_[1].

The concept is illustrated in Fig.4.2. The curve shows the energy gain in a RF structure, see e.g. Eq.3.17. The phase convention is such that the bunch head, i.e. earlier arriving particles, is at smaller phases (\(a_{1},\,a_{2}\)). Energy gain is only for points in the positive (upper) half-plane.

In a RF period, and limiting ourselves to the half-plane of energy gain, the synchronous phase can be chosen at either positive or negative slope of the "RF curve". If the positive slope is chosen, i.e., the synchronous phase of \(P_{2}\) is \(3\pi/2<\phi<2\pi\), leading particles in the bunch (\(a_{2}\)) enter the RF cavity with a phase which makes them to gain less energy than the synchronous particle (the relative energy variation results \(\delta<0\)). Therefore, their velocity relative to the synchronous particle is

Figure 4.2: Phase stability in a linac at weakly relativistic velocities

smaller (particles are slower) and, as the beam propagates through the accelerator, they become closer to the bunch centre. Similarly, trailing particles (\(b_{2}\)) gain more energy (\(\delta>0\)), therefore they get a larger velocity than the synchronous particle (particles are faster), and they eventually approach the bunch centre. This repeats as leading and trailing particles exchange their positions inside the bunch, i.e., the particle's motion is bounded. The opposite happens if the synchronous particle \(P_{1}\) lies on the negative slope of the energy gain curve: in this case leading (\(a_{1}\)) and trailing particles (\(b_{1}\)) progressively move far from the synchronous particle. After some time, particles get lost in the accelerator.

In summary, if the particles' initial phase and energy are close to those of the synchronous particle, and if this sits on a stable RF phase, _oscillations_ in the _longitudinal phase space_ (\(\phi\), \(\delta\)) can happen, which ensure the bunch motion to be stable.

Phase stability is derived below from the single particle's equations of motion. Reduced variables in the longitudinal phase space (\(\phi\), \(\delta\)) are adopted, which describe the motion of the generic particle relative to the coordinates of the synchronous particle:

\[\left\{\begin{array}{l}\Delta t=t-t_{s}\\ \\ \phi=\psi-\psi_{s}=\omega\Delta t=kz\\ \\ w=\Delta E-\Delta E_{s}\end{array}\right. \tag{4.1}\]

\(\Delta E\) is the particle's energy gain. In the following, we will assume \(v_{x}\), \(v_{y}<<v_{z}\approxeq|\vec{v}|\), and the independent spatial coordinate along the linac is indicated with \(s\). We have:

\[\left\{\begin{array}{l}\frac{d\phi}{ds}=\omega\frac{d}{ds}(t-t_{s})=\omega \left(\frac{1}{v_{z}}-\frac{1}{v_{z,s}}\right)\approxeq-\omega\frac{v_{z}-v_{z,s}}{v_{z,s}^{2}}\\ \\ w=\frac{1}{2}m_{0}(v_{z}^{2}-v_{z,s}^{2})=\frac{1}{2}m_{0}(v_{z}+v_{z,s})(v_{ z}-v_{z,s})\approxeq m_{0}v_{z,s}(v_{z}-v_{z,s})\end{array}\right. \tag{4.2}\]

The approximate equalities are for \(v\approx v_{s}\). The second equation is substituted into the first one, and the first derivative of the reduced energy is calculated:

\[\left\{\begin{array}{l}\frac{d\phi}{ds}=-\frac{\omega}{m_{0}v_{z,s}^{3}}w\\ \\ \frac{dw}{ds}=\frac{d}{ds}\left(\Delta E-\Delta E_{s}\right)=q\,E_{z,0}\,\frac {d}{ds}\left[\int_{0}^{s}ds\cos\psi-\int_{0}^{s}ds\cos\psi_{s}\right]=\\ \\ =q\,E_{z,0}\left[\cos(\phi+\psi_{s})-\cos\psi_{s}\right]\approxeq-q\,E_{z,0} \Delta\left(\cos\psi_{s}\right)=q\,E_{z,0}\sin\psi_{s}\cdot\phi\end{array}\right. \tag{4.3}\]

The approximate equality in the second equation is for \(\psi\approx\psi_{s}\). Another derivation with respect to \(s\) leads to:

\[\left\{\begin{array}{l}\frac{d^{2}\phi}{ds^{2}}=-\frac{\omega}{m_{0}v_{z,s}^ {3}}\frac{dw}{ds}=-\frac{q\,E_{z,0}\omega\sin\psi_{s}}{m_{0}v_{z,s}^{3}}\phi \equiv-\Omega_{l}^{2}\phi\\ \\ \frac{d^{2}w}{ds^{2}}=q\,E_{z,0}\,\sin\psi_{s}\,\frac{d\phi}{ds}=-\frac{q\,E_{z,0}\omega\sin\psi_{s}}{m_{0}v_{z,s}^{3}}w\equiv-\Omega_{l}^{2}w\end{array}\right. \tag{4.4}\]where the first derivatives were replaced according to their expressions in Eq. 4.3. We introduced the _longitudinal angular frequency_\(\Omega_{l}\):

\[\Omega_{l}:=\sqrt{\frac{q\,E_{z,0}\omega\sin\psi_{s}}{m_{0}v_{z,s}^{3}}} \tag{4.5}\]

In conclusion, the longitudinal motion of weakly relativistic particles in a linac is described by equations of a quasi-harmonic oscillator (the oscillation frequency still depends from the particle's velocity), which ensures limited orbit excursion in the phase space. The Motion is stable if the following conditions are met:

1. all particles have phase and velocity close to those of the synchronous particle, i.e., the spread in phase and relative energy is \(<<1\);
2. the synchronous phase is chosen such that \(\Omega_{l}^{2}>0\), or \(q\,E_{z,0}\,\sin\,\psi_{s}>0\) for an accelerating field which goes like \(\sim\cos\,\psi\). For example, the convention \(q\,E_{z,0}>0\) leads to energy gain for \(\cos\,\psi_{s}>0\), and the synchronous phase has to be chosen in the range \(0\,<\,\psi_{s}\,<\,\pi/2\).

#### Adiabatic Damping

The dependence of \(\Omega_{l}\) from \(v_{z,s}\) implies that, as the bunch's average velocity increases during acceleration, the oscillation frequency reduces. If the variation of particle's velocity is so slow that \(\Delta v_{z,s}<<v_{z,s}\) in a period of oscillation, the motion in the longitudinal phase space can be approximated to that of a pure harmonic oscillator. Then, the particle's orbit in the phase space can be represented by an ellipse, whose extreme points have coordinates [0, \(w_{max}\)] and [\(\phi_{max},0\)]. The ellipse area is \(\pi\phi_{max}w_{max}\), and it is approximately constant over one period.

Recalling Eq. 4.3 and assuming \(v_{z,s}\approx const.\), we find:

\[\left\{\begin{array}{l}\frac{dw}{ds}=q\,E_{z,0}\,\sin\,\psi_{s}\cdot\phi\\ \\ \frac{d\phi}{ds}=-\frac{\omega}{m_{0}v_{z,s}^{3}}w\end{array}\right.\Rightarrow \left\{\begin{array}{l}ds=\frac{dw}{q\,E_{z,0}\,\sin\,\psi_{s}\cdot\phi}\\ \\ \phi d\phi=-\frac{\omega}{m_{0}v_{z,s}^{3}q\,E_{z,0}\,\sin\,\psi_{s}}wdw\equiv \alpha wdw\end{array}\right. \tag{4.6}\]

By integrating the lower equation on the r.h.s. in the ranges [0, \(\phi_{max}\)] and [0, \(w_{max}\)], we get:

\[\left\{\begin{array}{l}\phi_{max}^{2}-\alpha\,w_{max}^{2}=C_{1}\equiv 0\\ \\ \phi_{max}\,w_{max}=C_{2}\neq 0\end{array}\right.\Rightarrow\left\{\begin{array}{l }\phi_{max}\propto\alpha^{1/4}\propto\frac{1}{v_{z,s}^{3/4}}\\ \\ \frac{w_{max}}{E_{s}}\propto\frac{\alpha^{-1/4}}{v_{z,s}^{2}}\propto\frac{1}{v_ {z,s}^{3/4}}\end{array}\right. \tag{4.7}\]

Equation 4.7 shows that, in the non-relativistic regime, both the linac _phase acceptance_\(\phi_{max}\) and the _relative energy acceptance_\(w_{max}\) decrease with the beam's average velocity.

When the particle enters the ultra-relativistic regime \(v_{z}\approx v_{z,s}\to c\), Eq. 24 predicts \(\frac{d\phi}{ds}\to 0\), \(\frac{d^{2}w}{ds^{2}}\propto\frac{d\phi}{ds}\to 0\). Namely, the longitudinal oscillations tend to disappear. If the bunch is short enough with respect to the RF wavelength, and it is accelerated on-crest not to sample much of the RF field "curvature", all particles will have approximately the same RF phase, i.e., they will all gain the same amount of energy. Consequently, the initial _absolute_ energy spread is preserved, while the _relative_ energy spread decreases with the beam energy, \(\delta=\frac{\Delta E(s)}{E_{z}}\propto\frac{1}{\gamma(s)}\).

Since both the linac energy acceptance in the non-relativistic case and the beam's relative energy spread in the ultra-relativistic limit were derived in the approximation of slow velocity variation, their reduction with the beam's mean energy is called _adiabatic damping_.

##### 4.2.2.1 Discussion:Proton Injector

What is the number of longitudinal oscillations of a proton beam injected into a 30 m-long S-band linac (\(f_{RF}=3\) GHz)? The initial kinetic energy is \(T_{i}=10\) MeV. The beam is accelerated by a peak accelerating gradient \(E_{z,0}=25\) MV/m, and the synchronous phase is \(\psi_{s}=\pi/4\).

Owing to the fact that \(\Omega_{l}^{2}\sim 1/v_{z,s}^{3}\), the highest frequency of longitudinal oscillations is at the injection point (lowest beam energy), the smallest frequency is at the end of acceleration. Since Eq. 24 was derived for the independent variable \(s\), the calculation of the frequency in units of inverse time requires \(\Omega_{l}^{2}\) to be multiplied by \(v_{z,s}^{2}\):

\[\Omega_{l}(t)=\frac{2\pi}{T_{s}}=\sqrt{\frac{q\,E_{z,0}\omega\sin\psi_{s}}{p_{z,s}}} \tag{28}\]

The frequency at the beginning and at the end of the accelerator can be calculated once the longitudinal momentum at the beginning and at the end of acceleration is found, respectively. This can be evaluated in turn via the kinetic energy:

\[\left\{\begin{aligned} & T_{i}=10\,\text{MeV}\\ & T_{f}=T_{i}+eE_{z,0}L\cos(\frac{\pi}{4})=540.3\,\text{MeV}\\ \end{aligned}\right. \tag{29}\]

\[\Rightarrow\left\{\begin{aligned} & p_{s,i}c=\sqrt{E_{i}^{2}-(m_{p}c^{2})^{2}}= \sqrt{T_{i}^{2}+2T_{i}m_{p}c^{2}}=137.3\,\text{MeV}\\ & p_{s,f}c=\sqrt{T_{f}^{2}+2T_{f}m_{p}c^{2}}=1142.6\,\text{MeV} \end{aligned}\right. \tag{30}\]

By replacing these momenta in Eq. 28, we find \(\Omega_{l}^{i}(t)/(2\pi)=136\) MHz and \(\Omega_{l}^{f}(t)/(2\pi)=47\) MHz.

Although these instantaneous values look extremely high, one has also to consider that the particles, though in weakly relativistic regime, take extremely short time (compared to the human scale) to pass through the linac. Indeed, the instantaneous velocity, either at the injection or at the extraction point, is \(\beta c=pc^{2}/E\)which amounts to 0.145 \(c\) and 0.773 \(c\), respectively. Hence, the time a proton would take to pass 30 m at the smallest and largest velocity would be, respectively, 0.69 \(\upmu\)s and 0.13 \(\upmu\)s. At the end, the number of longitudinal oscillations the protons have effectively completed during acceleration is expected to be within the interval [47 MHz \(\cdot\) 0.13 \(\upmu\)s-136 MHz \(\cdot\) 0.69 \(\upmu\)s] \(\approx\) [6 \(-\) 94].

An exact calculation of the number of oscillations should take into consideration the variation of longitudinal velocity and momentum through the linac, assuming a linear variation of the particle's total energy with \(s\), i.e., \(\gamma(s)=\gamma_{0}+\frac{qE_{z,0}}{m_{\,p}c^{2}}s\equiv\gamma_{0}+\alpha s\), and \(\gamma_{0}=1.01\). In this case, we calculate the number of oscillations as the ratio of the travelling time and the instantaneous oscillation period:

\[N_{osc}=\frac{\Delta t}{T_{l}}=\int_{0}^{L}\frac{ds}{v_{z}(s)}\frac{\Omega_{l} (t)}{2\pi}=\frac{1}{2\pi}\int_{0}^{L}\frac{cds}{\beta_{z}(s)}\sqrt{\frac{ \omega qE_{z,0}\sin\psi_{s}}{p_{z}(s)}}=\]

\[=\frac{\sqrt{\omega qE_{z,0}\sin\psi_{s}}}{2\pi c}\int_{0}^{L}\frac{ds}{\beta _{z}(s)}\sqrt{\frac{c}{\beta_{z}(s)E(s)}}=\frac{1}{2\pi}\sqrt{\frac{\omega qE _{z,0}\sin\psi_{s}}{m_{p}c^{3}}}\int_{0}^{L}\frac{ds}{\beta_{z}^{3/2}(s)\gamma ^{\prime}(s)}=\]

\[=\frac{1}{2\pi}\sqrt{\frac{\omega qE_{z,0}\sin\psi_{s}}{m_{\,p}c^{3}}}\int_{0 }^{L}ds\frac{\gamma^{2}}{(\gamma^{2}-1)^{3/2}}=\frac{1}{2\pi}\sqrt{\frac{ \omega qE_{z,0}\sin\psi_{s}}{m_{\,p}c^{3}}}\int_{0}^{L}ds\frac{(\gamma_{0}+ \alpha s)^{2}}{\left[(\gamma_{0}+\alpha s)^{2}-1\right]^{3/2}}\approxeq 43\]

According to Eq. 4.11, \(N_{osc}\sim\frac{1}{\gamma\sqrt{m_{0}c^{2}}}\sim\frac{\sqrt{m_{0}c^{2}}}{E}\). Thus, the number of oscillations of lighter particles is smaller than for heavier particles, for the _same_ total energy. Namely, lighter particles become ultra-relativistic sooner, and their relative longitudinal position is frozen earlier.

#### 4.2.2 Discussion: Electronic Capture in a RF Gun

An RF cavity characterized by \(v_{ph}\approx c\), denominated _RF Gun_, can be used as very first accelerating stage of non-relativistic electrons, as long as the accelerating field is high enough to allow the electrons to enter the ultra-relativistic regime in less than one RF period. In other words, the relative shift of the e.m. wave with respect to the synchronous particle ("slippage") in one period has to be small enough to allow an energy gain much larger than the particle's initial energy. In most advanced RF Guns, an infrared (IR) laser impinges on the metallic or semiconductor surface (the "cathode") of the inner back face of the cavity. Electrons are emitted by photoelectric effect, commonly at kinetic energies lower than 10 eV. What is the minimum peak electric field in the RF Gun to "capture" the electrons?

The electron-wave slippage length per unit of RF phase is:

\[\frac{dl}{d\phi}=\frac{(v_{ph}-v_{0})dt}{d\phi}=\frac{c(1-\beta_{0})dt}{d \phi}=\frac{c}{\omega}(1-\beta_{0})=\frac{\lambda_{RF}}{2\pi}(1-\beta_{0})\]In order for the energy gain during such slippage to be much larger than the initial particle's energy (i.e., in proximity of the cathode surface), the peak field has to be:

\[\begin{array}{l}\Delta E\approx q\,E_{z,0}\,\frac{dI}{d\phi}=q\,E_{z,0}\lambda _{RF}\,\frac{1-\beta_{0}}{2\pi}\gg\gamma_{0}m_{0}c^{2}\\ \\ \Rightarrow E_{z,0}\gg\frac{2\pi m_{0}c^{2}}{q\lambda_{RF}}\left(\frac{\gamma_ {0}}{1-\beta_{0}}\right)\end{array} \tag{4.13}\]

For electrons accelerated in an S-band Gun (\(f_{RF}=3\) GHz) and therein emitted with an initial kinetic energy \(T_{0}\approx 10\) eV, one finds:

\[\begin{array}{l}\gamma_{0}=\frac{m_{e}c^{2}+T_{0}}{m_{e}c^{2}}=1.00002\\ \\ \Rightarrow\beta_{0}=\sqrt{1-\frac{1}{\gamma_{0}^{2}}}=0.006\\ \\ \Rightarrow E_{z,0}\gg 32\mathrm{MV/m}\end{array} \tag{4.14}\]

State-of-the-art RF Guns in the S-band to X-band frequency range, run at peak fields as high as 60-250 MV/m.

#### Momentum Compaction

Dipole fields are used to bend beam particles in linacs as well as in synchrotrons [2]. According to Eq.2.5, for any given bending field, particles of different longitudinal momenta will follow different curvature radii, and therefore their path lengths will be different. The variation of relative longitudinal momentum \(\delta\), curvature radius \(R\) and magnetic field \(B_{y}\), are related each other by

\[\begin{array}{l}\frac{dp_{z}}{p_{z}}=\frac{dR}{R}+\frac{dB_{y}}{B_{y}}=\frac {dR}{R}\left(1+\frac{R}{B_{y}}\frac{dB_{y}}{dR}\right)\equiv\frac{1}{\alpha_{ c}}\frac{dR}{R}\\ \\ \Rightarrow\alpha_{c}:=\frac{dR/R}{dp_{z}/p_{z}}=\frac{dL/L}{\delta}\end{array} \tag{4.15}\]

\(L\) is the path length of the on-energy, synchronous particle. \(\alpha_{c}\) is called _momentum compaction_ and it quantifies the variation of path length of an off-energy particle (\(\delta\neq 0\)) relative to the path length of the synchronous particle.

The transverse coordinates of the generic particle w.r.t. the synchronous particle are described hereafter in the _Frenet-Serret frame of coordinates_. The motion of the synchronous particle defines the reference orbit, whose longitudinal curvilinear coordinate is \(s\). At any \(s\), the longitudinal velocity of the particle is tangent to the instantaneous orbit of radius \(R(s)\), namely, \(\vec{v}_{z}=v_{z}\hat{s}\). The transverse plane is orthogonal to \(\hat{s}\), such that the generic particle is distant \(x\) and \(y\) from the synchronous particle, in the bending plane and in the plane orthogonal to it, respectively. The reference system is illustrated in Fig. 4.3-left plot.

A particle initially aligned to the synchronous particle but with a lower longitudinal momentum, is bent on an orbit of smaller curvature radius. The distance along the \(x\)-axis between the two particles, at any point \(s\) of the reference orbit, is \(x(s)\), see Fig. 4.3-right plot. We define _linear momentum-dispersion function_, simply dispersion hereafter, the quantity:

\[D_{x}(s):=\frac{x(s)}{\delta} \tag{4.16}\]

The difference in curvature radius of the two orbits in Fig. 4.3, where particle-1 is intended to be the synchronous particle, results:

\[\begin{array}{l}dR=\frac{C_{2}-C_{1}}{\delta b}=\frac{1}{\delta b}\left( \int ds_{2}-\int ds_{1}\right)=\frac{1}{\delta b}\int d\theta\left[\left(R_{1 }+x\right)-R_{1}\right]=\\ \\ =\frac{1}{\delta b}\int xd\theta=\langle x\rangle_{\theta}\end{array} \tag{4.17}\]

\(C_{1},C_{2}\) are the path lengths of the two particles from the common origin to any arbitrary \(s\), and \(\theta_{b}\) is the total bending angle of the synchronous particle. By recalling Eq. 4.15, we find:

\[\alpha_{c}=\frac{dR/R}{\delta}=\frac{1}{R}\frac{\langle x\rangle_{\theta}}{ \delta}=\frac{(D_{x})_{\theta}}{R}=\frac{1}{R\delta b_{b}}\int D_{x}d\theta= \frac{1}{C}\int\frac{D_{x}(s)}{R(s)}ds \tag{4.18}\]

\(C=R\theta_{b}\) is the total nominal path length, and \(R(s)\) the _local_ curvature radius evaluated along the nominal path (derived from the change of variable \(d\theta=ds/R(s)\)). Clearly, if the orbit is closed, the integral is closed as well, \(C\) is the ring circumference, and \(\alpha_{c}\) is the average value of the ratio \(D_{x}/R\) along the closed orbit. A magnetic lattice in which all dipole magnets have the same curvature radius is said _isomagnetic_.

It should be noted that \(\alpha_{c}\) can be either positive, null or negative, and that it receives non-zero contribution only from the dispersion defined along curved paths, e.g., inside dipole magnets. This implies that \(D_{x}\neq 0\) along a drift (for which \(R(s)\rightarrow\infty\)) does not contribute to \(\alpha_{c}\) at first order in \(\delta\). However, if \(D_{x}\) is non-zero in a drift, quadrupole magnets can modify its value at a successive dipole magnet, so that, at the end, \(\alpha_{c}\) can be tuned via a suitable manipulation of \(D_{x}\) both inside and outside the dipole magnets.

Owing to the relation \(D_{x}\sim\frac{dx}{dp_{z}}\sim\frac{dR}{dp_{z}}\sim\frac{1}{B_{y}}\) (the last relation is from Eq.2.5), the dispersion function results to be an intrinsic property of the magnetic lattice.

Figure 4.3: Left: Frenet-Serret coordinate system. Right: top-view of orbit dispersion of an off-energy particle in a dipole magnet

Since also the curvature radius is a property of the lattice (\(R\theta_{b}=l_{b}\)), it turns out that \(\alpha_{c}\sim\frac{D_{x}}{R}\) inherits the magnetic lattice.

The term "linear" attributed above to \(D_{x}\) and \(\alpha_{c}\) refers to the order in \(\delta\) taken to describe the particle's motion: \(x=D_{x}\delta+o(\delta^{2})\). In general, higher orders can be considered, so that higher order energy dispersion and momentum compaction can be defined, e.g., \(\alpha_{c}=\alpha_{1}+\alpha_{2}\delta+\alpha_{3}\delta^{2}+\cdots\).

#### Transition Energy

Off-energy ultra-relativistic particles in a synchrotron travel on slightly different orbits, which result in slightly different revolution frequencies. This amounts to \(\omega_{s}=\beta_{z,s}c/R_{s}\) for the synchronous particle in an isomagnetic lattice. The relative variation of revolution frequency per unit deviation of the longitudinal momentum is called _slip factor_:

\[\eta:=\frac{d\omega/\omega_{s}}{dp_{z}/p_{z,s}} \tag{4.19}\]

From the definition of \(\omega_{s}\) we have:

\[\begin{array}{l}\frac{d\omega}{\omega_{s}}=\frac{d\beta_{z}}{\beta_{z,s}}- \frac{dR}{R_{s}}=\frac{d\beta_{z}}{\beta_{z,s}}-\alpha_{c}\frac{dp_{z}}{p_{z, s}}=\frac{dp_{z}}{p_{z,s}}\left[\left(\frac{dp_{z}}{d\beta_{z}}\right)^{-1} \frac{p_{z,s}}{\beta_{z,s}}-\alpha_{c}\right]=\\ \\ =\frac{dp_{z}}{p_{z,s}}\left[\frac{1}{\gamma^{3}m_{0}c}\frac{p_{z,s}}{\beta_{z,s}}-\alpha_{c}\right]=\delta\left(\frac{1}{\gamma^{2}}-\alpha_{c}\right)\\ \\ \Rightarrow\eta=\frac{1}{\gamma^{2}}-\alpha_{c}\end{array} \tag{4.20}\]

and we calculated \(\frac{dp_{z}}{d\beta_{z}}=\frac{d(\gamma\beta_{z})}{d\beta_{z}}m_{0}c=\gamma^ {3}m_{0}c\). Typically, \(0<\alpha_{c}\ll 1\) in synchrotrons.

Equation 4.20 suggests that, for any given \(\alpha_{c}\) determined by the magnetic lattice, there exists a value of the beam energy \(\gamma_{tr}=1/\sqrt{|\alpha_{c}|}\), named _transition energy_, making \(\eta(\gamma_{tr})=0\). The transition energy determines a swap of the sign of revolution frequency variation per unit of relative energy deviation.

If the beam's energy is approximately constant along one or several turns and such that \(\eta>0\) (\(\gamma<\gamma_{tr}\) or "below transition"), particles at \(\delta>0\) will take a shorter time to make a turn than the synchronous particle, by virtue of their higher angular frequency. If \(\eta<0\) instead (\(\gamma>\gamma_{tr}\) or "above transition"), particles at \(\delta>0\) will take a longer time. This latter result is also named "negative mass" behaviour because more energetic particles become slower.

Phase stability in a synchrotron is ensured by a different synchronous phase depending whether the beam is below or above transition. As shown in Fig. 4.4, \(\psi_{s}\) of a beam above transition has to be chosen in a way that leading particles (\(a_{1}\)) gain more energy (\(\delta>0\)) than the synchronous particle. Doing so, they take a longer time to make a turn, thus they arrive later at the RF cavity on successive turns, and therefore they move towards the synchronous particle. A similar bounded motion also happens for the trailing particles (\(b_{1}\)), and _synchrotron oscillations_ are established. For the same reason, \(\psi_{s}\) of a beam below transition has to be chosen such that leading particles (\(a_{2}\)) gain less energy than the synchronous particle, etc.

The working point \(\eta=0\) is critical for the longitudinal stability. When the beam energy crosses the transition energy during an energy ramp, the synchronous phase chosen below transition becomes unstable (see Fig. 4.4). A fast switch of the synchronous phase must therefore be implemented to keep the beam motion stable above transition.

A lattice with \(\eta=0\) can nevertheless be conceived. It is said to be _isochronous_ because particles take the same time to make a turn independently from their energy deviation, see Eq. 4.19. The synchronous phase, however, has to be chosen \(\psi_{s}=2\pi n\), \(n\in\mathbb{N}\) (on-crest acceleration). Otherwise, since the particles' arrival time at the RF cavity is frozen, they will continue gaining more (less) energy than the synchronous particle. The beam energy spread would then grow indefinitely, until far off-energy particles cannot be safely kept on a closed orbit, and get lost. At the on-crest phase, instead, and if the bunch is short enough not to sample too much RF curvature, the energy gain per turn can be made approximately equal for all particles, the bunch length and the relative energy spread remain approximately constant.

In synchrotrons, \(\alpha_{c}\) is typically in the range \(10^{-5}-10^{-3}\). For example, for electron's total energy of 2 GeV, \(\frac{1}{\gamma^{2}}\approx 10^{-7}\) and \(\eta\approx-\alpha_{c}<0\). For protons at the same energy, however, \(\frac{1}{\gamma^{2}}=0.2\) and \(\eta\approxeq\frac{1}{\gamma^{2}}>0\). While electrons are always injected into synchrotrons above transition by virtue of their small rest energy, protons are often injected below transition. These have to rapidly cross the transition energy until they reach the target energy. For high intensity proton beams, the effect of inter-particle Coulomb interactions ("space charge" forces) changes sign at transition, causing a sudden change and oscillation of the bunch length, hence a dilution of the charge density, an increase of the beam energy spread, and possible particles loss. Moreover, if the instability is not suddenly damped, space charge-induced energy spread can produce a spread in \(\gamma_{tr}\) internal to the bunch, so that particles would not cross the transition simultaneously.

Figure 4.4: Left: phase stability in a synchrotron below (\(\eta<0\)) and above transition energy (\(\eta>0\))

Common remedies, denominated _fast \(\gamma_{tr}\) jump_ schemes, foresee the artificial increase of the transition crossing speed by means of fast pulsed quadrupole magnets. These are arranged in doublets in the proton synchrotrons PS and SPS at CERN. The modification to the dispersion function is intended to increase \(\alpha_{c}\), thus to lower \(\gamma_{tr}\), while ideally keeping the transverse dynamics unperturbed outside the doublets. While the beam energy ramp in a proton synchrotron can be of the order of \(d\gamma/dt\approx 10-100\ s^{-1}\), the required variation of the magnets' strength can be implemented in milliseconds, so that \(d\gamma_{tr}/dt\approx 10^{3}-10^{4}\ s^{-1}\).

##### Discussion: Momentum Compaction of a Drift Section

By virtue of Eq. 4.19, \(\eta\) behaves as a generalized momentum compaction. Namely, it is able to quantify the relative longitudinal shift of off-energy particles even for \(\alpha_{c}=0\), such as in a non-dispersive drift section. In this case, what is the relative particles' slippage, as function of their relative momentum deviation?

The deviation in revolution frequency in the definition of \(\eta\) can be intended here as the deviation in arrival time at a given \(s\)-coordinate along the drift or, equivalently, the deviation in longitudinal position \(\Delta z\) after a given time interval \(T_{s}\). In the approximation of small deviation of the longitudinal momentum, \(\delta\ll 1\):

\[\begin{array}{c}\frac{d\omega}{\alpha_{s}}=-\frac{\Delta T}{T_{s}}=-\frac{ \beta_{z,s}}{\beta_{z,s}}\frac{\Delta T}{T_{s}}\approx\frac{L-L_{s}}{L_{s}}=- \frac{\Delta z}{L_{s}}=\eta\delta=\frac{\delta}{\gamma_{s}^{2}}\\ \Rightarrow\frac{\Delta z}{\delta}=-\frac{L_{s}}{\gamma_{s}^{2}}\end{array} \tag{4.21}\]

\(\Delta z<0\) means that, assuming two particles occupying initially the same position, a more energetic particle moves ahead of the on-energy particle after a path length long \(L_{s}\). The longitudinal slippage of particles along a drift section due to their relative spread of velocities is suppressed by Special Relativity by a factor \(1/\gamma^{2}\).

The slip factor is usually neglected in linacs at high beam energies. For typical values \(\delta\leq 0.1\%\) in electron linacs, particles' longitudinal shift along tens' of meters is at nm scale, \(\Delta z\sim\frac{L\delta}{\gamma^{2}}\leq 0.01\mu\)m. The effect can be relevant for bunches produced already at \(\mu\)m-scale length and at energies as low as \(\sim\)10 MeV, such as in ultrafast electron diffraction sources, in which few microns bunch lengthening can be accumulated for \(\delta\sim 1\%\) over a 10 cm-long drift section.

##### Discussion: Magnetic Bunch Length Compression

Free-electron lasers are nowadays the most advanced light sources in x-rays. Electron bunches of high charge density are accelerated in a RF linac to multi-GeV final energy, then made to wiggle in magnetic devices, named "undulators", to emit highly energetic, collimated radiation pulses. The radiation intensity is exponentially amplified along the undulator in proportion to the electron bunch peak current. For this reason, the initial bunch duration, typically in the range of 5-20 ps fwhm, is compressed at intermediate linac energies to obtain peak currents at (multi-)kA level.

Bunch length compression is commonly accomplished in a 4-dipoles chicane (or a number of them), as shown in Fig. 4.5. We want to show that if a \(z\)-correlated momentum spread \(\delta(z)\) is imparted to the ultra-relativistic beam, the bunch can be time-compressed by a large factor in virtue of the chicane momentum compaction \(\alpha_{c}\).

By simplifying the bunch longitudinal dynamics to a two-particle model, the initial bunch length is just \(\Delta z_{i}=l_{b,i}=s_{2}-s_{1}\), with \(s_{1}\) taken along the reference trajectory. By virtue of the difference in longitudinal momentum and of the definition of \(\alpha_{c}\) in Eq. 4.15, the two particles will run over different path lengths. Their relative distance at the chicane exit will be:

\[\begin{array}{l}\Delta z_{f}=l_{b,\,f}=(s_{2}+L_{2})-(s_{1}+L_{1})=(s_{2}-s _{1})+(L_{2}-L_{1})=l_{b,i}+\Delta L\\ \\ \Rightarrow l_{b,\,f}=l_{b,i}+\Delta L=l_{b,i}+\alpha_{c}L_{1}\delta\end{array} \tag{4.22}\]

The relative momentum spread is generated by the linac upstream of the chicane. The energy gain is:

\[\Delta\,E(z)=e\,\Delta\,V_{0}\cos\phi_{RF}=e\,\Delta\,V_{0}\cos(\omega_{RF}t)= e\,\Delta\,V_{0}\cos(k_{RF}z) \tag{4.23}\]

From this, the \(z\)-correlated relative momentum spread is calculated in the ultra-relativistic approximation:

\[\delta=\tfrac{\Delta\,p_{z}}{p_{z,0}}\approx\tfrac{\Delta\,E}{E_{0}}=\tfrac{1 }{E_{0}}\tfrac{dE}{dz}\,\Delta z_{i}=-\tfrac{l_{b,i}}{E_{0}}k_{RF}e\,\Delta\, V_{0}\sin\phi_{RF}\equiv-l_{b,i}\,h_{z} \tag{4.24}\]

The quantity:

\[h_{z}:=\tfrac{1}{E_{0}}\tfrac{dE(z)}{dz}=\tfrac{e\,\Delta\,V_{0}}{E_{0}}k_{RF} \sin\phi_{RF} \tag{4.25}\]

is the _linear energy chirp_, relative to the beam mean energy \(E_{0}\) at the chicane. When the intrinsic beam energy spread is much smaller than the energy spread imparted by the linac, it results \(h\approx\tfrac{\sigma_{h}}{\sigma_{z,l}}\).

Figure 4.5: Three-particle model of a bunch travelling through a 4-dipoles magnetic chicane. The leading particle (3) is at lower energy than the synchronous particle (2), e.g. identified with the bunch center of mass (cm). The trailing particle (1) is at higher energy. By virtue of the different path lengths, the bunch head and tail eventually catch up with the center of mass (cm), and the bunch duration is shortened. (Original picture courtesy of M. Venturini)

The _linear compression factor_ is defined as the ratio of initial and final bunch length. It is calculated by inserting Eq. 4.24 into Eq. 4.22:

\[\begin{array}{l}l_{b,\,f}=l_{b,i}\;(1-\alpha_{c}L_{1}h_{z})\\ \\ \Rightarrow C:=\left|\frac{l_{b,i}}{l_{b,f}}\right|=\frac{1}{\left|1-\alpha_{c} L_{1}h_{z}\right|}\end{array} \tag{4.26}\]

In summary, the curved path in dipole magnets forces particles to shift one respect to another in proportion to their relative energy deviation, and despite their ultra-relativistic velocity. If the product of momentum compaction and relative momentum deviation is large enough, the shift can be comparable to the initial bunch length, which is therefore reduced. In fact, the bunch is shortened if \(sign(\alpha_{c}h_{z})>0\) and still \(\alpha_{c}L_{1}h_{z}<1\). Since \(\alpha_{c}<0\) in a 4-dipole chicane (more energetic particles are bent by smaller angles, thus they run shorter path lengths), it must be \(h_{z}<0\). This means that particles ahead of the on-energy particle (\(dz>0\)) must have lower energy (\(dE<0\)).

As a quantitative case study, let us consider an electron beam accelerated off-crest by an S-band (\(f_{RF}=3\) GHz) linac of total peak voltage \(\Delta\,V_{0}=500\) MV. The average beam energy at the chicane is specified to be \(E_{0}=1\) GeV. The on-energy path length through the chicane is \(L_{1}=\)10 m, and \(\alpha_{c}=-0.01\). If \(C=10\) is requested, for example, then Eq. 4.26 provides \(\phi_{RF}=\arcsin(-0.2865)=-16.6^{\circ}\) far from the accelerating crest (\(h_{z}=-9\) m\({}^{-1}\)).

#### Phase Stability in a Synchrotron

Particles in a high energy synchrotron show longitudinal oscillations in analogy to those in a low energy linac [1]. The driving force is also in this case the RF field. However, since all particles in the synchrotron travel at approximately the same velocity \(v_{z}\approx c\), their relative slippage is not due anymore to the spread of velocities, but to the different curved path lengths induced by a spread in longitudinal momentum, in accordance to Eq.2.5. Conditions of longitudinal stability will be analysed in the following for constant beam mean energy. In practice, the energy gain per turn provided by the RF cavity is assumed to replenish the energy loss due to radiation emission in dipole magnets. The case of energy ramp is discussed then.

The synchronous particle is, by definition, on-energy, and its revolution frequency \(\omega_{s}\) is a sub-harmonic of \(\omega_{RF}\). Its RF phase \(\psi_{s}\) determines a constant energy gain per turn \((\delta E)_{s}\):

\[\begin{array}{l}\omega_{RF}=\frac{d\psi_{s}}{dt}\equiv h\omega_{s}=-h\, \frac{d\theta_{s}}{dt}\\ \\ (\delta E)_{s}=q\,V_{0}\cos\psi_{s}\end{array} \tag{4.27}\]

and \(\theta_{s}\) is the deflection angle along the synchrotron circumference. The coefficient \(h\in\hat{\mathbb{N}}\), called _harmonic number_, is the number of RF cycles per revolution period. It is typically in the range \(10^{2}\)-\(10^{3}\) for \(f_{RF}=0.1-0.5\) GHz. The minus sign in the upper equation is to show that earlier particles with respect to the synchronous one are at smaller absolute phases.

The reduced variables to describe the longitudinal motion of the generic particle are \(\phi=\psi-\psi_{s}\) and \(w=\frac{2\pi\delta E}{\omega_{s}}\). Their first derivative with respect to time is:

\[\left\{\begin{array}{l}\frac{d\phi}{dt}=\frac{d(\psi-\psi_{s})}{dt}=-h\, \frac{(d\theta-d\theta_{s})}{dt}=-h\,\Delta\omega\\ \\ \frac{dw}{dt}\approx\frac{\Delta(\delta E)}{T_{0}}T_{0}=\delta E(\psi)-\delta E (\psi_{s})=q\,V_{0}(\cos\psi-\cos\psi_{s})\end{array}\right. \tag{4.28}\]

where the time-variation of the energy gain difference is taken over a revolution period \(T_{0}\).

The derivative of the relative phase in Eq. 4.28 is manipulated by replacing \(\Delta\omega\) with \(\eta\) (see Eq. 4.19). Doing so, the longitudinal momentum deviation \(\Delta p_{z}\) in the definition of the slip factor is approximated with the total momentum deviation (\(\Delta p_{x}\), \(\Delta p_{y}<<\Delta p_{z}\approxeq\Delta p\)). Then, the total momentum deviation is converted to total energy deviation:

\[\begin{array}{l}dp=d(\beta\gamma)m_{0}c=d\left(\gamma\sqrt{1-\frac{1}{\gamma ^{2}}}\right)m_{0}c=m_{0}cd\left(\sqrt{\gamma^{2}-1}\right)=\\ \\ =m_{0}c\frac{d\left(\sqrt{\gamma^{2}-1}\right)}{d\gamma}d\gamma=m_{0}c\gamma \,\frac{\sqrt{1-\beta^{2}}}{\beta}d\gamma=\frac{m_{0}c^{2}}{\beta c}d\gamma= \frac{dE}{\beta c}\end{array} \tag{4.29}\]

The Top row of Eq. 4.28 becomes:

\[\begin{array}{l}\frac{d\phi}{dt}=-h\eta\omega_{s}\frac{\Delta p_{z}}{p_{z,s }}\approxeq-\frac{h\eta\omega_{s}}{p_{z,s}}\frac{\Delta E}{\beta_{s}c}\equiv \frac{h\eta\Delta E}{R_{s}p_{s}}=-\frac{h\eta\omega_{s}}{2\pi\,R_{s}\,p_{z,s} }w;\\ \\ \frac{d^{2}\phi}{dt^{2}}=-\frac{h\eta\omega_{s}}{Cp_{z,s}}\frac{dw}{dt}=- \frac{h\eta\omega_{s}q\,V_{0}}{Cp_{z,s}}(\cos\psi-\cos\psi_{s})\end{array} \tag{4.30}\]

The "equivalent curvature radius" \(R_{s}\) for the synchronous particle was introduced to make the polygonal path through the synchrotron a circumference of total length \(C=\frac{2\pi\beta_{s}c}{\omega_{s}}=2\pi\,R_{s}\). It should be noted that, in general, \(R_{s}\) is different from the local curvature radius of the individual dipole magnets.

In the approximation of _small oscillation amplitudes_, i.e., \(\psi\approx\psi_{s}\) and \(w\approx w_{s}\), the second time-derivative of \(\phi\) and \(w\) becomes:

\[\left\{\begin{array}{l}\frac{d^{2}\phi}{dt^{2}}\approxeq\frac{h\eta\omega_{ s}q\,V_{0}}{Cp_{z,s}}\Delta(\cos\psi_{s})=-\frac{h\eta\omega_{s}q\,V_{0}}{Cp_{z,s} }\sin\psi_{s}\cdot\phi\equiv-\Omega_{s}^{2}\phi\\ \\ \frac{d^{2}w}{dt^{2}}\approxeq-q\,V_{0}\frac{d}{dt}\,\Delta(\cos\psi_{s})=q\,V _{0}\sin\psi_{s}\,\frac{d\phi}{dt}=-\frac{h\eta\omega_{s}q\,V_{0}}{Cp_{z,s}} \sin\psi_{s}\cdot w=-\Omega_{s}^{2}w\end{array}\right. \tag{4.31}\]

We introduced the _synchrotron angular frequency_:

\[\Omega_{s}:=\frac{2\pi}{T_{s}}=\sqrt{\frac{q\,V_{0}\eta\omega_{RF}\,\sin\psi_ {s}}{Cp_{z,s}}} \tag{4.32}\]This has the same form of the angular frequency derived in Eq. 4.8 for weakly relativistic motion in a linac, but here replacing \(E_{z,0}\to\eta\,V_{0}/C\).

In conclusion, when the bunch width in phase and energy is small, i.e., \(\Delta z_{b}\ll\lambda_{RF}\) and \(\delta\ll 1\), the longitudinal motion in a synchrotron is stable if \(\Omega_{s}^{2}>0\). In this case, the particle's motion in the longitudinal phase space \((\phi,\,w)\) describes an ellipse. Its projection onto the \(\phi\) and the \(w\)-axis corresponds to Eq. 4.31. The number of synchrotron oscillations per revolution period is named _synchrotron tune_, \(Q_{s}=\frac{\Omega_{s}}{\omega_{s}}=\frac{\Omega_{s}}{(\beta_{s}c/R_{s})}\). For \(\Omega_{s}\sim\) kHz and \(\omega_{s}\sim\) MHz, one synchrotron oscillation is completed in \(\sim\)1000 turns or so.

#### Constant of Motion

Equation 4.30 describes particle's motion for arbitrarily large oscillation amplitudes. The driving force can be derived from a scalar potential:

\[\begin{array}{l}\frac{d^{2}\phi}{dt^{2}}=F(\phi)=-\frac{dU(\phi)}{d\phi},\\ \\ \Rightarrow U(\phi)=-\int_{0}^{\phi}F(\phi)d\phi=-\frac{\Omega_{s}^{2}}{\sin \psi_{s}}\left(\sin\psi\,-\phi\cos\psi_{s}\right)+I_{0}\end{array} \tag{4.33}\]

The constant of integration \(I_{0}\) can be found as function of the particle's phase \(\phi\) and angular frequency \(\vec{\phi}\), by multiplying both terms of Eq. 4.30 for \(\dot{\phi}\), by integrating them in \(dt\), and finally integrating the r.h.s in \(d\phi\):

\[\begin{array}{l}\dot{\phi}\vec{\phi}=\frac{\Omega_{s}^{2}}{\sin\psi_{s}} \left[\cos(\phi+\psi_{s})-\cos\psi_{s}\right]\dot{\phi};\\ \frac{d}{dt}\left(\frac{\dot{\phi}^{2}}{2}\right)=\frac{\Omega_{s}^{2}}{\sin \psi_{s}}\left[\cos(\phi+\psi_{s})-\cos\psi_{s}\right]\frac{d\phi}{dt};\\ \frac{\dot{\phi}^{2}}{2}-\frac{\Omega_{s}^{2}}{\sin\psi_{s}}\left(\sin\psi\,- \phi\cos\psi_{s}\right)=T(\dot{\phi})+U(\phi)=I_{0}\end{array} \tag{4.34}\]

The first term of Eq. 4.34-bottom row depends only from the "phase velocity" \(\dot{\phi}\), therefore it can be associated to a kinetic energy \(T(\dot{\phi})\). The second term, made of an oscillatory and a linear function of \(\phi\), can be interpreted as a potential energy \(U(\phi)\). Consequently, \(I_{0}\) is the particle's total energy, in proper units of the reduced variables, and it is a constant of motion. This fact should not surprise because the non-interacting particles beam in the accelerator behaves as a conservative system, as long as the particles' total energy does not vary on average over one turn. The individual kinetic and potential contributions to the total energy are, of course, allowed to vary, which makes synchrotron oscillations to happen.

Figure 4.6 offers a graphical representation of the energy gain (Eq. 4.27) and of the RF potential (Eq. 4.33) as function of the particle's phase. The potential is evaluated for three values of the synchronous phase, \(\psi_{s}=\pi/2,\,2\pi/3,\,5\pi/6\). In proximity of \(\psi_{s}\), \(U(\phi)\) shows a _potential well_, around which the motion remains stable, i.e.,

the orbit in the longitudinal phase space is closed (ellipse) because the RF force is restoring. This is the case of the small oscillation amplitudes in Eq. 31.

We anticipate that, as shown in the figure, the width of the potential well, both in phase and in amplitude, depends from the value of \(\psi_{s}\). The absolute level of the well does not change over time for \(\psi_{s}=\pi/2\), which corresponds to null acceleration (see Eq. 27). For different values of \(\psi_{s}\), instead, the level of the potential well varies over consecutive RF cycles, though the beam energy is still approximately constant on average over each individual turn.

#### RF Acceptance

For sufficiently large oscillation amplitudes in Eq. 30, the particle's orbit in the longitudinal phase space can be unbounded. The phase space trajectory which constitutes the boundary between bounded and unbounded motion is called _separatrix_.

Figure 6: From top to bottom: effective energy gain, normalized RF potential, and separatrix, as function of the particle’s phase. \(U\) and \(\delta\) are plotted for 3 values of the synchronous phase. The potential for the stationary bucket (\(\psi_{s}=\pi/2\)) is amplified by a factor 10 for better visibility of the potential well. In this figure, \(E_{0}=3\) GeV, \(T_{0}=1\)u s, \(q\,V_{0}=1\) MeV, \(\Omega_{s}=3\) kHz

[MISSING_PAGE_FAIL:87]

#### Stationary Bucket

One could now wonder which is the value of \(\psi_{s}\) that maximizes in absolute sense \(G(\psi_{s})\), i.e., the RF energy acceptance. First, the range of \(\psi_{s}\) which ensures phase stability has to be identified. This is illustrated in Fig. 4.4: \(0\leq\psi_{s}\leq\pi/2\) for \(\eta<0\) and \(3\pi/2\leq\psi_{s}\leq 2\pi\) for \(\eta>0\). Since \(G(\psi_{s})\) is a monotonic function of \(\psi_{s}\), it is maximized by \(\psi_{s}=\pi/2\) above transition, and by \(\psi_{s}=3\pi/2\) below transition. In both cases, \(\hat{G}=\sqrt{2}\). Both those phases correspond to null acceleration (see Eq. 4.27), and the RF bucket is said _stationary_.

For values of \(\psi_{s}\neq\pi/2,\,3\pi/2\), the absolute level of the potential well changes with the phase as a consequence of some net acceleration over time, the RF bucket area is reduced and made asymmetric in phase (see Fig. 4.6). This suggests that for maximizing the beam injection efficiency from a booster ring, i.e. to maximize the number of particles collected by the storage ring in the RF bucket, the injection should be accomplished at fixed beam energy in the main accelerator (i.e., maximum RF bucket area).

Let us assume a storage ring above transition. The _maximum_ RF energy acceptance is found from Eq. 4.36 evaluated at \(\psi=\psi_{s}=\pi/2\), or equivalently from 4.37 evaluated at \(\psi_{s}=\pi/2\):

\[|\hat{\delta}_{acc}|=\tfrac{1}{\pi\,Q_{s}}\tfrac{q\,V_{0}}{E_{0}}=\sqrt{\tfrac {2\beta_{s}}{\pi h|\eta|}}\sqrt{\tfrac{q\,V_{0}}{E_{0}}} \tag{4.38}\]

Equation Eq. 4.36 is particularly helpful to calculate the total _bucket area_ of the stationary bucket, which therefore results the maximum bucket area among all possible synchronous phases. At first, we evaluate the separatrix equation for \(\psi_{s}=\pi/2\):

\[\begin{split}&\delta_{sep}(\psi_{s}=\pi/2)=\pm\tfrac{1}{2\pi\,Q_{s }}\tfrac{q\,V_{0}}{E_{0}}\sqrt{2(1+\sin\,\psi)}=\pm\tfrac{1}{2\pi\,Q_{s}} \tfrac{q\,V_{0}}{E_{0}}\sqrt{2(1+\cos\phi)}=\\ &=\pm\tfrac{1}{\pi\,Q_{s}}\tfrac{q\,V_{0}}{E_{0}}\sqrt{\cos^{2} \tfrac{\phi}{2}}=\pm\tfrac{1}{\pi\,Q_{s}}\tfrac{q\,V_{0}}{E_{0}}\cos\tfrac{ \phi}{2}=\hat{\delta}_{acc}\cos\tfrac{\phi}{2}\end{split} \tag{4.39}\]

The bucket area can now be calculated:

\[A_{bk}=2\int_{-\pi}^{\pi}\delta_{sep}(\phi)d\phi=2|\hat{\delta}_{acc}|\int_{- \pi}^{\pi}\cos\tfrac{\phi}{2}d\phi=\tfrac{2}{\pi\,Q_{s}}\tfrac{q\,V_{0}}{E_{0 }}\cdot 4=8|\hat{\delta}_{acc}| \tag{4.40}\]

The phase acceptance \(\delta\phi_{acc}\) can also be calculated as the distance in phase of the two points at which the separatrix crosses the horizontal axis. For the stationary bucket, Eq. 4.39 says that this happens at \([-\pi,0]\) and \([\pi,0]\), so that \(\delta\phi_{acc}=2\pi\). Any different synchronous phase from that one of the stationary bucket implies a smaller acceptance (see Fig. 4.6).

#### Discussion: Short Bunches in a Storage Ring

Beam dynamics in a synchrotron is periodic, and particles can be stored for long time. In this case, they emit radiation continuously, and such emission leads to equilibriumbeam sizes independent from the beam parameters at injection. Since the previous analysis applies to the particles' motion in a steady-state condition, we want to use the constant of motion to find the relationship between bunch duration and energy spread at equilibrium, in the approximation of small oscillation amplitudes.

Equation 4.31 says that the generic particle's orbit in the longitudinal phase space (\(\phi,\dot{\phi}\)) is an ellipse. Let us consider the particle representative of the bunch envelope. The ellipse horizontal and vertical semi-axis has coordinates [\(\phi_{max}\), 0] and [0, \(\dot{\phi}_{max}\)]. In the approximation of small amplitudes of oscillation (\(\phi\approx 0\)) in the stationary bucket (\(\psi_{s}=\pi/2\)), Eq. 4.34 becomes:

\[\begin{array}{l}\frac{\dot{\phi}^{2}}{2}-\Omega_{s}^{2}\left[\sin(\phi+ \frac{\pi}{2})-\phi\cos\frac{\pi}{2}\right]=\frac{\dot{\phi}^{2}}{2}-\Omega_{s }^{2}\cos\phi\approx\frac{\dot{\phi}^{2}}{2}-\Omega_{s}^{2}\left(1-\frac{\phi ^{2}}{2}\right)=const.\\ \\ \Rightarrow\dot{\phi}^{2}+\Omega_{s}^{2}\phi^{2}=I_{1}=const.\end{array} \tag{4.41}\]

At the two extremes of the semi-axis, Eq. 4.41 becomes:

\[\left\{\begin{array}{l}\phi=0\Rightarrow\dot{\phi}=\dot{\phi}_{max}=\sqrt{ I_{1}}=-\left(\frac{h\eta\alpha_{s}}{p_{s}C}\right)w_{max}\\ \\ \dot{\phi}=0\Rightarrow\Omega_{s}\phi=\Omega_{s}\phi_{max}=\sqrt{I_{1}}=- \left(\frac{h\eta\alpha_{s}}{p_{s}C}\right)w_{max}\end{array}\right. \tag{4.42}\]

and we made use of the expression of \(\dot{\phi}\) in Eq. 4.30. From the bottom equation it follows:

\[\phi_{max}=-\left(\frac{h\eta\alpha_{s}}{p_{s}C}\right)\frac{w_{max}}{\Omega_{ s}}=-\frac{h\eta c}{\Omega_{s}\,R_{s}}\frac{w_{max}\omega_{s}}{2\pi\,E_{0}} \approxeq\left(\frac{h\alpha_{c}}{Q_{s}}\right)\delta_{max} \tag{4.43}\]

The very last equality is obtained by recalling the definition of \(Q_{s}\) and the approximation is for particles at ultra-relativistic velocities above transition energy (\(\beta_{s}\to 1\), \(\eta\to-\alpha_{c}\)).

The full-width bunch duration is finally estimated:

\[\Delta t_{b}=\frac{2\phi_{max}}{\alpha_{RF}}=(2\delta_{max})\frac{h\alpha_{c} \omega_{s}}{h\alpha_{s}\Omega_{s}}=(2\delta_{max})\alpha_{c}\sqrt{\frac{p_{s} C}{h\alpha_{s}\alpha_{c}\,Q_{0}}}\propto\sqrt{\frac{\alpha_{c}}{\alpha_{ RF}\,V_{0}}} \tag{4.44}\]

This result suggests that bunch duration in a synchrotron can be shortened by means of an arrangement of the lattice to produce a small momentum compaction--so-called _low-\(\alpha\)_ mode of operation--and/or a high peak RF voltage at any fixed RF frequency (i.e., a high time-slope of the accelerating voltage).

For typical parameters of a modern electron light source like \(V_{0}\approx 2\) MV, \(E_{0}\sim 3\) GeV and \(Q_{S}\sim 0.003\), Eq. 4.38 predicts \(\delta_{acc}\approx 7\%\). In reality, nonlinearities associated to higher order terms of the momentum compaction may severely limit the RF energy acceptance, often reduced to 2-4%.

The corresponding electron bunch duration can be estimated by considering that the beam energy spread at equilibrium is typically a small fraction of the RFenergy acceptance, and often close to \(\delta_{max}=0.1\%\). We then assume \(\omega_{s}\approx 10\) MHz, \(h=800\) and \(\alpha_{c}=10^{-3}\). It results \(\Delta t_{b}\approx 23\) ps. Most recent storage rings with 10-fold smaller \(\alpha_{c}\) (so-called "fourth generation") can accommodate 3-fold shorter bunches at low currents. However, RF cavities tuned at a higher harmonic of the fundamental frequency are commonly used to flatten the potential well, hence to elongate the bunch, diluting the charge density, to eventually minimize particles' interaction internal to the bunch. Typical elongations by a factor 3-5 can force the bunch duration to \(\sim\)\(60-150\) ps fwhm, for ring circumferences in the range 0.2-2 km, and beam energies in the range 1-6 GeV.

#### Energy Ramp

We consider a synchrotron in energy ramp mode, where the beam energy is increased turn-by-turn by virtue of "excess" of energy gain in the RF cavity. During the energy ramp, the closed orbit in the accelerator has to be kept (approximately) constant because the local curvature radius \(r\) in the dipole magnets is fixed, as dictated by the geometry of the vacuum chamber. By recalling the relationship between longitudinal momentum and curvature radius in the presence of Lorentz's force in Eq.2.5, the aforementioned condition implies a variation of the dipole magnetic field with time, \(\dot{p}_{z}=q\,\dot{B}_{y}r\). Hence, the energy gain per turn can be expressed as follows:

\[\begin{array}{l}(\Delta E)_{turn}=(\Delta p_{z})_{turn}\frac{c^{2}}{v_{z}}= q\,\dot{B}_{y}r\,T_{0}c=q\,\dot{B}_{y}r\,2\pi\,R_{s}\equiv q\,V_{0}cos(\psi_{s}- \psi_{0})\\ \\ \Rightarrow\psi_{s}(t)=\psi_{0}+\arccos\left(2\pi\,R_{s}r\,\frac{\dot{B}_{y}} {q\,V_{0}}\right)\end{array} \tag{4.45}\]

We find that the synchronous phase must vary during the energy ramp, according to the ramp of the magnetic field.

Moreover, by virtue of a non-zero slip factor (whose meaning applies in this case to the motion of the synchronous particle in the presence of variation of the reference energy), we expect that also the revolution frequency varies. This has to be kept synchronous to the radiofrequency, which therefore has to vary as well:

\[\begin{array}{l}f_{RF}(t)=\frac{h\alpha_{b}(t)}{2\pi}=\frac{h}{2\pi\,R_{s}} \,\beta_{s}(t)c=\frac{h}{2\pi\,R_{s}}\frac{p_{z,s}(t)}{\gamma(t)m_{0}}=\frac{ h}{2\pi\,R_{s}}\frac{q\,B_{y}(t)rc^{2}}{\sqrt{(p_{z,s}c)^{2}+(m_{0}c^{2})^{2}}}=\\ \\ =\frac{h}{2\pi\,R_{s}}\frac{q\,B_{y}(t)rc^{2}}{\sqrt{(q\,B_{y}(t)rc)^{2}+(m_{0} c^{2})^{2}}}=\frac{hc}{2\pi\,R_{s}}\left(\frac{qr}{m_{0}c}\right)\frac{B_{y}(t)}{ \sqrt{\left[1+\left(\frac{qr}{m_{0}c}\right)^{2}B_{y}(t)^{2}\right]}}\end{array} \tag{4.46}\]

For \(p_{z}c\gg m_{0}c^{2}\), \(f_{RF}/h\to f_{s}\), i.e. the revolution frequency in the ultra-relativistic limit. As an example, Fig. 4.7 shows the variation with time of \(B_{y}\), \(\Delta f_{RF}\) and \(\psi_{s}\) for protons in energy ramp in the Tevatron collider.

#### Discussion:Tevatron Proton Collider

The Tevatron is a proton-antiproton synchrotron collider, initially designed for a kinetic energy ramp 10-400 GeV, later on upgraded to 150-980 GeV. It was put in shut down in 2011. The ring circumference is C = 6.28 km, the harmonic number is \(h=1110\), and the RF peak voltage \(V_{0}=2\) MV. The superconducting dipole magnets are 10 m-long, each dipole bending by 0.74 deg. We assume the energy ramp is concluded in \(\Delta t_{r}=1\) min. Finally, the gamma-transition is set to 10.

We would like to calculate: (i) the variation of the radio-frequency during the energy ramp in the range 10-400 GeV, and in the range 150-980 GeV, assuming the dipole curvature radius is kept constant, (ii) the average energy gain per turn and the rate of increase of the dipoles' field, for the upgraded energy ramp, and (iii) the synchrotron frequency at the initial and at the final energy, for the upgraded energy ramp.

\(\Delta f_{RF}\) can be calculated by means of Eq. 4.46, where a constant beam orbit implies \(r\approx const\). The curvature radius is \(r=10\) m/\(0.013\) rad = 775 m. The dipole field at the beginning and at the ed of the energy ramp is calculated through \(p_{z}=eB_{y}r\). The field ranges [0.05-1.73] T in the low energy ramp, and [0.65-4.22] T in the high energy ramp. The corresponding variation of radiofrequency is \(\Delta f_{RF}=-195\) kHz and \(\Delta f_{RF}=-1\) kHz, respectively. The main RF is in the range 45-53 MHz.

The average energy gain per turn is calculated from the variation of the longitudinal momentum corresponding to the kinetic energy range 150-980 GeV. Since the protons are ultra-relativistic, we can write \(\langle(\Delta E)_{turn}\rangle=\langle\beta c(\Delta p)_{turn}\rangle\approx c \Delta p\,\frac{T_{0}}{\Delta t_{r}}=0.3\) MeV. The field rate is retrieved from the momentum rate, \((\Delta p)_{turn}\approx er\,\dot{B}\,T_{0}\), which provides \(\dot{B}\approx 0.06\) T/s.

Finally, Eq. 4.32 allows us to calculate the synchrotron frequency at the initial and final beam energy. But first, the slip factor and the synchronous phase have to be found. For the former, we assume that the linear optics remains substantially unchanged during the energy ramp, which implies a constant momentum compaction \(|\alpha_{c}|=1/\gamma_{tr}^{2}=0.01\). Since the \(\gamma\)-factor amounts to \(E/(m_{p}c^{2})=[161, 1046]\) at the

Figure 4.7: The protons’ kinetic energy in the Tevatron collider is assumed to be varied from 150 GeV to 980 GeV in 1 minute. The dipole field increases over time with a parabolic ramp. The synchronous phase and the radiofrequency are varied as well. The following parameters were considered: 6.28 km-long circumference, 10 m-long superconducting dipole magnets, \(h=1110\), and 0.74 deg dipole bending angle (kept constant during the energy ramp)two kinetic energies of 150 GeV and 980 GeV, the slip factor's absolute value is substantially \(\alpha_{c}\).

The synchronous phase is retrieved from the peak voltage and the energy gain per turn calculated above, which have to satisfy \((\Delta E)_{turn}=eV_{0}\cos\psi_{s}\). Hence, \(\psi_{s}=1.4\) rad (approximately 81\({}^{\circ}\) far from the accelerating crest), and \(\frac{\Omega_{s}}{2\pi}=\) [0.39, 0.16] kHz.

#### Summary

Single particle longitudinal dynamics in a high energy synchrotron can be summarized as follows.

* When the approximation of small oscillation amplitudes is met, i.e., \(\Delta z_{b}\ll\lambda_{RF}\) and \(\delta\ll 1\), the motion is stable for those synchronous phases which make \(\Omega_{s}^{2}>0\) (Eq. 32).
* For large amplitudes of oscillation, the motion is stable within the phase space region delimited by the separatrix, namely, the RF bucket (Eq. 36). Its half-height is the RF energy acceptance of the accelerator, which depends from the synchronous phase.
* The synchronous phase which maximizes the RF bucket area is that for null acceleration (stationary bucket, Eq. 39). The bucket area is linearly proportional to the bucket height. The phase acceptance is also maximum and equal to \(2\pi\).
* The RF energy acceptance is weakly dependent but still maximized by a large product \(\omega_{RF}V_{0}\), and small \(\alpha_{c}\) (Eq. 38). However, for any equilibrium energy spread, the bunch duration shows the opposite dependence and therefore it is squeezed in proportion to the enlargement of the RF energy acceptance (Eq. 44).
* When operated in energy ramp mode, the synchronous phase and the frequency of the RF cavity have to vary with time in order to keep the closed orbit and the energy gain per turn approximately constant, and in accordance to the variation of the dipoles' magnetic field (Eqs. 45 and 46).

### Transverse Dynamics

#### Multipolar Field Expansion

The sequence of magnetic elements of an accelerator is called _magnetic lattice_[2; 3]. Dipole and quadrupole magnets allow control of the particles' transverse motion at first order in the particle's coordinates. Their principle of operation is sketched in Fig. 8. In synchrotrons, other multipole magnets are adopted for control of the dynamics at higher orders. This suggests the need for a description of the magnetic field components through an expansion around the reference orbit.

Let us consider an electromagnet with Fe-poles at distance \(R_{b}\) from the orbit, and centered on it. \(R_{b}\) is called _bore radius_; in the case of a dipole magnet bending in the horizontal plane, it corresponds to its half-gap. If we describe \(B(x,\,y)\) as an analytical continuous function of the spatial coordinates, we can expand it in Taylor series around the reference orbit. For the vertical field component in proximity of \(y=0\):

\[\left\{\begin{array}{l}B_{y}(x)\approxeq b_{0}+b_{1}\left(\frac{x}{R_{b}} \right)+b_{2}\left(\frac{x}{R_{b}}\right)^{2}+...=\sum_{0}^{\infty}\frac{R_{b} ^{m}}{m!}\left(\frac{\partial^{m}B_{y}}{\partial x^{m}}\right)\left(\frac{x}{ R_{b}}\right)^{m}\\ b_{m}:=\frac{R_{b}^{m}}{m!}\left(\frac{\partial^{m}B_{y}}{\partial x^{m}} \right)_{y=0}\quad\quad m\in\mathbb{N}\end{array}\right. \tag{4.47}\]

\(m\) is the order of expansion, where \(m=0\) is for a dipole, \(m=1\) for a quadrupole, \(m=2\) for a sextupole, etc.

Magnets which generate a Lorentz's force dependent _only_ from the coordinate in the force plane (\(F_{x}=F_{x}(x),\,F_{y}=F_{y}(y)\)) are said _normal_. In _skew_ magnets, \(F_{x}=F_{x}(x,\,y),\,F_{y}=F_{y}(x,\,y)\). The most general superposition of normal and skew

Figure 4.8: Top: 2 m-long dipole magnet of the Elettra booster ring. Cu-coils surrounding the north and south Fe-pole generate a magnetic field (\(H\)) which bends charged particles via Lorentz’s force (\(F\)). Bottom: quadrupole magnet on a rotating coil measurement bench. On the right, black arrows are magnetic field lines, red arrows are forces exerted on a positive charge entering the plane, and blue lines show a path concatenated to the coils. (Photos credit D. Castronovo, Elettra Sincrotrone Trieste)

field components, at any order, can be written in a compact complex notation:

\[B_{y}+i\,B_{x}=\sum_{0}^{\infty}\,\left(b_{m}+ia_{m}\right)\left(\frac{x+iy}{R_{ b}}\right)^{m} \tag{4.48}\]

where \(b_{m}\) are non-zero in normal magnets and \(a_{m}\) are non-zero in skew magnets. For example, in a normal quadrupole \(b_{1}=R_{b}\frac{\partial\,B_{y}}{\partial x}\), \(a_{1}=0\). In a skew quadrupole, \(b_{1}=0\), \(a_{1}=R_{b}\frac{\partial\,B_{x}}{\partial x}\).

Magnets built on purpose with a single field order are named "separate function" magnets. On the opposite, magnets built with superposed field orders are "combined function" magnets. The most common of this kind are dipole magnets with integrated quadrupole field gradient, hence \(B_{y}=b_{0}+b_{1}\frac{x}{R_{b}}\), \(B_{x}=b_{1}\frac{y}{R_{b}}\).

Magnets deviate the particles' trajectory in analogy to lenses in geometric optics. A magnetic lattice made of dipole and quadrupole magnets only implies forces at most linear with the particle's coordinates, and the particle's transverse motion is said _linear optics_. Sextupole, octupole and higher order magnets contribute to the _nonlinear optics_.

Maxwell's equations forbid the abrupt transition from a magnetic field region to a field-free region. Indeed, the region in proximity of the magnets' edges contain residual field components, not described by the aforementioned formulas. Such _fringe fields_ typically contribute poorly to the overall focusing property of the magnet. However, they can be important when a large number of magnets is installed in a tight lattice. In this case, interference of fringe fields of adjacent magnets can lead to a noticeable modification of the optics compared to the ideal "hard-edge" field model.

#### Quadrupole Magnet

A quadrupole magnet is made of four center-symmetric Fe-poles surrounded by coils. These generate the magnetic field, which is deviated by the high relative permeability of the poles towards the center of the magnet, see Fig. 4.8-right plot. For symmetry, the field is null at the exact center of the magnet, while the field lines are more dense in proximity of the poles. Moreover, owing to \(\vec{\nabla}\,\times\,\vec{B}=0\) in the region internal to the magnet, a normal quadrupole shows \(\frac{\partial\,B_{y}}{\partial x}=\frac{\partial\,B_{y}}{\partial y}\). This implies that a positive gradient \(g=\frac{\partial\,B_{y}}{\partial x}\) acting as a restoring or focusing force in the horizontal plane, corresponds to a defocusing force in the vertical plane, and viceversa. Let us verify this with the notation introduced in Eq. 4.48:

\[B_{y}+i\,B_{x}=b_{1}\frac{x}{R_{b}}-a_{1}\frac{y}{R_{b}}+ib_{1}\frac{y}{R_{b}}+ ia_{1}\frac{x}{R_{b}} \tag{4.49}\]

\[\Rightarrow\left\{\begin{array}{l}B_{y}=b_{1}\frac{x}{R_{b}}=\frac{\partial \,B_{y}}{\partial x}x\equiv gx\\ \\ B_{x}=b_{1}\frac{y}{R_{b}}=\frac{\partial\,B_{y}}{\partial x}\,y=\frac{\partial \,B_{x}}{\partial y}\,y=gy\end{array}\right. \tag{4.50}\]The Lorentz's force results:

\[\vec{F}=\left|\begin{array}{ccc}\hat{x}&\hat{y}&\hat{z}\\ 0&0&v_{z}\\ gy&gx&0\end{array}\right|=-(qv_{z}gy)\hat{x}+(qv_{z}gx)\hat{y} \tag{4.51}\]

In summary, Lorentz's force in a normal quadrupole is linearly proportional to the particle's distance from the magnet center, equal in strength but opposite in sign along the horizontal and the vertical axis. The constant of linear proportionality is the field gradient. At any point of coordinates \(|x,\,y|<R_{b}\), the field vector can be decomposed in a horizontal and a vertical component.

The field gradient can be expressed as function of \(R_{b}\) and of the Ampere-turns \(NI\) by evaluating the magnetic field curvilinear integral along the path in Fig. 4.8 (blue dashed arrows). For simplicity, we assume a relative permeability of the poles much larger than in vacuum (\(\mu_{r}\gg\mu_{0}\)):

\[\begin{array}{l}\hat{\oint}\vec{H}d\vec{s}=\int_{1}\vec{H}d\vec{s}+\int_{2} \vec{H}d\vec{s}\int_{3}\vec{H}d\vec{s}=\int_{0}^{R_{b}}H(r)dr+\int_{1}^{2}\vec{ H}d\vec{s}+\int_{2}^{3}H_{y}dx=\\ \\ =\int_{0}^{R_{b}}\frac{1}{\mu_{0}}\frac{\partial B_{r}}{\partial r}rdr+\int_{ 1}^{2}\frac{1}{\mu_{r}}\vec{B}d\vec{s}+0\approxeq\int_{0}^{R_{b}}\frac{g}{\mu _{0}}rdr=\frac{gR_{b}^{2}}{2\mu_{0}}\equiv NI.\\ \\ \Rightarrow g=\frac{2\mu_{0}NI}{R_{b}^{2}}\end{array} \tag{4.52}\]

Hence, the field gradient is made stronger by a smaller bore radius, for any given Ampere-turns. In practice, the size of \(R_{b}\) has to be traded off with the space required to accommodate the beam vacuum chamber.

In the literature, a quadrupole is said to be _focusing_ (QF) when it pushes off-axis particles towards the axis in the _horizontal_ plane. The opposite is for a _defocusing_ quadrupole (QD). Because of its symmetry, a QF can be made QD by simply rotating the magnet by \(90^{\circ}\). A normal quadrupole rotated by \(45^{\circ}\) becomes a skew quadrupole.

A QF becomes a QD for a particle which inverts its direction of motion through the magnet. Since this is equivalent to the change of sign of the particle's charge, some circular colliders have been conceived with two distinct vacuum chambers in the same magnets. The two colliding beams made of particles with opposite charge (e.g., electron-positron, electron-proton, or proton-antiproton colliders), travel on opposite directions. In this way, dipole and quadrupole magnets have exactly the same effect on the two counter-propagating beams, which therefore share the same linear optics.

#### Strong Focusing

Strong focusing, also named _Alternating-Gradient Focusing_, was first conceived by N. Christofilos in 1949 but not published. In 1952, the strong focusing principle was independently developed by E. Courant, M. S. Livingston, H. Snyder and J. Blewett at Brookhaven National Laboratory, then quickly deployed on the Alternating Gradient Synchrotron at the Brookhaven National Laboratory, USA.

The equations of motion for the generic particle are derived below in the presence of dipoles and quadrupoles only. For this reason, they are said to be in the _linear approximation_. At this stage, the geometry of the accelerator is arbitrary. The derivation is done with the following prescriptions.

* Since we are also interested to the particle's motion on a closed orbit, it is convenient to change the independent variable from \(t\) (time) to \(s\), the latter being the longitudinal spatial coordinate along the orbit in the Frenet-Serret reference system. In a synchrotron \(s\in[0,\,C]\). This change of variable implies \(\frac{d}{dt}\rightarrow\frac{d}{ds}\frac{ds}{dt}=v_{z}\frac{d}{ds}\).
* The break of azimuthal symmetry of the magnetic field in a synchrotron in general implies different field values at different points along the accelerator: \(B_{y}=B_{y}(s)\), \(\frac{\partial\,B_{y}}{\partial x}=g(s)\).
* The generic particle is assumed to travel close to the reference orbit, \(x\), \(y<<r\), \(R_{b}\). The local curvature radius \(r\) can therefore be expanded at first order in \(x\) around the _reference_ curvature radius \(R\), or \(r=R(1+x/R)\).
* The generic particle is also assumed to be off-energy, \(p_{z}=p_{z,s}(1+\delta)\), \(\delta<<1\). The synchronous particle's longitudinal momentum \(p_{z,s}\) is assumed to be constant (locally in a linac, on average over one turn in a synchrotron).

Newton's equation for the horizontal plane is:

\[m\ddot{x}(t)-m\dot{\theta}(t)^{2}r=-qv_{z}B_{y};\] \[\ddot{x}(s)-\frac{1}{v_{z}^{2}}\frac{v_{z}^{2}}{r(s)}=-\frac{1}{ v_{z}^{2}}\frac{qv_{z}B_{y}(s)}{m};\] \[\ddot{x}(s)\approx\frac{1}{R(s)}\left[1-\frac{x}{R(s)}\right]- \frac{q\,B_{y}(s)}{p_{z}};\] \[\ddot{x}(s)\approx\frac{1}{R(s)}\left[1-\frac{x}{R(s)}\right]- \frac{q\,B_{0}(s)}{p_{z,s}}\left[1+\frac{g(s)}{B_{0}(s)}x(s)\right](1-\delta); \tag{4.53}\] \[\ddot{x}(s)\approx\frac{1}{R(s)}-\frac{x}{R(s)^{2}}-\frac{1}{R(s )}+\frac{\delta}{R(s)}-\left[\frac{q\,g(s)}{p_{z,s}}\right](1-\delta)x(s);\] \[\Rightarrow\ddot{x}(s)+\left[k(s)(1-\delta)+\frac{1}{R(s)^{2}} \right]x(s)=\frac{\delta}{R(s)}\]

We have introduced the _normalized quadrupole gradient_\(k=\frac{eg}{p_{z}}\). In practical units:

\[k[m^{-2}]=0.3\frac{g[T/m]}{p_{z,s}[GeV/c]} \tag{4.54}\]

We find a similar equation for the vertical plane, for which \(R\rightarrow\infty\):

\[\ddot{y}(s)=\frac{q\,B_{x}(s)}{p_{z}};\] \[\ddot{y}(s)\approx\left[\frac{q\,g(s)}{p_{z,s}}y(s)\right](1- \delta); \tag{4.55}\] \[\Rightarrow\ddot{y}(s)-k(s)(1-\delta)y(s)=0\]Equations 4.53 and 4.55 are called _Hill's equations_. They describe a _quasi-harmonic_ oscillator, i.e., an oscillator in which the angular frequency is a function of the independent variable \(s\). We draw the following observations.

1. Consistently with Maxwell's equations, \(k(s)\) has opposite sign in Eqs. 4.53 and 4.55. As previously observed, this means that a QF (QD) is focusing in the x-plane (y-plane) and defocusing in the y-plane (x-plane). This suggests that a sequence of QF and QD could be suitably arranged to guarantee stability along the orbit in both transverse planes.
2. The field gradient can be re-written in terms of the field index introduced for weak focusing in Eq. 2.11: \[k=\frac{q\,g}{p_{z,s}}=\frac{q}{q\,B_{0}\,R}\frac{\partial\,B_{y}}{\partial x}= -\frac{1}{R^{2}}n\] (4.56) In this case, the dipolar and the gradient field component are intended to belong to distinct magnets (dipole and quadrupole, respectively). Since commonly \(|k|>>\frac{1}{R^{2}}\), i.e. the focusing force by quadrupoles is much stronger than weak focusing due to the orbit curvature in the dipoles, one has \(|n|>>1\), which implies a large number of oscillations per turn. Such operating mode is named _strong focusing_, as opposite to weak focusing in betatrons and cyclotrons.
3. In the presence of normal magnets only, Hill's Eqs. 4.53 and 4.55 apply to the linearly independent variables \(x(s)\), \(y(s)\). Their oscillations around the orbit are called _betatron oscillations_.
4. Hill's equations are linear in \(\delta\). This leads to a residual focusing term \(k\delta<<k\) which constitutes a _chromatic_ perturbation to the on-energy linear optics. The chromatic term on the r.h.s. of Eq. 4.53 determines a particular solution of the differential equation, due to the orbit distortion of off-energy particles in the magnets.

##### Discussion: FODO Lattice

Let us demonstrate that an ideal magnetostatic force focusing simultaneously in the horizontal and vertical plane cannot exist. This shall bring us to the conclusion that alternated restoring strengths can be implemented, instead, and that under certain circumstances this guarantees stability in both planes along the beam line.

Ideally, we would like to have:

\[F_{x}=0,\quad F_{y}=-k_{y}\,y,\quad F_{y}=-k_{z}z \tag{4.57}\]

At the same time, in the region occupied by the beam and internal to the magnet, \(\vec{\nabla}\vec{B}=0\), which implies null divergence of the total Lorentz's force:

\[\tfrac{\partial\,F_{x}}{\partial x}+\tfrac{\partial\,F_{y}}{\partial y}+\tfrac {\partial\,F_{z}}{\partial z}=0\ \ \Rightarrow\ \ k_{y}=-k_{z} \tag{4.58}\]

This proves the thesis.

The principle of alternating strong quadrupole gradients is illustrated in Fig. 4.9. A series of QFs and QDs, denominated FODO lattice in the literature, can guarantee stable motion in _both_ transverse planes if a proper choice of the gradients and the distances between the magnets is done. This basically corresponds to a situation in which, in each plane, the particle's distance from the magnetic axis of QFs is systematically larger than that in QDs. Since the quadrupole force is proportional to the particle's distance from the magnet's center (see Eq. 4.51), the restoring force of QFs is always larger than the repulsive one in QDs, and the particle continues oscillating around the reference orbit.

The same lattice, however, can lead to unstable motion, in either one plane or the other or both, if the quadrupoles' gradient and their relative distance is not chosen properly. Either too weak or too strong gradients can lead to instability: the particle starts deviating more and more from the orbit, until it gets lost on the vauum chamber.

#### Principal Trajectories

The solution of Hill's complete equation in Eq. 4.53 is the linear superposition of the solution of the homogeneous equation \(x_{\beta}(s)\) and the particular solution \(x_{D}(s)\). The former is function of initial conditions \(x_{0}\), \(x_{0}^{\prime}\), and it is said to describe the _betatron motion_. The latter is proportional to the relative momentum deviation through the momentum-dispersion function, \(x_{D}(s)=D(s)\delta\), and for this reason it is said to describe the _dispersive motion_:

\[\left\{\begin{aligned} x(s)=x_{\beta}(s)+x_{D}(s)=C(s)x_{0}+S(s)x_{0 }^{\prime}+D(s)\delta\\ x^{\prime}(s)=x_{\beta}^{\prime}(s)+x_{D}^{\prime}(s)=C^{\prime}(s) x_{0}+S^{\prime}(s)x_{0}^{\prime}+D^{\prime}(s)\delta\end{aligned}\right. \tag{4.59}\]

The derivatives are all intended with respect to the independent variable \(s\). For the sake of brevity, the suffix \(x\) of the optical functions \(C\), \(S\), \(D\) is suppressed. An identical formalism applies to the vertical plane.

Figure 4.9: Particle’s trajectory in a FODO for a stable (left) and unstable motion (right), in the horizontal (top) and vertical plane (bottom). Focusing quadrupoles in one plane behave as defocusing quadrupoles in the other plane

\(C(s)\), \(S(s)\) are called _principal trajectories_, \(D(s)\) is the dispersion function previously introduced in Eq. 4.16. \(C\), \(S\) and \(D\) are all solutions of Hill's equation, in the homogeneous and complete form, respectively. Equation 4.59 can be re-written in vectorial form by collecting the principal trajectories in a _transfer matrix_:

\[\vec{x}\left(s\right)=\binom{x}{x^{\prime}}_{s}=\binom{C}{C^{\prime}}\;S^{ \prime}\bigg{)}_{0\to s}\binom{x}{x^{\prime}}_{0}=M_{(0\to s)}\vec{x}_{0} \tag{4.60}\]

The determinant \(\det M=CS^{\prime}-SC^{\prime}\equiv W(s)\) is the _Wronskian_ associated to Hill's second order homogeneous differential equation.

The initial conditions of \(C\), \(S\) are chosen as follows: \(C(0)=S^{\prime}(0)=1\), \(C^{\prime}(0)=S(0)=0\). Their properties are studied by introducing a "frictional" or "dissipative" term \(\xi\) in the homogeneous Hill's equation. Hereafter, we neglect the chromatic focusing error \(k\delta\) for simplicity. The dissipative Hill's equations become:

\[\left\{\begin{array}{ll}C^{\prime\prime}+\xi C^{\prime}+kC=0&\quad\cdot(-S) \\ \\ S^{\prime\prime}+\xi S^{\prime}+kS=0&\quad\cdot(C)\end{array}\right. \tag{4.61}\]

The two equations are multiplied as indicated in Eq. 4.61 and summed to obtain:

\[\begin{array}{ll}\left(CS^{\prime\prime}-SC^{\prime\prime}\right)+\xi\left( CS^{\prime}-SC^{\prime}\right)=0;\\ \\ W^{\prime}+\xi W=0;\\ \\ \Rightarrow\;W(s)=W(0)\exp\left(-\int_{0}^{s}\xi(s^{\prime})ds^{\prime}\right) \\ \\ \Rightarrow\;W=1\,\Leftrightarrow\,\xi(s)=0\;\;\forall s\end{array} \tag{4.62}\]

We find that the Wronskian is constant (unitary) under two assumptions:

1. the motion is purely linear in the particle's coordinates (Eq. 4.53),
2. the motion is free of frictional forces (Eq. 4.62).

The physical meaning of the Wronskian is elucidated by considering a particle's trajectory in the transverse phase space \((x,\,x^{\prime})\). An arbitrarily small element of phase space area around the trajectory is, in general, expressed through the cross product \(A(s)\hat{k}=d\vec{x}(s)\times d\vec{x}^{\prime}(s)\), with \(d\vec{x}\), \(d\vec{x}^{\prime}\) the vectors pointing from the representative point of the particle to the vertices of the surface element, see Fig. 4.10.

In general, the two vectors are functions of \(s\), i.e., we assume the existence of a _linear map_\(M_{(0\to s)}\) under which the particle's coordinates and therefore the area transform:\[\left\{\begin{array}{l}d\vec{x}=d\vec{x}(dx_{0},dx_{0}^{\prime})\approxeq\left( \frac{\partial x}{\partial x_{0}}dx_{0},\,\frac{\partial x}{\partial x_{0}^{ \prime}}dx_{0}^{\prime}\right)\\ \\ d\vec{x}^{\prime}=d\vec{x}^{\prime}(dx_{0},dx_{0}^{\prime})\approxeq\left( \frac{\partial x^{\prime}}{\partial x_{0}}dx_{0},\,\frac{\partial x^{\prime}}{ \partial x_{0}^{\prime}}dx_{0}^{\prime}\right)\\ \end{array}\right. \tag{4.63}\]

\[\Rightarrow\vec{A}(s)=\left|\begin{array}{cc}\hat{\vec{i}}&\hat{\vec{j}}& \hat{\vec{k}}\\ \frac{\partial\vec{x}}{\partial x_{0}}dx_{0}&\frac{\partial x}{\partial x_{0} ^{\prime}}dx_{0}^{\prime}&0\\ \left|\frac{\partial x^{\prime}}{\partial x_{0}}dx_{0}&\frac{\partial x^{ \prime}}{\partial x_{0}^{\prime}}dx_{0}^{\prime}&0\\ \end{array}\right|= \tag{4.64}\]

\[=\left(\frac{\partial x}{\partial x_{0}^{\prime}}\frac{\partial x^{\prime}}{ \partial x_{0}^{\prime}}-\frac{\partial x}{\partial x_{0}^{\prime}}\frac{ \partial x^{\prime}}{\partial x_{0}}\right)dx_{0}dx_{0}^{\prime}\hat{\vec{k}}=\]

\[=\left(CS^{\prime}-S^{\prime}C\right)\vec{A}(0)=\vec{A}(0)\]

We conclude that the _Jacobian determinant_ of the function \(\vec{f}=(\vec{x}(x_{0},x_{0}^{\prime}),\,\vec{x^{\prime}}\)\((x_{0},x_{0}^{\prime}))\) is the Wronskian of the homogeneous Hill's equation. This implies that, under conditions 1. and 2. above, the _phase space area_ in proximity of a particle's trajectory is _preserved_, i.e., it is a constant of motion. This is the enunciation of _Liouville's theorem_ restricted to linear motion. The theorem will be demonstrated later on in a more general form in the framework of Hamiltonian dynamics.

#### Transfer Matrices

Hill's equation resembles that of a harmonic oscillator. This suggests solutions for \(C\) and \(S\) in the form of cos- and sin-like functions, respectively. Their amplitude and phase has to contain information on the magnetic focusing.

Our guess is in Eq. 4.65 below. We demonstrate _a posteriori_ that those functions indeed satisfy the homogeneous Eqs. 4.53 and 4.55. We restrict ourselves to the case of _constant focusing_ magnets, i.e., \(k(s)=const.\) and \(R(s)=const.\) inside each element of the accelerator, though \(k,\,R\) can certainly vary from element to element. In other words, \(k\) and \(R\) are Heaviside step-functions with argument \(s\)In case a magnetic element has a longitudinal field gradient, we could still apply the assumption of constant focusing to longitudinal slices of the magnets, such that the focusing properties within each slice are approximately constant. The principal trajectories are properly defined within each constant focusing element as follows:

\[\left\{\begin{array}{l}C(s)=\cos\left(\sqrt{K}s\right)\\ S(s)=\frac{1}{\sqrt{K}}\sin\left(\sqrt{K}s\right)\end{array}\right.,\;\;\;K:=k +\frac{1}{R^{2}} \tag{4.65}\]

We find for example for the \(C(s)\) function:

\[C^{\prime\prime}+kC=\cos\left(\sqrt{K}s\right)\cdot k+k\cdot\cos\left(\sqrt{K} s\right)=0 \tag{4.66}\]

The particular solution \(D(s)\) can in turn be expressed as function of the principal trajectories \(C(s)\), \(S(s)\):

\[D(s)=S(s)\int_{0}^{s}\frac{C(s^{\prime})}{R(s^{\prime})}ds^{\prime}-C(s)\int_{ 0}^{s}\frac{S(s^{\prime})}{R(s^{\prime})}ds^{\prime} \tag{4.67}\]

and we demonstrate below that such expression satisfies the complete Hill's equation for unitary \(\delta\). In the following, we make use of the fact that both \(C\), \(S\) satisfy Hill's homogeneous equation, and that \(W(C,S)\)=1:

\[\begin{array}{l}D^{\prime\prime}=\frac{d}{ds}D^{\prime}=\frac{d}{ds}\left[S ^{\prime}\int\frac{C(s^{\prime})}{R(s^{\prime})}ds^{\prime}+\frac{SC}{R}-C^{ \prime}\int\frac{S(s^{\prime})}{R(s^{\prime})}ds^{\prime}-\frac{CS}{R}\right]= \\ =S^{\prime\prime}\int\frac{C(s^{\prime})}{R(s^{\prime})}ds^{\prime}+\frac{S^{ \prime}C}{R}-C^{\prime\prime}\int\frac{S(s^{\prime})}{R(s^{\prime})}ds^{\prime }-\frac{C^{\prime}S}{R}=\\ =S^{\prime\prime}\int\frac{C(s^{\prime})}{R(s^{\prime})}ds^{\prime}-C^{\prime \prime}\int\frac{S(s^{\prime})}{R(s^{\prime})}ds^{\prime}+\frac{CS^{\prime}- SC^{\prime}}{R}=\\ =-KS\int\frac{C(s^{\prime})}{R(s^{\prime})}ds^{\prime}+KC\int\frac{S(s^{\prime })}{R(s^{\prime})}ds^{\prime}+\frac{W}{R}=\\ =-KD+\frac{1}{R}\end{array} \tag{4.68}\]

In each transverse plane, the optical functions in Eqs. 4.65 and 4.67 can be used to build a \(3\times 3\)_transfer matrix_, so generalizing Eq. 4.60 to the inclusion of dispersive motion, but still constant longitudinal momentum. This way, a sequence of accelerator elements can be described as the ordered product of matrices reflecting the actual sequence of the accelerator components. For example, in the horizontal plane (the suffix \(x\) is suppressed for brevity) and with notation \(\phi=s\sqrt{K}\):

\[\begin{array}{l}\vec{x}=\begin{pmatrix}x\\ x^{\prime}\\ \delta\end{pmatrix}_{s}=\begin{pmatrix}C&S&D\\ C^{\prime}&S^{\prime}&D^{\prime}\\ 0&0&1\end{pmatrix}_{0\to s}\vec{x}_{0}=\\ \\ =\begin{pmatrix}\cos\phi&\frac{1}{\sqrt{K}}\sin\phi&\frac{1}{RK}(1-\cos\phi)\\ -\sqrt{K}\sin\phi&\cos\phi&\frac{1}{R\sqrt{K}}\sin\phi\\ 0&0&1\end{pmatrix}_{x,s}\vec{x}_{0}\equiv M_{x}\vec{x}_{0}\end{array} \tag{4.69}\]For the vertical plane in the absence of vertical dispersion, the generic transfer matrix in Eq. 4.69 is specialized to the case \(R\to\infty\) and \(k\to-k\):

\[\vec{y}=\begin{pmatrix}y\\ y^{\prime}\\ \delta\end{pmatrix}_{s}=\begin{pmatrix}\cos\phi&\frac{1}{\sqrt{-k}}\sin\phi&0 \\ -\sqrt{-k}\sin\phi&\cos\phi&0\\ 0&0&1\end{pmatrix}_{y,s}\quad\vec{y}_{0}\equiv M_{y}\vec{y}_{0} \tag{4.70}\]

To describe particle's motion in the linear and non-frictional approximation, the product of matrices has to be still a matrix with unitary determinant, as prescribed by Eq. 4.62. Namely, the individual matrices have to belong to an algebric group. This is the case of _symplectic_ matrices. We recall that M is a symplectic matrix if it satisfies \(M^{T}GM=G\), with \(G\) the anti-symmetric singular matrix

\[G=\begin{pmatrix}0&1&0&...&0\\ -1&0&0&...&0\\ 0&0&1&...&0\\ 0&-1&0&...&0\\ 0&0&0&...&1\\ 0&0&0&-1&0\end{pmatrix}_{n\times n}=\begin{pmatrix}O&I_{n}\\ -I_{n}&O\end{pmatrix} \tag{4.71}\]

In summary, the condition \(\det M=1\) for each of the transfer matrices describing the accelerator is necessary but no sufficient to describe a phase space area-preserving map. If \(M\) is symplectic, instead, the condition \(\det M=1\) is automatically satisfied (though not demonstrated here), and the product of N arbitrary symplectic matrices will still give a correct description of the area-preserving map. As long as an accelerator behaves as a linear and conservative system, all its elements have to be represented by symplectic matrices.

##### 4.3.5.1 Discussion: Drift, Dipole, Quadrupole

We want to calculate \(3\times 3\) transfer matrices for a drift (straight section in the absence of any external field), a dipole magnet and a quadrupole magnet. The dipole will be approximated to a small bending angle \(\theta\ll 1\). The quadrupole's matrix will be calculated in _thin lens_ approximation, according to which the quadrupole's length \(l_{q}\to 0\) but the integrated gradient is non-zero, \(kl_{q}=const\).

The transfer matrix of a drift section is calculated from Eqs. 4.69 and 4.70 by specifying \(R\to\infty\) and \(k=0\):

\[M_{dr,x}=M_{dr,y}=\begin{pmatrix}1&L&0\\ 0&1&0\\ 0&0&1\end{pmatrix} \tag{4.72}\]

The _focal length_ is defined by \(u^{\prime}=u_{0}/f\) (\(u=x\), \(y\)), hence \(f=1/m_{21}\). As expected, it is infinite in a drift because the particle's angular divergence does not change in that element.

The matrices of a separate function dipole magnet of arc-length \(l_{d}\) and bending angle \(\theta\) can be calculated from Eqs. 4.69 and 4.70 by specifying \(k=0\). In the horizontal plane, \(\phi_{x}=s/R=l_{d}/R=\theta\). In the vertical plane, \(\theta=0\) and \(R\to\infty\). We find:

\[M_{d,x}=\begin{pmatrix}\cos\theta&R\sin\theta&R(1-\cos\theta)\\ -\frac{1}{R}\sin\theta&\cos\theta&\sin\theta\\ 0&0&1\end{pmatrix} \tag{4.73}\]

\[M_{d,y}=\begin{pmatrix}1&l_{d}&0\\ 0&1&0\\ 0&0&1\end{pmatrix}\]

Since Eq. 4.73 assumes that the field at the entrance and at the exit of the dipole magnet lies on a plane perfectly orthogonal to the direction of the particle's motion, the dipole is classified as "sector". This kind of magnet results a perfect drift in the vertical plane. In the horizontal, it shows a focal length \(f=-R/\sin\theta\). For completeness, we report that a dipole magnet with non-zero edge angles, also called "rectangular", shows a similar focusing property, but in the vertical plane only. In both a sector and a rectangular dipole, the focal length goes like \(f\propto R^{2}/l_{d}\sim(l_{d}B_{y}^{2})^{-1}\), i.e., the stronger the integrated field is, the shorter the focal length is, i.e., the stronger the focusing provided by the magnet will be.

If we now assume \(\theta\ll 1\) and we expand the matrix terms to first order in \(\theta\):

\[M_{d,x}\approx\begin{pmatrix}1&l_{d}&\frac{\theta l_{d}}{2}\\ 0&1&\theta\\ 0&0&1\end{pmatrix}\Rightarrow\left\{\begin{array}{l}D_{x}\approxeq\frac{ \theta l_{d}}{2}=D_{x}^{\prime}\frac{l_{d}}{2}\\ D_{x}^{\prime}\approxeq\theta\end{array}\right. \tag{4.74}\]

For small bending angle, the derivative of the dispersion generated by the dipole magnet is approximately the bending angle, and the dispersion function propagates as if it originated at the middle of the dipole with slope \(\theta\).

The transport matrices representative of a quadrupole magnet long \(l_{q}\) are calculated in thin lens approximation. We first specialize Eqs. 4.69 and 4.70 with \(R\to\infty\), then take the limit \(l_{q}\to 0\) but for \(kl_{q}=const\):

\[M_{q,x}=\begin{pmatrix}\cos(\sqrt{k}l_{q})&\frac{1}{\sqrt{k}}\sin(\sqrt{k}l_{ q})&0\\ -\sqrt{k}\sin(\sqrt{k}l_{q})&\cos(\sqrt{k}l_{q})&0\\ 0&0&1\end{pmatrix}\to\begin{pmatrix}1&0&0\\ -kl_{q}&1&0\\ 0&0&1\end{pmatrix} \tag{4.75}\]

\[M_{q,y}=M_{q,x}(k\to-k)\to\begin{pmatrix}1&0&0\\ kl_{q}&1&0\\ 0&0&1\end{pmatrix}\]

As expected, the focal length \(|f|=\frac{1}{|kl_{q}|}\) has opposite sign in the two transverse planes. In fact, the thin lens approximation applies as long as \(l_{q}\ll f\).

#### 4.3.5.2 Discussion: Dog-Leg

Two consecutive dipole magnets, identical but with opposite sign of the curvature radius, can be used to translate the beam with respect to its initial direction of motion. Demonstrate that such a "dog-leg" configuration translates the beam by exactly the amount of dispersion function excited by the lattice, and that the beam direction is not changed because the derivative of the dispersion function at the exit of the line is null.

If \(R\theta=L\) is the first sector dipole's length with positive radius and bending angle, then the second dipole has identical geometry but negative curvature radius and negative angle. The \(3\times 3\) transfer matrix of the system is:

\[\begin{split} M_{x}&=\begin{pmatrix}1&R\sin\theta &R(1-\cos\theta)\\ 0&1&\sin\theta\\ 0&0&1\end{pmatrix}\begin{pmatrix}1&R\sin\theta&-R(1-\cos\theta)\\ 0&1&-\sin\theta\\ 0&0&1\end{pmatrix}=\\ &\\ &=\begin{pmatrix}1&2R\sin\theta&-R\sin^{2}\theta\\ 0&1&0\\ 0&0&1\end{pmatrix}\end{split} \tag{4.76}\]

\[\Rightarrow\left\{\begin{aligned} D_{x}&=M_{13}=-R\sin^{2} \theta=-d\\ D_{x}^{\prime}&=M_{23}=0\end{aligned}\right. \tag{4.77}\]

#### 4.3.6 Periodic Motion

In this Section, the special case of betatron motion in a periodic lattice, such as in a synchrotron, is considered [4]. Synchrotrons are made of a sequence of \(N\) identical "cells", each cell accommodating a series of dipole and quadrupole magnets. The presence of dipoles give them the name of "arcs", or "achromat" if the horizontal dispersion function and its derivative are both zero at the entrance and at the exit of the cell. \(N\) is called _superperiod_ of the synchrotron.

Each cell is long \(L_{cell}\) so that \(NL_{cell}=2\pi\,R_{s}=C\). A cell can be represented by a transfer matrix \(M\), as shown in Fig. 4.11. The synchrotron total matrix (in each transverse plane) is \(M_{t}=\prod_{i=1}^{N}M_{i}=M_{i}^{N}\). In general, \(M_{x}\neq M_{y}\). In an isomagnetic lattice with \(N_{d}\) dipoles per cell, the dipole bending angle is \(\theta_{d}=\frac{2\pi}{N_{d}}\). For example, Fig. 4.11 sketches a "double-bend" arc lattice with superperiod \(N\)=12, where each vertex of the polygonal is the center of a dipole magnet. Particles' motion is said to be "stable" if particles can travel on the reference closed orbit, or in proximity of it, for a very large number of turns, ideally \(n\rightarrow\infty\).

Henceforth, we will adopt the notation \(u=x\), \(y\) for all quantities related to the betatron motion, unless differently specified. The periodic homogeneous Hill's equation is:

\[\left\{\begin{array}{l}u^{\prime\prime}(s)+K(s)u(s)=0\\ \\ K(s)=K(s+L_{cell})=K(s+C)\end{array}\right. \tag{4.78}\]

The closed form solution of Hill's equation for the whole periodic lattice can be built as a linear superposition of the elements of a basis of the matrix \(M\). In each transverse plane, the basis is given by the eigen-vectors \(\vec{u}_{j}^{*}\) (\(j=1\), 2) of the matrix. Given \(\lambda_{j}\) the eigenvalues of \(M\), there exists a diagonal matrix \(\Lambda=(\lambda_{1},\lambda_{2})I\) and an invertible matrix \(P\) such that \(M=P^{-1}\,\Lambda\,P\). The eigenvalues satisfy:

\[M\vec{u}_{j}^{*}(s)=\lambda_{j}\vec{u}_{j}^{*} \Rightarrow \vec{u}(s)=\sum_{j=1}^{2}A_{j}\vec{u}_{j}^{*} \tag{4.79}\]

Since the matrix for the whole accelerator still has to contain finite elements after \(n\) turns, i.e., the particle's coordinates shall not diverge to infinite for \(n\to\infty\), it results:

\[|\lim_{n\to\infty}M^{Nn}\vec{u}(0)|=|\lim_{n\to\infty}M^{Nn}\sum_{j=1}^{2}A_{j }\vec{u}_{j,0}^{*}|=|\lim_{n\to\infty}\sum_{j=1}^{2}A_{j}\lambda_{j}^{Nn}\vec{ u}_{j,0}^{*}|=\]

\[\leq|A_{1}\vec{u}_{1,0}^{*}|\lim_{n\to\infty}|\lambda_{1}|^{Nn}+|A_{2}\vec{u}_{ 2,0}^{*}|\lim_{n\to\infty}|\lambda_{2}|^{Nn}<\infty\]

\[\Rightarrow|\lambda_{j}|\leq 1 \tag{4.80}\]

The eigenvalues of \(M\) satisfy the equation \(\det\,(M-\lambda_{j}\,I)=0\). We write \(M\) with generic terms and develop the equation accordingly, to retrieve additional properties of the eigenvalues:

Figure 4.11: Schematic of a synchrotron with superperiod \(N\)=12 and double bend arc lattice

\[det\left[\left(\begin{matrix}a&b\\ c&d\end{matrix}\right)-\left(\begin{matrix}\lambda_{j}&0\\ 0&\lambda_{j}\end{matrix}\right)\right]=det\left(\begin{matrix}a-\lambda_{j}&b \\ c&d-\lambda_{j}\end{matrix}\right)= \tag{4.81}\] \[=\lambda_{j}^{2}-(a+d)\lambda_{j}+(ad-bc)=\lambda_{j}^{2}-Tr(M) \lambda_{j}+\det M=\] \[=\lambda_{j}^{2}-Tr(M)\lambda_{j}+\det M=0,\]

The solutions are:

\[\lambda_{1,2}=\frac{Tr(M)\pm\sqrt{Tr(M)^{2}-4\det M}}{2}\Rightarrow\left\{ \begin{matrix}\lambda_{1}\cdot\lambda_{2}=\det M=1\\ \lambda_{1}+\lambda_{2}=Tr(M)\end{matrix}\right. \tag{4.82}\]

Equations 4.80 and 4.82 imply \(|Tr(M)|=|\lambda_{1}+\lambda_{2}|\leq|\lambda_{1}|+|\lambda_{2}|\leq 2\). We demonstrate below that \(Tr(M)=\pm 2\Leftrightarrow\lambda_{j}=\pm 1\), i.e., \(M=\pm I\).

**Lemma 1**. For \(\lambda_{1}=x_{1}+iy_{1}\), \(\lambda_{2}=x_{2}+iy_{2}\), it results:

\[\lambda_{1}\lambda_{2}=(x_{1}x_{2}-y_{1}y_{2})+i(x_{1}y_{2}+x_{2}y_{1})=1 \Rightarrow\left\{\begin{matrix}x_{1}x_{2}=1+y_{1}y_{2}\\ x_{1}y_{2}=-x_{2}y_{1}\end{matrix}\right. \tag{4.83}\]

**Lemma 2**. Let us assume generic \(\lambda_{j}\in\mathbb{C}\) and consider the extreme condition \(\lambda_{1}+\lambda_{2}=Tr(M)=\pm 2\). It follows that \(y_{1}+y_{2}=0\) or \(y_{1}=-y_{2}\), and \(x_{1}+x_{2}=\pm 2\). By virtue of Lemma 1, \(x_{1}y_{2}=-x_{2}y_{1}\Rightarrow x_{1}=-x_{2}y_{1}/y_{2}=x_{2}\Rightarrow x _{1}=x_{2}=\pm 1\)\(\Rightarrow\lambda_{1}=\lambda_{2}=\pm 1\)\(\Rightarrow\)\(\Lambda=\pm I\) and \(M=P^{-1}\,\Lambda\,P=\pm I\). Viceversa, if \(M=\pm I\) then \(Tr(M)=\pm 2\).

When \(M=\pm I\), the periodic system is unconstrained (any arbitrary initial condition satisfies the equation of motion) and, owing to Eq. 4.80, any infinitesimal perturbation \(\epsilon\) to the lattice such that \(|\lambda_{j}+\epsilon|>1\) will lead to unstable motion. This brings to a tighter _necessary_ condition for long-term stability of single particle's motion in the 2-D phase space: \(|Tr(M)|<2\). "Stability" means here bounded values of the particle's coordinates over an arbitrarily large number of passes through a lattice represented by the 2\(\times\)2 transfer matrix \(M\).

#### 4.3.7 Betatron Function

Floquet's theorem (demonstrated below) applied to a periodic lattice allows us to write the two linearly independent solutions of Hill's equation as periodic functions of \(s\) with period \(C\), multiplied by a complex phase:

\[\left\{\begin{array}{l}u_{1,2}(s)=p_{1,2}(s)e^{\pm i(\mu(s)-\mu(0))},\quad \mu(s)\in\mathbb{R}e\ \forall s\\ \\ p_{1}(s)=p_{2}(s)\equiv p(s)=p(s+C)\ \forall s\end{array}\right. \tag{4.84}\]The two amplitudes \(p_{1},\ p_{2}\) can be made equal without loss of generality because the two exponential functions are already linearly independent. The functions \(p\), \(\mu\) are intended to be defined in the transverse plane \(u\).

With \(u_{1,2}(0)=p_{1,2}(0)\) the solutions at the initial position \(s=0\), and by virtue of the periodicity in Eq. 4.84, the two solutions after one turn result:

\[\begin{array}{l}u_{1,2}(C)=p_{1,2}(C)e^{\pm i(\mu(C)-\mu(0))}=p_{1,2}(0)e^{ \pm i(\mu(C)-\mu(0))}=\\ =u_{1,2}(0)e^{\pm i(\mu(C)-\mu(0))}\equiv Mu_{1,2}(0)\end{array} \tag{4.85}\]

The very last equality demonstrates that \(e^{\pm i(\mu(C)-\mu(0)}\) are the eigenvalues \(\lambda_{1,2}\) of \(M\). The real quantity \(\mu(C)-\mu(0)\) is called "characteristic exponential coefficient" of the homogeneus Hill's equation. Clearly, those eigenvalues satisfy Eqs. 4.80 and 4.82, and in particular:

\[|Tr(M)|=|\lambda_{1}+\lambda_{2}|=|2\cos\Delta\mu|<2\Leftrightarrow\Delta\mu \neq p\pi,\,p\in\mathbb{N} \tag{4.86}\]

In the literature, \(u_{1},\,u_{2}\) are written in terms of a positive-definite amplitude \(\beta(s):=p^{2}(s)\) named _betatron function_:

\[\left\{\begin{array}{l}u_{1,2}(s)=\sqrt{\beta(s)}(s)e^{\pm i(\mu(s)-\mu(0))} \\ \\ \beta(s)=\beta(s+C)\end{array}\right. \tag{4.87}\]

The generic solution of Hill's homogeneous equation is therefore:

\[\begin{array}{l}u(s)=a_{1}u_{1}(s)+a_{2}u_{2}(s)=a_{1}\sqrt{\beta(s)}(s)e^{ i(\mu(s)-\mu_{0}))}+a_{2}\sqrt{\beta(s)}(s)e^{-i(\mu(s)-\mu_{0})}=\\ =\sqrt{2J}\sqrt{\beta(s)}\cos(\mu(s)-\mu_{0}+\phi_{0}),\end{array} \tag{4.88}\]

where the constants are \(\sqrt{J}=\sqrt{2a_{1}a_{2}}\) and \(\tan\phi_{0}=i\,\frac{a_{1}-a_{2}}{a_{1}+a_{2}}\). All quantities are intended to be defined in the \(u-\)transverse plane. We draw the following observations.

* The amplitude \(\sqrt{2J}\) is a constant of motion, called _single particle invariant_.
* The particle's transverse position along the accelerator is proportional to \(\sqrt{\beta_{u}}\), which contributes to determining the amplitude of oscillation.
* \(\Delta\mu(s)=\mu(s)-\mu_{0}+\phi_{0}\) assumes the meaning of a _relative phase advance_ of the betatron oscillation.
* Floquet's theorem states the existence a solution of the periodic Hill's equation, i.e., the equation defined for a _periodic lattice_. The solution implies a _periodic betatron function_. Since \(\beta(s)\) is an analytic continuous (i.e., differentiable) function of \(s\), its derivative \(\beta^{\prime}(s)\) is also periodic in \(s\).

As a consequence of Floquet's theorem (Eq.4.84), the betatron phase advance and the betatron function are intrinsically connected. We explicit their relation by substituting the solution \(u(s)=p(s)e^{i\mu(s)}\) into Hill's equation:

\[\begin{array}{l}u=pe^{i\mu},\\ u^{\prime}=p^{\prime}e^{i\mu}+ip\mu^{\prime}e^{i\mu},\\ u^{\prime\prime}=\left[p^{\prime\prime}+i2p^{\prime}\mu^{\prime}+ip\mu^{ \prime\prime}-p\mu^{\prime 2}\right]e^{i\mu}\\ \\ \Rightarrow u^{\prime\prime}+ku=0,\\ \left(p^{\prime\prime}-p\mu^{\prime 2}\right)+i\left(2p^{\prime}\mu^{ \prime}+p\mu^{\prime\prime}\right)+kp=0\\ \left(p^{\prime\prime}+kp\right)-p\mu^{\prime 2}+i\left(2p^{\prime}\mu^{ \prime}+p\mu^{\prime\prime}\right)=0\end{array} \tag{4.89}\]

The real and the imaginary part of the r.h.s. of the last expression have to be individually zero. The latter one gives:

\[\begin{array}{l}\frac{\mu^{\prime\prime}}{\mu^{\prime}}=-\frac{2p^{\prime}} {p};\\ \\ \frac{d}{ds}\left(\ln\mu^{\prime}\right)=-\frac{d}{ds}\left(\ln p^{2}\right)+ \ln c,\ \ \ \ \ln c\equiv 0;\\ \\ \int ds(l.h.s.)=\int ds(r.h.s.);\\ \\ \mu^{\prime}=\frac{1}{p^{2}};\\ \\ \Rightarrow\mu(s)-\mu_{0}=\int_{0}^{s}\frac{ds^{\prime}}{\beta(s^{\prime})} \end{array} \tag{4.90}\]

The number of betatron oscillations per turn:

\[Q_{u}=\frac{\Delta\mu_{u}}{2\pi}=\oint\frac{ds}{\beta_{u}(s)}\equiv\frac{2\pi \,R_{s}}{\beta_{u}} \tag{4.91}\]

is called _betatron tune_. The pair \([Q_{x},Q_{y}]\) is the synchrotron _working point_. The integer part of \(Q_{u}\) is usually in the range 10-100, thus orders of magnitude larger than the synchrotron tune \(Q_{s}\) (see Eq.4.32).

The last equality on the r.h.s. of Eq.4.91 defines an _average_ betatron function evaluated over the particle's closed orbit. It is important to notice that, in general, \(\frac{1}{\beta_{u}}=\langle\frac{1}{\beta_{u}}\rangle\neq\frac{1}{\langle\beta _{u}\rangle}\). Figure4.12 illustrates the generic solution \(u(s)\), expressed as function of \(\beta_{u}(s)\) and \(\overline{\beta_{u}}\).

#### Floquet's Theorem

Floquet's theorem is evoked in Eq.4.84 for particle's motion in a 2-D phase space (\(u\), \(u^{\prime}\)), with \(u\) the particle's position relative to the reference orbit, and \(u^{\prime}=du/ds\)the angular divergence. The most general enunciation assumes a periodic system with period \(C\):

\[\left\{\begin{array}{l}f^{\prime}(s)=A(s)\,f(s)\\ A(s)=A(s+C)\end{array}\right. \tag{4.92}\]

We now apply Eq. 4.92 to the particle's position and angular divergence, \(f\to u\), \(f^{\prime}\to u^{\prime}\). In our notation, \(A(s)\) is still a periodic function, given by the combination of principal trajectories and arbitrary initial values \(u_{0}\), \(u_{0}^{\prime}\):

\[\left\{\begin{array}{l}u^{\prime}=Au\\ u^{\prime\prime}=Au^{\prime}+A^{\prime}u=A(Au)+A^{\prime}u=(A^{2}+A^{\prime}) u\equiv Ku\end{array}\right. \tag{4.93}\]

The second expression shows that Hill's equation for the betatron motion describes a special "Floquet's system", where \(K(s)=K(s+C)\).

**Lemma 1**: if \(u_{1}(s)\), \(u_{2}(s)\) are linearly independent solutions of Eq. 4.93, then \(\tilde{u}_{1}=u_{1}(s+C)\), \(\tilde{u}_{2}=u_{2}(s+C)\) are also linearly independent solutions.

Firstly, we verify that \(\tilde{u}_{j}\), \(j=1,2\), is a solution of Eq. 4.93 (the subscript is removed for brevity):

\[\tilde{u}^{\prime}=A(s+C)\tilde{u}=A(s)\tilde{u} \tag{4.94}\]

by virtue of the periodicity of \(A(s)\).

Secondly, we verify the condition of linear independence by demonstrating that the wronskian of the two solutions is non-zero for at least one value of the \(s\)-coordinate in the range of existence \(s\in[0,C]\). To do this, we remind that the vectors \(\tilde{\tilde{u}}_{j}\) are obtained by a single-turn transformation of vectors \(\tilde{u}_{j}\) via the transfer matrix \(M_{t}\):

\[\tilde{\tilde{u}}=M_{t}\tilde{u}\quad\Rightarrow\quad\left\{\begin{array}{ l}\tilde{u}_{j}=M_{11}u_{j}+M_{12}u_{j}^{\prime}\\ \tilde{u}_{j}^{\prime}=M_{21}u_{j}+M_{22}u_{j}^{\prime}\end{array}\right. \tag{4.95}\]

Figure 4.12: Betatron oscillation in the x-plane. Particle’s position x(s) is evaluated for the s-dependent \(\beta_{x}(s)\) (solid line) and the average betatron function \(\overline{\beta_{x}}\) (dashed line)

The wronskian is:

\[\begin{split}\begin{vmatrix}\tilde{u}_{1}&\tilde{u}_{2}\\ \tilde{u}_{1}^{\prime}&\tilde{u}_{2}^{\prime}\end{vmatrix}=& \begin{pmatrix}M_{11}u_{1}+M_{12}u_{1}^{\prime}\end{pmatrix}\begin{pmatrix}M_{21} u_{2}+M_{22}u_{2}^{\prime}\end{pmatrix}-\begin{pmatrix}M_{11}u_{2}+M_{12}u_{2}^{ \prime}\end{pmatrix}\begin{pmatrix}M_{21}u_{1}+M_{22}u_{1}^{\prime}\end{pmatrix}= \\ &=M_{11}M_{22}u_{1}u_{2}^{\prime}+M_{12}M_{21}u_{1}^{\prime}u_{2}-M_{11}M_{22}u_{ 2}u_{1}^{\prime}-M_{12}M_{21}u_{2}^{\prime}u_{1}=\\ &=M_{11}M_{22}(u_{1}u_{2}^{\prime}-u_{2}u_{1}^{\prime})+M_{12}M_{21}(u_{1}^{ \prime}u_{2}-u_{2}^{\prime}u_{1})=\\ &=M_{11}M_{22}-M_{12}M_{21}=1\ \ \forall s\end{split}\]

where we used the fact that \(u_{1}\), \(u_{2}\) are linearly independent solutions, and that \(M_{t}\) is symplectic.

**Lemma 2**: given the "fundamental" matrices \(U=\begin{pmatrix}u_{1}&0\\ 0&u_{2}\end{pmatrix}\) and \(\tilde{U}=\begin{pmatrix}\tilde{u}_{1}&0\\ 0&\tilde{u}_{2}\end{pmatrix}\), there exists a non-singular matrix \(R\) such that \(\tilde{U}=U\,R\).

It has to be:

\[\begin{split}\tilde{U}_{11}&=\tilde{u}_{1}=u_{1}\,R_{11}\\ \tilde{U}_{12}&=0=u_{1}\,R_{12}\Rightarrow R_{12}=0\\ \tilde{U}_{21}&=0=u_{2}\,R_{21}\Rightarrow R_{21}=0\\ \tilde{U}_{22}&=\tilde{u}_{2}=u_{2}\,R_{22}\end{split} \tag{4.96}\]

Therefore \(R\) is diagonal. We also have \(\det(\tilde{U})=\det(U)\det(R)\Rightarrow\det(R)\neq 0\), i.e., \(R\) is invertible. Owing to the existence of the logarithm matrix for \(R\), we can write \(R=e^{QC}\), where in general \(Q_{ij}\in\mathbb{C}\). By virtue of its unitary determinant, it must be:

\[R(s)=\begin{pmatrix}e^{iq(s)C}&0\\ 0&e^{-iq(s)C}\end{pmatrix},\ \ q(s)\in\mathbb{R}e \tag{4.97}\]

**Lemma 3**: given the matrix \(P(s)=U(s)e^{-Qs}\), it results \(P(s+C)=P(s)\ \forall s\).

This is demonstrated by calculating \(P(s+C)\) and using Lemma 2:

\[\begin{split} P(s+C)&=U(s+C)e^{-Q(s+C)}=\tilde{U}e^{-QC}e^{ -Qs}=U\,Re^{-QC}e^{-Qs}=\\ &=U\,Ie^{-Qs}=P(s);\end{split}\]

\[\Rightarrow\left\{\begin{array}{ll}U(s)&=\,P(s)e^{Qs}\\ P(s)&=\,P(s+C)\end{array}\right. \tag{4.98}\]

Equation 4.98 concludes the demonstration of Floquet's theorem. Indeed, we can define a real function \(\mu(s)=q(s)C\) so that:

\[\left\{\begin{array}{ll}U_{11}=u_{1}(s)&=\,p_{1}(s)e^{i\mu(s)}\\ U_{22}=u_{2}(s)&=\,p_{2}(s)e^{-i\mu(s)}\end{array}\right. \tag{4.99}\]

and \(p_{j}\) is periodic with period \(C\).

#### Courant-Snyder Invariant

A complete description of betatron motion needs the knowledge not only of the particle's position (Eq. 4.88), but also of its angular divergence. This is the angle between the vector of particle's transverse and longitudinal momentum, evaluated at any point along the accelerator. We remind that in the Frenet-Serret coordinate system, the longitudinal versor is tangent to the orbit. Thus, the divergence identifies the particle's instantaneous direction of motion:

\[u^{\prime}(s)=\frac{p_{u}}{p_{z}}=\frac{v_{u}}{v_{z}}=\frac{1}{v_{z}}\frac{du}{ dt}=\frac{du}{ds}, \tag{4.100}\]

and by virtue of Eq. 4.88

\[\begin{split} u^{\prime}(s)&=\frac{du}{ds}= \frac{\sqrt{2J}}{2\sqrt{\beta}}\frac{d\beta}{ds}\cos\Delta\mu-\sqrt{2J\beta} \sin\Delta\mu\cdot\frac{d\mu}{ds}=\\ &=\sqrt{2J}\left[\frac{1}{2\sqrt{\beta}}\frac{d\beta}{ds}\cos \Delta\mu-\frac{1}{\sqrt{\beta}}\sin\Delta\mu\right]=\\ &=-\sqrt{\frac{2J}{\beta}}\left(\alpha\cos\Delta\mu+\sin\Delta \mu\right),\end{split} \tag{4.101}\]

\[\begin{split}\alpha:=-\frac{\beta^{\prime}}{2}\end{split}\]

Equation 4.101 shows that the single particle's betatron oscillations can be described by the so-called _Courant-Snyder_ parameters \(\alpha_{u}(s)\), \(\beta_{u}(s)\), or _Twiss functions_. This description is equivalent to that given in terms of principal trajectories in Eqs. 4.69 and 4.70. In fact, the two initial conditions \(u_{0}\), \(u^{\prime}_{0}\) are here replaced by the constants \(2J\), \(\mu_{0}\), while the linearly independent transfer functions \(C_{u}(s)\), \(S_{u}(s)\) are substituted by the linearly independent functions \(\beta_{u}(s)\), \(\alpha_{u}(s)\).

By definition, the locations at which \(\alpha(\bar{s})=0\) correspond to a maximum or minimum of \(\beta(s)\). Since we can freely define the initial betatron phase \(\mu(\bar{s})=\mu_{0}=0\), we have from Eq. 4.101 that \(\alpha(\bar{s})=0\) identifies the points along the accelerator in correspondence of which the particle assumes a local maximum or minimum distance from the reference orbit. This is the case, for example, of a particle in the middle of a quadrupole magnet, see Fig. 4.9.

Let us now assume that the full set of functions \(u(s)\), \(u^{\prime}(s)\), \(\beta(s)\) and \(\alpha(s)\) are known at a given \(s\). Since \(J\) is a constant of motion, we can retrieve it from the knowledge of the full set of data. After we will have found an expression for it, we will demonstrate that it is indeed a constant of motion under the assumptions 1. and 2. (see Eq. 4.62), which were used to derive the homogeneous Hill's Eqs. 4.53 and 4.55.

At first, we recall the solution of the homogeneous Hill's equation and its first derivative:

\[\left\{\begin{split} u(s)=\sqrt{2J\beta}\cos\Delta\mu\\ u^{\prime}(s)=-\sqrt{\frac{2J}{\beta}}\left(\alpha\cos\Delta\mu+ \sin\Delta\mu\right)\end{split}\right. \tag{4.102}\]By substituting the first equation into the second one:

\[\begin{split}& u^{\prime}=-\sqrt{\frac{2J}{\beta}}\left(\alpha\frac{u }{\sqrt{2J\beta}}+\sqrt{1-\frac{u^{2}}{2J\beta}}\right)=-u\frac{\alpha}{\beta} -\sqrt{\frac{2J}{\beta}-\frac{u^{2}}{\beta^{2}}};\\ &\left(u^{\prime}+u\frac{\alpha}{\beta}\right)^{2}+\frac{u^{2}}{ \beta^{2}}=\frac{2J}{\beta};\\ &\Rightarrow\left\{\begin{array}{l}2J=u^{2}\left(\frac{\alpha^ {2}+1}{\beta}\right)+2uu^{\prime}\alpha+\beta u^{\prime 2}=\gamma u^{2}+2\alpha uu^{ \prime}+\beta u^{\prime 2},\\ \\ \gamma(s):=\frac{1+\alpha(s)^{2}}{\beta(s)}\end{array}\right.\end{split} \tag{4.103}\]

By virtue of Eq. 4.103, the single particle invariant \(2J\) is also called _Courant-Snyder invariant_. The choice of the two linearly independent Courant-Snyder (C-S) parameters among \(\alpha\), \(\beta\), \(\gamma\) is indeed irrelevant, being the third one dependent from the other two.

To demonstrate that the expression on the r.h.s of Eq. 4.103 is a constant of motion, we consider a generic solution of Hill's homogeneous equation and define a wrong-skian as follows:

\[\left\{\begin{array}{l}u(s)=a_{1}u_{1}(s)+a_{2}u_{2}(s)\\ u_{1}=\sqrt{\beta}e^{i\mu}\end{array}\right.\Rightarrow V(s)=\left|\begin{array} []{ll}u&u^{\prime}\\ u_{1}&u^{\prime}_{1}\end{array}\right|=uu^{\prime}_{1}-u^{\prime}u_{1} \tag{4.104}\]

We observe that:

\[\begin{split}&\frac{dV}{ds}=u^{\prime}u^{\prime}_{1}+uu^{\prime \prime}_{1}-u^{\prime\prime}u_{1}-u^{\prime}u^{\prime}_{1}=uu^{\prime\prime}_{ 1}-u^{\prime\prime}u_{1}=u(-ku_{1})-(-ku)u_{1}=\\ &\quad=-kuu_{1}+kuu_{1}=0\\ &\Rightarrow V(s)=const.\quad\forall s\end{split} \tag{4.105}\]

\(V(s)\) is expressed in terms of the Courant-Snyder parameters:

\[\begin{split}& V(s)=uu^{\prime}_{1}(s)-u^{\prime}u_{1}(s)=u \left(\frac{1}{2}\frac{\beta^{\prime}}{\sqrt{\beta}}e^{i\mu}+i\sqrt{\beta} \frac{e^{i\mu}}{\beta}\right)-u^{\prime}u_{1}=\\ &\quad\quad=u\frac{e^{i\mu}}{\sqrt{\beta}}\left(-\alpha+i\right)- u^{\prime}u_{1}=u\sqrt{\beta}e^{i\mu}\left(\frac{i-\alpha}{\beta}\right)-u^{ \prime}u_{1}=\\ &\quad\quad=uu_{1}\left(\frac{i-\alpha}{\beta}\right)-u^{\prime}u _{1}=u_{1}\left[u\left(\frac{i-\alpha}{\beta}\right)-u^{\prime}\right]\\ &\quad\quad\Rightarrow VV^{*}=u_{1}u^{*}_{1}\left[u^{2}\left( \frac{1+\alpha^{2}}{\beta^{2}}\right)+\frac{2\alpha}{\beta}uu^{\prime}+u^{ \prime 2}\right]=\\ &\quad\quad\quad=\gamma u^{2}+2\alpha uu^{\prime}+\beta u^{\prime 2 }\end{split} \tag{4.106}\]

We conclude that since \(V(s)\) is a constant of motion, \(|V|^{2}=VV^{*}\) is constant as well, and this is exactly the expression of \(2J\) in Eq. 4.103.

#### Phase Space Ellipse

Hill's equation for the betatron motion in a periodic lattice describes a quasi-periodic oscillator. That is, the particle's motion in the phase space (\(u\), \(u^{\prime}\)) is bounded and, in the most general case, the orbit at any given \(s\) maps an ellipse. The dependence of the C-S parameters from the \(s\)-coordinate suggests that, contrary to a pure harmonic oscillation (i.e., constant angular frequency), the ellipse orientation and ellipticity changes at different \(s\).

According to Eq. 4.102, the particle's lateral position oscillates within an envelope of amplitude \(\sqrt{2J\beta}\). Local maxima or minima are reached for:

\[\begin{array}{l}x^{\prime}=0\Rightarrow\cos\Delta\mu=-\frac{\sin\Delta\mu}{ \alpha}\Rightarrow\cos^{2}\Delta\mu=\frac{1}{1+\alpha^{2}}\\ \\ \Rightarrow|x|=\frac{\sqrt{2J\beta}}{\sqrt{1+\alpha^{2}}}\leq\sqrt{2J\beta} \end{array} \tag{4.107}\]

Fig. 4.9 suggests that this is common to happen in quadrupole magnets.

At all points of the lattice where \(\alpha=0\), the particle's maximum spatial and angular excursion, not to be met simultaneously, are \(|\hat{u}|=\sqrt{2J\beta}\) and \(|\hat{u}^{\prime}|=\sqrt{2J/\beta}\), respectively. The C-S invariant introduced in Eq. 4.103 becomes:

\[\begin{array}{l}2J(\alpha=0)=\gamma u^{2}+\beta u^{\prime 2}=\frac{u^{2}}{ \beta}+\beta u^{\prime 2}=2J\frac{u^{2}}{|\hat{u}|^{2}}+2J\frac{u^{\prime 2}}{| \hat{u}^{\prime}|^{2}}\\ \\ \Rightarrow\frac{u^{2}}{|\hat{u}|^{2}}+\frac{u^{\prime 2}}{|\hat{u}^{\prime}|^{2}}= 1\\ \\ \Rightarrow|\hat{u}||\hat{u}^{\prime}|=2J=\frac{Area}{\pi}\end{array} \tag{4.108}\]

Eq. 4.108 describes an up-right phase space ellipse, i.e., the ellipse axes are aligned to the Cartesian axes (\(u\), \(u^{\prime}\)). The area of the ellipse is just the particle's C-S invariant in units of \(\pi\), i.e., the phase space area enclosed by the particle's orbit is a constant of motion, in the assumption of linear and non-dissipative forces. The phase space ellipse for generic C-S parameters is illustrated in Fig. 4.13. We summarize our findings below.

* Betatron motion of the generic particle is represented by an ellipse in the phase space (\(u\), \(u^{\prime}\)), on which the particle's representative pint lies during the particle's motion in the accelerator. The ellipse area in units of \(\pi\) is the particle's C-S invariant, i.e., it is a constant of motion in the approximation of linear and non-dissipative dynamics.
* The ellipse orientation and ellipticity are uniquely determined by the C-S parameters. By virtue of \(s\)-dependent focusing in Hill's equation, the C-S parameters vary with \(s\), and so the ellipse does. In general, we can draw a different ellipse in correspondence of each \(s\) along the accelerator. The particle's coordinates at any \(s\) belong to the ellipse evaluated at that point.

* If the motion is periodic, the ellipse remains identical at any given \(s\) over consecutive turns (because the C-S parameters are periodic with period equal to the synchrotron circumference). Turn-by-turn, the particle's coordinates at that specific \(s\) will map the ellipse as the betatron phase advances.
* The Particle's position is proportional to \(\sim\)\(\sqrt{\beta}\), its angular divergence to \(\sim\)\(\frac{1}{\sqrt{\beta}}\). The condition \(\alpha=0\) implies a local maximum or minimum of the betatron function, and the corresponding phase space ellipse is up-right. The individual particle's position and angular divergence do not necessarily reach a maximum there, depending from the betatron phase advance. However, the absolute maximum position and angular divergence become accessible, and equal to \(|\hat{u}|=\sqrt{2J\beta}\) and \(|\hat{u}^{\prime}|=\sqrt{2J/\beta}\), respectively.

#### Floquet's Normalized Coordinates

The phase space representation of particle's betatron motion is expected to reduce to a purely harmonic oscillator if the dependence of the oscillation amplitude from the \(s\)-coordinate could be removed. To show this, let us derive Hill's homogeneous equation under a transformation of coordinates normalized to the local betatron function.

The betatron phase advance becomes the independent variable:

\[\left\{\begin{array}{l}s\rightarrow\Delta\mu=Q\theta\equiv\phi\\ u\to w=\frac{u}{\sqrt{\beta}}\end{array}\right.\Rightarrow\left\{ \begin{array}{l}w=\sqrt{2J}\cos\phi\\ w^{\prime}=\frac{dw}{d\phi}=-\sqrt{2J}\sin\phi\end{array}\right. \tag{4.109}\]\(Q\) is the betatron tune, the pair \((J,\phi)\) is called _action-angle_ variables, and \(w,\,w^{\prime}\) are the "Floquet's normalized coordinates". An ellipse in the phase space \((u,\,u^{\prime})\) becomes a circle of radius \(\sqrt{2J}\) in the phase space \((w,\,w^{\prime})\). As the particle advances along the accelerator coordinate \(s\), the representative point lying on the circle moves by an angle \(\phi(s)\). As expected, Hill's homogeneous equation becomes that of a pure harmonic oscillator (the angular frequency does not depend from \(s\) anymore):

\[w^{\prime\prime}=\frac{dw^{\prime}}{d\phi}=-w\ \ \Rightarrow\ \ w^{\prime\prime}+w=0 \tag{4.110}\]

It is straightforward to show that, if \(\theta\) is taken as independent variable instead, then the phase space orbit still maps onto an ellipse, and the angular frequency is just the betatron tune:

\[w^{\prime\prime}(\theta)=-\,Q^{2}\sqrt{2J}\cos(Q\theta)\ \ \Rightarrow\ \ w^{\prime\prime}+Q^{2}w=0 \tag{4.111}\]

Since \(\Delta\mu_{t}=2\pi\,Q\) is the single turn phase advance in a synchrotron, a particle moves in this normalized phase space by a corresponding angle \(\theta_{t}=2\pi\).

#### 4.3.11.1 Discussion: Periodic Momentum-Dispersion Function

The solution of the Hill's complete equation for the periodic motion, denominated _periodic momentum-dispersion function_, or simply "periodic dispersion", is by definition an eigenvector of the one-turn \(3\times 3\) transfer matrix, in analogy to Eq. 4.87 for the betatron motion. Let us find an expression for \(\eta\), \(\eta^{\prime}\) as function of the principal trajectories, by exploiting the one-turn matrix properties (trace and determinant).

Being the periodic dispersion an eigen-vector of the one-turn matrix, it must satisfy:

\[\begin{pmatrix}\eta\\ \eta^{\prime}\\ 1\end{pmatrix}_{s}=\begin{pmatrix}C(s)&S(s)&D(s)\\ C^{\prime}(s)&S^{\prime}(s)&D^{\prime}(s)\\ 0&0&1\end{pmatrix}\begin{pmatrix}\eta\\ \eta^{\prime}\\ 1\end{pmatrix}_{0} \tag{4.112}\]

The periodicity over one turn implies \((\eta,\,\eta^{\prime},\,1)_{0}^{t}=(\eta,\,\eta^{\prime},\,1)_{s}^{t}\). We can therefore solve Eq. 4.112 for \(\eta\) and \(\eta^{\prime}\) at the arbitrary coordinate \(s\). By using the relation \(C+S^{\prime}=Tr(M_{t})=2\cos\,\Delta\mu\) and \(\det M=CS^{\prime}-S^{\prime}C=1\), we find:

\[\begin{cases}\eta(s)=\frac{[1-S^{\prime}(s)]D(s)+S(s)D^{\prime}(s)}{2(1-\cos \Delta\mu)}\\ \\ \eta^{\prime}(s)=\frac{[1-C(s)]D^{\prime}(s)+C^{\prime}(s)D(s)}{2(1-\cos\Delta \mu)}\end{cases} \tag{4.113}\]

It emerges that \(D(s)\) is the dispersive element of the transfer matrix, while \(\eta(s)\) is the periodic solution of Hill's complete equation. When \((\eta,\,\eta^{\prime})\) are calculated along the beam line starting from initial values \((0,\,0)\), it results from Eq. 4.112 that, at the generic coordinate \(s\), \((\eta,\,\eta^{\prime})_{s}=(D,\,D^{\prime})_{s}\). This justifies the naming of dispersion function for \(D(s)\) adopted so far, for a beam line in which \(D_{0}=D_{0}^{\prime}=0\).

#### Equivalence of Matrices

Solutions of Hill's homogeneous equation can be expressed equivalently in terms of principal trajectories (see Eq. 4.60) and C-S parameters (see Eq. 4.102). The equivalence implies that principal trajectories mapping the particle's coordinates from one point to another in the accelerator, can be expressed as a combination of C-S parameters evaluated at the beginning and at the end of the line, and viceversa.

The former correspondence is found by recalling \(u\), \(u^{\prime}\) in Eqs. 4.59 and 4.102 (all optical functions refer to the same transverse plane):

\[\left\{\begin{aligned} & u(s)=Cu_{0}+Su^{\prime}_{0}=\sqrt{2J\beta} \cos\Delta\mu\\ & u^{\prime}(s)=C^{\prime}u_{0}+S^{\prime}u^{\prime}_{0}=-\sqrt{ \frac{2J}{\beta}}\left(\sin\Delta\mu+\alpha\cos\Delta\mu\right)\end{aligned}\right. \tag{4.114}\]

The Initial conditions \(C(0)=S^{\prime}(0)=1,C^{\prime}(0)=S(0)=0\), \(\beta(0)=\beta_{0},\alpha(0)=\alpha_{0}\), \(\Delta\mu(0)=0\) are used to find:

\[\left\{\begin{aligned} & u(0)=u_{0}=\sqrt{2J\beta_{0}}\\ & u^{\prime}(0)=u^{\prime}_{0}=-\alpha_{0}\sqrt{\frac{2J}{\beta_ {0}}}\end{aligned}\right. \tag{4.115}\]

and therefore:

\[\left\{\begin{aligned} & C=\sqrt{\frac{\beta}{\beta_{0}}}\cos \Delta\mu+S\frac{\alpha_{0}}{\beta_{0}}\\ & C^{\prime}=-\frac{\sin\Delta\mu+\alpha\cos\Delta\mu}{\sqrt{ \beta\beta_{0}}}+S^{\prime}\frac{\alpha_{0}}{\beta_{0}}\end{aligned}\right. \tag{4.116}\]

We now impose \(CS^{\prime}-SC^{\prime}=1\) to find \(S^{\prime}\):

\[\begin{aligned} & S^{\prime}\sqrt{\frac{\beta}{\beta_{0}}}\cos \Delta\mu+SS^{\prime}\frac{\alpha_{0}}{\beta_{0}}+S\left(\frac{\sin\Delta\mu+ \alpha\cos\Delta\mu}{\sqrt{\beta\beta_{0}}}\right)-SS^{\prime}\frac{\alpha_{0 }}{\beta_{0}}=1;\\ & S^{\prime}=\frac{1}{\sqrt{\frac{\beta}{\beta_{0}}}\cos\Delta \mu}\left[1-S\frac{(\sin\Delta\mu+\alpha\cos\Delta\mu)}{\sqrt{\beta\beta_{0}}} \right]\end{aligned} \tag{4.117}\]

Finally, we remind that \(S\) is a solution of Hill's equation, and thereby it can be used to evaluate the C-S invariant at \(s=0\) and at any downstream coordinate. We choose the second point in correspondence of a local maximum of the betatron function (\(\alpha=0\) or \(\beta=1/\gamma\)):

\[\begin{aligned} &\beta S^{\prime 2}+2\alpha SS^{\prime}+ \gamma\,S^{2}=\beta_{0}S(0)^{\prime 2}+2\alpha_{0}S(0)S^{\prime}(0)+\gamma_{0}S(0)^{2}= \beta_{0};\\ &\beta S^{\prime 2}+\frac{S^{2}}{\beta}=\beta_{0};\\ &\frac{S^{2}}{\beta\beta_{0}}=1-\frac{\beta_{0}}{\beta}S^{\prime 2 }\end{aligned} \tag{4.118}\]

We conclude that: (i) since \(S(0)=0\), it has to be \(S\sim\sin\Delta\mu\); (ii) it does not depend from \(\alpha_{0}\), therefore it does not depend from \(\alpha\) either; (iii) it has to be a length. These properties are all simultaneously satisfied by \(S=\sqrt{\beta\beta_{0}}\sin\Delta\mu\). This is nowplugged into the expressions for \(C\), \(C^{\prime}\) and \(S^{\prime}\) (Eqs. 4.116 and 4.117) to obtain the following correspondence:

\[M=\begin{pmatrix}C&S\\ C^{\prime}&S^{\prime}\end{pmatrix}=\begin{pmatrix}\sqrt{\frac{\beta}{\beta_{0}}} \left(\cos\Delta\mu+\alpha_{0}\sin\Delta\mu\right)&\sqrt{\beta_{0}\beta}\sin \Delta\mu\\ \frac{(\alpha-\alpha_{0})\cos\Delta\mu-(1+\alpha\alpha_{0})\sin\Delta\mu}{ \sqrt{\beta_{0}\beta}}\,\,\sqrt{\frac{\beta_{0}}{\beta}}\,\,(\cos\Delta\mu- \alpha\sin\Delta\mu)\end{pmatrix} \tag{4.119}\]

\[\Rightarrow M_{t}:=M_{turn}(\alpha=0)=\begin{pmatrix}\cos\Delta\mu_{t}&\beta \sin\Delta\mu_{t}\\ -\frac{1}{\beta}\sin\Delta\mu_{t}&\cos\Delta\mu_{t}\end{pmatrix} \tag{4.120}\]

The one-turn matrix \(M_{t}\) for a periodic lattice was conveniently chosen in correspondence of a point of local maximum or minimum of the betatron function. The stability condition \(|Tr(M_{t})|<2\) is satisfied by \(\Delta\mu_{t}=2\pi\,Q_{u}\neq n\pi\,,n\in\mathbb{N}\) and \(Q_{u}\) the betatron tune. The condition \(Q_{u}=n/2\) identifies the so-called "integer" and "half-integer" resonance (see later). It is immediate to see from Eq. 4.119 that for _periodic_ C-S parameters, we can always write \(Tr(M_{t})=2\cos\Delta\mu_{t}\).

#### Non-Periodic Motion

We may ask ourselves what is the meaning of the C-S parameters, i.e., a phase-amplitude representation of particle's motion, in a non-periodic lattice, like that one of a single-pass linac.

Naively, since the particle "does not know" whether the lattice it is going to pass through is periodic or not, we might infer that the C-S formalism introduced for periodic motion applies to a non-periodic lattice as well. If we keep the solution of Hill's equation as in Eq. 4.88, we find the differential equation the betatron function has to satisfy in a generic non-periodic system. We adopt the notation \(C=\cos\Delta\mu\), \(S=\sin\Delta\mu\) for brevity (not to be confused with the principal trajectories), and \(\beta(s)=p^{2}(s)\):

\[\begin{array}{l}u=a\sqrt{\beta}\cos\Delta\mu\equiv apC,\\ u^{\prime}=ap^{\prime}C-apS\mu^{\prime},\\ u^{\prime\prime}=ap^{\prime\prime}C-ap^{\prime}S\mu^{\prime}-ap^{\prime}S\mu^ {\prime}-ap(C\mu^{\prime 2}+S\mu^{\prime\prime}).\\ \\ \mu^{\prime}=\frac{1}{p^{2}},\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\mu^{\prime \prime}=-\frac{2}{p^{3}}p^{\prime}\\ \\ \Rightarrow u^{\prime\prime}+Ku=0;\\ ap^{\prime\prime}C-ap^{\prime}\frac{S}{p^{2}}-ap^{\prime}\frac{S}{p^{2}}-ap \frac{C}{p^{4}}+apS^{2}\frac{p^{\prime}}{p^{3}}+aKpC=0;\\ C\left(p^{\prime\prime}+Kp-\frac{1}{p^{3}}\right)-S\left(-\frac{2p^{\prime}}{ p^{2}}+\frac{2p^{\prime}}{p^{2}}\right)=0;\\ \\ \Rightarrow p^{\prime\prime}+Kp=\frac{1}{p^{3}}\end{array} \tag{4.121}\]By replacing the definition \(p(s)=\sqrt{\beta(s)}\) into the last expression of Eq. 4.121, we find:

\[\tfrac{1}{2}\beta\beta^{\prime\prime}-\tfrac{1}{4}\beta^{\prime 2}+K\,\beta^{2}=1 \tag{4.122}\]

We draw the following observations.

* When \(K=0\), such as in a drift section, the solution of the equation for initial conditions \(\beta(0)=\beta_{0}\), \(\beta^{\prime}(0)=0\) is \(\beta(s)=\beta_{0}+\tfrac{s^{2}}{\beta_{0}}\). Namely, the betatron function grows quadratically from the minimum value \(\beta_{0}\).
* When \(|K|\gg\left[\tfrac{1}{\beta^{2}},\left(\tfrac{1}{\beta}\tfrac{d\beta}{ds} \right)^{2}\right]\), the equation reduces to \(\beta^{\prime\prime}=-2K\,\beta\). For example, this is the case of a betatron function far from a minimum, or smooth optics in the presence of moderate strengths. The solution is either oscillatory or exponential depending on the sign of K (for example, passing through a focusing or defocusing quadrupole magnet, respectively).

We conclude that, if Eq. 4.122 has solution, the C-S invariant introduced in Eq. 4.103 is also well-defined in a non-periodic lattice, and subject to the same conditions of linearity and non-frictional forces discussed above.

Single particle's betatron motion in the transverse phase space along a non-periodic lattice is illustrated in Fig. 4.13-right plot. The periodic boundary conditions imposed to \(\beta_{u},\alpha_{u}\) in a synchrotron--if a periodic solution exists--are replaced by arbitrary initial conditions \(\beta_{u,0}\), \(\alpha_{u,0}\), and \(\Delta\mu_{u,0}=0\). For any given lattice--i.e., a sequence of transfer matrices known in terms of principal trajectories \(C\), \(S\)--\(\beta_{u}(s)\), \(\alpha_{u}(s)\) are uniquely defined at any \(s\) by virtue of Eq. 4.119. Moreover, for any given particle's invariant \(J\), the particle's coordinates \(u(s)\), \(u^{\prime}(s)\) are also uniquely determined by virtue of Eqs. 4.102 and 4.90.

Yet, we have not formally justified the adoption of Floquet's solution for the non-periodic Hill's equation with arbitrary variable coefficient \(K(s)\). To do so, we may proceed in two complementary ways.

First, the non-periodic problem can be reduced to a periodic one, by demonstrating that for any choice of \(\beta_{0}\), \(\alpha_{0}\), there exists a lattice making the beam line periodic, which would allow us to invoke Floquet's theorem. This condition is always satisfied by the inverse matrix of the original line, or \(MM^{-1}=I\). We note that in this case \(Tr\,(MM^{-1})=2\), contrary to the prescription for stability in Eq. 4.80 (we will show that such lattice satisfies the so-called "integer resonance"), and that the initial conditions are unconstrained. The impasse is removed as long as we require the motion to be bounded on a single-pass only, and the initial conditions are arbitrarily set by the user.

Second, we note that any accelerator lattice can be modeled as a series of constant focusing elements, in the sense of Eq. 4.65. This allows the non-periodic Hill's equation with arbitrary variable coefficient \(K(s)\) to be reduced locally to an equation with constant coefficient \(K\), which has solution for any \(K\) (either oscillatory or hyperbolic, depending on the sign of \(K\)).

In conclusion, the equivalence of matrices in Eq. 4.119 is formally preserved for an arbitrary non-periodic lattice as long as a piece-wise constant focusing representation of the beam line is allowed. The description of the motion is exact up to the chosen order of expansion of the matrix elements in the particle's coordinates.

#### Summary

The analysis of the single particle transverse dynamics has brought to the following findings.

* Alternated strong quadrupole focusing can guarantee stable motion in both transverse planes as predicted by Hill's equations, forcing particles to move in proximity of the reference orbit.
* Particle's motion in the phase space \((u,u^{\prime})\) is that of a quasi-harmonic oscillator. The Courant-Snyder parameters describe a phase space ellipse, on which the particle's representative point lies. The ellipse area is an invariant of the particle's motion in the assumption of linear, non-dissipative dynamics.
* Linear optics is described through the product of symplectic matrices. These can be written either in terms of the geometric parameters of the accelerator component, i.e., principal trajectories (Eq. 4.65), or of the lattice Courant-Snyder parameters (Eq. 4.119).
* Long-term stability in a periodic system requires that the trace of the \(2\times 2\) one-turn transfer matrix be \(|Tr(M_{t})|<2\).

### 4.4 Beam Envelope

#### Statistical Emittance

Several hundreds' million particles can be stored in a bunch, in linacs as well in synchrotrons. While particle tracking codes are able to model the motion of individual particles, the theory of single particle dynamics can be used to reduce the complexity of the problem by keeping track of the beam envelope, i.e., of the bunch as a whole [5].

It was shown that the C-S parameters determine the shape and the orientation of a particle's phase space ellipse. The ellipse area--the C-S particle's invariant--depends from the particle's initial conditions (Eq. 4.102), which vary from particle to particle. If the motion of all beam particles is described by the same C-S parameters defined by the lattice (Eq. 4.119), then all ellipses are _omothetic_, at any \(s\), and the beam is said to be _matched_ to the lattice.

The first consequence of this is that, as long as phase space ellipses can be defined, i.e., as long as the motion is linear and non-dissipative, particles' orbits in phase space never overlap. The second implication is that the ensemble of representative points of the charge distribution can be described by an "envelope ellipse" containing all or, conventionally, a given percentage of beam particles, as shown in Fig. 4.14. The phase space area occupied by the discrete distribution of \(N\) particles is therefore approximated to the envelope ellipse area, called _beam emittance_.

Let us assume for simplicity a monochromatic beam (i.e., vanishing energy spread) in non-dispersive motion. Conventionally, and especially for lepton beams, the envelope ellipse semi-axes are made to coincide with the Root-Mean-Square (rms) values \(\sigma_{u}=\sqrt{\langle u^{2}\rangle}\), \(\sigma_{u^{\prime}}=\sqrt{\langle u^{\prime 2}\rangle}\) of the distribution, where for simplicity we assume a distribution centered both in \(u\), \(u^{\prime}\), i.e. \(\langle u\rangle=\langle u^{\prime}\rangle=0\). Owing to the correspondence of ellipse axes and second order momenta of the charge distribution function, the envelope ellipse phase space area is called _root-mean-square statistical emittance_ or, simply, rms emittance. In the absence of \((u,u^{\prime})\) correlation, it results (compare with Eq. 4.108):

\[\epsilon_{u}=\frac{Area}{\pi}=\sigma_{u}\sigma_{u^{\prime}} \tag{4.123}\]

In the most general case \(\langle uu^{\prime}\rangle\neq 0\), the envelope ellipse axes (directions \(w,w^{\prime}\) in Fig. 4.14) can be rotated with respect to the Cartesian axes by the angle \(\theta\) which minimizes the spread in position along the rotated axes. Namely, \(\theta\) satisfies the least square condition:

\[\frac{d}{d\theta}\sum_{i}w_{i}^{2}=\frac{d}{d\theta}\sum_{i}\left(u_{i}^{ \prime}\cos\theta-u_{i}\sin\theta\right)^{2}=0 \tag{4.124}\]

Figure 4.14: Omothetic phase space ellipses containing different percentages of beam particles

The expansion of the expression above leads to:

\[\begin{array}{l}\frac{d\langle w^{2}\rangle}{d\theta}=-\frac{2}{N}\sum_{i}\left(u _{i}^{\prime 2}\sin\theta\cos\theta+u_{i}u_{i}^{\prime}\cos^{2}\theta-u_{i}u_{i}^{ \prime}\sin^{2}\theta-u_{i}^{2}\sin\theta\cos\theta\right)=\\ =-\frac{2}{N}\sum_{i}\left[u_{i}^{\prime 2}\frac{\sin 2\theta}{2}-u_{i}u_{i}^{ \prime}(1-2\cos^{2}\theta)-u_{i}^{2}\frac{\sin 2\theta}{2}\right]=\\ =-\frac{1}{N}\sum_{i}\left[(u_{i}^{\prime 2}-u_{i}^{2})\sin 2\theta+2u_{i}u_{i}^{ \prime}\cos 2\theta\right]=\\ =-\frac{1}{N}\sum_{i}\cos 2\theta\left[(u_{i}^{\prime 2}-u_{i}^{2})\tan 2 \theta+2u_{i}u_{i}^{\prime}\right]=0\end{array}\]

\(\Rightarrow\tan(2\theta)=\frac{2\langle uu^{\prime}\rangle}{\langle u^{2} \rangle-\langle u^{\prime 2}\rangle}\),

\[\begin{array}{l}\langle w^{2}\rangle=\frac{1}{2}\left(\langle u^{2}\rangle+ \langle u^{\prime 2}\rangle+\frac{2\langle uu^{\prime}\rangle}{\sin 2\theta} \right),\\ \langle w^{\prime 2}\rangle=\frac{1}{2}\left(\langle u^{2}\rangle+ \langle u^{\prime 2}\rangle-\frac{2\langle uu^{\prime}\rangle}{\sin 2\theta} \right)\end{array}\]

By applying Eq. 4.123 to the new orthogonal axes \(w,\,w^{\prime}\), and with the help of Eq. 4.125, one finds:

\[\begin{array}{l}\epsilon_{u}=\sigma_{w}\sigma_{w^{\prime}}=\frac{1}{2}\left( \langle u^{2}\rangle+\langle u^{\prime 2}\rangle+\frac{2\langle uu^{\prime} \rangle}{\sin 2\theta}\right)^{1/2}\left(\langle u^{2}\rangle+\langle u^{\prime 2} \rangle-\frac{2\langle uu^{\prime}\rangle}{\sin 2\theta}\right)^{1/2}=\\ =\left[\left(\langle u^{2}\rangle+\langle u^{\prime 2}\rangle\right)^{2}- \frac{4\langle uu^{\prime}\rangle^{2}}{\sin^{2}2\theta}\right]^{1/2}=\\ =\left[\left(\langle u^{2}\rangle+\langle u^{\prime 2}\rangle\right)^{2}-4 \langle uu^{\prime}\rangle^{2}\left(1+\cot^{2}2\theta\right)\right]^{1/2}=\\ =\left\{\left(\langle u^{2}\rangle+\langle u^{\prime 2}\rangle\right)^{2}-4 \langle uu^{\prime}\rangle^{2}\left[1+\frac{\left(\langle u^{2}\rangle- \langle u^{\prime 2}\rangle\right)^{2}}{4\langle uu^{\prime}\rangle^{2}} \right]\right\}^{1/2}=\\ =\frac{1}{2}\left[4\langle u^{2}\rangle\langle u^{\prime 2}\rangle-4 \langle uu^{\prime}\rangle\right]^{1/2}=\\ =\sqrt{\langle u^{2}\rangle\langle u^{\prime 2}\rangle-\langle uu^{\prime} \rangle^{2}}\end{array}\]

Averages are intended over the particles' ensemble:

\[\langle u^{2}\rangle=\frac{1}{N}\sum_{i=1}^{N}u_{i}^{2},\qquad\langle u^{ \prime 2}\rangle=\frac{1}{N}\sum_{i=1}^{N}u_{i}^{\prime 2},\qquad\langle uu^{ \prime}\rangle=\frac{1}{N}\sum_{i=1}^{N}u_{i}u_{i}^{\prime}\]

Some additional algebric manipulation of Eq. 4.126 gives:

\[\begin{array}{l}\epsilon_{u}=\sqrt{\frac{1}{N}\sum_{i}u_{i}^{2}\frac{1}{N} \sum_{i}u_{i}^{\prime 2}-\left(\frac{1}{N}\sum_{i}u_{i}u_{i}^{\prime}\right)^{2}}=\\ =\frac{1}{\sqrt{2}N}\sqrt{\sum_{i}\sum_{j}(u_{i}u_{j}^{\prime}-u_{j}u_{i}^{ \prime})^{2}}=\frac{1}{N}\sqrt{2\sum_{i}\sum_{j}A_{ij}^{2}}\end{array}\]

The rms emittance can therefore be interpreted as the rms value of the phase space areas of triangles (\(A_{ij}\)) made of any two points of the distribution and connected to the origin. However, if the transfer map of the accelerator is not linear, triangles do not transform into triangles necessarily. This suggests that the rms emittance is _not_ preserved under _nonlinear_ transformations.

The percentage of particles contributing to the rms emittance depends from the 2-D distribution function in the phase space. Since in many cases the charge distribution in accelerators is Gaussian, or similar to that, it is instructive to consider a Gaussian distribution hereafter. In such case, it is possible to calculate analytically the percentage of particles contributing to the rms emittance and, in a more general case, to the phase space area extending over an arbitrary number of sigmas in \(u\) and \(u^{\prime}\).

Let us consider the centered 2-D Gaussian distribution in Fig. 4.14; we assume for simplicity \(\langle uu^{\prime}\rangle=0\):

\[f(u,u^{\prime})=\frac{1}{2\pi\sigma_{u}\sigma_{u^{\prime}}}e^{-\frac{u^{2}}{2 \sigma_{u}^{2}}}e^{-\frac{u^{\prime 2}}{2\sigma_{u^{\prime}}^{2}}} \tag{4.129}\]

Projection of the distribution onto the \(u\), \(u^{\prime}\) axis provides, of course, a 1-D Gaussian distribution in the \(u\), \(u^{\prime}\) coordinate, respectively. Projection of the distribution onto the \((u,u^{\prime})\) plane with a cut-off at \(k\)-sigmas both in \(u\), \(u^{\prime}\) generates an ellipse of area \(k^{2}\sigma_{u}\sigma_{u}^{\prime}\). The ellipse equation is:

\[\frac{u^{2}}{k^{2}\sigma_{u}^{2}}+\frac{u^{\prime 2}}{k^{2}\sigma_{u^{\prime}} ^{2}}=1\quad\text{ \ \ \ or \ \ \ \ }\frac{u^{2}}{2\sigma_{u}^{2}}+\frac{u^{\prime 2}}{2\sigma_{u^{\prime}}^{2}}= \frac{k^{2}}{2} \tag{4.130}\]

The relative fraction of particles contained in \(k\)-sigmas and normalized to 1 is, by virtue of Eq. 4.130:

\[P=\frac{f(0,0)-f(x,x^{\prime};k)}{\hat{f}(x,x^{\prime})}=1-\frac{f(x,x^{\prime };k)}{f(0,0)}=1-e^{-\frac{k^{2}}{2}} \tag{4.131}\]

The rms emittance \(\epsilon_{u}\) corresponds to \(k=1\) and therefore to \(P(k=1)=39\%\). We find, for example, \(P(1.5)=67.5\%\), \(P(2)=86\%\), \(P(2.45)=95\%\), \(P(3)=98.9\%\), and \(P(4)=99.97\%\).

#### Transverse Beam Matrix

The equivalence of the envelope ellipse area of a matched beam and the statistical emittance is shown below by recalling the expression of particle's position and angular divergence as solutions of Hill's equation, and by averaging the particle's coordinates over the particles' ensemble. This translates into an average over the particles' invariant \(J_{i}\) and initial betatron phase \(\mu_{0,i}\). We find:\[\begin{array}{l}\langle u^{2}\rangle=\frac{1}{N}\sum_{i=1}^{N}2J_{i}\,\beta(s) \cos^{2}\Delta\mu_{i}(s)=\beta(s)\langle J\rangle\\ \langle u^{\prime 2}\rangle=\frac{1}{N}\sum_{i=1}^{N}\frac{2J_{i}}{\beta(s)}\left[ \alpha(s)\cos\Delta\mu_{i}(s)+\sin\Delta\mu_{i}(s)\right]^{2}=\gamma\,(s) \langle J\rangle\\ \langle uu^{\prime}\rangle=-\frac{1}{N}\sum_{i=1}^{N}2J_{i}\,\cos\Delta\mu_{i }(s)\left[\alpha(s)\cos\Delta\mu_{i}(s)+\sin\Delta\mu_{i}(s)\right]=-\alpha(s )\langle J\rangle\\ \Rightarrow\epsilon_{u}=\sqrt{\langle u^{2}\rangle\langle u^{\prime 2} \rangle-\langle uu^{\prime}\rangle^{2}}=\langle J\rangle\sqrt{\beta\gamma- \alpha^{2}}=\langle J\rangle\\ \Rightarrow\left\{\begin{array}{l}\sigma_{u}^{2}(s)=\langle u^{2}(s)\rangle =\epsilon_{u}\beta(s)\\ \\ \sigma_{u}^{\prime 2}(s)=\langle u^{\prime 2}(s)\rangle=\epsilon_{u}\gamma(s)\\ \\ \langle uu^{\prime}\rangle=-\epsilon_{u}\alpha(s)\end{array}\right.\end{array} \tag{4.132}\]

The condition \(\alpha_{u}=0\) identifies a local maximum or minimum of the beam's rms size (minimum or maximum of the rms angular divergence). The latter identities can be cast in matrix form:

\[\sigma:=\epsilon_{u}\begin{pmatrix}\beta_{u}&-\alpha_{u}\\ -\alpha_{u}&\gamma_{u}\end{pmatrix}\equiv\begin{pmatrix}\langle u^{2}\rangle& \langle uu^{\prime}\rangle\\ \langle uu^{\prime}\rangle&\langle u^{\prime 2}\rangle\end{pmatrix} \tag{4.133}\]

The matrix \(\sigma\) for the \(u\)-transverse plane is said (covariant) _beam matrix_ and it satisfies \(\sqrt{\det\sigma}=\epsilon_{u}\). It is essential to keep in mind that, while the C-S invariant is the single particle's constant of motion, the emittance is a quantity referring to an ensemble of particles. Indeed, Eq. 4.126 tells us that the emittance of a single particle is zero.

In order to find the transformation rule of \(\sigma\) in the presence of a transfer matrix \(M\), we first consider the generic coordinates \((u,u^{\prime})\), and calculate:

\[\begin{array}{l}\vec{u}^{T}\sigma^{-1}\vec{u}=(u,u^{\prime})\frac{1}{ \epsilon^{2}}\epsilon\begin{pmatrix}\gamma_{u}&\alpha_{u}\\ \alpha_{u}&\beta_{u}\end{pmatrix}\begin{pmatrix}u\\ u^{\prime}\end{pmatrix}=\frac{1}{\epsilon}(u,u^{\prime})\begin{pmatrix}\gamma _{u}u+\alpha_{u}u^{\prime}\\ \alpha_{u}u+\beta_{u}u^{\prime}\end{pmatrix}=\\ =\frac{1}{\epsilon}(\gamma u^{2}+2\alpha uu^{\prime}+\beta u^{\prime 2})= \frac{2J}{\epsilon}=const.\end{array} \tag{4.134}\]

Since the single particle's phase space vector transforms as \(\vec{u}=M\vec{u}_{0}\), Eq. 4.134 can be re-written as follows:

\[\begin{array}{l}\vec{u}^{T}\sigma^{-1}\vec{u}=\left(M\vec{u}_{0}\right)^{T} \sigma^{-1}M\vec{u}_{0}=\vec{u}_{0}^{T}M^{T}\sigma^{-1}M\vec{u}_{0}\equiv \vec{u}_{0}^{T}\sigma_{0}^{-1}\vec{u}_{0}\\ \Rightarrow M^{T}\sigma^{-1}M=\sigma_{0}^{-1};\\ \sigma_{0}=\left(M^{T}\sigma^{-1}M\right)^{-1}=M^{-1}\sigma\left(M^{T}\right) ^{-1};\\ M\sigma_{0}M^{T}=MM^{-1}\sigma\left(M^{T}\right)^{-1}M^{T};\\ \Rightarrow\sigma\left(s\right)=M\sigma_{0}M^{T}\end{array} \tag{4.135}\]In practical situations, if the beam is not matched to the lattice, a preliminary manipulation of the rms beam size and divergence is needed in order to adapt the beam's C-S parameters to the design values. This procedure is commonly carried out in the transverse planes with solenoid fields and/or quadrupole magnets, and it is called _optics matching_.

When a non-zero rms energy spread \(\sigma_{\delta}\) is considered, the beam envelope is modified by the presence, if any, of the dispersion function (as for the beam size) and of its first derivative (as for the beam divergence). Owing to the linear superposition of the solutions \(u_{\beta}\) and \(u_{D}=D_{u}\delta\) in Eq. 4.59, the most general expression for the rms beam size and angular divergence becomes:

\[\left\{\begin{aligned} &\sigma_{u}(s)=\sqrt{(u_{\beta}^{2}(s))+ \langle u_{D}^{2}(s)\rangle}=\sqrt{\epsilon_{u}\beta_{u}(s)+(D_{u}(s)\sigma_{ \delta})^{2}}\\ &\sigma_{u^{\prime}}(s)=\sqrt{({u^{\prime}_{\beta}}^{2}(s))+({u^ {\prime}_{D}}^{2}(s))}=\sqrt{\epsilon_{u}\gamma_{u}(s)+\big{(}D^{\prime}_{u}( s)\sigma_{\delta}\big{)}^{2}}\end{aligned}\right. \tag{4.136}\]

We remind for completeness that, since the dispersion function is also a solution of Hll's complete equation, it is a periodic function when the magnetic lattice is periodic, and so its derivative is. In this case, the periodic dispersion function is noted as \(\eta(s)=\eta(s+C)\), \(\eta^{\prime}(s)=\eta^{\prime}(s+C)\)\(\forall s\), see Eq. 4.113.

#### Transfer of Courant-Snyder Parameters

Equation 4.135 says that if the lattice map is described by principal trajectories, the beam's representative ellipse can be tracked by transporting the beam's C-S parameters through the lattice. The transfer matrix mapping the C-S parameters through the accelerator is found below by using the definition of principal trajectories:

\[\left\{\begin{aligned} & u=Cu_{0}+Su^{\prime}_{0}\quad\cdot(S^{ \prime})\\ & u^{\prime}=C^{\prime}u_{0}+Su^{\prime}_{0}\quad\cdot(-S)\\ & u=Cu_{0}+Su^{\prime}_{0}\quad\cdot(-C^{\prime})\\ & u^{\prime}=C^{\prime}u_{0}+Su^{\prime}_{0}\quad\cdot(C)\\ \end{aligned}\right.\Rightarrow(CS^{\prime}-SC^{ \prime})u_{0}=u_{0}=(S^{\prime}u-Su^{\prime})\]

Then, the C-S invariant is calculated:

\[\begin{aligned} &\gamma_{0}u_{0}^{2}+2\alpha_{0}u_{0}u^{ \prime}_{0}+\beta_{0}u^{\prime 2}_{0}=\\ &=\gamma_{0}(S^{\prime}u-Su^{\prime})^{2}+2\alpha_{0}(S^{\prime} u-Su^{\prime})(Cu^{\prime}-C^{\prime}u)+\beta_{0}(Cu^{\prime}-C^{\prime}u)^{2}=\\ &=(\gamma_{0}S^{\prime 2}-2\alpha_{0}C^{\prime}S^{\prime}+\beta_{0}C^{ \prime 2})u^{2}+\\ &\quad\quad-2(\gamma_{0}SS^{\prime}-\alpha_{0}CS^{\prime}-\alpha _{0}SC^{\prime}+2\beta_{0}CC^{\prime})uu^{\prime}+\\ &\quad\quad+(\gamma_{0}S^{2}-2\alpha_{0}CS+\beta_{0}C^{2})u^{ \prime 2}\equiv\\ &\equiv\gamma u^{2}+2\alpha uu^{\prime}+\beta u^{\prime 2}\end{aligned}\right.\]Equating member-to-member in the previous expression we find:

\[\begin{pmatrix}\beta\\ \alpha\\ \gamma^{\prime}\end{pmatrix}_{s}=\begin{pmatrix}C^{2}&-2CS&S^{2}\\ -CC^{\prime}&CS^{\prime}+SC^{\prime}&-SS^{\prime}\\ C^{\prime 2}&-2C^{\prime}S^{\prime}&S^{\prime 2}\end{pmatrix}\begin{pmatrix} \beta\\ \alpha\\ \gamma^{\prime}\end{pmatrix}_{0} \tag{4.137}\]

In reality, the second order momenta of the charge distribution can be retrieved from measurements of the beam size and divergence at a certain location in the accelerator. Equation 4.126 is then used to calculate \(\epsilon_{u}\). The correspondence established by Eq. 4.133 allows one to determine the C-S parameters associated to the charge distribution. If the "beam C-S parameters" are matched to the "lattice C-S parameters" at a specific location, then the beam dynamics can be predicted and controlled from that point on by just looking to the beam envelope. Namely, the beam dynamics is reduced to the evolution of the C-S parameters (in geometric sense, of the beam envelope ellipse) through the lattice according to Eq. 4.137.

#### Discussion:Low-\(\beta\) Insertion

Straight sections, or drifts, are present in linacs and synchrotrons to accommodate, for example, beam diagnostics, insertion devices, detectors, etc. It is sometimes convenient to design the beam optics such that the transverse beam sizes along the drift section are minimized. These sections are therefore named _low-\(\beta\) insertions_. What are the C-S parameters at the entrance of the drift which, for any given drift length \(L\), minimize the betatron function overall? What is the maximum phase advance allowed by such special configuration?

For symmetry, a minimum value of the beam size averaged over the drift implies a beam waist, i.e., a minimum of the betatron function, in the middle of the section. The two transverse planes behave, of course, identically. It is then convenient to define the origin of the \(s\)-axis at the middle point of the drift. In each plane, the drift transfer matrix is built with principal trajectories \(C=S^{\prime}=1\), \(C^{\prime}=0\), \(S=L\) (see Eq. 4.72). Then, the transfer matrix for the C-S parameters with initial conditions \(\beta_{0}>0\), \(\alpha_{0}=0\) is determined according to Eq. 4.137. The Betatron function propagates from the waist towards the drift ends as follows:

\[\left\{\begin{aligned} \beta(s)&=C(s)^{2}\beta_{0}-2C(s)S(s) \alpha_{0}+S(s)^{2}\gamma_{0}=\beta_{0}-2s\alpha_{0}+s^{2}\gamma_{0}=\beta_{0} +\frac{s^{2}}{\beta_{0}}\\ \alpha(s)&=-CC^{\prime}\beta_{0}+(CS^{\prime}+SC^{ \prime})\alpha_{0}-SS^{\prime}\gamma_{0}=-\frac{s}{\beta_{0}}\end{aligned}\right. \tag{4.138}\]

By construction, \(\beta(s)\) is already set to its minimum value \(\beta_{0}\) at the midpoint (\(s=0\)). Its value is minimized along the whole drift by a suitable choice of \(\beta_{0}\), which is found by imposing:

\[\left(\frac{d\beta(s)}{d\beta_{0}}\right)_{s=L/2}=1-\frac{L^{2}}{4\beta_{0}^{2 }}\equiv 0\ \ \Rightarrow\ \ \beta_{0,opt}=\frac{L}{2}. \tag{4.139}\]It follows from Eq. 4.138 that the C-S parameters at the edges of the drift are \(\beta(L/2)=L\), \(\alpha(L/2)=-1\) (the beam is diverging if seen from the midpoint, converging if seen from the opposite direction).

Figure 4.15 shows the betatron function in a low-\(\beta\) insertion for optimal and non-optimal initial C-S parameters. Any deviation from the initial conditions found above leads to a larger average betatron function, although a local minimum can still be obtained.

The betatron phase advance can be calculated from Eq. 4.90:

\[\Delta\mu=\int_{-L/2}^{L/2}\frac{1}{\beta(s)}ds=\int_{-L/2}^{L/2}\frac{1}{\beta _{0}+\frac{s^{2}}{\beta_{0}}}ds=2\arctan\left(\frac{L}{2\beta_{0}}\right)= \frac{\pi}{2} \tag{4.140}\]

The phase advance tends to \(\pi\) when \(\frac{L}{\beta_{0}}\rightarrow\infty\).

#### Longitudinal Beam Matrix

Equation 4.31 for the particle's longitudinal oscillations is formally identical to Hills' equation for the betatron motion, Eq. 4.78. In the former case, the oscillations are defined in the longitudinal phase space (\(z\),\(\delta\)), \(z\) and \(\delta\) being the reduced variables for the particle's longitudinal position and relative longitudinal momentum. When the latter is constant on average over one turn, or approximately constant during adiabatic acceleration, the longitudinal motion is (approximately) purely harmonic.

By virtue of such similarity, the longitudinal motion can be described with formalism identical to that adopted so far for the transverse planes, see Eq. 4.102, with the additional feature of a constant amplitude of oscillation (independent from \(s\)). Hence, longitudinal C-S parameters, C-S invariant, statistical emittance, and beam matrix, can be equivalently defined. The single particle's longitudinal invariant \(J_{z}\) is

Figure 4.15: Betatron function along a drift section. The symmetric solution (solid, \(\alpha_{0}=1\), \(\beta_{0}=L\)) determines the minimum value of the betatron function averaged over the whole section. All other solutions can lead to even lower local minima, but the lower the minimum is, the faster the rising of the function elsewhere will be, so that the average value is eventually larger (dashed, \(\alpha_{0}=0.5\), \(\beta_{0}=L/2\); dot-dashed, \(\alpha_{0}=2\), \(\beta_{0}=2L\))

constant as long as the motion can be approximated to a linear and non-dissipative one.

The longitudinal beam matrix \(\Sigma\) transforms as in Eq. 4.135:

\[\Sigma(s)=M\,\Sigma_{0}M^{T}=\epsilon_{z}\begin{pmatrix}\beta_{z}&-\alpha_{z}\\ -\alpha_{z}&\gamma_{z}\end{pmatrix}=\begin{pmatrix}\langle z^{2}\rangle& \langle z\delta\rangle\\ \langle z\delta\rangle&\langle\delta^{2}\rangle\end{pmatrix} \tag{4.141}\]

The linear energy correlation along the bunch (linear energy chirp, see Eq. 4.25) and the beam total relative energy spread result:

\[\begin{array}{c}h=\frac{\langle z\delta\rangle}{\langle z^{2}\rangle}=- \frac{\alpha_{z}}{\beta_{z}}\\ \\ \sigma_{\delta}=\sqrt{\langle\delta^{2}\rangle}=\epsilon_{z}\gamma_{z}\end{array} \tag{4.142}\]

The Beam's longitudinal emittance is:

\[\epsilon_{z}=\sqrt{det(\Sigma)}=\sqrt{\langle z^{2}\rangle\langle\delta^{2} \rangle-\langle z\delta\rangle^{2}}\rightarrow\sigma_{z}\sigma_{\delta} \tag{4.143}\]

and the limit is for null energy chirp. In this case, the energy spread is said to be _uncorrelated_.

#### Normalized Emittance

The transverse rms emittance is also named _geometric_ (also _natural_ in synchrotrons) to recall its geometric meaning in the phase space, \(\epsilon_{u}\sim\sigma_{u}\sigma_{u^{\prime}}\). By virtue of Eq. 4.132, it is also a _constant of motion_ under the assumption of linear and non-dissipative dynamics.

But, if the beam's mean energy is not constant, \(\epsilon_{u}\) is not expected to be constant anymore because a frictional force is introduced, which translates for example into an instantaneous variation of the particle's angular divergence via Eq. 4.100. To quantify the emittance variation with beam's energy, we first consider instantaneous longitudinal acceleration, small energy spread (\(\sigma_{\delta}\ll 1\)), and we assume for simplicity \(\alpha_{u}=0\):

\[\epsilon_{u}=\sigma_{u}\sigma_{u^{\prime}}=\frac{\sigma_{u}\sigma_{p_{u}}}{p_{ z,s}}=\left(\frac{\sigma_{u}\sigma_{p_{u}}}{m_{0}c}\right)\frac{1}{\beta_{z}\gamma} \tag{4.144}\]

\(\beta_{z},\,\gamma\) are the common Lorentz factors. It results that the geometric emittance is inversely proportional to the beam's mean energy because pure longitudinal acceleration does affect neither the beam size nor the transverse momentum, but only the longitudinal momentum. In practice, the higher the energy is, the more collimated the beam will be.

To compare the emittance of accelerated beams somehow independently from the final beam's energy, an _energy-normalized emittance_ is introduced (simply "normalized emittance" hereafter):

\[\epsilon_{n,u}:=\beta_{z}\gamma\epsilon_{u}\approxeq\gamma\epsilon_{u} \tag{4.145}\]

and the approximation is for ultra-relativistic particles (\(\beta_{z}\approxeq\beta\to 1\)). It results that \(\epsilon_{n,u}\) is also defined in statistical sense, therefore it is constant under the same conditions which apply to \(\epsilon_{u}\).

In analogy to the motion in the transverse planes, the longitudinal rms emittance is \(\epsilon_{z}=\sigma_{z}\sigma_{\delta}\propto\frac{1}{\beta_{z}\gamma}\). Here, we are assuming that particles are in a deep relativistic regime, so that their velocity is approximately \(c\), in spite of their relative momentum deviation, and therefore the bunch length is constant (this is exact in high energy linacs; other effects concur to the determination of emittances in a synchrotron, which will be treated later on). Consequently, the normalized longitudinal emittance is also defined according to Eq. 4.145:

\[\epsilon_{n,z}:=\beta_{z}\gamma\epsilon_{z}=\beta_{z}\gamma\cdot\frac{\sigma_{ z}\sigma_{p_{z}}}{p_{z,s}}=\frac{\sigma_{z}\sigma_{p_{z}}}{m_{0}c}\approxeq \frac{\sigma_{z}\sigma_{E}}{\beta_{z}m_{0}c^{2}}\approxeq\sigma_{z}\sigma_{\gamma} \tag{4.146}\]

The last two approximations use \(\Delta p=\Delta E/(\beta c)\), by neglecting the transverse momenta with respect to the longitudinal (\(p_{x}\), \(p_{y}<<p_{z}\approxeq p\)), and by taking the ultra-relativistic limit \(\beta_{z}\to 1\). Equation 4.146 says that in the presence of linear, ultra-relativisitic and non-dissipative motion, the product of bunch length and _uncorrelated_ energy spread is a constant of motion.

#### Beam Brightness

Many applications of particle accelerators, and especially at high beam energies, require not only a high charge spatial density, but also the capability of the beam to be collimated by strong magnetic field gradients. In other words, the beam envelope should have both small transverse size and angular divergence, which translate into small transverse emittance.

The charge density evaluated over the transverse 4-D phase space and the bunch duration is called 5-D _peak brightness_:

\[B=\frac{Q}{\epsilon_{x}\epsilon_{y}\sigma_{t}}=\frac{I}{\epsilon_{x}\epsilon_{ y}} \tag{4.147}\]

Its energy-normalized counterpart is defined in terms of the normalized transverse emittances:

\[B_{n}=\frac{I}{\epsilon_{x,n}\epsilon_{y,n}} \tag{4.148}\]

In the ultra-relativistic limit, \(B\approxeq\gamma^{2}B_{n}\).

In high energy light sources, the spectral bandwidth of the emitted radiation can be affected by the beam's relative energy spread \(\sigma_{\delta}\). For this reason the 6-D beam brightness is conveniently introduced:

\[\mathbb{B}=\frac{I}{\epsilon_{x}\epsilon_{y}\sigma_{\delta}} \tag{4.149}\]

Its energy-normalized counterpart, defined in the ultra-relativistic limit, makes use of the normalized longitudinal emittance in Eq. 4.146:

\[\mathbb{B}_{n}=\frac{Qc}{\epsilon_{x,n}\epsilon_{y,n}\sigma_{z}\sigma_{\gamma}} \tag{4.150}\]

In the ultra-relativisitc limit, \(\mathbb{B}\approxeq\gamma^{3}\mathbb{B}_{n}\).

Light sources emitting coherent radiation can also be interested to high peak currents, because this drives the intensity of the emitted radiation. Incoherent light sources and colliders, instead, are in most cases dealing with the average peak current. This is the total beam charge evaluated over a turn in synchrotrons, the beam duty cycle in linacs. In general, if the beam is made of a train of \(n_{b}\) bunches and the repetition rate of the train production is \(f_{t}\), the 5-D _average brightness_ results:

\[\langle B\rangle=B\cdot\sigma_{t}\cdot f_{t}\cdot n_{b}=\frac{\langle I\rangle }{\epsilon_{x}\epsilon_{y}} \tag{4.151}\]

#### Discussion: Normalized Emittance of a Large Energy Spread Beam

Let us demonstrate that for an ultra-relativistic but not monochromatic particle beam, the rms normalized transverse emittance is proportional to the beam momentum spread, and that its expression reduces to Eq. 4.145 for vanishing energy spread.

Equation 4.144 shows that the normalized emittance is defined in terms of second order momenta of the spatial and transverse momentum distribution, and we recall that \(u^{\prime}=p_{u}/p_{z}\). We find:

\[\epsilon_{n,u}^{2}=\frac{\langle u^{2}\rangle\langle p_{u}^{2}\rangle}{m_{0}^{ 2}c^{2}}-\frac{\langle up_{u}\rangle^{2}}{m_{0}^{2}c^{2}}=\langle u^{2}\rangle \langle\beta_{z}^{2}\gamma^{2}u^{\prime 2}\rangle-\langle u\beta_{z}\gamma u^{ \prime}\rangle^{2} \tag{4.152}\]

In the simpler case \(\langle u^{\prime}p_{z}\rangle\approx 0\) we can write:

\[\epsilon_{n,u}^{2}\approxeq\langle u^{2}\rangle\langle u^{\prime 2}\rangle \langle\beta_{z}^{2}\gamma^{2}\rangle-\langle uu^{\prime}\rangle^{2}\langle \beta_{z}\gamma\rangle^{2} \tag{4.153}\]

With the definition of relative momentum spread

\[\sigma_{\delta}^{2}=\frac{\langle p^{2}\rangle-\langle p\rangle^{2}}{\langle p \rangle^{2}}\approxeq\frac{\langle\beta_{\varepsilon}^{2}\gamma^{2}\rangle- \langle\beta_{z}\gamma\rangle^{2}}{\langle\beta_{z}\gamma\rangle^{2}}, \tag{4.154}\]Eq. 4.153 becomes:

\[\begin{split}\epsilon_{n,u}^{2}&\approx\langle u^{2} \rangle\langle u^{\prime 2}\rangle\langle\beta_{z}\gamma\rangle^{2}(1+\sigma_{ \delta}^{2})-\langle uu^{\prime}\rangle^{2}\langle\beta_{z}\gamma\rangle^{2}= \langle\beta_{z}\gamma\rangle^{2}\left(\epsilon_{u}^{2}+\sigma_{u}^{2}\sigma_ {u^{\prime}}^{2}\sigma_{\delta}^{2}\right)=\\ &=\langle\beta_{z}\gamma\rangle^{2}\epsilon_{u}^{2}\left[1+\left( 1+\frac{\langle uu^{\prime}\rangle^{2}}{\epsilon_{u}^{2}}\right)\sigma_{ \delta}^{2}\right];\\ \epsilon_{n,u}&=\langle\beta_{z}\gamma\rangle \epsilon_{u}\sqrt{1+\sigma_{\delta}^{2}\left(1+\alpha_{u}^{2}\right)},\end{split} \tag{4.155}\]

which reduces to Eq. 4.145 for \(\sigma_{\delta}\to 0\).

Owing to \(\alpha_{u}(s)\neq 0\), the rms transverse normalized emittance evaluated for a not negligible relative energy spread is not constant any longer, even for beam transport at constant mean energy. This has some analogy to the case of an effective emittance in the presence of chromatic motion, where dispersion affects the beam size and divergence. Thereby, the rms emittance calculated from the second order momenta of the distribution (which are now contributed from betatron _and_ dispersive motion, see Eq. 4.136) is apparently larger than the emittance in the presence of betatron oscillations only.

In the presence of acceleration, the normalized emittance of a large energy spread beam is not constant even in case of "smooth optics", or \(\langle\alpha_{u}\rangle\approx 0\) along the line, because \(\epsilon_{n,u}\sim\gamma\epsilon_{u}\sigma_{\delta}\sim 1/\gamma\) (the absolute uncorrelated energy spread is assumed to be approximately constant, see Eq. 4.7 and discussion there). This is consistent with the fact that the statistical emittance in Eq. 4.155 is at third order in the particle coordinates, \(\epsilon_{n,u}\sim\sigma_{u}\sigma_{u^{\prime}}\sigma_{E}\). Namely, the single particle's motion is nonlinear, and the beam's optics is affected by chromatic aberrations.

#### Discussion: Ultimate Beam Emittance

What is the smallest transverse emittance of a particle beam?

Since the answer implies the smallest scale (both in space and momentum) of an ensemble of massive particles, it has to have quantistic nature. Heisenberg's Uncertainty Principle applied to the horizontal plane of motion states:

\[\Delta x\,\Delta p_{x}\geq\frac{h}{4\pi} \tag{4.156}\]

where \(h\) is the Planck's constant, and the widths are intended to be standard deviations. The beam angular divergence is \(\Delta x^{\prime}=\frac{\Delta p_{x}}{p_{z}}=\frac{\Delta p_{x}}{\beta_{z} \gamma m_{0}c}\), and the equation can be re-written as follows:

\[\begin{split}&\epsilon_{x}=\Delta x\,\Delta x^{\prime}=\frac{ \Delta x\,\Delta p_{x}}{\beta_{z}\gamma\mu_{0}c}\geq\frac{1}{4\pi\beta_{z} \gamma}\frac{h}{m_{0}c}=\frac{1}{\beta_{z}\gamma}\frac{\lambda c}{4\pi}\\ &\Rightarrow\epsilon_{n,u}\geq\frac{\lambda_{C}}{4\pi}\end{split} \tag{4.157}\]

The _Compton wavelength_\(\lambda_{C}=\frac{h}{m_{0}c}\) is the wavelength of a photon whose energy is equal to the rest energy of the massive particle: \(E=hv_{c}=\frac{hc}{\lambda_{C}}=m_{0}c^{2}\), and \(p=\frac{h}{\lambda_{c}}=m_{0}c=\frac{E}{c}\) (see De Broglie's wave-particle duality in Eq.1.41), where \(\lambda_{C}=2.426\cdot 10^{-6}\)\(\mu\)m for an electron. State-of-the-art electron beam normalized emittances have been measured at \(\sim 10^{-3}\)\(\mu\)m level for \(\sim\)fC beam charges, in the keV beam energy range.

## References

* [1] J. Le Duff, Dynamics and acceleration in linear structures, longitudinal beam dynamics in circular accelerators, in _Proceedings of CERN Accelerator School: 5th General Accelerators Physical Course_, CERN 94-01, vol. I, ed. by S. Turner, Geneva, Switzerland (1994), pp. 277-311
* [2] J. Rossbach, P. Schmuser, Basic course on accelerator optics, in _Proceedings of CERN Accelerator School: 5th General Accelerators Physical Course_, CERN 94-01, vol. I, ed. by S. Turner, Geneva, Switzerland (1994), pp. 17-68
* [3] M. Sands, _The Physics of Electron Storage Rings: An Introduction_, SLAC-121, UC-28 (ACC) (Stanford Linear Accelerator Center, Menlo Park, CA, USA, 1979), pp. 18-69
* [4] M. Martini, _An Introduction to Transverse Beam Dynamics in Accelerators_, CERN/PS 96-11 (PA). Lecture at the Joint University Accelerators School, Archamps, France (1996), pp. 1-43, 84-91
* [5] J. Buon, Beam phase space and emittance, in _Proceedings of CERN Accelerator School: 5th General Accelerators Physical Course_, CERN 94-01, vol. I, ed. by S. Turner, Geneva, Switzerland (1994), pp. 17-88

