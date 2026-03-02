Particle accelerators devoted to emission of radiation are named "light sources". Nowadays, the most advanced and powerful light sources driven by RF accelerators are electron storage rings and linac-driven free-electron lasers (FELs). Short electron linacs also drive Inverse Compton Scattering light sources, which extend the photon energy to \(\gamma\)-rays (up to \(\sim\)MeV photon energy). In the following, these three types of light sources are discussed.

From the middle of the XX century until 1980s, two generations of synchrotrons were developed for particle physics experiments. In spite of the attention of radiology industry already from the 1920s for x-ray emission, synchrotron radiation from dipole magnets of those machines was an undesired effect, reducing the beam energy and radio-activating vacuum and RF components of the accelerator. But, some pioneering scientists realized soon that x-rays could be used for optical experiments of matter physics. Since 1990s, 3rd generation synchrotron light sources started to be specifically designed and built for providing light pulses at high repetition rate, to several photon beamlines simultaneously. New magnetic elements, denominated _insertion devices_ (IDs), were installed in the straight sections to enhance and shape the radiation emission, in addition to dipole radiation.

More recently, a 4th generation of storage ring light sources is being built worldwide. These machines are also called _diffraction limited storage rings_: a tight and strong focusing magnetic lattice based on multi-bend cells minimizes the electron beam emittance to unprecedented levels. Light pulses emitted by such collimated electron beams show a higher degree of transverse coherence in x-rays than in any preceding circular light source.

FELs started developing in their most powerful single-pass configuration since 1990s. Today, they are complementary to storage rings as for many aspects, and characterized by extremely high peak power, spectral brightness (6-D photon density), short pulse duration, and coherence. Next generation of high gain FELs is targetingsub-femtosecond pulse durations, TW-level peak power, full coherence at multi-keV photon energies, and MHz repetition rate. Since a single FEL facility usually serves one to few beamlines only simultaneously, compactness and larger experimental fan-out are also features worth of some attention in the community.

Compton sources are incoherent light sources based on scattering of a \(\sim\)10-100s MeV energy electron beam and an IR or UV external laser. By virtue of the relativistic Doppler effect, the scattered radiation easily reaches 10s of keV to MeV photon energies. Compton sources are typically far more compact than x-ray FELs, but at the expense of peak power. Next generation of Compton sources aims at lager peak and average power, sub-picosecond pulse duration, and tuneable repetition rate up to MHz. So-called gamma-gamma colliders based on multi-GeV electron linacs have been conceived as Compton sources in which the back-scattered photon energy approaches the electron beam energy.

### 10.1 Brilliance

#### Practical Meaning

One of the figures of merit of light sources is the 6-D photon density [1, 2]. It is named _spectral brightness_ or _brilliance_, and it has some analogy to the particle beam brightness introduced in Eq. 4.149. The Brilliance describes the effective radiation flux per unit spectral bandwidth (\(d\omega/\omega\)), conventionally at the location of emission. It is meant to be a peak or an average quantity depending on if the flux is intended to be a peak (i.e., instantaneous) or time-averaged quantity (e.g., over the beam train duration, or a turn in a synchrotron).

Whenever the source, i.e. a charged particle beam, has non-zero transverse emittance, the spectral flux (or spatio-angular spectral density, defined as the spectral intensity \(\phi\) per unit area and unit angular cone of emission) is diminished by the convolution of the intrinsic photon beam sizes and the charged particle beam sizes:

\[B_{r}=\frac{\phi}{4\pi^{2}\Sigma_{x}\Sigma_{x^{\prime}}\Sigma_{y}\Sigma_{y^{ \prime}}}, \tag{10.1}\]

where we defined:

\[\begin{array}{l}\phi=\frac{dN_{ph}}{dtd\omega/\omega},\\ \\ \Sigma_{u}=\sqrt{\sigma_{u}^{2}+\sigma_{r}^{2}}\Sigma_{u^{\prime}}=\sqrt{\sigma _{u^{\prime}}^{2}+\sigma_{r^{\prime}}^{2}}\\ \\ \sigma_{u}=\sqrt{\varepsilon_{u}\beta_{u}+(D_{x}\sigma_{\delta})^{2}}\to\sqrt{ \varepsilon_{u}\beta_{u}}\\ \\ \sigma_{u^{\prime}}=\sqrt{\varepsilon_{u}\gamma_{u}+(D_{x}^{\prime}\sigma_{ \delta})^{2}}\to\sqrt{\varepsilon_{u}/\beta_{u}}\\ \\ \sigma_{r}=\sqrt{\varepsilon_{r}\beta_{r}}\\ \\ \sigma_{r^{\prime}}=\sqrt{\varepsilon_{r}/\beta_{r}}\end{array} \tag{10.2}\]\(\varepsilon_{u},\varepsilon_{r}\) are, respectively, the geometric emittance of the charged beam (\(u=x\), \(y\)) and the "intrinsic" geometric emittance of the light pulse (identical in both planes). Hereafter, "intrinsic" refers to the properties of an ideal monochromatic (\(\Delta\lambda\ll\lambda\)) light pulse emitted by a point-like source (\(\varepsilon_{u}\approx 0\)). We anticipate that \(\varepsilon_{r}=\sigma_{r}\sigma_{r^{\prime}}=\frac{\lambda}{4\pi}\) (the derivation is postponed). This is the case, for example, of an ideal laser beam, which features a symmetric Gaussian intensity distribution in the transverse planes (\(TEM_{00}\) mode), and \(\lambda\) is the central wavelength of the narrow band laser pulse.

In the above expressions, the C-S formalism is identically applied to the charged and the photon beam. In this case, the betatron function is simply defined as function of the light beam's emittance and spot size, and it actually depends from the type of magnetic insertion used for stimulating the emission of radiation. In most cases, and for maximizing the brilliance, the charged beam is assumed to emit radiation in correspondence of a waist (\(\alpha_{u}=0\)), in a dispersion-free region (\(D_{x}=0\), \(D_{x^{\prime}}=0\)).

If we describe a light pulse through a 6-D photon distribution, we can apply Liouville's theorem and find that brilliance is a conserved quantity in the absence of light absorption. If the phase space area is defined in terms of statistical parameters of the photon distribution (i.e., second order momenta or rms emittance), then the additional constraint of linear optics is required to make the brilliance a conserved quantity.

A photon beamline is an optical system aiming at manipulating and transporting the photon pulse to the experimental end-station. It includes a large variety of optical elements, as sketched in Fig. 10.1-left plot, to direct and focus radiation (mirrors), to filter it in wavelength (monochromators, such as gratings and crystals) or angle (mask, pinhole, slits). Unavoidable optical aberrations, micro-roughness, slope errors and thermal deformation of the optical surfaces, diffraction effects, and transmission efficiency of 0.1-1% are common challenges of real x-ray beamlines. They all contribute to a reduction of the brilliance _at the sample_.

But, why a high brilliance is important for experiments at light sources, and why should it be preserved down to the sample? The answer is in the need of high intensity at (sub-)micron-scale spot sizes at the sample for high spatial resolution, as well as at the entrance of a monochromator for high energy resolution.

If the source is not brilliant enough, a small spot size can still be produced by physically cutting the light pulse, but at the expense of intensity. Alternatively, the original spot size can be imaged to a smaller spot size ("demagnified"), but at the

Figure 10.1: Left: optical components of an x-ray beamline. Right: in red, a higher (in blue, smaller) brilliance pulse impinges on the optical elements with a smaller (larger) footprint

expense of large angular divergence. This would imply larger apertures, severe optical aberrations, larger size of the mirrors, hence more expensive specifications for the flatness of the mirrors' surface, and higher slope error.

We now understand that a higher brilliance at the source, i.e. smaller size and angular divergence, possibly accompanied by relatively narrow bandwidth, favors the preservation of the brilliance through the beamline. In short, a highly brilliant light source gives access to high energy resolution, small spot on the sample, accompanied by high intensity.

#### Optics Matching, Diffraction Limit

It is easy to show that for the realistic case of non-zero particle beam emittance, the brilliance in Eq. 10.1 is maximized by the electron beam transversely _matched_ to the intrinsic photon beam sizes, or \(\beta_{x,e}=\beta_{y,e}=\beta_{r}\). In this case:

\[\hat{B}_{r}=\frac{\phi}{4\pi^{2}(\varepsilon_{x,e}+\varepsilon_{r})( \varepsilon_{y,e}+\varepsilon_{r})}=\frac{2}{1+\kappa}\frac{\phi}{\lambda^{2} }\approxq\left\{\begin{array}{ll}\frac{2\phi}{\lambda^{2}},&\kappa\ll 1\\ \\ \frac{\phi}{\lambda^{2}},&\kappa\approx 1\end{array}\right. \tag{10.3}\]

The r.h.s. of Eq. 10.3 is evaluated for a charged beam satisfying the so-called _diffraction limit_ in the horizontal plane, \(\varepsilon_{x,e}=\varepsilon_{r}=\frac{\lambda}{4\pi}\). The vertical emittance is expressed as function of the horizontal emittance through the coupling factor \(\kappa\leq 1\) (see Eq. 5.53). Equation 10.3 highlights that, under proper matching, a flat beam (\(\varepsilon_{y}\ll\varepsilon_{x}\)) guarantees a brilliance as twice as that one of a round beam at full coupling (\(\varepsilon_{y}\approx\varepsilon_{x}\)). The absolute maximum of the brilliance is for an ideally zero-emittance--i.e., point-like--charged beam, and it amounts to \(4\phi/\lambda^{2}\).

"Diffraction limit" refers to the capability of a particle beam of emitting radiation characterized by a high degree of transverse _coherence_, essentially by virtue of the source's small size and divergence. Coherence can be intended in turn as a high degree of flatness of the wavefront of the far field. This translates, for example, into the capability of radiation of producing interference fringes once diffracted through properly sized holes. Of course, a particle beam satisfying the diffraction limit at \(\lambda\) will produce even more coherent radiation at any longer wavelength.

Optics matching of a particle beam in a light source aims at forcing the transverse charge distribution to betatron functions which maximize the brilliance from IDs. We consider a specific class of IDs, called _undulators_. An undulator is basically made of two opposite arrays of magnetic poles of alternated polarity. Electrons wiggle through the undulator and emit synchrotron radiation stimulated by Lorentz's force. A detailed treatment of undulator radiation will be given later on. Here, we anticipate that the transverse intensity distribution of radiation emitted by a monochromatic Gaussian charged beam in a long undulator (\(L_{u}\)) is also approximately Gaussian, and its intrinsic transverse size and angular divergence can be approximated to:\[\left\{\begin{array}{l}\sigma_{r}=\frac{\sqrt{\lambda L_{u}}}{2\pi}\\ \sigma_{r^{\prime}}=\sqrt{\frac{\lambda_{s}}{4\overline{L_{u}}}}\end{array} \right.\Rightarrow\left\{\begin{array}{l}\sigma_{r}\sigma_{r^{\prime}}= \varepsilon_{r}=\frac{\lambda}{4\pi}\\ \beta_{r}=\frac{\sigma_{r}}{\sigma_{r^{\prime}}}=\frac{L_{u}}{\pi}\\ \gamma_{r}=\frac{\sigma_{r^{\prime}}}{\sigma_{r}}=\frac{\pi}{L_{u}}\end{array}\right. \tag{10.4}\]

Since ID segments in storage rings are long \(\sim\)\(1-5\) m for practical reasons, \(\beta_{r}\approx 0.3-1.5\) m.

Unfortunately, optimal matching of the electron beam at the ID location is often prevented by other constraints to the optics design related to, e.g., minimum equilibrium emittance, chromaticity control, large dynamic aperture, etc. Consequently, the "geometric" component of \(B_{r}\), i.e., the denominator of Eq. 10.1, can assume a variety of values depending from the actual value of the coupling factor and the ratio of the charged and photon beam betatron function. For the horizontal beam emittance at the diffraction limit, it results:

\[\begin{array}{l}\chi:=\frac{B_{r}}{\phi/(4\pi^{2})}=\left[\sqrt{\varepsilon_ {x}\beta_{x}+\varepsilon_{r}\beta_{r}}\sqrt{\frac{\varepsilon_{x}}{\beta_{x} }+\frac{\varepsilon_{r}}{\beta_{r}}}\sqrt{\varepsilon_{y}\beta_{y}+ \varepsilon_{r}\beta_{r}}\sqrt{\frac{\varepsilon_{y}}{\beta_{y}}+\frac{ \varepsilon_{r}}{\beta_{r}}}\right]^{-1}=\\ \\ =\left[\sqrt{\varepsilon_{r}(\beta_{x}+\beta_{r})}\sqrt{\varepsilon_{r}(\frac {1}{\beta_{x}}+\frac{1}{\beta_{r}})}\sqrt{\varepsilon_{r}(\kappa\beta_{y}+ \beta_{r})}\sqrt{\varepsilon_{r}(\frac{\kappa}{\beta_{y}}+\frac{1}{\beta_{r}} )}\right]^{-1}=\\ \\ =\frac{1}{\varepsilon_{r}^{2}}\frac{\beta_{r}\sqrt{\beta_{x}\beta_{y}}}{( \beta_{x}+\beta_{r})\sqrt{(\kappa\beta_{y}+\beta_{r})(\beta_{y}+\kappa\beta_{ r})}}\end{array} \tag{10.5}\]

We discriminate four cases:

1. \(\kappa\approx 1\) (_full coupling_): \(\chi=\frac{1}{\varepsilon_{r}^{2}}\frac{\beta_{r}\sqrt{\beta_{x}\beta_{y}}}{( \beta_{x}+\beta_{r})(\beta_{y}+\beta_{r})}\) \(\rightarrow\) max. for \(\beta_{x}=\beta_{y}=\beta_{r}=L_{u}/\pi\), \(\hat{\chi}=\frac{1}{4\varepsilon_{r}^{2}}\)
2. \(\kappa\ll 1\) & \(\beta_{y}\approx\beta_{r}\) (_flat & matched beam_): \(\chi=\frac{1}{\varepsilon_{r}^{2}}\frac{\sqrt{\beta_{x}\beta_{r}}}{(\beta_{x} +\beta_{r})}\) \(\rightarrow\) max. for \(\beta_{x}=\beta_{y}=\beta_{r}=L_{u}/\pi\), \(\hat{\chi}=\frac{1}{2\varepsilon_{r}^{2}}\)
3. \(\kappa\ll 1\) & \(\kappa\beta_{y}\approx\beta_{r}\) (_flat & mismatched beam_): \(\chi=\frac{1}{\sqrt{2}\varepsilon_{r}^{2}}\frac{\sqrt{\beta_{x}\beta_{r}}}{( \beta_{x}+\beta_{r})}\) \(\rightarrow\) max. for \(\beta_{x}=\beta_{r}=L_{u}/\pi\), \(\hat{\chi}=\frac{1}{2\sqrt{2}\varepsilon_{r}^{2}}\)
4. \(\kappa\ll 1\) & \(\beta_{y}\approx\kappa\beta_{r}\) (_flat & over-matched beam_): \(\chi=\frac{1}{\sqrt{2}\varepsilon_{r}^{2}}\frac{\sqrt{\beta_{x}\beta_{r}}}{( \beta_{x}+\beta_{r})}\) \(\rightarrow\) max. for \(\beta_{x}=\beta_{r}=L_{u}/\pi\), \(\hat{\chi}=\frac{1}{2\sqrt{2}\varepsilon_{r}^{2}}\)

As expected, the brilliance is maximized by a diffraction-limited, flat beam light source, with both horizontal and vertical betatron function matched to the intrinsic betatron function of the photon beam at the ID (case 2). Matching means here that the charge and the photon beam distributions can be represented in the transverse phase space by omothetic ellipses, see Fig. 10.2.

#### Central Cone

Undulator radiation emitted on-axis is characterized by a higher brilliance and a larger degree of transverse coherence compared to far off-axis emission. For this reasons a beamline usually includes a "front-end" area, where optical elements limit the angular acceptance of the beamline in order to match the central angular cone of emission.

In case of undulator radiation, the _coherent flux_\(F_{coh}\) is estimated as the fraction of the total spectral flux \(F=\phi/(\Sigma_{x}\,\Sigma_{y})\) contained in the intrinsic central angular cone of emission:

\[\begin{split} F_{coh}&:=F\,\frac{\delta\theta_{x} \delta\theta_{y}}{\Sigma_{x^{\prime}}\Sigma_{y^{\prime}}}=F\,\frac{\sigma_{r^{ 2}}^{2}}{\sqrt{\frac{\varepsilon_{x}}{\beta_{x}}+\frac{\varepsilon_{r}}{\beta_ {r}}}\sqrt{\frac{\varepsilon_{y}}{\beta_{y}}+\frac{\varepsilon_{r}}{\beta_{r} }}}=F\,\frac{\sigma_{r^{\prime}}^{2}}{\frac{\varepsilon_{r}}{\beta_{r}}\sqrt{1 +\frac{\beta_{r}}{\beta_{x}}\frac{\varepsilon_{x}}{\varepsilon_{r}}\sqrt{1+ \frac{\beta_{r}}{\beta_{y}}\frac{\varepsilon_{y}}{\varepsilon_{r}}}}}=\\ &=\frac{F}{\sqrt{\Big{(}1+\frac{\delta_{r}}{\beta_{x}}\frac{ \varepsilon_{x}}{\varepsilon_{r}}\Big{)}\Big{(}1+\frac{\delta_{r}}{\beta_{y} }\frac{\varepsilon_{y}}{\varepsilon_{r}}\Big{)}}}\rightarrow\,\frac{F}{\sqrt {2}}\end{split} \tag{10.6}\]

The limit is for the coherent flux maximized by the same conditions of maximum brilliance, i.e., by a diffraction-limited, flat (\(\varepsilon_{y}\ll\varepsilon_{x}\)), and matched particle beam. \(F_{coh}\,\rightarrow\,F\) when \(\varepsilon_{x}\), \(\varepsilon_{y}\,\rightarrow\,0\), that is, radiation emitted by a point-like source is 100% transversely coherent.

### Coherence

Coherence of light generally refers to the capability of predicting the e.m. field at any point \(P_{2}\) of the radiation pattern from the knowledge of the field at a nearby point \(P_{1}\). This implies a well-defined correlation of the e.m. field vectors, or phase relation, which is in turn at the origin of the interference pattern of electric fields of same polarization in the well-known Young's double slit experiment, see Fig. 10.3.

Figure 10.2: Superposition of intrinsic photon beam (yellow) and electron beam (green) phase space area, and effective radiation envelope (blue). On the left, the electron beam is mismatched, and its emittance is larger than the intrinsic photon beam emittance. On the right, the electron beam is matched to the light pulse, and its emittance is close to the diffraction limitIn such scheme, the total field intensity (normalized to \(2Z_{0}\)) recorded at the observation point P, and produced by the linear superposition \(U(\vec{r}_{1},\vec{r}_{2},\)\(t_{1},\)\(t_{2})\) of two e.m. plane waves, is:

\[I_{P}=\langle UU^{*}\rangle=\left\langle\left[E_{1}e^{i(\vec{k} \vec{r}_{1}-\omega t_{1})}+E_{2}e^{i(\vec{k}\vec{r}_{2}-\omega t_{2})}\right] \left[E_{1}e^{-i(\vec{k}\vec{r}_{1}-\omega t_{1})}+E_{2}e^{-i(\vec{k}\vec{r}_{2 }-\omega t_{2})}\right]\right\rangle=\] \[=\langle|E_{1}|^{2}\rangle+\langle|E_{2}|^{2}\rangle+\langle E_{ 1}E_{2}\rangle e^{i\left(\vec{k}\Delta\vec{r}-\omega\Delta t\right)}+\langle E _{2}E_{1}\rangle e^{-i\left(\vec{k}\Delta\vec{r}-\omega\Delta t\right)}=\] \[=\langle|E_{1}|^{2}\rangle+\langle|E_{2}|^{2}\rangle+2\langle E_ {1}E_{2}\rangle\cos\left(\vec{k}\Delta\vec{r}-\omega\Delta t\right)=\] \[=I_{1}+I_{2}+2\sqrt{I_{1}I_{2}}\cos\left(\vec{k}\Delta\vec{r}- \omega\Delta t\right) \tag{10.7}\]

For simplicity, we assumed monochromatic waves of same wave number \(k\). The two waves have amplitudes \(E_{1}\), \(E_{2}\). \(\Delta\vec{r}=\vec{r}_{1}-\vec{r}_{2}\), \(\Delta t=t_{1}-t_{2}\) are, respectively, the path length difference of the two waves to reach P, and the relative delay with which they were generated. This is usually zero in Young's experiment, and thereby \(I_{p}=I_{p}(\Delta\vec{r})\). \(\langle...\rangle\) denotes an ensemble statistical average. For non-stationary states, such as light pulses, the average is calculated over many shots. In case of stationary states, such as plane parallel waves (\(\vec{r}=z\)), the ensemble average can be replaced with a time average, \(\langle...\rangle=\frac{1}{T}\int...dt\).

Equation 10.7 shows that the maximum total intensity is obtained for relative phases multiple of \(2\pi\), i.e., the two waves are emitted "in phase". We also see that the relative phase--and therefore coherence--is, in general, a spatio-temporal quantity. By posing conditions to the smallness of either the pure spatial or the pure temporal relative phase, a distinction between longitudinal and transverse coherence can be made, although they remain different aspects of the same physical property.

Figure 10.3: An incoherent monochromatic light source is collimated through a pinhole. The size is assumed to be such that the emerging light is now (partially) coherent. As a result of diffraction at two downstream apertures, an interference pattern appears on the screen after a large number of pulses is collected

Longitudinal coherence is often associated--but not limited--to a narrow spectral bandwidth. For a finite pulse duration, the narrowest bandwidth is determined by the Fourier limit, i.e. \(\Delta t\Delta v\geq 1/2\). Transverse coherence is usually related to the capability of collimating the light pulse down to small spatial size and angular divergence. Hence, the pulse would ideally approach the transverse emittance of a laser beam.

Since an e.m. wave passing through a suitably small pinhole can produce a diffraction pattern analogue to that in Fig. 10.3, we infer that diffraction can be intended as a special case of interference of a wave with itself. Diffraction and interference are diagnostic tools to quantify the degree of coherence of a light source.

#### Correlation Functions

In the "classical" picture of far field e.m. radiation, the Glauber's normalized first order correlation function of the electric field is:

\[g_{1}(\vec{r}_{1},\,t_{1};\,\vec{r}_{2},\,t_{2})=\frac{\langle E^{*}(\vec{r}_{1 },\,t_{1})E(\vec{r}_{2},\,t_{2})\rangle}{\sqrt{\langle|E(\vec{r}_{1},\,t_{1})| ^{2}\rangle\langle|E(\vec{r}_{2},\,t_{2})|^{2}\rangle}} \tag{10.8}\]

In case of stationary states the result does not depend from \(t_{1}\), but only from the delay \(\tau=t_{1}-t_{2}\):

\[g_{1}(\tau)=\frac{\langle E^{*}(t)E(t+\tau)\rangle}{\langle|E(t)|^{2}\rangle} \leq\frac{\sqrt{\langle|E(t)|^{2}\rangle\langle|E(t+\tau)|^{2}\rangle}}{\langle |E(t)|^{2}\rangle}=g_{1}(0)=1 \tag{10.9}\]

The result \(g_{1}(\tau)\leq g_{1}(0)\) is obtained by means of the Cauchy-Schwarz inequality \(|\langle\vec{u},\,\vec{v}\rangle|^{2}\leq\langle\vec{u},\,\vec{u}\rangle\cdot \langle\vec{v},\,\vec{v}\rangle\). Then, by virtue of the oscillating behaviour of the electric field amplitude with time, \(\langle|E(t+\tau)|^{2}\rangle=\langle|E(t)|^{2}\rangle\forall\tau\), and eventually \(g_{1}(0)=1\).

In a Michelson interferometer, the radiation pulse is split in two components. A time delay is introduced, and the two pulses are eventually recombined (see Fig. 1.1). The intensity of the resulting field at any position on the screen where the pulses impinge, can be measured as a function of the time delay. The _visibility_ of the resulting interference pattern is:

\[v=\frac{I_{max}-I_{min}}{I_{max}+I_{min}} \tag{10.10}\]

where \(I_{max}\), \(I_{min}\) are, respectively, the maximum and minimum intensity of the interference pattern. It can be shown that for monochromatic waves of same polarization:

\[v(\lambda)=\frac{2\sqrt{I_{1}I_{2}}}{I_{1}+I_{2}}|g_{1}(\vec{r}_{1},\,t_{1};\, \vec{r}_{2},\,t_{2})| \tag{10.11}\]

where \(I_{1}\), \(I_{2}\) are the total intensities of the plane waves. It is then apparent from Eqs. 10.11 and 10.9 that a fully coherent light pulse (e.g., a single frequency emission, such as an ideal laser or a monochromatic plane parallel wave) has \(v=|g_{1}^{SF}(\tau)|=|g_{1}(0)|=1\).

As a matter of fact, monochromatic light is characterized by \(g_{1}^{SF}(\tau)=e^{-i\omega_{0}\tau}\), while for Gaussian chaotic light, \(g_{1}^{GC}(\tau)=e^{-i\omega_{0}\tau-\frac{\pi}{2}\left(\frac{\tau}{\tau_{c}} \right)^{2}}\). The "coherence time" of light \(\tau_{c}\), treated later on, is generally inversely proportional to the spectral bandwidth. We have the following limits:

\[\left\{\begin{array}{ll}\lim_{\tau\to\infty}g_{1}^{GC}(\tau)=0,\\ \lim_{\tau_{c}\to\infty}g_{1}^{GC}(\tau)=g_{1}^{SF}(\tau)\forall\tau\end{array} \right.\left\{\begin{array}{ll}\lim_{\tau\to\infty}|g_{1}^{GC}(\tau)|=0\\ \lim_{\tau_{c}\to\infty}|g_{1}^{GC}(\tau)|=1\end{array}\right. \tag{10.12}\]

The visibility ranges from 0 for incoherent light pulses, to 1 for fully coherent pulses. Anything in between is described as "partially coherent".

Correlation functions of the electric field can be defined up to an arbitrarily high order. For example, the normalized second order correlation function is:

\[g_{2}(\vec{r}_{1},t_{1};\vec{r}_{2},t_{2})=\frac{(E^{*}(\vec{r}_{1},t_{1})E^{* }(\vec{r}_{2},t_{2})E(\vec{r}_{1},t_{1})E(\vec{r}_{2},t_{2}))}{\langle|E(\vec{ r}_{1},t_{1})|^{2}\rangle\langle|E(\vec{r}_{2},t_{2})|^{2}\rangle} \tag{10.13}\]

In the "classical" picture of electric fields, we can re-order them to express \(g_{2}\) in terms of intensities. A plane parallel wave in a stationary state will have:

\[g_{2}(\tau)=\frac{\langle I(t)I(t+\tau)\rangle}{\langle I(t)\rangle^{2}} \Rightarrow g_{2}(0)=\frac{\langle I(t)^{2}\rangle}{\langle I(t)\rangle^{2}}\geq 1 \tag{10.14}\]

and the result on the r.h.s. is again by virtue of the Cauchy-Schwarz inequality (see Eq. 10.9 with \(|v|=1\)). It then emerges that a fully coherent light pulse has \(g_{2}(\tau)=g_{2}(0)=1\).

It can be shown that chaotic light of all kinds has \(g_{2}(\tau)=1+|g_{1}(\tau)|^{2}\). Hence, we have the following limits:

\[\left\{\begin{array}{ll}\lim_{\tau\to\infty}g_{2}^{GC}(\tau)=1,\\ \\ \lim_{\tau_{c}\to\infty}g_{2}^{GC}(\tau)=2\end{array}\right. \tag{10.15}\]

#### Transverse Coherence

At first order in the sense of Glauber's correlation functions, the _transverse coherence length_\(L_{c,\perp}\) is the width (in each transverse plane) of the function \(g_{1}\) in Eq. 10.8, evaluated as function of the lateral distance \(\vec{r_{1}}-\vec{r_{2}}\) for a constant \(\tau\), or \(g_{1}(\vec{r}_{1},\vec{r}_{2})\). The "degree of transverse coherence" is [2]:

\[\xi_{c}=\frac{\int\int|g_{1}(\vec{r}_{1},\vec{r}_{2})|^{2}\langle I(\vec{r}_{ 1})\rangle\langle I(\vec{r}_{2})\rangle d\vec{r}_{1}d\vec{r}_{2}}{\left[\int \langle I(\vec{r}_{1})\rangle d\vec{r}_{1}\right]^{2}}\leq 1 \tag{10.16}\]

In a more naive picture, we can think of \(\xi_{c}\to 1\) like if there exist a maximum angular divergence of the light pulse which preserves a phase relation of the e.m.

field evaluated at two transverse locations of the wavefront. Such an angle is said _coherence angle_, and it is illustrated in Fig. 10.4-left plot.

In order to estimate the coherence angle, let us consider the path length difference of two rays emitted, respectively, on-axis and at the edge of the source. The two rays are emitted at the same time \(t=0\). If the observer is on-axis and at a distance \(l_{1}\gg d\) from the source, with \(d\) the source's full transverse size and such that \(\theta\ll 1\), we have:

\[\Delta l=l_{2}-l_{1}=l_{2}(1-\cos\theta)\approxeq\frac{l_{2}\theta^{2}}{2} \approxeq\frac{d\theta}{4} \tag{10.17}\]

The distance over which the two rays, initially in phase (\(\Delta\phi=0\)), become out of phase (\(\Delta\phi=\pi\)), defines the coherence angle for the specified wavelength of emission:

\[\Delta\phi=\omega\Delta t=2\pi\,\nu\,\frac{\Delta l}{c}\equiv\pi\,\Rightarrow \,\Delta l\approxeq\frac{\lambda}{2}\Rightarrow\theta_{c}=\frac{2\lambda}{d} \tag{10.18}\]

Such an estimate is consistent with, and made more accurate by, the Uncertainty Theorem of Fourier theory [3]. This states that the product of the effective widths of a function and of its Fourier transform (FT)--which thereby constitute a Fourier pair--must be equal or larger than 1/2. We apply the Theorem to the lateral spatial coordinate \(x\) of a photon. Since FT(\(x\)) = 1/\(\lambda_{x}\), and by recalling \(\frac{1}{\lambda_{x}}=\frac{k_{x}}{2\pi}=\frac{p_{x}}{h}\), we find:

\[\Delta x\,\Delta(\lambda_{x}^{-1})=\Delta x\,\frac{\Delta\lambda_{x}}{\lambda_ {x}^{2}}=\frac{\Delta x\,\Delta k_{x}}{2\pi}\geq\frac{1}{2}\Rightarrow\Delta x \,\Delta\,p_{x}\geq\frac{h}{2} \tag{10.19}\]

Owing to the fact that the angular divergence of radiation relates to the transverse momentum of the e.m. wave, which is assumed here to be monochromatic at the wavelength \(\lambda\), we find the lower limit of the product of the spatial and angular divergence of the light pulse:

\[\Delta\theta=\frac{\Delta p_{x}}{p_{z}}\approx\frac{\Delta p_{x}}{p}=\frac{ \Delta p_{x}}{(h\nu/c)}=\frac{\Delta p_{x}}{hk}\Rightarrow\Delta x\,\Delta \theta\hbar k\geq\frac{h}{2}\Rightarrow\left\{\begin{array}{l}\sigma_{\theta, c}=\frac{\lambda}{4\pi}\,\frac{1}{\sigma_{x}}\\ \\ \theta_{c}=0.44\frac{\lambda}{\Delta x}\end{array}\right. \tag{10.20}\]

We used \(\Delta x=\sqrt{2\pi}\,\sigma_{x}\) and \(\Delta\theta=\sqrt{2\pi}\,\sigma_{\theta,c}\) for the standard deviations, and \(\Delta\theta=2\sqrt{2\ln 2\theta_{c}}\) for fwhm quantities. One can notice that the same result is obtained

Figure 10.4: Left: two rays emitted by a particle beam travel different path lengths up to the observer. If the path is long enough, they reach the out-of-phase condition. Right: a non-zero source size generates light diffraction at a downstream pinhole, as function of the angular acceptance determined by the pinhole size and the source-pinhole distance

by evaluating the minimum transverse rms phase space area occupied by a photon according to Heisenberg's Uncertainty Principle, i.e., \(\sigma_{x}\sigma_{p_{x}}\geq\hbar/2\). The transverse coherence length is just \(L_{c,\perp}=l_{1}\theta_{c}\) (either in rms or fwhm sense).

The estimate of the coherence angle in Eq. 10.18 can be retrieved equivalently from the consideration that, according to geometric optics, the first minimum of the diffraction intensity pattern generated by a point-like source when the emitted light passes through a circular pinhole of diameter \(2R\), is in correspondence of the angle \(\theta=\lambda/(2R)\), see Fig. 10.4-right plot. This condition is smeared if the source has a non-zero transverse half-size \(\Delta x\), because of the angular divergence determined by the source-pinhole distance \(L\), \(\Delta\theta=\Delta x/L\). In other words, the smaller the source size is, the more apparent the diffraction effect becomes at any given \(\lambda\), and ideally at any \(\lambda\) for a zero-emittance source. We will observe diffraction at \(\lambda\) as long as \(\Delta\theta=\frac{\Delta x}{L}\leq\frac{\lambda}{R}\). That is, as long as the angular acceptance determined by the pinhole is smaller than the coherence angle:

\[\tfrac{R}{L}\leq\tfrac{\lambda}{\Delta x}\equiv\theta_{c} \tag{10.21}\]

In conclusion, two definitions of \(L_{c,\perp}\) were given. The former one, related to Eq. 10.16, is often used to characterize a collected light beam by measuring the spatial visibility of interfering split pulses. The latter definition of \(L_{c,\perp}\), related to the coherence angle in Eqs. 10.20 or 10.21, can be used to estimate the angular acceptance of a beamline in order to either exploit coherence properties of the central cone of radiation or, on the contrary, to avoid diffraction effects due to emerging transverse coherence.

As a matter of fact, Eq. 10.21 shows that transversely incoherent radiation can be made coherent (in a specific wavelength range) by physically selecting a small transverse portion of the incoming light beam, e.g., with a pinhole. Doing so, the downstream beamline will receive light from an "effective" source of smaller size than the actual one, thus closer to or below the diffraction limit, though at the expense of a reduced intensity.

#### Longitudinal Coherence

In analogy to first order transverse coherence, the _coherence time_\(\tau_{c}\) is the width of the function \(g_{1}\) in Eq. 10.8, evaluated as function of the relative delay \(\tau\) for fixed points \(\vec{r}_{1}\), \(\vec{r}_{2}\) of the collected intensity pattern, or \(g_{1}(\tau)\). The rms value of \(\tau_{c}\) for an arbitrary intensity distribution is [2]:

\[\tau_{c,rms}=\int_{-\infty}^{\infty}|g_{1}(\tau)|^{2}d\tau \tag{10.22}\]

where for a Gaussian function \(\tau_{c,rms}\approxeq 0.85\tau_{c,hwhm}\). The longitudinal coherence length is \(L_{c,\parallel}=c\tau_{c}\). Equation 10.22 is commonly adopted to retrieve \(\tau_{c}\) from a set of measurements of the radiation spectral intensity pattern imaged on a screen. In analogy to Eq. 10.16, it can be used to characterize a collected light beam.

A more intuitive connection of \(\tau_{c}\) to the spectral bandwidth of a light pulse can be found by looking the geometry in Fig. 10.4-right plot. Since the first minimum of the interference fringes is at \(\theta=\lambda/(2R)\) for fully coherent radiation, we require that the spectral bandwidth \(\Delta\lambda\) of the actual pulses be small enough not to perturb the pattern:

\[\Delta\theta=\tfrac{\Delta\lambda}{2R}\ll\theta\Rightarrow\tfrac{\Delta\lambda }{\lambda}=\tfrac{\Delta v}{v}\ll 1 \tag{10.23}\]

Given the narrow bandwidth of partially coherent radiation, we wonder what is the corresponding fraction of the pulse duration over which the phase relation of electric fields of slightly different frequencies is still preserved. The two frequencies at spectral distance \(\Delta\omega\) travelling for a time interval \(\Delta t\) accumulate a phase difference \(\Delta\phi=\Delta\omega\Delta t\). The time the fields associated to the two frequencies take to become opposite in phase is, by definition, the coherence time:

\[\Delta\phi=\Delta\omega\Delta t=\Delta\omega\tau_{c}\equiv\pi\Rightarrow L_{c, \parallel}=c\tau_{c}=\tfrac{c}{2\Delta\omega}=\tfrac{\lambda^{2}}{2\Delta\lambda} \tag{10.24}\]

As already for the transverse coherence length in Eq. 10.20, the longitudinal coherence time is equivalently retrieved from the Fourier's Uncertainty Theorem, \(\Delta\nu\Delta t\geq 1/2\), with rms quantities \(\Delta\nu=\sqrt{2\pi}\sigma_{v}\), \(\Delta t=\sqrt{2\pi}\sigma_{t}\), or from Heisenberg's Uncertainty Principle applied to the longitudinal rms phase space area, \(\sigma_{E}\sigma_{t}\geq\hbar/2\)[3].

Equation 10.24 implies that a single frequency has infinite coherence time. Viceversa, if \(\tau_{c}\) is smaller than the finite duration of a particle beam source, and if the beam source has an internal structure that allows coherent (i.e., in phase) emission, we may imagine the beam emitting "spikes" of radiation, each spike being emitted independently from (i.e., not correlated to) the others, but individually coherent. This is the case of a free-electron laser in regime of so-called Self-Amplified Spontaneous Emission (SASE), as discussed later on.

So as a pinhole is used at beamlines to obtain some higher degree of transverse coherence, a monochromator is often adopted to further select a small spectral portion of the incoming broad-band radiation. Since the selection is physical in the dispersive plane of the monochromator, a larger coherence time, i.e. a narrower bandwidth, is obtained at the expense of reduced intensity.

#### Discussion: Degeneracy Parameter

Equations 10.18 and 10.24 for the coherence angle and the coherence time suggest that, for any given transverse and longitudinal size of the source, and therefore of the emitted light pulse, it is more and more difficult to obtain a high degree of coherence at shorter and shorter wavelengths. We corroborate this observation by exploiting the definition of brilliance and its dependence from the central wavelength of emission.

Let us consider a certain number of photons \(N_{ph}\) in a light pulse of central wavelength \(\lambda\) and spectral bandwidth \(\Delta\lambda\). For simplicity, we assume the photonsuniformly distributed in a pulse duration \(\Delta t\). The number of photons contained in one longitudinal coherence length is:

\[N_{c,\parallel}=N_{ph}\frac{L_{c,\parallel}}{c\Delta t}=\frac{I_{r}}{c}\frac{ \lambda^{2}}{2\Delta\lambda} \tag{10.25}\]

and \(I_{r}\) is the total intensity.

By substituting \(I_{r}\) from Eq. 10.25 in the definition of brilliance for a diffraction limited, matched, flat particle beam (Eq. 10.3), we find:

\[\hat{B}_{r}=\frac{2I_{r}}{\lambda^{2}(\Delta\lambda/\lambda)}=\frac{4}{ \lambda^{2}(\Delta\lambda/\lambda)}\frac{cN_{c,\parallel}(\Delta\lambda/ \lambda)}{\lambda}=\frac{4cN_{c}}{\lambda^{3}} \tag{10.26}\]

We introduced the notation \(N_{c,\parallel}=N_{c}\) in the r.h.s. of Eq. 10.26 to stress out that, since the brilliance is estimated for a source at the diffraction limit, the emitted photons are by definition transversely coherent. Hence, \(N_{c}\) is the number of photons transversely _and_ longitudinally coherent, and \(\hat{B}_{r}\) is the total 6-D photon density emitted by a diffraction-limited source, evaluated in a "coherence volume" \(\lambda^{3}\).

Equation 10.26 shows that, for any given brilliance of radiation centered at \(\lambda_{0}\), the number of fully coherent photons becomes favorably larger at any \(\lambda\geq\lambda_{0}\). The number of fully coherent photons \(N_{c}\) or, more precisely, the number of photons per coherence phase space volume and coherence time--i.e., the number of photons per "coherent mode"--is called _degeneracy parameter_, \(\delta_{w}\). If \(D\leq 1\) is the light source duty cycle (\(D=1\) for \(\delta_{w}\) of a single pulse, and \(B_{r}\) is the peak brilliance), it follows from Eq. 10.26:

\[\delta_{w}(\lambda)\equiv N_{c}(\lambda)=\frac{DB_{r}\lambda^{3}}{4c}=8.34 \cdot 10^{-25}D\lambda^{3}[\text{\AA}]B_{r}[\frac{\#ph}{mm^{2}mrad^{2}sec0.1\% bw}] \tag{10.27}\]

Since photons are bosons, we can have \(\delta_{w}\geq 1\). However, it is only with the advent of undulators in most recent storage ring light sources (\(B_{r}(5\text{\AA})\sim 10^{22}\) in conventional units) and, especially, of free-electron lasers (\(B_{r}(1\text{\AA})\sim 10^{32}\) in conventional units), that values \(\delta_{w}\approx 1\) and \(\delta_{w}\gg 1\), respectively, have been achieved in the x-ray region [1].

#### Intensity Enhancement

Equations 10.20 and 10.24 suggest that a charge distribution compact enough in the 3-D space tends to radiate coherently. Namely, the instantaneous emission of a distribution which behaves as a point-like charge--say, a "macro-particle"--tends to be transversely and longitudinally coherent because all photons are emitted with approximately the same spatio-temporal phase.

To be more rigorous, however, the degree of coherence of radiation emitted at a certain wavelength is established by the ratio of beam size and coherence length, either transverse or longitudinal. Thus, compactness _per se_', defined with respect to the wavelength scale, is not a sufficient condition for coherent emission. We show below that, instead, it is so for enhancement of the radiation intensity, in proportion to the number of radiating beam particles.

We consider a bunch of \(N_{b}\gg 1\) particles spatially distributed in space with vectors \(\vec{r}_{j}\), \(j=1\),..., \(N_{b}\), emitting radiation at the central frequency \(\omega\). To simplify the math, we assume pure monochromatic radiation, \(\Delta\omega\ll\omega\). The emission is said _incoherent_ if the charges assume a random phase distribution. In this case the total radiated field (intensity) is the linear superposition of the field (linear sum of intensity) of single particle emissions: \(I_{tot}(\omega)=N_{b}\,I_{r}(\omega)\).

The amplitude of the electric field radiated by the \(j\)-th particle is \(E_{0j}\). The electric field in the far field approximation is described as a travelling wave with wave-vector \(\vec{k}_{j}=\hat{n}_{j}\frac{\omega}{c}\). The total field vector is:

\[\vec{E}_{tot}(\omega)=\sum_{j=1}^{N_{b}}\vec{E}_{j}e^{i(\omega t-\vec{k}_{j} \vec{r}_{j})} \tag{10.28}\]

The radiation intensity, or number of photons per unit of time, is proportional to the total e.m. energy and it can therefore be calculated by means of the Poynting vector (see Eqs. 7.6 and 7.7):

\[I_{tot}(\omega)=\frac{dN_{ph}}{dt}=\frac{1}{\hbar\omega}\frac{dE(\omega)}{dt} =\frac{1}{\hbar\omega}\int_{\Omega}|\vec{S}\cdot\hat{n}|\,R^{2}d\Omega=\left( \frac{ce_{0}}{2\hbar\omega}\right)\int_{\Omega}|\vec{E}_{tot}(\omega)|^{2}R^{ 2}d\Omega \tag{10.29}\]

We now expand the absolute value of the total electric field. The frequency dependence is collapsed into a normalized amplitude \(E_{j}(\omega)=E_{0j}e^{i\omega t}\). Next, we assume particles' spacing in all directions be much smaller than the (central) wavelength of emission, or \(|\Delta\vec{r}|\ll\lambda\):

\[|\vec{E}_{tot}(\omega)|^{2}=|\sum_{j=1}^{N_{b}}E_{j}(\omega)e^{-ik \hat{n}_{j}\vec{r}_{j}}|^{2}=\sum_{j=1}^{N_{b}}E_{j}(\omega)e^{-ik\hat{n}\vec{r }_{j}}\sum_{m=1}^{N_{b}}E_{m}^{*}(\omega)e^{ik\hat{n}\vec{r}_{m}}=\] \[=\sum_{j=1}^{N_{b}}|E_{j}(\omega)|^{2}+\sum_{j\neq m}^{N_{b}}E_{j }(\omega)E_{j}^{*}(\omega)e^{i\pi\frac{\omega(\vec{r}_{m}-\vec{r}_{j})}{ \lambda}}\approxeq\] \[\approxeq\sum_{j=1}^{N_{b}}|E_{j}|^{2}+\sum_{j\neq m}^{N_{b}}E_{j }E_{j}^{*}=N_{b}|E_{j}|^{2}+N_{b}(N_{b}-1)|E_{j}|^{2}=N_{b}^{2}|E_{j}|^{2} \tag{10.30}\]

A more accurate analysis should consider a 3-D charge distribution function, whose Fourier transform would contribute through "form factors" to the coherent intensity. In the simplified assumption of no correlation between longitudinal and transverse distribution functions, and by substituting Eq. 10.30 into Eq. 10.29, we end up with [7]:

\[\begin{array}{l}I_{tot}(\omega)=\frac{|\vec{E}_{tot}(\omega)|^{2}}{2Z_{0}} \approxeq N_{b}I_{r}(\omega)+N_{b}(N_{b}-1)I_{r}(\omega)|f_{\perp}(\omega)|^ {2}|f_{\parallel}(\omega)|^{2}\approxeq\\ \\ \approxeq N_{b}I_{r}(\omega)\left[1+N_{b}|f_{\perp}(\omega)|^{2}|f_{\parallel}( \omega)|^{2}\right]\end{array} \tag{10.31}\]

For example, the form factor of a Gaussian distribution function is \(|f(\omega_{u})|^{2}=e^{-k_{u}^{2}\sigma_{u}^{2}/2}\), where \(\omega_{u}=ck_{u}\) and \(u=x\), \(y\), \(z\). \(|f(\omega_{u})|\to 0\) when \(\sigma_{u}\gg\lambda_{u}\), and \(|f(\omega_{u})|\to 1\) when \(\sigma_{u}\ll\lambda_{u}\). In the former case, the intensity of the emitted radiation is just the single particle emission times the number of emitters ("incoherent emission"). On the contrary, if the beam charge is distributed over spatial scales smaller than the wavelength of interest, and for large number of emitters, the total intensity of emitted radiation at that wavelength goes like the intensity of the single particle emission times the _square_ of the number of emitters ("coherent emission" or, more precisely, "intensity enhancement").

The latter effect is exploited, for example, in electron linacs, where the bunch is time-compressed and squeezed to small spot sizes. It is then sent through a small aperture or a metallic foil to generate coherent diffraction or transition radiation, respectively. Bunch shortening is sometimes approached in storage rings, where a nonlinear dependence of the intensity of dipole synchrotron radiation from the bunch charge is observed. In both cases, coherent emission can be obtained at IR-THz frequencies. EUV and x-ray free-electron lasers exploit the same physics, but in order to have large intensity at shorter wavelengths, electron bunches are internally micro- or nano-bunched. The electron clusters are separated by the central wavelength of emission, each cluster being much shorter than that.

### 10.3 Undulator Spontaneous Radiation

In its simplest configuration, an undulator [4, 5] is made of two opposite periodic arrays of magnetic poles of alternated polarity, through which the charged particles, electrons hereafter, wiggle by virtue of the Lorentz's force--see Fig. 10.5. The magnetic poles can be either permanent magnets (e.g., rare earths) or electro-magnets, or hybrid elements. Depending if the electrons travel in a vacuum chamber internal to the undulator, or if the electrons and the undulator are both inside a tank under vacuum, the ID is said "out-of-vacuum" or "in-vacuum", respectively. "Planar" undulators have planar geometry of the arrays. Their relative shift in the longitudinal direction determines the plane of oscillation of the electrons, thus control of the electric field polarization (linearly horizontal, vertical, inclined, circular, etc.).

One of the most common designs is the out-of-vacuum variable gap, planar, variably polarized undulator (APPLE-II). More complex designs have up to 6 degrees of freedom (APPLE-X, with four independent arrays and variable gap both in the horizontal and in the vertical plane). Cryogenic and superconducting undulators have been developed with the main purpose of increasing the magnetic field at shorter undulator periods. Typical dimensions of an undulator are 1-5 m total length, 1-10 cm magnetic period, 3-20 mm gap. A single segment can be allocated in a storage ring straight section. Tens' of such segments in a series constitute the undulator line of a free-electron laser.

#### Central Wavelength

The central wavelength of undulator emission is determined by the constructive interference of field emitted at homologous points of consecutive periods. Being the _undulator spontaneous radiation_ (spontaneous means here incoherent) a special case of synchrotron radiation, the radiated intensity is proportional to the magnetic field, and the central cone at each _local_ source point has a characteristic aperture \(\sim\)\(1/\gamma\). However, this implies that the overlap of consecutive wavefronts, especially at high beam energies, is only possible if the electrons wiggle around the magnetic axis with a relatively small oscillation amplitude. This imposes in turn a moderate magnetic field.

Let us consider for simplicity a planar undulator for horizontally polarized light, see Fig. 10.5-top plot. The vertical magnetic field is ideally a sinusoidal function

Figure 10.5: Top view of electron trajectory in a planar linearly polarized insertion device (top), and APPLE-II type undulator configured for diverse polarization statesof the longitudinal coordinate \(z=v_{z}t\) along the ID: \(B_{y}=B_{0y}\sin\left(\frac{2\pi v_{z}t}{\lambda_{u}}\right)\), with \(v_{z}\approx const\) the particle's longitudinal velocity, and \(\lambda_{u}\) the undulator period. The particle's horizontal velocity is calculated from the Lorentz's force:

\[F_{x}=\gamma m_{0}a_{x}=m_{0}\gamma\,\frac{dv_{x}}{dt}=-ev_{z}B_{y}=-ev_{z}B_{0 y}\sin\left(\frac{2\pi v_{z}t}{\lambda_{u}}\right);\]

\[\int dv_{x}=-\left(\frac{eB_{0y}}{\gamma m_{0}}\right)\int dz^{\prime}\sin \left(\frac{2\pi z^{\prime}}{\lambda_{u}}\right)=\frac{1}{\gamma}\left(\frac{ eB_{0y}\lambda_{u}}{2\pi m_{0}}\right)\cos\left(\frac{2\pi z}{\lambda_{u}}\right); \tag{10.32}\]

\[\beta_{x}=\frac{v_{x}}{c}=\frac{1}{\gamma}\left(\frac{eB_{0y}\lambda_{u}}{2 \pi m_{0}c}\right)\cos\left(\frac{2\pi z}{\lambda_{u}}\right)\equiv\frac{K}{ \gamma}\cos\left(\frac{2\pi z}{\lambda_{u}}\right)\]

where we introduced the _undulator parameter_\(K=\left(\frac{eB_{0y}\lambda_{u}}{2\pi m_{e}c}\right)=0.934\,B_{0y}[T]\lambda_{u} [\mathrm{cm}]\) in practical units, and for the electron rest mass.

The generic particle's longitudinal velocity is:

\[v_{z}=c\left(\beta^{2}-\beta_{x}^{2}-\beta_{y}^{2}\right)^{1/2} \approxeq c\left[1-\frac{1}{\gamma^{2}}-\left(\frac{K}{\gamma}\right)^{2}\cos^ {2}(k_{u}z)\right]^{1/2}= \tag{10.33}\] \[=c\left[1-\frac{1}{\gamma^{2}}-\left(\frac{K}{\gamma}\right)^{2} \frac{1+\cos(2k_{u}z)}{2}\right]^{1/2}=c\left[1-\frac{1}{\gamma^{2}}\left(1+ \frac{K^{2}}{2}\right)-\frac{K^{2}}{2\gamma^{2}}\cos(2k_{u}z)\right]^{1/2}\approxeq\] \[\approxeq c\left[1-\frac{1}{2\gamma^{2}}\left(1+\frac{K^{2}}{2} \right)-\frac{K^{2}}{4\gamma^{2}}\cos(2k_{u}z)\right]=\langle v_{z}\rangle-c \frac{K^{2}}{4\gamma^{2}}\cos(2k_{u}z)\]

and \(\langle v_{z}\rangle=const.\) is the average velocity evaluated along one undulator period. It can be shown that for a helically polarized undulator of same peak field, \(K\rightarrow\sqrt{2}\,K\) and \(v_{z}=\langle v_{z}\rangle=const\) due to purely geometric considerations.

If \(K<1\), the radiation cones emitted along the trajectory overlap, so allowing constructive interference of the wavefronts to build up. The condition of constructive interference is illustrated in red in Fig. 10.5, where light is assumed to be emitted at a generic angle \(\theta\) from the on-axis direction. During the time \(\Delta t_{1}=\frac{\lambda_{u}\cos\theta}{c}\) the wavefront emitted at A reaches B, the radiating electron, initially in A, reaches B in an interval \(\Delta t_{2}=\frac{\lambda_{u}}{v_{z}}\). Constructive interference happens if the path length difference \(c(\Delta t_{2}-\Delta t_{1})\) is equal to one period of oscillation, \(\lambda\), of the e.m. field, i.e., the two wavefronts result in phase:

\[c(\Delta t_{2}-\Delta t_{1})=c\frac{\lambda_{u}}{v_{z}}-\lambda_{u}\cos\theta \equiv\lambda \tag{10.34}\]

In the ultra-relativistic limit \(\gamma\gg 1\), small observation angles \(\theta\ll 1\), and small oscillation amplitudes \(K/\gamma\ll 1\), Eq. 10.34 is re-written as follows:

\[\frac{\lambda}{\lambda_{u}}+1-\frac{\theta^{2}}{2}=\frac{1}{\beta _{z}}\approxeq\left[\sqrt{\beta^{2}-\beta_{x}^{2}-\beta_{y}^{2}}\right]^{-1} \approxeq\left[\sqrt{1-\frac{1}{\gamma^{2}}-\langle\beta_{x}^{2}\rangle} \right]^{-1}= \tag{10.35}\] \[\left[\sqrt{1-\frac{1}{\gamma^{2}}-\frac{1}{2}\frac{K^{2}}{\gamma ^{2}}}\right]^{-1}\approxeq 1+\frac{1}{2\gamma^{2}}\left(1+\frac{K^{2}}{2} \right);\] \[\Rightarrow\lambda=\frac{\lambda_{u}}{2\gamma^{2}}\left(1+\frac{K^ {2}}{2}+\gamma^{2}\theta^{2}\right)\]Equation 10.35 establishes a one-to-one relationship between the central wavelength of undulator emission and the particle's energy. It is more and more accurate for a large number of periods, such that \(\beta_{x}\) in Eq. 10.32 can be replaced with its average value over \(z\). By neglecting the vertical motion we also implied \(\beta_{y}^{2}\ll\frac{K^{2}}{\gamma^{\prime 2}}\).

The on-axis resonant wavelength and photon energy in practical units are:

\[\begin{array}{l}\varepsilon_{ph}[eV]=hv_{n}=9509\frac{E[GeV]^{2}}{\lambda_{u }[mm]\left(1+\frac{K^{2}}{2}\right)}\\ \\ \lambda[nm]=\frac{c}{v}=\frac{1241.5}{\varepsilon_{ph}[eV]}\end{array} \tag{10.36}\]

It is instructive to notice that Eq. 10.34 for on-axis emission can be written as \(\frac{\lambda_{u}}{v_{z}}=\frac{\lambda}{c-v_{z}}\). This states that the time the electron takes to travel \(\lambda_{u}\) is equal to the time the light takes to slip ahead the electron by \(\lambda\). The radiation _slippage_ (relative to the source electron) after \(N_{u}\) periods is therefore \(N_{u}\lambda\). This is the minimum duration of the radiation pulse, independently from the bunch duration.

For any given undulator period, the central wavelength can be continuously varied by tuning, for example, the beam energy. However, this usually implies a re-tuning of the entire accelerator, and might be not that practical for fine scans. Hence, a variable \(K\) is adopted through "variable gap" or "variable phase" undulators, in which the density of the magnetic field lines is varied with a remote control of, respectively, the undulator gap and the arrays relative phase, as shown in Fig. 10.5. It turns out that the higher intensity of radiation is at longer wavelengths, because these are obtained with a larger \(K\), which corresponds to stronger centripetal acceleration and therefore larger radiated power (see Eq. 7.8). The shortest wavelength is emitted on-axis (\(\theta=0\), see Eq. 10.35).

#### 10.3.1.1 Discussion: Doppler Effect in an Undulator

Equation 10.35 for the central wavelength of undulator emission can be retrieved starting from the particle's rest frame, where the undulator field is seen by the particle as a travelling wave. How long is the undulator period in the particle's rest frame? What is the wavelength of the scattered radiation in the laboratory frame, if elastic scattering is assumed and the particle's recoil is neglected?

In the particle's rest frame, the undulator period is shortened to \(\lambda^{\prime}_{u}=\lambda_{u}/\gamma\). The undulator field, static in the laboratory frame, is seen by the particle as an e.m. wave of period \(\lambda^{\prime}_{u}\). The wave scatters on the electron and, by virtue of the relativistic Doppler effect (see Eq. 1.48), its wavelength in the laboratory frame results :

\[\begin{array}{l}\lambda=\gamma\lambda^{\prime}_{u}(1-\beta_{z}\cos\theta)= \lambda_{u}\left(1-\sqrt{1-\frac{1}{\gamma^{2}}-\beta_{y}^{2}-\beta_{x}^{2}} \cos\theta\right)\approxeq\\ \\ \approxeq\lambda_{u}\left[1-\left(1-\frac{1}{2\gamma^{2}}-\frac{K^{2}}{4\gamma ^{2}}\right)\left(1-\frac{\theta^{2}}{2}\right)\right]=\frac{\lambda_{u}}{2 \gamma^{2}}\left(1+\frac{K^{2}}{2}+\gamma^{2}\theta^{2}\right)+o\left(\frac{ \theta^{2}}{\gamma^{2}}\right)\end{array} \tag{10.37}\]This derivation highlights the physical origin of the \(\lambda\sim\gamma^{-2}\) dependence, i.e., relativistic length contraction of the undulator period (factor \(\gamma^{-1}\)), and the relativistic Doppler effect of the central frequency (factor \((2\gamma)^{-1}\)).

#### Spectral Width, Angular Divergence

Equation 10.35 for the central wavelength is satisfied also by harmonics \(\lambda_{n}=\lambda/n\), of the fundamental, \(n\in\dot{\mathbb{N}}\). Since the minimum duration of the spontaneous radiation pulse is the radiation slippage, and since this is determined by the fundamental wavelength of emission, the slippage of any hamonic is the same of the fundamental. Thus, we have [5, 6]:

\[\begin{array}{l}\Delta t_{n}=\frac{N_{u}n\lambda_{u}}{c}=\frac{N_{u}\lambda }{c}=\Delta t\\ \\ \Rightarrow\Delta v_{n}=c\frac{\Delta\lambda_{u}}{\lambda_{n}^{2}}\approx \frac{1}{\Delta t_{n}}=\frac{c}{N_{u}\lambda}\\ \\ \Rightarrow\frac{\Delta\lambda_{u}}{\lambda_{n}}=\frac{\Delta v_{n}}{v_{n}} \approx\frac{1}{nN_{u}}\end{array} \tag{10.38}\]

In summary, higher harmonics (\(n>1\)) are emitted in a pulse of same duration of the fundamental emission. Their absolute energy bandwidth is the same of the fundamental, while their _relative_ bandwidth is \(n-\)times smaller.

The spectral intensity distribution--simply "spectrum" hereafter--is proportional to the Fourier transform of the electric field emitted along the \(N_{u}\) undulator periods. Let us assume for simplicity a monochromatic wave at the fundamental frequency \(\omega_{0}=2\pi c/\lambda\). Since the number of periods is finite and the field amplitude is constant, the spectrum is the sinc function (see later Fig. 10.11 and inset):

\[\begin{array}{l}E(t)=\left\{\begin{array}{l}E_{0}e^{i\omega_{0}t},\,|t|\, \leq\,\frac{T}{2}=\frac{N_{u}\lambda}{c}\\ \\ 0,\,otherwise\end{array}\right.;\\ \\ \left\{\begin{array}{l}E(\omega)=\int_{-T/2}^{T/2}E_{0}e^{-i(\omega-\omega_{0 })t}dt=-\frac{E_{0}}{i(\omega-\omega_{0})}\left[e^{-i(\omega-\omega_{0})T/2}- e^{i(\omega-\omega_{0})T/2}\right]=\\ \\ =E_{0}\frac{2sin(\frac{\omega_{0}T}{2})}{\Delta\omega}=E_{0}T\cdot sinc(\xi), \\ \\ \xi:=\frac{(\omega-\omega_{0})T}{2}=\pi\,N_{u}\,\frac{\Delta\omega_{0}}{\omega_ {0}};\\ \\ I(\omega)=\frac{|E(\omega)|^{2}}{2Z_{0}}=\frac{E_{0}^{2}T^{2}}{2Z_{0}}sinc^{2} (\xi).\end{array}\right.\\ \\ \end{array} \tag{10.39}\]

Since the fwhm bandwidth of the sinc function is \(\Delta\omega\approx\frac{2\pi}{T}=\frac{2\pi c}{2N_{u}\lambda}\), the intrinsic relative bandwidth is \(\frac{\Delta\omega}{\omega_{0}}\sim\frac{1}{N_{u}}\), as already in Eq. 10.38. That is, the larger the number of undulator periods is, the more effective the spectral filtering due to the constructive interference is, the narrower is the intrinsic relative bandwidth of the undulator. This,however, can be enlarged by non-zero dimensions of the electron beam, as discussed below.

Equation 10.35 shows that off-energy electrons will radiate at slightly different wavelengths. The effect can be neglected as long as the relative bandwidth enlargement due to the beam's energy spread is negligible with respect to the intrinsic relative bandwidth. This leads to an upper limit for the beam's energy spread:

\[\tfrac{\Delta\lambda_{n}(\gamma)}{\lambda_{n}}=2\,\tfrac{\Delta\gamma}{\gamma} \approx\sigma_{\delta}\ll\tfrac{\Delta\lambda_{n}}{\lambda_{n}}\approx\tfrac{ 1}{nN_{u}} \tag{10.40}\]

This prescription is usually met at electron synchrotrons for fundamental emission, where \(\sigma_{\delta}\leq 0.1\%\) and \(\tfrac{1}{N_{u}}\approx 1\%\). Some bandwidth enlargement is expected at higher harmonics, e.g. \(n\geq 10\).

Equation 10.35 also shows that electrons travelling with angular divergence \(\Delta\theta\) will radiate at slightly different wavelengths. At such off-axis directions of observation, some red-shifted bandwidth enlargement is expected. By recalling the intrinsic bandwidth in Eq. 10.38, this effect is negligible as long as:

\[\begin{array}{l}\tfrac{\Delta\lambda_{n}(\theta)}{\lambda_{n}}=\tfrac{1}{ \lambda_{n}}\tfrac{\lambda_{n}}{2n}\Delta\theta^{2}=\tfrac{\lambda_{n}}{2} \Delta\theta^{2}\ll\tfrac{1}{nN_{u}}\\ \\ \Rightarrow\sigma_{\theta}\ll\sqrt{\tfrac{2\lambda}{\lambda_{n}}\tfrac{1}{nN_ {u}}}=\tfrac{1}{\gamma}\sqrt{\tfrac{1+K^{2}/2}{nN_{u}}}\approx\tfrac{1}{ \gamma}\sqrt{\tfrac{1}{nN_{u}}}\end{array} \tag{10.41}\]

The r.h.s. of Eq. 10.41, obtained for \(K<1\), describes the characteristic angular divergence of the \(n\)-th harmonic of undulator radiation, or central cone. It suggests that, although the local emission has angular divergence \(1/\gamma\), the constructive interference imposes a tighter condition to the overall overlap of the wavefronts, so that the angular divergence of the radiation pulse exiting the undulator is smaller than that of synchrotron radiation, in proportion to the number of undulator periods. Higher harmonics, i.e. shorter wavelengths, show smaller angular divergence.

#### Discussion: Intrinsic Size of Undulator Radiation

Obtain the expressions for the _intrinsic_ transverse size and angular divergence of the fundamental transverse mode [5; 6] anticipated in Eq. 10.4, starting from the spectral and angular distribution of the undulator spontaneous radiation.

The rms angular divergence \(\sigma_{r^{\prime},n}\) is calculated from Eq. 10.41 for the \(n\)-th harmonic by assuming \(K<1\) and by replacing the relativistic \(\gamma\)-factor with the expression for the central wavelength (see Eq. 10.35). An undulator length \(L_{u}=N_{u}\lambda_{u}\) is used:

\[\sigma_{r^{\prime},n}\approx\tfrac{1}{\alpha}\sqrt{\tfrac{2\lambda}{\lambda_{ n}}\tfrac{1}{nN_{u}}}\approx\tfrac{1}{2\sqrt{2}}\sqrt{\tfrac{2\lambda_{n}}{ \lambda_{u}}\tfrac{\lambda_{u}}{L_{u}}}=\sqrt{\tfrac{\lambda_{n}}{4L_{u}}}, \tag{10.42}\]

where \(\alpha=2\sqrt{2}=2.8\) is a numerical factor to pass from full width \(\Delta\theta\) to rms value of the angular divergence, intermediate to 3.46 (uniform) and 2.36 (Gaussian). According to Eq. 10.20, the rms emittance of a transversely coherent light beam is \(\varepsilon_{r}=\sigma_{r}\sigma_{r^{\prime}}=\lambda/(4\pi)\). It follows \(\sigma_{r,n}=\tfrac{\varepsilon_{r}}{\sigma_{r^{\prime},n}}\approx\tfrac{ \sqrt{\lambda_{n}L_{u}}}{2\pi}\).

#### Dipole, Wiggler, Undulator

Historically, IDs were introduced in synchrotrons to complement synchrotron radiation from dipole magnets, and in particular:

* to shift the critical photon energy to higher values by virtue of a larger magnetic field (see Eq. 7.24);
* to increase the radiated power, in proportion to a larger number of magnetic poles;
* to increase the spectral brightness, e.g. by producing narrower intrinsic spectral bandwidth.

In practice \(B_{0y}\ \propto\lambda_{u}\), which typically leads to \(K\ <\ 10\) for \(\lambda_{u}\ <\ 5\) cm (still, higher K can be produced at shorter \(\lambda_{u}\) with specific magnetic designs and technologies). At such period lengths, \(N_{u}\geq 100\) periods can be assembled in an ID to be accommodated in a synchrotron's straight section, which is commonly long \(\sim 5\) m or so. The moderate deflection of the electrons' orbit for \(K\leq 1\) classifies the ID as an _undulator_.

Longer periods allow \(K>10\), and the ID can be classified as a _wiggler_. Owing to the larger magnetic field, electrons in a wiggler have large oscillation amplitude, and the wavefront interference, otherwise present in undulators, gets lost. That is, a wiggler behaves as a series of (strong) dipole magnets.

Given the similarity in the physics of emission shared by dipole, undulator and wiggler magnets, we can estimate the total average (over one turn) radiated power in the three cases starting from Eq. 7.15, assuming \(N_{b}\) electrons in a bunch of average current \(\langle I\rangle\)[1, 4, 5, 6].

For the dipole magnet, it is simple to find the radiated power in a turn and, from that, along a magnet of length \(L=R\theta_{d}\), assuming an isomagnetic lattice:

\[\begin{array}{l}P_{d,turn}[kW]=\frac{N_{b}U_{0}}{T_{0}}=\frac{\langle I \rangle U_{0}}{e}=26.5E^{3}[GeV]B[T]\langle I\rangle[A]\\ \\ P_{d,L}[kW]=P_{d,turn}[kW]\cdot\frac{\theta_{d}}{2\pi}=14.06E^{4}[GeV]\langle I \rangle[A]L[m]/R^{2}[m]\end{array} \tag{10.43}\]

The wiggler total radiation intensity is the incoherent sum of synchrotron radiation emitted at \(N_{u}\)-poles, thus the total intensity is \(N_{u}\)-fold that from a single dipole of same magnetic field. The total power from a wiggler magnet long \(L_{w}\) (a dipole magnet is equivalent to 2 wiggler periods) is calculated as the fraction \(\frac{L_{w}/2}{2\pi\,R}\) of the synchrotron radiation power emitted along the whole circumference:

\[\begin{array}{l}P_{w}[kW]=\frac{N_{b}U_{0}}{T_{0}}\frac{L_{w}}{4\pi R}=\frac {\langle I\rangle[A]U_{0}[kV]}{4\pi}\frac{0.3B[T]L_{w}[m]}{E[GeV]}=0.633E^{2}[ GeV]\langle B^{2}[T]\rangle\langle I\rangle[A]L_{w}[m]\end{array} \tag{10.44}\]

and \(\langle B^{2}[T]\rangle\) is the rms wiggler field squared.

The average power from an undulator is estimated through Eq. 7.12 with the prescription of spectro-angular filtering provided by the ID. According to Eq. 10.35, the relative bandwidth observed off-axis at the generic angle \(\theta\) is:

\[\begin{array}{l}\frac{\Delta\lambda}{\lambda}=\frac{\lambda(\theta)-\lambda (0)}{\lambda(0)}=\frac{\nu^{2}\theta^{2}}{1+\frac{K}{2}}\end{array} \tag{10.45}\]To estimate the power emitted only in the central cone, we restrict the observation angle to the characteristic angular divergence of undulator spontaneous radiation \(\theta\approx\frac{1}{\gamma}\sqrt{\frac{1}{N_{u}}}\) (see Eq. 10.41), to obtain \(\frac{\Delta\lambda}{\lambda}\approx\frac{1}{N_{u}\left(1+\frac{K^{2}}{2} \right)}\). The dipole radius is replaced by the undulator magnetic field according to \(R=\frac{p_{e}}{eB}\approx\frac{\lambda_{u}}{2\pi\,m_{e}cK}\frac{E}{c}\). Finally, if particles take the time interval \(\Delta t\) to pass the undulator length \(L_{u}\approxeq c\Delta t\), the turn-averaged power has to be re-scaled by the fraction \(\Delta t/T_{0}\):

\[P_{u}=N_{b}\,P_{sr}\,\frac{\Delta\lambda}{\lambda}\,\frac{\Delta t}{T_{0}}= \frac{1}{4\pi\varepsilon_{0}}\frac{2}{3}e^{2}c\gamma^{4}\,\frac{4\pi^{2}m_{e}^ {2}c^{2}}{\lambda_{u}^{2}E^{2}}\frac{K^{2}}{N_{u}\left(1+\frac{K^{2}}{2} \right)}\frac{N_{b}\Delta t}{T_{0}}\frac{ec}{ec}=\frac{2\pi}{3}\frac{e}{ \varepsilon_{0}}\frac{\langle I\rangle N_{u}\gamma^{2}}{\lambda_{u}}\frac{K^ {2}}{\left(1+\frac{K^{2}}{2}\right)} \tag{10.46}\]

An ab initio calculation would provide the following expression, in practical units, for the spectral intensity of undulator spontaneous emission at the \(n\)-th harmonic of the fundamental (\(n\) odd), within the central angular cone \(2\pi\,\sigma_{r^{\prime}}^{2}\):

\[\begin{array}{ll}\phi_{n}&=\,\frac{dN_{r}}{dtd\omega/\omega}=2\pi\,\sigma_{ r^{\prime}}^{2}\left(\frac{d\phi_{n}}{d\Omega}\right)_{\theta=\psi=0}=\frac{1}{2} \pi\,\alpha N_{u}\,\frac{\langle I\rangle}{e}\left(\frac{K^{2}}{1+\frac{K^{2} }{2}}\right)n[JJ]_{n}^{2}=\\ &=\,\frac{1.43}{2}\cdot 10^{14}N_{u}\,\left(\frac{K^{2}}{1+\frac{K^{2}}{2}} \right)n[JJ]_{n}^{2}\langle I\rangle[A]\ \ \left[\frac{\#ph}{sec\cdot 0.1\%bw}\right]\end{array} \tag{10.47}\]

where \(\alpha=1/137\) is the fine structure constant. \([JJ]_{n}=1\) for helically polarized undulators, while it is a combination of Bessel functions for linearly polarized light:

\[[JJ]_{n}=(-1)^{\frac{n-1}{2}}J_{\frac{n-1}{2}}(\frac{na_{u}^{2}}{2(1+a_{u}^{2} )})-J_{\frac{n+1}{2}}(\frac{na_{u}^{2}}{2(1+a_{u}^{2})}),\,a_{u}=K^{2}/2 \tag{10.48}\]

When the lowest harmonics (\(n\leq 3\)) are included in the total intensity, the transverse mode of radiation cannot be any longer approximated by a Gaussian, and the factor 1/2 in Eq. 10.47 is removed.

By recalling Eq. 10.4, the central angular cone has amplitude:

\[2\pi\,\sigma_{r^{\prime}}^{2}=\frac{\pi\lambda}{2nL_{u}}=\frac{\pi}{N_{u}} \frac{\left(1+\frac{K^{2}}{2}\right)}{4n\gamma^{2}} \tag{10.49}\]

Inserted into Eq. 10.47, this shows that in the central cone \(\frac{d\phi_{n}}{d\Omega}\propto N_{u}^{2}\). On the contrary, emission from a wiggler or a dipole shows \(\phi\propto N_{u}\) (\(N_{u}=1\) in a dipole, see also Eqs. 7.29 and 10.44). The \(N_{u}^{2}\)-dependence of the central cone of undulator radiation is the result of constructive interference, and a signature of partial coherence. Still, the total intensity is linearly proportional to the number of particles (so as in a wiggler and a dipole, see Eqs. 7.31 and 10.44) and therefore it is _not_ coherent emission in the sense of Eq. 10.31. This can be rephrased by stating that undulator spontaneous radiation is produced as a coherent sum of fields emitted by one electron (Gaussian laser-like mode with rms size and angular divergence in Eq. 10.4)over many consecutive periods (\(I_{r}\propto N_{u}^{2}\)). However, it remains an incoherent sum of wave trains emitted by the population of electrons (\(I_{r}\propto N_{e}\)).

In summary, dipole magnet synchrotron radiation, multipole wiggler emission and undulator spontaneous radiation have the same physical origin, but the output spectral and angular characteristics are eventually determined by the specificity of the electrons' trajectory in the magnetic device. A qualitative comparison of typical angular and spectral distribution of radiation from a dipole magnet and IDs is illustrated in Fig. 10.6.

#### Harmonic Emission

Despite Eq. 10.35 is satisfied by any harmonic of the fundamental wavelength, emission of _sub_-harmonics from wigglers or undulators is suppressed by the electron trajectory through the ID [5]. Indeed, the particle's trajectory with local versor \(\hat{n}=(-\hat{x}\cos\phi\sin\theta,\,-\hat{y}\sin\phi\sin\theta,\,-\hat{z} \cos\theta)\) in the Frenet-Serret coordinates system, is embedded in the expression of the radiated electric field (see Eq. 7.5). The spectral distribution of the emitted radiation is proportional to the Fourier transform of the electric field (see Eq. 7.25), hence of the particle's trajectory:

\[\frac{dU}{d\Omega}=\int\frac{dP}{d\Omega}dt=c\varepsilon_{0}\int|R\bar{E}(t )|^{2}dt=2c\varepsilon_{0}\int|R\bar{E}(\omega)|^{2}d\omega\Rightarrow\frac{ d^{2}U}{d\Omega d\omega}=2c\varepsilon_{0}|R\bar{E}(\omega)|^{2} \tag{10.50}\]

Figure 10.6: Electron trajectory, characteristic angular divergence (left) and spectrum (right) of radiation from dipole magnet, multipole wiggler, undulator. Zoom: spectral intensity from a planar undulator optimized for the 3rd harmonic of 2.7 keV, in the presence of pin-hole for angular selection of on-axis radiation (red, solid), and without angular selection (blue, dashed). A large angular acceptance leads to red-shifted bandwidth enlargement of the odd harmonics, as well as to observation of even harmonics off-axis (see next Section)

For a _planar, linearly polarized_ ID, we draw the following observations.

* If \(K\ll 1\) (undulator), the particle's longitudinal velocity in Eq. 10.33 is approximately constant, \(v_{z}\approx\langle v_{z}\rangle=const\). The trajectory is a pure sinusoidal oscillation, whose FT is a monochromatic emission. Since in the particle's reference frame the motion is a purely transverse harmonic oscillation, the angular distribution of radiation is that one of an electric dipole (see Fig. 10.7-top row). In the laboratory reference frame, radiation is boosted to a forward, collimated, centered emission (see also Fig. 7.2).
* If \(K\gg 1\) (wiggler), Eq. 10.33 describes the superposition of a pure sinusoidal oscillation and a _higher_ frequency perturbation to the particle's velocity, of the order of \(cK^{2}/\gamma^{2}\) (see Fig. 10.7-bottom row). The FT of such trajectory has higher harmonic contents, namely, sub-harmonics of the fundamental emission are suppressed. The particle's motion in the reference frame moving at \(\langle v_{z}\rangle\) with respect to the laboratory has a net longitudinal component: the orbit is a "figure-8", which can be described as the superposition of a transverse and a longitudinal electric dipole. In the laboratory reference frame, they generate respectively forward off-axis and on-axis emission, at higher harmonics of the fundamental (see also Fig. 7.2). But, since the motion is anti-symmetric in proximity of the ID axis and symmetric off-axis, only odd harmonics are permitted on-axis, and only even harmonics off-axis.
* Any intermediate value of \(K\) determines a situation intermediate to the two extreme cases above. For example, the emission from a linearly polarized undulator with \(K\geq 1\) shows odd harmonics on-axis and even harmonics off-axis. The higher the \(K\) is, the larger the contribution of higher harmonics to the whole radiated energy will be. At large \(K\)-values, the distinction between undulator and wiggler becomes more vague.

Figure 10.7: Top view of particle’s trajectory through an undulator (top) and a wiggler magnet (bottom), and angular distribution of radiation, in the particle’s rest frame and in the laboratory frame, where the particle moves at ultra-relativistic velocity. The geometry of the particle’s trajectory determines the emission of higher odd and even harmonics of the fundamental

Another way of looking to the spectral properties of emission from IDs is by considering the duration of the light flash seen by the observer. For an ideal undulator, the observer--assumed on-axis--is illuminated by radiation emitted along the device with no interruption. Since the spectral bandwidth is inversely proportional to the time duration of the pulse, it will be narrow in proportion of the number of undulator periods (see Eq. 10.38).

In a wiggler, the observer will receive most of the light when the particles travel parallel to the axis of the device, i.e., along portions of the trajectory far off-axis. This way, the radiation pulse at the exit of a wiggler can be seen as the incoherent superposition of light flashes transversely separated but periodically spaced. In the frequency domain, this is equivalent to a rich harmonic content, in proportion to the deflection parameter. At very high frequencies, the natural enlargement of each spectral line, as already discussed for a dipole magnet, smears the spectrum, which in fact results a _continuum_ (see Fig. 10.6-right plot). The intensity is amplified with respect to a dipole magnet of same magnetic field, by just the number of wiggler periods.

#### 10.3.4.1 Discussion:Transverse Coherence of Light Sources

We want to show that the intrinsic degree of transverse coherence of undulator radiation is larger than that of dipole radiation at the same wavelength. What is the size of a pinhole at distance L = 15 m from the ID, to make the undulator radiation fully transversely coherent? What electron beam emittance would be required to ensure fully transversely coherent radiation?

Let us consider an electron beam energy E = 2.5 GeV, and transverse beam size at the dipole magnet and at the ID \(\sigma_{x,\,y}\approx 30\,\mu\)m. The undulator fundamental emission is at \(\lambda=0.5\) nm, from \(N_{u}=100\) periods, each period long \(\lambda_{u}=2\) cm. For simplicity, synchrotron radiation from the dipole magnet is assumed to be at the critical frequency.

The relative degree of transverse coherence is evaluated in terms of the ratio of the typical angular divergence of radiation and the coherence angle \(\theta_{c}\approx\frac{\lambda}{4\pi\sigma_{x,\,y}}=1.3\,\mu\) rad (see Eq. 10.20). The ratio for the dipole and the undulator amounts to, respectively, \((\theta_{c}\gamma)^{-1}\approx 150\) and \(\left(\theta_{c}\gamma\sqrt{N_{u}}\right)^{-1}\approxeq 15\) (see Eq. 10.41). It results that both sources are far from transverse coherence at 0.5 nm, but undulator radiation is \(\sim\)10-fold closer to that than synchrotron radiation.

The pinhole radius needed to push the undulator radiation to the diffraction limit, i.e., to provide full transverse coherence, is estimated through Eq. 10.21, giving \(R\leq\frac{\lambda L}{4\pi\sigma_{x,\,y}}=20\,\mu\)m. Worth to notice, the intrinsic angular divergence of undulator radiation is approximately 6-fold larger than the coherence angle (see Eq. 10.4).

Transversely coherent radiation could be naturally produced if the electron beam transverse emittances satisfy the diffraction limit condition at 0.5 nm, i.e., \(\varepsilon_{x,\,y}<\frac{\lambda}{4\pi}=40\) pm rad, or \(\varepsilon_{n}<\frac{\gamma\lambda}{4\pi}=0.2\,\mu\)m rad.

#### Discussion: Longitudinal Coherence in Storage Rings

What is the wavelength range in which an electron bunch in a storage ring could radiate in regime of intensity enhancement? Does intensity enhancement imply longitudinal coherence, or viceversa, in an undulator and in a dipole magnet, assuming no monochromatization of radiation?

For a quantitative answer, typical parameters at medium energy storage ring light sources are considered, such as 3 GeV-energy, 1.6 nC-bunch charge, and rms bunch length of 9 mm. For simplicity, we ignore the transverse beam dimensions ("pencil beam" approximation). Let us assume a dipole curvature radius \(R=10\) m, number of undulator periods \(N_{u}=100\), and undulator emission optimized at \(\lambda=0.1\) nm in \(n=5\)th harmonic of the fundamental.

According to Eq. 10.31, intensity enhancement is present for \((N_{b}-1)|f_{\parallel}(\omega)|^{2}>1\). For a Gaussian beam \(|f_{\parallel}(\omega)|^{2}=e^{-\omega^{2}\sigma_{t}^{2}}\), hence:

\[\begin{split}& N_{b}e^{-\omega^{2}\sigma_{t}^{2}/2}\\ & N_{b}>e^{\frac{(2\pi c\sigma_{t})^{2}}{2\lambda^{2}}}\\ &\sigma_{z}<\frac{\lambda}{2\pi}\sqrt{\ln\,\mathrm{N_{b}}}\end{split} \tag{10.51}\]

Coherent emission in the sense of Eq. 10.30 starts appearing at \(\lambda>1\) cm (microwaves). As a matter of fact, the formation of density modulations internal to the bunch due to the so-called "microwave instability" can amplify the emission at wavelengths shorter than \(\sigma_{z}\), typically down to the IR-THz region.

A large degree of longitudinal coherence can be obtained if the bunch length is shorter than the longitudinal coherence length. In the limit of vanishing bunch length, a single mode would be emitted, i.e., a fully coherent pulse. By recalling Eq. 10.24 and the definition of critical frequency of synchrotron radiation in Eq. 7.24, we impose:

\[2\pi\,\sigma_{z}\leq\frac{\lambda^{2}}{2\Delta\lambda}=\frac{c}{2\Delta v} \approx\frac{c}{4v_{c}}=\frac{c}{4}\,\frac{1}{\frac{3}{4\pi}\frac{c\nu^{3}}{R }}\approx\frac{R}{\nu^{3}}=0.05\,\mathrm{nm} \tag{10.52}\]

A similar consideration for the undulator leads to:

\[2\pi\,\sigma_{z}\leq\frac{\lambda^{2}}{2\Delta\lambda}\approx n\,N_{u}\, \frac{\lambda}{2}=25\,\mathrm{nm} \tag{10.53}\]

It emerges that both synchrotron radiation and undulator spontaneous radiation are longitudinally incoherent radiation for realistic bunch durations. Since the longitudinal coherence of dipole radiation is constrained by the critical frequency (Eq. 10.52), there is no practical combination of parameters which guarantee some intrinsic degree of longitudinal coherence. Instead, an undulator can be configured and tuned to a central wavelength approaching the bunch length (Eq. 10.53). For example, partial longitudinal coherence could be generated by a far-IR undulator traversed by a \(\sim\)mm long beam.

By comparing Eqs. 10.52, 10.53 with Eq. 10.51 we deduce that, for typical beam and ID parameters at storage ring light sources:

* for synchrotron radiation in proximity or _above_ the critical frequency and for undulator spontaneous radiation, the condition of intensity enhancement automatically implies some degree of longitudinal coherence;
* for synchrotron radiation _below_ the critical frequency, longitudinal coherence implies intensity enhancement; this is not necessarily true for undulator radiation.

#### 10.3.4.3 Discussion: Longitudinal Coherence of Harmonic Emission

Let us demonstrate that the longitudinal coherence length of undulator fundamental and harmonic emission is the same, and that it can be lengthened by tuning the undulator to a sub-harmonic of the fundamental.

\(L_{c,\parallel}\) is calculated from Eq. 10.24 for both the fundamental emission at \(\lambda\) and harmonic emission at \(\lambda_{n}=\lambda/n\). The intrinsic relative bandwidth of undulator radiation satisfies \(\frac{\Delta\lambda_{n}}{\lambda_{n}}=\frac{1}{n}\frac{\Delta\lambda}{\lambda}\) (see Eq. 10.38). Hence:

\[\begin{array}{l}L_{c,\parallel}(\lambda_{n})=\frac{\lambda_{n}^{2}}{2 \Delta\lambda_{n}}=\frac{n\lambda}{2\Delta\lambda}\lambda_{n}=\frac{\lambda^ {2}}{2\Delta\lambda}=L_{c,\parallel}(\lambda)\\ \\ \Rightarrow L_{c,\parallel}(\lambda_{n})=\frac{N_{n}\lambda}{2}\end{array} \tag{10.54}\]

We suppose that the undulator can be tuned to a sub-harmonic \(\lambda^{\prime}=n\lambda\) of the original fundamental, which is now generated as a higher harmonic of order \(n\). In this case, the radiation slippage along the undulator changes because the fundamental wavelength has changed, and it amounts to \(\Delta t=N_{u}\lambda^{\prime}=nN_{u}\lambda\). Equation 10.38 prescribes an \(n\)-fold smaller energy bandwidth, \(\Delta v_{n}=\frac{c}{nN_{u}\lambda}\). The relative bandwidth at the wavelength \(\lambda_{n}=\lambda\) is:

\[\begin{array}{l}\frac{\Delta\lambda}{\lambda}\equiv\frac{\Delta\lambda_{u}} {\lambda_{n}}=\frac{\Delta v_{n}}{v_{n}}=\frac{c}{nN_{u}\lambda}\frac{\lambda} {c}=\frac{1}{nN_{u}}\\ \\ \Rightarrow L_{c,\parallel}(\lambda)=\frac{\lambda^{2}}{2\Delta\lambda}=n \frac{N_{u}\lambda}{2}\end{array} \tag{10.55}\]

As demonstrated in Eq. 10.54, it still holds \(L_{c,\parallel}(\lambda^{\prime})=L_{c,\parallel}(\lambda)\). However, the absolute value of the longitudinal coherence length evaluated at a _given_ wavelength is \(n\)-fold longer when radiation is generated as a higher harmonic (compare Eqs. 10.54 and 10.55).

### 10.4 Inverse Compton Scattering

Electron linear accelerators find application in so-called Inverse Compton Scattering light sources (ICS). Accelerated electrons interact in (quasi) head-on collision with an external IR or UV laser. Back-scattered photons are boosted in frequency by virtue of the relativistic Doppler effect, which applies first to the incoming radiation in the electrons' rest frame, then to the scattered wavelength in the laboratory frame. Owing to the similarity of this interaction with that one of electrons and undulator field in the electron's rest frame, an expression for the central wavelength in ICS similar to that of spontaneous undulator emission is obtained.

#### Thomson Back-Scattering

We assume a laser of wavelength \(\lambda_{L}\) is impinging on a relativistic electron beam at a positive angle \(\theta_{i}\ll 1\) in the laboratory frame. We apply Eq. 47 to calculate the laser wavelength in the electron's rest frame (where the laser source is moving), so that accepted quantities in Eq. 47 refer here to the laboratory frame (where the laser source is at rest):

\[\lambda^{\prime}_{L}=\frac{\lambda_{L}}{\gamma\left(1+\beta_{z}\cos\theta_{i }\right)} \tag{56}\]

Let us assume an elastic scattering on an ultra-relativistic electron (\(\gamma\gg 1\)) with no electron recoil, or _Thomson scattering_[7]. In other words, radiation instantaneously scattered at a small negative angle \(\theta_{s}\) in the laboratory frame, is seen by the electron with a wavelength \(\lambda^{\prime}_{s}\approx\lambda^{\prime}_{L}\) (the photon energy does not change). In the laboratory frame, the source of scattered radiation is moving at the electron's velocity and we therefore apply Eq. 48 to retrieve the wavelength of the wavefront propagating at the angle \(\theta_{s}\) (accounted quantities in the equation refer here to the electron's rest frame):

\[\begin{split}&\lambda_{s}\approx\lambda^{\prime}_{L}\gamma\left(1- \beta_{z}\cos\theta_{s}\right)\approx\lambda_{L}\frac{1-\beta\cos\theta_{s}}{1 +\beta\cos\theta_{i}}\approx\lambda_{L}\frac{1-\sqrt{1-\frac{1}{\gamma^{2}}} \left(1-\frac{\theta_{s}^{2}}{2}\right)}{1+\beta}\approx\\ &\approx\frac{\lambda_{L}}{2}\left[1-\left(1-\frac{1}{2\gamma^{2 }}\right)\left(1-\frac{\theta_{s}^{2}}{2}\right)\right]\approx\frac{\lambda_{ L}}{4\gamma^{2}}\left(1+\gamma^{2}\theta_{s}^{2}\right)+o\left(\frac{\theta_{s}^{2}}{ \gamma^{2}}\right)\end{split} \tag{57}\]

The no recoil approximation implies that in the frame where the electron is initially at rest, the electron energy--i.e., its rest energy--is not changed. That is, the electron's rest energy is much larger than the photon energy. The invariant mass of the electron-photon system is approximately the electron's rest energy. The initial 4-momentum in the laboratory frame is (see Eq. 36) \(p_{t}^{\mu}=\left(\frac{E_{e}+E_{L}}{c}\), \(\vec{p}_{e}+\vec{p}_{L}\right)\). The invariant mass in the laboratory frame is:

\[\begin{split}&\sqrt{p_{t}^{\mu}\,p_{t,\mu}c^{2}}=\sqrt{E_{e}^{2}+E_{L}^ {2}+2E_{e}E_{L}-p_{e}^{2}c^{2}-p_{L}^{2}c^{2}-2\vec{p}_{e}\vec{p}_{L}c^{2}}=\\ &=\sqrt{m_{e}^{2}c^{4}+2E_{e}E_{L}+2\frac{E_{e}}{\beta_{z}}E_{L} \cos\theta_{i}}\approxeq\sqrt{m_{e}^{2}c^{4}+4E_{e}E_{L}}=m_{e}c^{2}\sqrt{1+ \frac{4E_{e}E_{L}}{m_{e}^{2}c^{4}}}=\\ &=m_{e}c^{2}\sqrt{1+\frac{4\gamma^{2}E_{L}}{E_{e}}}=m_{e}c^{2} \sqrt{1+\frac{\hat{E}_{s}}{\hat{E}_{e}}}\to m_{e}c^{2}\iff E_{e}\gg\hat{E}_{ s}\end{split} \tag{58}\]A "recoil parameter" \(X:=\hat{E_{s}}/E_{e}\ll 1\) defines the transition from the exact "Compton regime" to the approximated "Thomson regime" (no recoil).

#### Angular and Spectral Distribution

Because of the analogy of ICS with the generation of undulator radiation, Eq. 10.57 has the same functional dependence of Eq. 10.35. But, by virtue of a laser wavelength much shorter (\(\leq\)1\(\mu\)m) than any magnetic undulator period (\(\geq\)1 cm), ICS easily boosts the on-axis scattered photon energy to hard x-rays and up to \(\gamma\)-rays (\(\sim\)MeV photon energy range), for only few 100s MeV electron energy. In spite of this, the Thomson regime is still a good approximation of the process in many practical cases (\(X\leq 1\%\)).

The total Thomson cross section is pretty small, \(\sigma_{T}\approx\frac{8\pi}{3}r_{e}^{2}=6.7\cdot 10^{-25}\)cm\({}^{2}\approx 1\) barn. Hence, high charge and large laser photon densities are required at the interaction point (IP) to produce substantial scattered radiation intensity. They are usually accompanied by small transverse sizes of the two beams at the IP, and by small incident scattering angle.

Since the emission is incoherent and no interference process intervenes, the characteristic angular divergence of the scattered radiation is \(1/\gamma\) and, so as for synchrotron radiation, \(\frac{(d\sigma/d\Omega)_{\theta_{s}=1/\gamma}}{(d\sigma/d\Omega)_{\theta_{s}= 0}}=\frac{1}{8}\) (see Eq. 7.22). But, for ICS it holds:

\[\sigma_{T,cone}=\int_{0}^{2\pi}d\phi\int_{0}^{1/\gamma}\frac{d\sigma}{d\Omega }\sin\theta d\theta\approxeq\frac{4\pi}{3}r_{e}^{2}=\frac{1}{2}\sigma_{T} \tag{10.59}\]

Namely, about half of the scattered photons are in the central cone of emission. By virtue of Eq. 10.57, these are also the most energetic photons.

In summary, photons are scattered at any angle and the spectral bandwidth is large. However, both intensity and photon energy distribution are peaked on-axis (so-called "Compton edge"). Here, the bandwidth is typically of the order of \(\sim\)1 \(-\) 10%, dominated by the angular acceptance \(\sim\gamma\theta_{s}\) of the system, the electron beam relative energy spread, and the laser relative spectral bandwidth.

The correlation between photon energy and angle of emission is described by Eq. 10.57, which can be re-written as:

\[E_{s}=E_{L}\ \frac{1+\beta\cos\theta_{i}}{1-\beta\cos\theta_{s}}\approxeq 2 \gamma^{2}E_{L}\frac{1+cos\theta_{i}}{1+\gamma^{2}\theta_{S}^{2}}\leq 4 \gamma^{2}E_{L} \tag{10.60}\]

For head-on collision, the correlation is illustrated in Fig. 10.8. We identify two relevant points (for \(\beta\to 1\)):

\[\left\{\begin{array}{l}E_{s,\theta_{s}=0}=4\gamma^{2}E_{L}\equiv\hat{E}_{s }\\ \\ E_{s,|\theta_{s}|\approx 1/\gamma}=2\gamma^{2}E_{L}=\frac{1}{2}\hat{E}_{s} \end{array}\right. \tag{10.61}\]

This allows a spectral selection of ICS radiation by means of a collimation system. Photons which are not scattered and propagate straight have energy \(E_{s,\theta_{s}=\pi}=E_{L}\).

[MISSING_PAGE_EMPTY:522]

Finally, \(\gamma^{2}\) from Eq. 10.64 is substituted back into the first line of Eq. 10.63:

\[\begin{array}{l}\frac{m_{e}^{2}c^{2}}{\left[\frac{h}{c}(v-v^{\prime})+m_{e}c \right]^{2}-\left(\frac{h}{c}\right)^{2}(v^{2}-2vv^{\prime}\cos\theta+v^{\prime 2 })}=1;\\ m_{e}^{2}c^{2}=\left(\frac{h}{c}\right)^{2}(v-v^{\prime})^{2}+m_{e}^{2}c^{2}+2 m_{e}h(v-v^{\prime})-\left(\frac{hv}{c}\right)^{2}-\left(\frac{hv^{\prime}}{c} \right)^{2}+2\left(\frac{h}{c}\right)^{2}vv^{\prime}\cos\theta;\\ m_{e}hv-m_{e}hv^{\prime}-\frac{h}{c}vv^{\prime}(1-\cos\theta)=0;\\ v^{\prime}=v\frac{m_{e}c}{m_{e}c+\frac{hv}{c}(1-\cos\theta)}.\\ \Rightarrow\lambda^{\prime}=\lambda+\frac{h}{m_{e}c}(1-\cos\theta).\end{array} \tag{10.65}\]

In conclusion, the scattered wavelength in the electron's rest frame is shifted by the Compton wavelength \(\lambda_{C}=\frac{h}{m_{e}c}\) (see also Eq. 4.157). As expected, the wavelength of non-interacting photons (\(\theta\) = 0) is not changed, while the maximum shift is by \(2\lambda_{C}\) for back-scattered light.

##### Discussion:Gamma-Gamma Collider

Gamma-gamma colliders have been proposed as Compton sources driven by multi-GeV-energy electron linacs, for high intensity-high energy photonic interactions for particle physics experiments. We demonstrate below that when the recoil parameter cannot be neglected (\(X\geq 1\)), the maximum energy of back-scattered photons in the laboratory frame approaches (from below) the ultra-relativistic electron energy.

At first, the back-scattered photon energy is calculated in the electrons' rest frame, as given by Eq. 10.65:

\[\begin{array}{l}\lambda_{f}^{e}=\lambda_{i}^{e}+2\lambda_{C};\\ \frac{c}{\lambda_{f}^{e}}=\frac{c}{\lambda_{i}^{e}}\frac{\lambda_{i}^{e}}{ \lambda_{i}^{e}+2\lambda_{C}}=\frac{c}{\lambda_{i}^{e}}\frac{1}{1+\frac{2 \lambda_{C}}{\lambda_{i}^{e}}};\\ v_{f}^{e}=v_{i}^{e}\frac{1}{1+\frac{2v_{i}^{e}}{c}}\end{array} \tag{10.66}\]

The relativistic Doppler effect in Eq. 1.48 is applied to Eq. 10.66 to express the photon frequencies in the laboratory frame, assuming head-on collision (\(\theta_{i}=\pi\)) and back-scattering (\(\theta_{s}=0\)):

Figure 10.9: Compton scattering in the frame where the electron is initially at rest

\[\begin{array}{l}v_{i}^{lab}=v_{i}^{e}\frac{1}{\gamma(1+\beta)},\ \ \ \ \ \ v_{f}^{lab}=v_{f}^{e}\frac{1}{\gamma(1-\beta)}\\ \Rightarrow v_{f}^{lab}\gamma(1-\beta)=v_{i}^{lab}\gamma(1+\beta)\frac{1}{1+ \frac{v_{i}^{lab}}{v_{C}}2\gamma(1+\beta)}\end{array} \tag{10.67}\]

We define \(E_{e}=\gamma m_{e}c^{2}\) the electron's energy before scattering, \(\hat{E}_{S,C}=hv_{f}^{lab}\) the maximum Compton back-scattered photon energy, \(\hat{E}_{L}=hv_{i}^{lab}\) the incident laser photon energy, and \(\hat{E}_{S,T}=hv_{i}^{lab}2\gamma^{2}(1+\beta)\approxeq 4\gamma^{2}E_{L}\) the maximum back-scattered photon energy in the Thomson regime (small recoil approximation, see Eq. 10.60). Then, Eq. 10.67 becomes:

\[\begin{array}{l}\hat{E}_{S,C}=\hat{E}_{L}\frac{1+\beta}{1-\beta}\frac{1}{1+ \frac{\hat{E}_{L}}{\gamma m_{e}c^{2}}2\gamma(1+\beta)}\approxeq\frac{4\gamma^ {2}\hat{E}_{L}}{1+\frac{4\gamma^{2}\hat{E}_{L}}{E_{e}}}=\frac{\hat{E}_{S,T}}{1+ \frac{\hat{E}_{S,T}}{E_{e}}}=\frac{X}{X+1}E_{e}\leq E_{e}\end{array} \tag{10.68}\]

### Free-Electron Laser

Free-electron lasers (FELs) are advanced light sources ranging from IR-THz to hard x-rays. They are characterized by high peak intensity, large degree of transverse and longitudinal coherence, accompanied by sub-picosecond to attosecond pulse duration. The low gain regime of FEL emission, also named "small signal", is introduced first. Although not explicitly discussed, it applies, for example, to optical-cavity FELs driven by electron storage rings. The formalism is then extended to the high gain regime, commonly driven by high brightness electron linacs. The FEL theory is limited to the longitudinal beam dynamics, i.e., for vanishing transverse emittances. 3-D effects are finally recalled for completeness.

#### Resonance Condition

The FEL process relies on the energy exchange between electrons and undulator radiation emitted by the electrons themselves. The exchange happens by virtue of the collinearity of the electron's transverse velocity (Eq. 10.32) and the transverse electric field of the co-propagating e.m. wave. In a planar undulator for horizontally polarized light, the electrons wiggle in the horizontal plane, and the amount of radiated power is [8]:

\[\begin{array}{l}\frac{dE}{dt}=-ec[JJ]\vec{\beta}\cdot\vec{E}=-ec[JJ]\beta_{x }E_{x}=-ec\frac{\hat{K}E_{x,0}}{\gamma}\cos(k_{u}z)\cos(kz-\omega t+\phi_{0})= \\ =-ec\frac{\hat{K}E_{x,0}}{2\gamma}\left[\cos\left[(k_{u}+k)z-\omega t+\phi_{0} \right]+\cos\left[(k_{u}-k)z+\omega t-\phi_{0}\right]\right]\equiv\\ \equiv-ec\frac{\hat{K}E_{x,0}}{2\gamma}\left[\cos(\psi+\phi_{0})+\cos\chi \right],\end{array} \tag{10.69}\]where the minus sign is conventionally adopted to indicate energy loss by the electron when the scalar product is positive, and is the electron-radiation "coupling factor", given by difference of Bessel functions with argument. The coupling factor takes into account the oscillation of the electron's longitudinal velocity in the horizontally polarized undulator (see Eq. 10.33). It applies to the electron-radiation interaction only, not to the definition of the central wavelength of emission. In a helically polarized undulator, the particle's longitudinal velocity is constant and.

Equation 10.69 can be intended as the superposition of the electrons with a forward () and a backward travelling wave. The phase of the latter one oscillates faster than the forward wave, therefore it can be neglected over many undulator periods:

(10.70)

The former is a _ponderomotive phase_, i.e., the phase of a wave co-propagating with the electrons, given in turn by the superposition of the travelling wave of the undulator spontaneous radiation () and the stationary wave describing the undulator magnetic field ().

Since both the electron's velocity and the electric field oscillate with time, the average energy exchange over many undulator periods would be null. Thus, a net energy exchange is guaranteed only by a synchronization of the two vectors, i.e., a constant ponderomotive phase:

(10.71)

We conclude that only radiation at the wavelength of the on-axis undulator spontaneous emission sustains the interaction with the electrons (see Eq. 10.35). In particular, Eq. 10.71 identifies the particle's "resonant energy",, which guarantees emission at a certain for any given undulator period and undulator field.

The synchronization implied by Eq. 10.71 can be re-written as:

(10.72)

That is, the particle's longitudinal velocity is equal to the phase velocity of the ponderomotive wave. In general, the synchronization or the resonant emission implies that at the end of the undulator, some electrons will have been subject to net energy loss (emission of radiation), others to net energy gain (absorption of radiation). This, however, does not imply any amplification of the output radiation yet, which indeed needs the electron dynamics to be taken into account, as explained below.

#### 10.5.2 Pendulum Equation

The dynamics of the \(j\)-th electron is described in the longitudinal phase space through the ponderomotive phase \(\psi_{j}\) and the relative energy deviation with respect to the resonant energy, \(\eta_{j}=\frac{\gamma_{j}-\gamma_{r}}{\gamma_{r}}\). The amplitude of the radiated electric field is assumed to be approximately constant at this stage (regime of "small signal") [4].

The time-derivative of the phase is calculated by considering the practical case \(\eta_{j}\ll 1\) :

\[\begin{array}{l}\frac{d\psi_{j}}{dt}=c\beta_{z,j}(k_{u}+k)-k\approxeq ck_{u} +ck\left[1-\frac{1}{2\gamma_{j}^{2}}\left(1+\frac{K^{2}}{2}\right)-1\right]=\\ =ck_{u}-\frac{ck}{2\gamma_{r}^{2}}\left(1+\frac{K^{2}}{2}\right)\frac{\gamma_ {r}^{2}}{\gamma_{j}^{2}}=ck_{u}-\frac{ck}{2\gamma_{r}^{2}}\left(1+\frac{K^{2}} {2}\right)\frac{1}{(1+\eta_{j})^{2}}=\\ \approxeq ck_{u}-\frac{ck}{2\gamma_{r}^{2}}\left(1+\frac{K^{2}}{2}\right)(1-2 \eta_{j})=ck_{u}-ck_{u}+2ck_{u}\eta_{j}=\\ =2ck_{u}\eta_{j}\end{array} \tag{10.73}\]

The time-derivative of the relative energy deviation is calculated by using Eq. 10.69:

\[\begin{array}{l}\frac{d\eta_{j}}{dt}=\frac{d}{dt}\frac{\gamma_{j}}{\gamma_{ r}}=\frac{1}{\gamma_{r}m_{e}c^{2}}\frac{dE_{j}}{dt}\approxeq-\left(\frac{eE_{x,0}\hat{K}}{2\gamma_{r}^{2}m_{e}c}\right)\cos\psi_{j}=-a\sin\phi_{j}\end{array} \tag{10.74}\]

where the amplitude \(a\) and the phase \(\phi_{j}=\psi_{j}+\pi/2\) were defined. By replacing Eq. 10.74 into the time-derivative of Eq. 10.73, one obtains the "pendulum equation":

\[\left\{\begin{array}{l}\frac{d\psi_{j}}{dt}=2ck_{u}\eta_{j}\\ \Rightarrow\frac{d^{2}\phi_{j}}{dt^{2}}=2ck_{u}\frac{d\eta_{j}}{dt}=-\left( \frac{eE_{x,0}k_{u}\hat{K}}{\gamma_{r}^{2}m_{e}}\right)\sin\phi_{j}\equiv- \Omega^{2}\sin\phi_{j}\\ \frac{d\eta_{j}}{dt}=-a\sin\phi_{j}\end{array}\right. \tag{10.75}\]

In the approximation of small signal, the system behaves as a conservative one, and the Hamiltonian is a constant of motion. Hamilton's equations \(\frac{d\eta}{dt}=-\frac{\partial H}{\partial\phi}\), \(\frac{d\phi}{dt}=\frac{\partial H}{\partial\eta}\) are satisfied by the following Hamiltonian:

\[\begin{array}{l}H(\phi,\eta)=ck_{u}\eta^{2}+\frac{\Omega^{2}}{2ck_{u}}(1- \cos\phi)\end{array} \tag{10.76}\]

The Hamiltonian along the separatrix and the separatrix equation are, respectively:

\[\begin{array}{l}H_{sep}=H(\pm\pi,0)=\frac{\Omega^{2}}{ck_{u}}\\ H(\phi,\eta)\equiv H_{sep}\Rightarrow\eta_{sep}=\pm\frac{\Omega}{ck_{u}}\cos \left(\frac{\phi}{2}\right)\propto\sqrt{E_{x,0}}\end{array} \tag{10.77}\]Figure 10.10 illustrates the phase space trajectories \(\eta(\phi)\) retrieved from Eq. 10.76, where the Hamiltonian is evaluated for different initial conditions (\(\phi_{0}=0\), \(\eta_{0}\neq 0\)). Particles in quadrant \(I\) have \(\eta>0\) and therefore they move forward in \(\phi\). Since their phase is \(0<\phi<\pi\), it turns out \(\frac{d\eta}{dt}<0\), i.e., particles are losing energy (see Eq. 10.75). For energy conservation, their energy is transferred to the radiated field. Similarly, particles in quadrant \(I\,I\) are moving back in phase but they still yield energy to the field. Particles in quadrants \(III\) and \(IV\) are absorbing energy from the field. The area delimited by the separatrix is called "FEL bucket".

##### Discussion:Undulator Momentum Compaction

What is the linear momentum compaction, or \(R_{56}\) transport matrix term, of an undulator made of \(N_{u}\) periods and resonant at the wavelength \(\lambda\)?

According to the definition of momentum compaction in Eq. 15, the longitudinal relative shift of off-momentum particles in a bunch is \(\Delta z=R_{56}\delta\). The shift can be calculated explicitly from Eq. 10.73, in the ultra-relativistic limit and under on-resonance condition:

\[\begin{array}{l}\frac{d\psi}{dt}=2ck_{u}\eta\approxeq 2ck_{u}\delta;\\ \\ \int d\psi=\int_{0}^{t}\omega dt^{\prime}\approxeq-\int_{0}^{z}\frac{\omega}{c }dz^{\prime}\approxeq 2k_{u}\delta\int_{0}^{L_{u}}ds;\\ \\ \Delta z=-2\frac{k_{u}}{k}N_{u}\lambda_{u}\delta=-2N_{u}\lambda\delta\\ \\ \Rightarrow R_{56}=-2N_{u}\lambda=-\frac{L_{u}}{\gamma^{2}}\left(1+\frac{K^{2}} {2}\right)\end{array} \tag{10.78}\]

Figure 10.10: Electron phase space trajectories (black solid) according to Eq. 10.76 in small signal regime, in the range \(\phi\in[-\pi,\pi]\). The separatrix is in dashed red. An initial electron distribution slightly above the resonant energy (dotted light green) evolves towards positive phases (dotted dark green). The opposite happens for electrons initially below resonance (not shown)

This is basically the \(R_{56}\) of a drift section long \(L_{u}=N_{u}\lambda_{u}\) (see Eq. 4.21), but modified by the wiggling trajectory of the particles in the presence of the undulator magnetic field.

#### Low Gain

The amplification of FEL radiation is calculated in terms of energy lost by the electrons. The small-signal approximation [6, 9] implies a small variation of the radiated field amplitude during the interaction with the electrons. In other words, the electrons' synchrotron oscillation period (\(\sim 1/\Omega\)) is assumed to be much longer than the time taken to travel the undulator length \(L_{u}=N_{u}\lambda_{u}\). This allows us to describe the variation with time of the generic particle's coordinates in the framework of a 2nd order perturbation theory, where the perturbation coefficient is \(\varepsilon=(\Omega L_{u}/c)^{2}\ll 1\):

\[\left\{\begin{array}{l}\phi(t)=\phi_{0}(t)+\varepsilon\frac{d\phi_{0}}{dt}dt +\varepsilon^{2}\frac{d^{2}\phi_{0}}{dt^{2}}dt^{2}+o(\varepsilon^{3})\equiv \phi_{0}(t)+\varepsilon\phi_{1}(t)+\varepsilon^{2}\phi_{2}(t)\\ \\ \eta(t)=\eta_{0}+\varepsilon\eta_{1}(t)+\varepsilon^{2}\frac{d\eta_{1}}{dt}dt +o(\varepsilon^{3})\equiv\eta_{0}+\varepsilon\eta_{1}(t)+\varepsilon^{2}\eta _{2}(t)\end{array}\right. \tag{10.79}\]

The 0th order solution assumes constant particle energy, therefore from Eq. 10.75:

\[\eta(t)\approxeq\eta_{0}=const.\Rightarrow\phi_{0}(t)=\int_{0}^{t}\frac{d \phi_{0}}{dt^{\prime}}dt^{\prime}=2ck_{u}\eta_{0}t+\theta_{0}\equiv\xi+\theta_ {0} \tag{10.80}\]

The 1st order solution is:

\[\left\{\begin{array}{l}\dot{\phi}_{1}=2ck_{u}\eta_{1}\\ \\ \dot{\eta}_{1}=-\frac{\Omega^{2}}{2ck_{u}}\sin(\phi_{0})=-\frac{\Omega^{2}}{2ck _{u}}\sin(\xi+\theta_{0})\end{array}\right. \tag{10.81}\]

\[\Rightarrow\left\{\begin{array}{l}\eta_{1}(t)=\int_{0}^{t}\frac{d\eta_{1}}{ dt^{\prime}}dt^{\prime}=\frac{\Omega^{2}}{(2ck_{u})^{2}\eta_{0}}\left[\cos(\xi+ \theta_{0})-\cos(\theta_{0})\right]\\ \\ \phi_{1}(t)=\int_{0}^{t}\frac{d\phi_{1}}{dt^{\prime}}dt^{\prime}=\int_{0}^{t}2ck _{u}\eta_{1}(t^{\prime})dt^{\prime}=\frac{\Omega^{2}}{2ck_{u}\eta_{0}}\left[ \frac{\sin(\xi+\theta_{0})-\sin(\theta_{0})}{2ck_{u}\eta_{0}}-t\cos(\theta_{0} )\right]\end{array}\right. \tag{10.82}\]

The 2nd order solution is:\[\begin{array}{l}\dot{\eta}_{2}=\frac{d}{dt}\left(\frac{d\eta_{1}}{dt}dt\right)= \frac{d\dot{\eta}_{1}}{dt}dt=-\eta_{0}\Omega^{2}\cos(\xi+\theta_{0})dt=\\ =-\left(\frac{1}{2ck_{u}}\frac{d\phi_{0}}{dt}dt\right)\Omega^{2}\cos(\xi+\theta _{0})=-\frac{\Omega^{2}}{2ck_{u}}\phi_{1}\cos(\xi+\theta_{0});\\ \Rightarrow\eta_{2}(t)=-\frac{\Omega^{2}}{2ck_{u}}\int_{0}^{t}dt^{\prime}\phi _{1}(t^{\prime})\cos(\xi(t^{\prime})+\theta_{0})=\\ =-\frac{\Omega^{4}}{(2ck_{u})^{3}\eta_{0}^{2}}\int dt^{\prime}\left[\sin(\xi+ \theta_{0})-\sin(\theta_{0})\right]\cos(\xi+\theta_{0})\\ \quad+\frac{\Omega^{4}}{(2ck_{u})^{2}\eta_{0}}\int dt^{\prime}t^{\prime}\cos( \theta_{0})\cos(\xi+\theta_{0})=\\ =-\frac{\Omega^{4}}{(2ck_{u})^{3}\eta_{0}^{2}}\left\{\frac{1}{4ck_{u}\eta_{0} }\left[\sin(\xi)\sin(\xi+2\theta_{0})\right]-\frac{\sin(\theta_{0})}{2ck_{u} \eta_{0}}\left[\sin(\xi+\theta_{0})-\sin(\theta_{0})\right]\right\}+\\ \quad-\frac{\Omega^{4}}{\eta_{0}}\frac{\cos(\theta_{0})}{(2ck_{u})^{4}\eta_{0 }^{2}}\left[\xi\sin(\xi+\theta_{0})+\cos(\xi+\theta_{0})-\cos(\theta_{0}) \right]=\\ =-\frac{\Omega^{4}}{(2ck_{u})^{4}\eta_{0}^{3}}\left[\frac{1}{2}\sin(\xi)\sin( \xi+\theta_{0})-\sin(\theta_{0})\sin(\xi+\theta_{0})+\sin^{2}(\theta_{0})+ \right.\\ \quad\left.-\xi\,\sin(\xi+\theta_{0})\cos(\theta_{0})-\cos(\theta_{0})\cos(\xi +\theta_{0})+\cos^{2}(\theta_{0})\right]\end{array} \tag{10.83}\]

Finally, we calculate the relative energy variation averaged over all electrons' initial phase \(\theta_{0}\):

\[\begin{array}{l}\langle\eta\rangle_{\theta_{0}}=\frac{1}{2\pi}\int_{0}^{2 \pi}\eta(t;\theta_{0})d\theta_{0}=\eta_{0}+0+\langle\eta_{2}\rangle_{\theta_{ 0}}=\\ =\eta_{0}+\frac{\Omega^{4}}{(2ck_{u})^{4}\eta_{0}^{3}}\left[\frac{\xi}{2}\sin( \xi)+\cos(\xi)-1\right]\end{array} \tag{10.84}\]

The _small-signal gain_ is the amount of energy transferred on average from all electrons to the radiated field, normalized to the resonance energy:

\[G_{ss}(t)=\eta_{0}-\langle\eta\rangle_{\theta_{0}}=-\frac{\Omega^{4}}{(2ck_{u })^{4}\eta_{0}^{3}}\left[\frac{\xi}{2}\sin(\xi)+\cos(\xi)-1\right] \tag{10.85}\]

At the end of the undulator it results \(\xi/2=ck_{u}\eta_{0}\cdot N_{u}\lambda_{u}/c=2\pi\,N_{u}\,\frac{\Delta\gamma} {\gamma_{r}}=\pi\,N_{u}\,\frac{\Delta\omega}{\omega_{r}}\equiv x\) (see also Eq. 10.39). The paremeter \(x\) is called _detuning parameter_, and Eq. 10.85 becomes:

\[\begin{array}{l}G_{ss}(L_{u})=-\frac{\Omega^{4}L_{u}^{3}}{16ck_{u}}\frac{1}{( ck_{u}\eta_{0}L_{u})^{3}}\left[\frac{\xi}{2}\sin(\xi)+\cos(\xi)-1\right]\propto\\ \propto-\frac{1}{x^{3}}\left[x\sin(2x)+\cos(2x)-1\right]=-\frac{d}{dx}sinc^{2 }(x)\end{array} \tag{10.86}\]

Equation 10.86 is called _Madey's theorem_ and it says that, by recalling Eq. 10.39, the small-signal gain is proportional to the negative derivative of the undulator spontaneous emission. Figure 10.11 shows that the gain is positive, i.e. the field is amplified, only if electrons' initial energy is slightly above the resonant energy (see also Fig. 10.10). The gain is maximum for \(\xi\approx 2\).

A monochromatic electron beam above resonance energy evolves as illustrated in Fig. 10.10. As the beam travels along the undulator, electrons shift towards positive phases (quadrants \(I\) and \(II\)), where they continue yielding energy to the radiated field. Electrons within the separatrix are trapped and perform closed orbits in the longitudinal phase space. As the radiated field amplitude increases, the separatrix height increases (see Eq. 10.77), and initially untrapped electrons can be trapped, so contributing to the gain. The radiative process starts being amplified with respect to the undulator spontaneous emission by the build-up of periodic \(\lambda\)-spacing of electron clusters into consecutive FEL buckets (intensity enhancement, see Eq. 10.31). Such a positive feedback instability justifies the naming of _stimulated emission_.

The abrupt growth of the field amplitude and the corresponding electrons' nonlinear dynamics is said "high gain" regime of the FEL. It normally requires a relatively long undulator traversed by a high brightness electron beam, as discussed in the following Section.

#### High Gain

The variation of the transverse electric field amplitude generated by the electron's transverse current density is described by the following 1-D wave equation [4, 8, 9]:

\[\left(\frac{\partial^{2}}{\partial z^{2}}-\frac{1}{c^{2}}\frac{\partial^{2}}{ \partial t^{2}}\right)E_{x}\left(z,t\right)=\mu_{0}\frac{\partial j_{x}}{ \partial t} \tag{10.87}\]

Figure 10.11: In the inset, constant amplitude electric field along 10 undulator periods. The FT of the radiated pulse of finite duration is a \(sinc^{2}\) function, shown in the main plot versus the detuning parameter (black solid). It represents the intensity of undulator spontaneous radiation normalized to the single particle’s emission. The small-signal gain (red dashed) is proportional to its derivative

A solution of the form \(E_{x}(z,t)=E_{x}(z)e^{i(kz-\omega t)}\) is plugged into Eq. 10.87 to find:

\[\left\{\begin{array}{l}\frac{\partial^{2}}{\partial z^{2}}E_{x}(z)e^{i(kz- \omega t)}=\frac{\partial}{\partial z}\left[E_{x}^{\prime}+ikE_{x}(z)\right]e^{ i(kz-\omega t)}=\left(E_{x}^{\prime\prime}+2ikE_{x}^{\prime}-E_{x}k^{2}\right)e^{ i(kz-\omega t)}\\ -\frac{1}{c^{2}}\frac{\partial^{2}}{\partial t^{2}}\left[E_{x}(z)e^{i(kz- \omega t)}\right]=\frac{\omega^{2}}{c^{2}}E_{x}(z)e^{i(kz-\omega t)}\end{array}\right. \tag{10.88}\]

Equation 10.88 can be simplified by virtue of the following assumptions.

* The field amplitude varies slowly, i.e., \(E_{x}^{\prime\prime}(z)\ll kE_{x}^{\prime}(z)\) or \(\frac{dE_{x}(z)}{dz}\ll\frac{E_{x}(z)}{\lambda}\) ("slow variable field approximation" or SVEA).
* The radiation slippage \(N_{u}\lambda\) is much shorter than the bunch length. Therefore, the wave is superimposed to the ultra-relativistic electrons along the whole undulator, through which the dispersion relation for the on-resonance wave is preserved, \(k^{2}-\frac{\omega^{2}}{c^{2}}=0\).
* In the ultra-relativistic limit, the ratio of transverse and longitudinal current density results \(j_{x}/j_{z}=v_{x}/v_{z}\approxeq v_{x}/c=\beta_{x}=\frac{\hat{K}}{\gamma}\cos (k_{u}z)\) (see Eq. 10.32, and the generalized undulator parameter for \(v_{z}\neq const\) in Eq. 10.69). Since \(v_{z}\) has a DC and an AC component (see Eq. 10.33), we also have \(j_{z}=j_{0}+j_{1}(z)\). In particular, the electron-wave synchronization leading to the resonance condition implies that \(j_{1}\) oscillates with the electron's ponderomotive phase introduced in Eq. 10.71. In conclusion, \(j_{x}=j_{z}\frac{\hat{K}}{\gamma}\cos(k_{u}z)=\frac{\hat{K}}{\gamma}\left[j_{0 }+j_{1}(\psi)e^{i[(k+k_{u})z-\omega t]}\right]e^{ik_{u}z}\) in Euler's notation.

With the aforementioned approximations applied to Eq. 10.88, Eq. 10.87 becomes:

\[\left[E_{x}^{\prime\prime}(z)+2ikE_{x}^{\prime}(z)+\left(\frac{ \omega^{2}}{c^{2}}-k^{2}\right)E_{x}(z)\right]e^{i(kz-\omega t)}\approx 2ikE_{x} ^{\prime}(z)e^{i(kz-\omega t)}=\] \[=\mu_{0}j_{1}(\psi)\frac{\hat{K}}{\gamma}\frac{\partial}{ \partial t}e^{i[(k+2k_{u})z-\omega t]}=-i\omega\mu_{0}j_{1}(\psi)\frac{\hat{K} }{\gamma}e^{i[(k+2k_{u})z-\omega t]};\] \[E_{x}^{\prime}(z)=-\mu_{0}j_{1}\frac{\omega}{k}\frac{\hat{K}}{ \hat{2}\gamma}e^{i[(k+2k_{u})z-\omega t]}e^{-i(kz-\omega t)}=-\mu_{0}cj_{1} \frac{\hat{K}}{2\gamma}e^{i2k_{u}z}\approx-\frac{\hat{K}}{4\varepsilon_{0}c} \frac{\langle j_{1}\rangle}{\langle\gamma\rangle}, \tag{10.89}\]

and the very last approximated equality is after averaging over many undulator periods, over all beam particles' phases.

The electron current density \(\langle j_{1}\rangle\) in Eq. 10.89 is the bunch peak current averaged over the transverse beam sizes; it is proportional to the correlation of the phases of \(N\) electrons in a bunch:

\[\langle j_{1}\rangle=-\frac{I}{2\pi\sigma_{\perp}^{2}}\frac{1}{N}\sum_{n=1}^{ N}e^{-i\psi_{n}} \tag{10.90}\]and therefore proportional to the FT of the current distribution or _bunching factor_\(b=|\langle e^{-i\psi}\rangle|\). This is 0 for randomly phased electrons, and 1 for phase separation between two electrons multiple of \(2\pi\).

It is convenient to re-write Eq. 10.89 in terms of an electric field amplitude normalized to the average current density. Doing so, the oscillation frequency in the FEL bucket for the pendulum equation in Eq. 10.75 results proportional to the "plasma frequency", \(\Omega^{2}\equiv(4ck_{u})^{2}A\rho^{3}\propto\omega_{p}^{2}\), where we introduce:

\[\left\{\begin{array}{l}\omega_{p}^{2}=\frac{n_{e}e^{2}}{m_{e}\varepsilon_{0 }}=\frac{N_{e}e}{2\pi\sigma_{\perp}^{2}c\sigma_{t}}\frac{e}{m_{e}\varepsilon_ {0}}=\frac{I}{I_{A}}\frac{2c^{2}}{\sigma_{\perp}^{2}}\\ A=\varepsilon_{0}E_{x}(z)\frac{\langle\gamma\rangle}{\hat{K}}\frac{ck_{u}}{2 \pi\sigma_{\perp}^{2}}\\ \\ \rho=\frac{1}{2\langle\gamma\rangle}\left(\frac{I}{I_{A}}\right)^{\frac{1}{3} }\left(\frac{\hat{K}}{k_{u}\sigma_{\perp}}\right)^{\frac{2}{3}}=\frac{1}{ \langle\gamma\rangle}\left(\frac{\hat{K}\omega_{p}}{4ck_{u}}\right)^{\frac{2 }{3}}\end{array}\right. \tag{10.91}\]

and \(I_{A}=4\pi\,\varepsilon_{0}m_{e}c^{3}/e=17045\) A is the Alfven current.

The so-called _FEL parameter_ or "Pierce's parameter" \(\rho\) is proportional, though weakly, to the electron beam brightness. For typical electron beam and undulator parameters for x-ray FELs like \(\gamma\sim 1-10\) GeV, \(I\sim kA\), \(\sigma_{\perp}\sim 10-100\) \(\mu\)m, \(\hat{K}\sim 1\), \(\lambda_{u}\sim 1-5\) cm, we find \(\rho\sim 0.01\%-0.1\%\) (still, some FEL designs may target one order of magnitude smaller or larger \(\rho\)).

Finally, the 1-D coupled Newton-Lorentz's equations of motion for the electrons and Maxwell's equation for the electric field can be written in the following compact form:

\[\left\{\begin{array}{l}\frac{d\psi}{dz}=2k_{u}\eta\\ \\ \frac{d\eta}{dz}=-\frac{\Omega^{2}}{2c^{2}k_{u}}\cos\psi\,=-8k_{u}\rho^{3}A \cos\psi\\ \\ \frac{dA}{dz}=k_{u}\langle e^{-i\psi}\rangle\end{array}\right. \tag{10.92}\]

The normalized field amplitude is derived further to get a third order differential equation:

\[\begin{array}{l}\frac{d^{3}A}{dz^{3}}=\frac{d}{dz}\left(-2ik_{u}^{2}\langle \eta e^{-i\psi}\rangle\right)=-2ik_{u}^{2}\left(\langle\frac{d\eta}{dz}e^{-i \psi}\rangle-i\langle\eta e^{-i\psi}\frac{d\psi}{dz}\rangle\right)=\\ \\ =i16k_{u}^{3}\rho^{3}A\langle e^{-i\psi}\,\cos\psi\rangle-4k_{u}^{3}\langle \eta^{2}e^{-i\psi}\rangle\end{array} \tag{10.93}\]

For electrons' energy on resonance, the second term on the r.h.s. of Eq. 10.93 is null, \(\langle\gamma\rangle=\gamma_{r}\Rightarrow\eta=0\). The first term averaged over all phases reduces to \(i8k_{u}^{3}\rho^{3}\). We then adopt the short notation \(\Gamma=2k_{u}\rho\), and search a solution of the form \(E_{x}(z)=E_{x,0}e^{\alpha z}\):\[E_{x}^{\prime\prime\prime}-i\,\Gamma^{3}E_{x}=0\Rightarrow\alpha^{3}=i\,\Gamma^{3 }\Rightarrow\alpha_{j}=\left\{\begin{array}{l}\frac{i+\sqrt{3}}{2}\,\Gamma\\ \frac{i-\sqrt{3}}{2}\,\Gamma\\ -i\,\Gamma\end{array}\right. \tag{10.94}\]

The three roots of the cubic equation describe, respectively, a growing, a decaying, and an oscillatory mode of the field amplitude. At the beginning of the undulator, the three modes compete with one another, i.e., the total electric field amplitude grows slowly with z ("lethargy"). For \(z\geq 1/\,\Gamma\), the exponentially growing mode (\(\alpha_{1}\)) dominates and the field intensity can be written as:

\[\begin{array}{l}I=\frac{|E_{x}(z)|^{2}}{2Z_{0}}=\frac{1}{2Z_{0}}\left(\frac{ E_{x,ln}}{3}\right)^{2}\cdot(e^{\alpha_{1}z}+e^{\alpha_{2}z}+e^{\alpha_{3}z})^{2} \approx\frac{1}{9}\,\frac{|E_{x,ln}|^{2}}{2Z_{0}}e^{\sqrt{3}\Gamma z}\\ \\ \Rightarrow P(z)\approx\frac{P_{0}}{9}e^{\frac{z}{2z}}\,,\,L_{g}=\frac{1}{ \sqrt{3}\Gamma}=\frac{\lambda_{u}}{4\pi\sqrt{3}\rho}\end{array} \tag{10.95}\]

Equation 10.94 points out that, unlike the small-signal regime, the high gain FEL is optimized by an ideally monochromatic electron beam at the resonant energy \(\langle\gamma\rangle=\gamma_{r}\). It can be shown that this applies also in the presence of detuning as long as \(\rho\ll 1\), which is the common regime of operation of short wavelength FELs. The characteristic length \(L_{g}\) is said _1-D gain length_, and the regime is said "high gain" because the radiated energy at the exit of a long undulator (\(z\gg 1/\,\Gamma\)) is orders of magnitude larger than that of the initial undulator spontaneous emission (\(z\leq 1/\,\Gamma\)).

#### Pierce's Parameter

The high gain FEL modelled by Eq. 10.95 is named _Self-Amplified Spontaneous Emission_ (SASE) because the equation describes the amplification of undulator spontaneous radiation emitted in the first segments of the undulator line (an estimate for the initial power \(P_{0}\) will follow). The model assumes that each new bunch of electrons accelerated in a single-pass linac reaches the undulator with a different, randomly distributed configuration of phases (see \(\langle j_{1}\rangle\) in Eq. 10.90). Hence, a statistical characterization of the FEL spectral intensity can be carried out, where many intense light pulses, each light pulse radiated by a large number of electrons, are considered.

The longitudinal coherence length (sometimes also _cooperation length_) of a SASE FEL is determined by the average radiation slippage over the characteristic scale of the FEL power growth, which is the gain length \(L_{g}\). In this sense, the slippage identifies the bunch slice through which the field phase is approximately constant:

\[l_{coh}\approx\frac{N_{u}\lambda}{L_{u}/L_{g}}\approx\frac{\lambda}{4\pi\rho} \tag{10.96}\]For example, if a bunch long \(\Delta z_{b}=30\)\(\upmu\)m is lasing at \(\lambda=1\) nm with \(\rho=0.1\%\), we expect on average \(\Delta z_{b}/l_{coh}\approx 400\) spikes in a pulse, each spike representing a longitudinally coherent mode. The 400 modes, however, are mutually incoherent. It follows that when \(\Delta z_{b}\leq l_{coh}\), a single longitudinal mode (single narrow spectral line) can be produced.

By virtue of the definition of longitudinal coherence length in Eq. 10.24, we find:

\[l_{coh}\approx\tfrac{\lambda}{4\pi\rho}=\tfrac{1}{2\pi}\tfrac{\lambda^{2}}{2 \Delta\lambda}\Rightarrow\rho\approx\tfrac{\Delta\lambda}{\lambda} \tag{10.97}\]

that is, the FEL parameter quantifies the FEL rms intrinsic relative bandwidth. Since the relative spectral width of each SASE spike is of the order of \(\lambda/\Delta z\), the average number of spikes in the spectrum will be \(\rho\Delta z/\lambda\approx 30\).

The resonance condition says that the FEL relative bandwidth is proportional to the electron beam's relative energy spread, since off-resonance electrons will tend to emit radiation at slightly different wavelengths. Hence, the intrinsic bandwidth is preserved as long as it is much larger than the relative energy spread evaluated on the scale of the cooperation length:

\[\tfrac{\Delta\lambda}{\lambda}=2\tfrac{\Delta\gamma}{\gamma}\approx\sigma_{ \delta}\Rightarrow\sigma_{\delta}<\tfrac{\Delta\lambda}{\lambda}\approx\rho, \tag{10.98}\]

This defines \(\rho\) as the upper limit to the beam rms _uncorrelated_ relative energy spread for not degrading the FEL brilliance.

As a matter of fact, a small energy spread ensures efficient electrons trapping into the FEL bucket (see Fig. 10.10). But, since FEL emission implies energy modulation of the trapped electrons, the beam's energy spread grows along the undulator, till electrons escape from the bucket (de-bunching), or they are too far from the resonance condition to contribute substantially to the gain. At this point, the amplification of the radiated field stops. Since the amount of energy transferred from the electrons to the radiated field is of the order of the energy modulation amplitude, the _saturation power_ level is:

\[P_{sat}\approx\tfrac{I\Delta E_{sat}}{e}\approx\tfrac{I(E)}{e}\sigma_{\delta, sat}\approx\rho\,P_{b} \tag{10.99}\]

and \(P_{b}\) is just the electron beam power. In conclusion, \(\rho\) is also the efficiency of energy transfer of the FEL process.

The initial radiation power level \(P_{0}\) introduced in Eq. 10.95 is estimated as follows. We observe that a fraction \(\rho\) of the electron beam power \(P_{b}\) is converted into radiation. Since the initial emission is governed by the initial bunching factor \(b_{0}\), it has to be \(P\sim\left|b_{0}\right|^{2}\). Let us define \(\left|b_{0}\right|^{2}\) for a Poisson-like electron distribution within the FEL spectral bandwidth \(\Delta v\). We get:

\[\left|b_{0}\right|^{2}=\tfrac{2e}{T}\Delta v=\tfrac{2ec}{T}\tfrac{\Delta \lambda}{\lambda^{2}}\approx 2ec\tfrac{\rho}{\lambda}\tfrac{\rho E}{eP_{sat}} \Rightarrow P_{0}\approx\rho\,P_{b}\left|b_{0}\right|^{2}\approx 2c\rho^{2} \,E/\lambda. \tag{10.100}\]

Short wavelength SASE FELs are commonly characterized by shot-noise and saturation power of the order of \(P_{0}\sim 10-100\) W and \(P_{sat}\sim 1-10\) GW, respectively.

At the saturation point, electrons have completed one-half of the synchrotron period in the FEL bucket. After that, they start absorbing energy from radiation. But, since the motion becomes chaotic, the FEL power oscillates around the saturation power level (assuming an undulator longer than the saturation length). The saturation length can be estimated from Eqs. 10.95 and 10.100 as \(L_{sat}\approx L_{g}\ln(9\frac{I}{2ec}\frac{\lambda}{\rho})\). In EUV and x-ray SASE FELs, the power saturation length is typically in the range 15-22 \(L_{g}\).

In summary, FEL emission in high gain regime is so intense because it is coherent in the sense of Eq. 10.31. Electrons are first modulated in energy by the radiated field (\(d\eta/dz\)). The undulator momentum compaction (\(R_{56}=2N_{u}\lambda\)) shifts the electrons back and forth depending on the sign of their relative energy deviation, so that they distribute in clusters inside the FEL bucket with periodic spacing of \(\lambda\) (\(d\psi/dz\)). The sharper the electron distribution is in the FEL bucket, the larger the bunching is, the larger the field amplitude becomes (\(dA/dz\)). The FEL bucket grows in height along the undulator by virtue of the growing field amplitude, and it starts trapping more and more electrons. The process stops when the electron beam uncorrelated energy spread is so enlarged by the FEL instability to prevent any additional net energy transfer.

#### Electron Beam Quality

The high gain regime of FELs requires a persistent interaction of electrons and radiated field along the whole undulator line. In the transverse plane, this is guaranteed by (i) negligible light diffraction along the characteristic length of power growth, or \(L_{g}<L_{R}=4\pi\sigma_{r}^{2}/\lambda\), with \(L_{R}\) the Rayleigh length, and by (ii) matched spot sizes of the two beams. The relative phase slippage due to electrons' betatron motion and energy spread could lead to loss of synchronism, which is therefore minimized by (iii) small electron beam's angular divergence, and (iv) small relative energy spread.

Conditions (ii)-(iii) translate into an electron beam ideally below the diffraction limit, \(4\pi\varepsilon_{x,\,y}\leq\lambda\). This determines in turn a high degree of transverse coherence (on-axis Gaussian mode), as suggested by Eq. 10.20. For an electron beam at the diffraction limit, condition (i) implies \(L_{g}<\langle\beta_{x,\,y}\rangle\), the latter being the average betatron function along the undulator, assuming smooth optics.

In summary, light diffraction, electron beam's transverse emittance and energy spread (so-called "3-D effects"), all contribute to a reduction of the lasing efficiency. This can be quantified by means of an effective 3-D gain length \(L_{g,3D}\approx L_{g,1D}(1+\chi)\), where typically \(\chi\sim 0.1-0.3\). At this point, one could note that the transverse momentum spread due to betatron motion scales as \(\Delta p_{\perp}\sim\sqrt{\varepsilon/\beta}\), whereas the transverse beam size scales as \(\sigma_{\perp}\sim\sqrt{\varepsilon\beta}\). These opposing scaling laws suggest there is some optimal value of \(\beta\) that, on top of light diffraction, will minimize \(L_{g,3D}\) for a given set of radiation and electron beam parameters.

#### Discussion: Brilliance of FELs and Synchrotron Light Sources

The peak brilliance of high gain x-ray FELs is several orders of magnitude larger than that of undulator emission at synchrotron light sources. Roughly speaking, this can be attributed to coherent emission in FELs during the exponential power growth, with respect to the incoherent emission in a short undulator (compare Eqs. 10.46 and 10.95).

We would like to quantify the ratio of peak brilliance at the two light sources in a more rigorous way, considering the undulator spontaneous radiation at the source point and in the presence of monochromatization. Typical parameters for emission in soft x-rays are considered, such as bunch charges \(Q_{fel}/Q_{sr}=0.1nC/1nC\), \(\rho=10^{-3}\), \(N_{u}=100\) in the short undulator, resolving power \(RP=10^{4}\) and transmission efficiency \(TR=1\%\) through the monochromator for the undulator spontaneous radiation. We assume same central wavelength in soft x-rays, similar spot size and angular divergence of the light pulse at the source. This allows us to reduce the brilliance ratio to the ratio of the spectral power.

With the short notation \(b=d\lambda/\lambda\), the ratio of peak brilliance _at the source_ (e.g., undulator exit) and after _monochromatization_ of the spontaneous emission, is:

\[\left.\begin{array}{c}\frac{\hat{B}_{fel}}{\hat{B}_{sr}}\right|_{source} \approx\left.\frac{(d\hat{P}/db)_{fel}}{(d\hat{P}/db)_{sr}}\approx\frac{(N_{b }^{2}/b)_{fel}}{(N_{b}/b)_{sr}}\approx\frac{N_{b,fel}/\rho}{10^{2}N_{u}} \approx 10^{8}\\ \\ \left.\frac{\hat{B}_{fel}}{\hat{B}_{sr}}\right|_{mono}\approx\frac{(d\hat{P}/ db)_{fel}}{(d\hat{P}/db)_{mono}}\approx\frac{N_{b,fel}/\rho}{10^{2}TR\cdot RP} \approx 10^{8}\end{array}\right. \tag{10.101}\]

where we used \(b_{fel}\approx\rho\) and \(b_{sr}\approx 1/N_{u}\). Equation 10.101 points out that some larger degree of longitudinal coherence of the spontaneous emission is obtained through monochromatization at the expense of a lower flux.

An explicit and more rigorous calculation can be done by means of Eq. 10.46 for undulator spontaneous emission (re-written by means of the resonance condition), and of Eq. 10.99 for FEL emission at saturation. Doing so, we assume a consistent set of parameters for the FEL such as \(E=2\) GeV and \(K\approx 1\) to get \(\lambda\approx 1\) nm. The ratio of linac-to-ring peak current is of the order of \(I_{lin}/I_{sr}\approx 1000\) A / 10 A:

\[\frac{\hat{B}_{fel}}{\hat{B}_{sr}}\approx\frac{(d\hat{P}/db)_{fel}}{(d\hat{P}/ db)_{sr}}\approx\frac{\rho I_{lin}E}{\epsilon\rho}\frac{3\epsilon_{0}}{ \pi\epsilon}\frac{\lambda}{I_{sr}N_{u}K^{2}}\approx 10^{8} \tag{10.102}\]

In conclusion, the large increase of peak brilliance in a high gain FEL compared to other x-ray light sources has to be attributed to coherent emission, intended as intensity enhancement, _and_ to a high degree of longitudinal coherence. One can notice, however, that the FEL power _at saturation_ shows a weaker dependence from the total number of particles than in fully coherent emission: \(P_{sat}\propto\rho I\propto N_{b}^{4/3}\). This reflects the loss of longitudinal coherence, or "debunching", associated to synchrotron oscillations of electrons in the FEL bucket. Namely, the maximum power is reached at the expense of some reduced longitudinal coherence.

Equation 10.102 is now revised to estimate the ratio of the _average_ brilliance. This is simply done by multiplying the above expression by the ratio of FEL-to-storage ring bunch duration, and FEL-to-storage ring repetition rate. Repetition frequencies of high brightness linacs span \(10^{2}\)-\(10^{6}\) Hz, while medium size, multi-GeV-energy storage ring light sources have harmonic number \(h\sim 500\) and revolution frequency of the order of 1 MHz or so. We find:

\[\frac{\langle B_{fel}\rangle}{\langle B_{sr}\rangle}\approx\frac{\hat{B}_{fel}}{ \hat{B}_{sr}}\cdot\frac{0.1ps}{100ps}\cdot\frac{(10^{2}-10^{6})Hz}{10^{9}Hz} \approx 10^{-2}-10^{2}, \tag{10.103}\]

with the lower limit for NC linacs, the upper limit for SC ones.

#### 10.5.6.2 Discussion: Is a Free-Electron Laser a Laser?

The FEL acronym points out that, on the one hand, the electrons are unbounded or "free" from atomic levels. On the other hand, it suggests a similarity with a conventional atomic laser by virtue of the large degree of transverse and longitudinal coherence accompanying lasing in the high gain regime. Nevertheless, the coherence of an atomic laser is properly described in terms of \(quantistic\) states of light, while the coherence of an FEL has so far been characterized only through Eqs. 10.9 and 10.14 in a \(classical\) formalism.

With the due aforementioned differences in mind, we may establish, with some imagination, a similitude between the process of electrons' manipulation and emission of radiation in an atomic laser, and in a high gain FEL, as sketched by the 4-step process in Table 10.1.

\begin{table}
\begin{tabular}{p{142.3pt}|p{113.8pt}|p{113.8pt}} \hline Step & Atomic Laser & Free-Electron Laser \\ \hline
1 & Stable configuration of atomic levels & Non-relativistic photo-electrons \\ \hline
2 & Inversion of electron population between two atomic levels by means of external energy pumping & Energy increase to GeV-level by means of RF acceleration \\ \hline
3 & Spontaneous emission of radiation by electrons migrating from upper to lower atomic level & Undulator spontaneous emission in the first undulator segments \\ \hline
4 & Stimulated radiation emission, nonlinearly amplified in intensity, narrow-band, and highly collimated in the direction of the seeding. An optical cavity narrows the spectral bandwidth & Radiation emission stimulated by spontaneous emission (or external laser in seeded-FELs), exponentially amplified in intensity, highly collimated. The long resonant undulator selects the central frequency, the spectral bandwidth narrows \\ \hline \end{tabular}
\end{table}
Table 10.1: Process leading to lasing in atomic lasers and in high gain FELs

## References

* [1] A. Balerna, S. Mobilio, Introduction to synchrotron radiation, in _Synchrotron Radiation: Basics, Methods and Applications_, ed. by S. Mobilio, F. Boscherini, C. Meneghini (Published by Springer, Berlin, Heidelberg, 2015), pp. 3-28. ISBN: 978-3-642-55314-1
* [2] D. Attwood, A. Sakdinawat, _X-rays and Extreme Ultraviolet Radiation_ (Published by Cambridge University Press, 2016), pp. 110-147, 227-278. ISBN: 9781107477629
* [3] P.A. Millette, The Heisenberg Uncertainty Principle and the Nyquist-Shannon Sampling Theorem. Progress Phys. **3**, 9-14 (2013)
* [4] M.R. Howells, B.M. Kincaid, The properties of undulator radiation, LBL-34751, in _Proceedings of the NATO Advanced Study Institute_, Maratea, Italy, ed. by A.S. Schlacher, F.J. Wuilleumier (1992)
* [5] R.P. Walker, Insertion devices: undulators and wigglers, in _Proceedings of CERN Accelerator School: Synchrotron Radiation and Free Electron Lasers_, Geneva, Switzerland. CERN 98-04, ed. by S. Turner (1998), pp. 129-190
* [6] K.-J. Kim, Z. Huang, R. Lindberg, _Synchrotron Radiation and Free-Electron Lasers: Principles of Coherent X-ray Generation_ (Published by Cambridge University Press, 2017), pp. 74-138. ISBN: 9781316677377
* [7] V. Berestetskii, E. Lifshitz, L. Pitaevskii, Quantum electrodynamics, in _Landau and Lifshitz, Course of Theoretical Physics_, vol. 4 (Published by Pergamon Press, New York, 1982)
* [8] C. Pellegrini, The history of X-ray free-electron lasers. Eur. Phys. J. H **37**, 659-708 (2012)
* [9] G. Dattoli, E. Di Palma, S. Pagnutti, E. Sabia, Free electron coherent sources: from microwave to X-rays. Phys. Rep. **739**, 1-51 (2018)

