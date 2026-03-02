## Colliders

In high energy colliders, two counter-propagating beams collide to produce new particles ("events"), aiming at investigate the nuclear and sub-nuclear structure of matter, verify theoretical particle models, increase the experimental precision of particles' parameters, etc. This Chapter treats the physics of beam-beam collision by introducing the concept of luminosity in a circular collider, strategies to maximize it, and limitations related to beam stability. It is complemented by considerations on linear colliders.

### Luminosity

If \(n_{1},\,n_{2}\) are the number of particles involved in a two-beam collision, the number of events produced in the unit of time is expected to be proportional to the beams' population, \(\dot{n}_{3}(t)\propto n_{1}\cdot n_{2}\). Moreover, the closer the two beams are one another at the interaction point (IP)--namely, the smaller the effective interaction area \(A\) is--, the higher the probability of their interaction will be. Hence, the event rate shall be proportional to the convolution of the beams' charge densities, \(\dot{n}_{3}(t)\propto\rho_{1}\rho_{2}\sim\frac{n_{1}n_{2}}{A}\).

In general, a specific class of events could only be produced with a certain probability among several "production channels" permitted by the beams interaction. In particle physics, such a probability is expressed via the _cross-section_ of the interaction, commonly in units of cm\({}^{2}\), and thereby \(\dot{n}_{3}(t)\propto\sigma\). While \(n_{1},\,n_{2},\,A\) are all features of the accelerator system, \(\sigma\) is independent from the way the interaction is put in place, i.e., it depends only from the physics of the particles' interaction.

In summary, the production rate of specific events can be expressed as the product of quantities related to the accelerator and the event cross-section [1]:

\[\frac{dn_{3}}{dt}=L(t)\sigma\;\;\Rightarrow\;\;n_{3}=\int_{t_{1}}^{t_{2}}\frac {dn_{3}}{dt}dt=\int_{t_{1}}^{t_{2}}\sigma\,L(t)dt\equiv\sigma\mathbb{L}(\Delta t) \tag{11.1}\]

[MISSING_PAGE_FAIL:263]

Equation 11.3 demonstrates the importance of a small beam geometric emittance to maximize the luminosity. Specially strong quadrupole magnets are commonly arranged in proximity of the IP to squeeze the betatron functions. Their high gradients become responsible of a large part of the storage ring chromaticity (see Eq. 6.29).

#### Discussion: Lifetime, Run Time and Preparation Time

According to Eq. 11.1, the instantaneous luminosity is constant as long as the number of particles per bunch remains constant over time. In reality, the beam current in storage rings decays with time: the beam lifetime, or \(\tau\) (see e.g. Eq. 8.65), can be at the level of a fraction of hour to several hours. If the run time interval \(t_{r}\) starts only after a beam time preparation long \(t_{p}\), what is the most convenient combination of time intervals which maximizes the integrated luminosity?

Assuming an exponential decay \(L(t)=L_{0}e^{-t/\tau}\), the average integrated luminosity available for data collection is [1]:

\[<\mathbb{L}>=\frac{\int_{0}^{t_{r}}L(t)dt}{t_{r}+t_{p}}=L_{0}\cdot\tau\,\frac{ 1-e^{-t_{r}/\tau}}{t_{r}+t_{p}} \tag{11.4}\]

The quantity \((\mathbb{L})\) is therefore maximized as function of the run time by:

\[\begin{array}{l}\frac{d<\mathbb{L}>}{dt_{r}}\equiv 0\ \ \Rightarrow\ \ \frac{L_{0}\tau}{(t_{r}+t_{p})^{2}}\left[\frac{1}{\tau}(t_{r}+t_{p})e^{-\frac {t_{r}}{\tau}}\,-(1-e^{-\frac{t_{r}}{\tau}})\right]=0;\\ t_{r}=\tau\,\ln\left(1+\frac{t_{r}+t_{p}}{\tau}\right)\equiv f(t_{r})\end{array} \tag{11.5}\]

Figure 11.1 illustrates the solution of Eq. 11.5 for three values of the ratio \(t_{p}/\tau\). The maximum of \(<\mathbb{L}>\) is obtained for \(t_{p}<\tau\). Indeed, this goes in the direction of the ideal case of very short preparation time and extremely long lifetime.

### Crossing Angle

Circular colliders usually adopt a non-zero crossing angle at the IP to avoid unwanted collisions between bunches of the two counter-propagating trains, which could dilute the charge density over time, generate spurious events, etc. The immediate geometric effect, however, is that of reducing the luminosity by virtue of an effective larger interaction area [1, 2].

We discriminate two contributions to the effective area. The first one is shown in Fig. 11.2-left plot: the projection of the longitudinal size of beam-2 (\(\sigma_{z,2}\)) onto the axis orthogonal to the motion of beam-1 (\(r_{1}\)) adds in quadrature to the natural size of beam-1 (\(\sigma_{x,1}\)). The second one is shown in Fig. 11.2-right plot: the projection of the horizontal size of beam-2 (\(\sigma_{x,2}\)) onto the longitudinal axis of the same beam (\(r_{2}\)) is seen by beam-1 as a longer interaction interval, i.e., it adds in quadrature to the natural length of beam-2 (\(\sigma_{z,2}\)).

In short, the two terms modify the instantaneous luminosity in Eq. 11.2 with an effective enlargement of the horizontal beam size and bunch lengthening. The latter effect is usually negligible with respect to the former one because strong focusing generates "pencil beams" at the IP, i.e., \(\sigma_{x}\ll\sigma_{z}\):

\[\sigma_{x}\rightarrow\sqrt{\sigma_{x}^{2}+\sigma_{z}^{2}\tan^{2 }\frac{\phi}{2}}\cdot\frac{\sqrt{\sigma_{z}^{2}+\sigma_{x}^{2}\tan^{2}\frac{ \phi}{2}}}{\sigma_{z}}=\] \[=\sigma_{x}\sqrt{1+\left(\frac{\sigma_{z}}{\sigma_{x}}\right)^{2 }\tan^{2}\frac{\phi}{2}}\sqrt{1+\left(\frac{\sigma_{x}}{\sigma_{z}}\right)^{2 }\tan^{2}\frac{\phi}{2}}\equiv\frac{\sigma_{x}}{S}; \tag{11.6}\] \[\Rightarrow L=\frac{n_{1}n_{2}}{4\pi\sigma_{x}\sigma_{y}}n_{b}f_{ ip}S,\ \ \ \ \ \ S(\sigma_{x}\ll\sigma_{z})\approx\frac{1}{\sqrt{1+\left(\frac{\sigma_{z}}{ \sigma_{x}}\right)^{2}\tan^{2}\frac{\phi}{2}}}\]

By definition, \(S\leq 1\ \forall\phi\geq 0\). For example, for beams stored to produce 7 TeV invariant mass in LHC, \(\phi\approx 0.3\) m rad, \(\sigma_{x}\approx 20\)m and \(\sigma_{z}\approx 8\) cm, giving \(S\approx 0.85\).

Figure 11.2: Beam-1 (\(b_{1}\)) and beam-2 (\(b_{2}\)) interacting at a total crossing angle \(\phi\) (angles and beams’ transverse and longitudinal sizes are sketched for illustrative purpose only). The longitudinal (left) and transverse size (right) of beam-2 is seen by beam-1 as an additional contribution to the transverse interaction area (\(\sigma_{z,2}\) and \(\sigma_{x,2}\) in bold black is projected onto the red segment)

### 11.3 Hourglass Effect

So far, the luminosity was defined assuming that the betatron functions at the IP are constant along the interaction region. The length of the effective interaction region is the bunch duration, which can span from tens' to few hundreds' of millimeters. In reality, since the IP is a drift section internal to a particle detector and, as said, the betatron functions have minima at the IP, the section is a low-\(\beta\) insertion, and \(\beta(s)\) varies according to Eq. 4.138 [1, 2]. It is easy to see that, for example, \(\beta^{*}\approx 3\) cm corresponds to a 2-fold larger \(\beta\) at the edges of an interaction region long 3 cm. Owing to Eq. 11.3, such growth of \(\beta(s)\) is expected to reduce the luminosity. Indeed, we show below that the effect becomes important for small \(\beta^{*}\) in the presence of long bunches, i.e., in proportion to the ratio \(\sigma_{s}/\beta^{*}\). In the literature, the enlargement of the beam size in proximity of a waist is called _hourglass effect_, by virtue of the shape of the beam envelope around the waist (see Fig. 4.15).

For simplicity, symmetric beams at the interaction region are considered. The variation of the beam size along the interaction region is:

\[\sigma_{u}^{*}=\sqrt{\epsilon_{u}\beta_{u}^{*}}\rightarrow\sigma_{u}=\sqrt{ \epsilon_{u}\beta_{u}^{*}\left(1+\frac{s^{2}}{\beta_{u}^{*2}}\right)}=\sigma_ {u}^{*}\sqrt{1+\left(\frac{s}{\beta_{u}^{*}}\right)^{2}} \tag{11.7}\]

We re-define for brevity \(n_{d}=2n_{1}n_{2}n_{b}f_{ip}\), and we introduce \(w_{u}(s)=s/\beta_{u}^{*}\). Then, the luminosity is calculated by integrating first in \(dx\), \(dy\), then in \(ds_{0}\) and finally in \(ds\):

\[L=\frac{n_{d}}{(\sqrt{2\pi})^{6}\sigma_{x}^{2}\sigma_{y}^{2}\sigma_{x}^{2}} \int\int\int\int e^{-\frac{s^{2}}{\sigma_{x}^{2}}e^{-\frac{y^{2}}{\sigma_{y}^ {2}}}e^{-\frac{s^{2}}{\sigma_{y}^{2}}}e^{-\frac{y^{2}}{\sigma_{y}^{2}}}}dxdydsds _{0}=\]

\[=\frac{n_{d}}{(\sqrt{2\pi})^{6}\sigma_{x}^{*2}\sigma_{y}^{*2}\sigma_{x}^{2}} \int\int\int\int\frac{e^{-\frac{s^{2}}{\sigma_{x}^{2}[1+w_{x}(s)^{2}]}}e^{- \frac{y^{2}}{\sigma_{y}^{2}[1+w_{y}(s)^{2}]}}e^{-\frac{s^{2}}{\sigma_{x}^{2}} }e^{-\frac{y_{0}^{2}}{\sigma_{x}^{2}}}dxdydsds_{0}=\]

\[=\frac{n_{d}\pi^{3/2}\sigma_{x}^{*}\sigma_{y}^{*}\sigma_{s}}{(\sqrt{2\pi})^{6 }\sigma_{x}^{*2}\sigma_{y}^{*2}\sigma_{x}^{2}}\int\frac{e^{-\frac{s^{2}}{ \sigma_{y}^{2}}}}{\sqrt{1+w_{x}(s)^{2}}\sqrt{1+w_{y}(s)^{2}}}ds=...\left[\frac {s}{\sigma_{s}}\rightarrow\,\xi\right]...\]

\[=\frac{n_{d}\pi^{3/2}}{(\sqrt{2\pi})^{6}\sigma_{x}^{*}\sigma_{y}^{*}\sigma_{s} }\int\frac{e^{-\xi^{2}}}{\sqrt{1+\left(\frac{\xi}{\beta_{x}^{*}/\alpha_{s}} \right)^{2}}\sqrt{1+\left(\frac{\xi}{\beta_{y}^{*}/\alpha_{s}}\right)^{2}}} \sigma_{s}d\xi=...\left[r_{u}:=\frac{\beta_{u}^{*}}{\sigma_{s}}\right]...\]

\[=\frac{n_{1}n_{2}n_{b}f_{ip}}{4\pi\sigma_{x}^{*}\sigma_{y}^{*}}\int\frac{1}{ \sqrt{\pi}}\frac{e^{-\xi^{2}}}{\sqrt{1+\left(\frac{\xi}{\xi_{x}}\right)^{2}} \sqrt{1+\left(\frac{\xi}{\xi_{y}}\right)^{2}}}d\xi\equiv L^{*}\cdot\mathbb{H}(r _{x},r_{y}) \tag{11.8}\]

For optics of the interaction region symmetric in the two transverse planes (\(\beta_{x}(s)=\beta_{y}(s)\)), the hourglass effect reduces the nominal luminosity \(L^{*}\) by the quantity:\[\mathbb{H}(r)=\int_{-\infty}^{+\infty}\frac{1}{\sqrt{\pi}}\,\frac{e^{-\xi^{2}}}{1+ \left(\frac{\varsigma}{r}\right)^{2}}d\varsigma=\sqrt{\pi}\,re^{r^{2}}[erf(r)]<1 \ \ \forall r \tag{11.9}\]

\(\mathbb{H}\) is a monotonic function of \(r\) with asymptotic value 1 for \(r=\frac{\beta^{*}}{\sigma_{s}}\rightarrow\infty\). For example, \(H\left(0.5\right)\approx 0.53\), \(H\left(1\right)\approx 0.75\), and \(H\left(2\right)\approx 0.9\).

#### Discussion: Luminosity of a Compton Source

The definition of \(L\) in Eq. 11.2 can be identically applied to the interaction of a charged particle beam and a photon beam, such as in an ICS light source. In this case, the number of events is the number of scattered photons, and the cross section in the no recoil approximation is the Thomson cross section: \(N_{ph}=L\sigma_{T}\). How does the luminosity depend from the laser parameters and the electron beam energy, for any given energy of the scattered photons?

For simplicity, we assume both beams to be round and perfectly matched, with identical beam sizes at the IP. The laser wavelength is \(\lambda_{L}\), and the electron beam is intended to be at the diffraction limit. If \(Q_{b}\) and \(U_{L}\) are the bunch charge and the laser pulse energy, respectively, the luminosity for a single interaction is:

\[\begin{array}{l}L_{ICS}=\frac{N_{e}N_{L}f_{ip}}{2\pi\sqrt{\sigma_{x,e}^{2}+ \sigma_{x,L}^{2}}\sqrt{\sigma_{y,e}^{2}+\sigma_{y,L}^{2}}}=\frac{1}{4\pi\sigma _{L}^{2}}\frac{Q_{b}}{e}\frac{U_{L}}{E_{L}}=\frac{1}{4\pi}\,\frac{Q_{b}}{e} \frac{U_{L}}{hc}\lambda_{L}\left(\frac{4\pi}{\beta_{L}\lambda_{L}}\right)=\\ \\ =\frac{1}{ehc}\,\frac{Q_{b}U_{L}}{\beta_{L}}\approx\,\frac{Q_{b}U_{L}}{\sigma_ {L}^{2}}\,\frac{\nu^{2}\lambda_{s}}{\pi ehc}\end{array} \tag{11.10}\]

The very last expression is derived by imposing the resonance condition \(\lambda_{s}\approx\lambda_{L}/(4\gamma^{2})\) for the ICS radiation collected on-axis and for head-on collision (see Eq. 10.57).

As expected, the number of ICS photons is proportional to the total bunch charge and the laser total pulse energy. Owing to the local interaction, \(L\) is maximized by small laser and electron beam transverse sizes at the IP. However, very tight waists would enhance the hourglass effect, thus degrading the effective luminosity. A trade-off in the control of the transverse sizes has to be reached, eventually.

The luminosity of the ICS light source is proportional to the electron beam energy. This means that, for any target \(\lambda_{s}\), \(\lambda_{L}\) could be made longer (in correspondence of which a larger laser power can be available) at higher electron beam energies. This, however, is at the expense of larger RF power as required by a longer accelerator or higher accelerating gradients. The luminosity is larger at higher energies of the scattered photons, which implies again higher electron beam energies.

### Beam-Beam Tune Shift

The mutual penetration of two colliding beams at the IP makes each beam subject to the e.m. field generated by the other beam [1, 3]. The effect is equivalent to focusing in both transverse planes. Assuming infinitely long ("coasting beam" approximation), transversely "round" Gaussian beams of identical sizes at the IP (\(\sigma_{x,1}=\sigma_{x,2}=\sigma_{y,1}=\sigma_{y,2}\)), the angular divergence ("kick") induced by the beam-beam interaction is:

\[\Delta r^{\prime}=\tfrac{\Delta p_{r}}{p_{z}}=\tfrac{1}{p_{z}}\int_{-\infty}^{+ \infty}\rho(r,s)E_{r}(r)ds=\tfrac{2r_{0}n_{1}}{\gamma}\tfrac{1}{r}\left(1-e^{ -\tfrac{r_{0}^{2}}{2\sigma_{r}^{2}}}\right)\approx\left(\tfrac{r_{0}n_{1}}{ \gamma}\tfrac{r}{\sigma_{r}^{2}}\right)_{r\ll\sigma_{r}} \tag{11.11}\]

The approximation is for a kick in proximity of beam's axis, where the largest portion of the charge distribution lies. Specialized to the case of flat beams, i.e., \(\sigma_{x}\gg\sigma_{y}\), the kick in each transverse plane becomes:

\[\Delta u^{\prime}\approx\frac{2r_{0}n_{1}}{\gamma}\frac{u}{\sigma_{u}(\sigma_ {x}+\sigma_{y})}\equiv\delta_{bb}u,\ \ u=x,\,y \tag{11.12}\]

The linearized _beam-beam effect_ in Eq. 11.12 is in fact the inverse of a focal length, \(\delta_{bb}=1/f_{bb}\). Such parasitic focusing is typically weaker in circular colliders with respect to linear colliders, by virtue of a larger interaction area. The perturbation has to be small enough not to disrupt the beams' quality over many consecutive hours of operation.

If the effect is strong enough, the transverse kick leads in turn to a variation of the particles' lateral position already internally to the interaction region:

\[\left\{\begin{array}{l}\Delta r^{\prime}=-\frac{r}{f_{bb}}\\ \Delta r=\Delta r^{\prime}\cdot\sigma_{z}\end{array}\right.\Rightarrow \Delta r=-\frac{\sigma_{z}}{f_{bb}}r \tag{11.13}\]

\[\Rightarrow D_{bb,u}:=\frac{\Delta u}{u}=-\frac{\sigma_{z}}{f_{bb,u}}=\frac{ 2r_{0}n_{1}}{\gamma}\frac{\sigma_{z}}{\sigma_{u}(\sigma_{x}+\sigma_{y})} \tag{11.14}\]

\(D_{bb}\) is called _disruption parameter_. When the kick is relatively "gentle" or \(D_{bb}<1\), the effect can be evaluated as a perturbation to the single-turn beam matrix. For flat beams, the larger contribution is in the vertical plane (see Eq. 11.12):

\[\tilde{M}_{t}=B\cdot M_{t}\cdot B=\begin{pmatrix}1&0\\ -\frac{\delta_{bb}}{2}&1\end{pmatrix}\begin{pmatrix}\cos\Delta\mu_{y}&\beta_{y }^{*}\sin\Delta\mu_{y}\\ -\frac{\sin\Delta\mu_{y}}{\beta_{y}^{*}}&\cos\Delta\mu_{y}\end{pmatrix} \begin{pmatrix}1&0\\ -\frac{\delta_{bb}}{2}&1\end{pmatrix}=\]

\[\approxeq\begin{pmatrix}\cos\Delta\mu_{y}-\frac{\delta_{bb}}{2}\beta_{y}^{*} \sin\Delta\mu_{y}&\beta_{y}^{*}\sin\Delta\mu_{y}\\ -\frac{1}{\beta_{y}^{*}}\left(\sin\Delta\mu_{y}+\frac{\delta_{bb}}{2}\cos \Delta\mu_{y}\right)&\cos\Delta\mu_{y}\end{pmatrix}= \tag{11.15}\]

\[\approxeq\begin{pmatrix}\cos 2\pi(Q_{y}+\xi_{y})&\beta_{y}^{*}\sin 2\pi(Q_{y} +\xi_{y})\\ -\frac{1}{\beta_{y}^{*}}\sin 2\pi(Q_{y}+\xi_{y})&\cos 2\pi(Q_{y}+\xi_{y})\end{pmatrix},\]where we introduced the _beam-beam tune shift_:

\[\xi_{y}=\frac{\delta_{bb}\beta_{y}^{*}}{4\pi}=\frac{r_{0}n_{1}\beta_{y}^{*}}{2\pi \,\gamma\,\sigma_{y}(\sigma_{x}+\sigma_{y})}, \tag{11.16}\]

we assumed \(\xi_{y}\ll 1\), and neglected higher orders in \(\delta_{bb}\). The tune-shift is usually at the level of \(\sim\)0.01 for collisions \(e^{-}e^{+}\), and \(<\)0.05 for \(pp\). An analogous expression is obtained for the horizontal plane, but smaller in proportion to the ratio of the transverse beam sizes at the IP.

The beam-beam tune shift should be small enough to guarantee stable particles' motion. Firstly, dangerous resonances should be avoided. Secondly, the overall stability of the periodic motion should be ensured by satisfying the condition \(|Tr(\tilde{M}_{t})|<2\). In the approximation of small beam-beam tune shift, the latter condition gives (in each transverse plane):

\[\begin{array}{l}2|\cos 2\pi\,(Q+\xi)|=2|\cos\Delta\mu\cos(2\pi\,\xi)-\sin \Delta\mu\sin(2\pi\,\xi)|\approx\\ \approx 2|\cos\Delta\mu-2\pi\xi\sin\Delta\mu|<2\\ \Rightarrow\xi\,<\,\frac{|\cot\Delta\mu|}{2\pi}\end{array} \tag{11.17}\]

At the same time, a small beam-beam tune shift limits the instantaneous luminosity. If \(\xi_{th}\) is the maximum beam-beam tune shift allowed in the vertical plane by beam stability, such that \(\xi_{y}\leq\xi_{th}\), then the maximum value of \(n_{1}\) is found by plugging Eq. 11.16 into Eq. 11.2, to get the tune-shift-limited luminosity:

\[\left\{\begin{array}{l}n_{1}\,<\,\xi_{th}\frac{2\pi\,\gamma\,\sigma_{x}( \sigma_{x}+\sigma_{y})}{r_{0}\beta_{y}^{*}},\\ \\ L=\frac{n_{1}n_{2}n_{b}f_{ip}}{4\pi\,\sigma_{x}^{*}\sigma_{y}^{*}}\,<\,\frac{ \gamma\,\xi_{th}}{2r_{0}\beta_{y}^{*}}\left(1+\frac{\sigma_{x}^{*}}{\sigma_{x} ^{*}}\right)n_{2}n_{b}f_{ip}\end{array}\right. \tag{11.18}\]

Unlike in storage rings, usually \(D_{bb}\gg 1\) in linear colliders, where the bunches are used once and therefore a stronger beam-beam effect can be tolerated. In practice, the deformation of the bunch transverse size along its duration profits of the parasitic e.m. focusing, which is stronger in the vertical plane for flat beams. This brings to even lower beam sizes in correspondence of the IP, and eventually to an enhancement of the luminosity (_pinch effect_) which counteracts the hourglass effect.

### Beam-Beam Lifetime

Colliding beams interact not only through the expected channels to produce new events, but also via Coulomb force [3]. This generates an energy change (beam-beam-beam Bremsstrahlung or radiative Bhabha scattering in \(e^{-}e^{-}\) and \(e^{-}e^{+}\) colliders) which can exceed the RF or the momentum acceptance. In such case, particles get lost and the beam current reduces with time. If \(\sigma_{loss}\) is the cross section describing the interaction and we define the event rate as the relative loss of beam particles per unit of time, then the _beam-beam lifetime_ can be expressed as function of the single bunch luminosity (see Eq. 11.3, \(n_{b}=1\)):

\[\frac{1}{\tau_{bb}}=\frac{1}{n_{1}}\frac{dn_{1}}{dt}=\frac{L\sigma_{loss}}{n_{1 }}\approx\frac{\sigma_{loss}}{n_{1}}\frac{n_{1}n_{2}}{4\pi\sigma_{x}^{*} \sigma_{y}^{*}}f_{ip}=\frac{n_{2}\,f_{ip}}{4\pi\sigma_{x}^{*}\sigma_{y}^{*}} \sigma_{loss} \tag{11.19}\]

Since the r.h.s. of Eq. 11.19 is approximately constant with time, Eq. 11.19 leads to an exponential decay with time of the bunch population, \(n_{1}=n_{0}e^{-t/\tau_{bb}}\). In general, \(\sigma_{loss}\,\propto 4r_{0}^{2}\alpha\) (dominated by Coulomb interaction), and it typically results \(\sigma_{loss}\,\leq\,0.2\) barn both at electron and proton colliders over a wide range of beam energies.

## References

* [1] W. Herr, B.J. Holzer, B. Muratori, Concept of luminosity, in _Elementary Particles-Accelerators and Colliders_, vol. 21C, ed. by S. Myers, H. Schopper (Published by Springer Materials, 2013). ISBN: 978-3-642-23052-3
* [2] K. Potter, Luminosity measurements and calculations, in _Proceedings of CERN Accelerator School: 5th General Accelerators Physical Course_, Geneva, Switzerland. CERN 94-01 vol. I, ed. by S. Turner (1994), pp. 117-130
* [3] H. Mais, C. Mari, Introduction to beam-beam effects, in _Proceedings of CERN Accelerator School: 5th General Accelerators Physical Course_, Geneva, Switzerland. CERN 94-01, vol. I, ed. by S. Turner (1994), pp. 499-524

