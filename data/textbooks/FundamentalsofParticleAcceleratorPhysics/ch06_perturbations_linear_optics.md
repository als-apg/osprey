Four main perturbations to linear optics are introduced in this Chapter: orbit distortion, tune resonances, linear chromaticity, and weak betatron coupling. The former is described in the framework of linear optics, in the presence of magnets' misalignment. When applied to periodic motion, it introduces betatron resonances. Both resonances and linear optics distortion due to particle's momentum deviation--the latter effect is denominated _linear chromaticity_,--force to the adoption of high order magnets to cure, thus to nonlinear optics. Finally, magnets' alignment errors lead to weak coupling of the betatron motion, partly invalidating the uncoupled Hill's equations.

### 11 Orbit Distortion

#### 11.1 Single Pass

A thin quadrupole magnet of integrated gradient \(kl\) and misaligned by \(\Delta u\) with respect to the beam path, imposes to the beam as a whole an angular deviation, a "kick" in jargon, like if it were a dipole. The scaling down in field order by misaligned magnets is a general effect denominated "feed-down". In a quadrupole, this can also be intended as an additional focusing proportional to the quadrupole-beam relative misalignment, or \(\Delta u^{\prime}=kl\Delta u\).

The generic particle's coordinates transform through the quadrupole according to Eq. 11:\[\begin{pmatrix}u_{q}\\ u_{q}^{\prime}\end{pmatrix}=\begin{pmatrix}1&0\\ kl&1\end{pmatrix}\begin{pmatrix}u_{0}\\ u_{0}^{\prime}\end{pmatrix}+\begin{pmatrix}0\\ kl\Delta u\end{pmatrix}\] \[\Rightarrow\begin{cases}u_{q}=u_{0}\\ u_{q}^{\prime}=(u_{0}^{\prime}+klu_{0})+kl\Delta u\equiv u_{\beta}^{\prime}+ \theta_{q}\end{cases} \tag{6.1}\]

Since the quadrupole is treated in thin lens approximation, Eq. 6.1 shows that the gradient error does not modify the initial particle's position, but it does the angular divergence.

We now assume a generic, not periodic beam line downstream of the quadrupole magnet. The beam line is ideally aligned to the beam original path. The transfer matrix \(M_{1,2}\) has the form of Eq. 4.119. Subscript "1" marks the location of the "perturbation", subscript "2" of the "observation" point downstream. To follow the motion of the beam as a whole downstream of the misaligned quadrupole, \(M\) is intended to be applied to the bunch centroid's coordinates:

\[\begin{pmatrix}u\\ u^{\prime}\end{pmatrix}=M_{1,2}\begin{pmatrix}u_{q}\\ u_{q}^{\prime}\end{pmatrix}=M_{1,2}\begin{pmatrix}u_{0}\\ u_{\beta}^{\prime}+\theta_{q}\end{pmatrix}\] \[\Rightarrow u(s)=\begin{bmatrix}m_{11}u_{0}+m_{12}(u_{0}^{\prime}+ klu_{0})\end{bmatrix}+m_{12}\theta_{q}\equiv u_{\beta}(u_{0},u_{0}^{\prime})+u_{ p}(\theta_{q}) \tag{6.2}\] \[\Rightarrow u_{p}(s)=kl\Delta u\sqrt{\beta_{1}\beta_{2}(s)}\sin\Delta \mu_{1,2}(s)\]

The Beam's position downstream of the kick is proportional to the (square root of) betatron function at the perturbation.

The Beam trajectory distortion induced by misaligned quadrupoles is commonly corrected with a series of small dipole magnets, named "steering" or "corrector" magnets, kicking the beam to restore it on--or in proximity of--the unperturbed trajectory. The Beam's position is recorded at several locations along the accelerator by detectors called "Beam Position Monitors" (BPMs).

The effect of a steering magnet on the beam trajectory is described by an equation of the same form of Eq. 6.2. We then infer that, for any target trajectory to be reached after correction, the kick from a steering magnet is minimized by a large betatron function at the steering magnet location (\(\beta_{1}\)). At the same time, the sensitivity of the beam position to the steerer is larger when the observation is done at a location of large betatron function (\(\beta_{2}\)), and for the steerer-to-BPM phase advance \(\Delta\mu=(2n+1)\pi/2\), \(n\in\mathbb{N}\).

#### Closed Orbit

In a synchrotron, the closed orbit is the reference orbit of the synchronous particle. Its transverse coordinates in the Frenet-Serret reference system are \((u,u^{\prime})=(0,0)\)\(\forall s\). A closed orbit perturbed by a dipolar kick like that one in Eq. 6.2 still has to satisfy the one-turn periodicity [1]:\[\begin{pmatrix}u\\ u^{\prime}\end{pmatrix}=M_{t}\begin{pmatrix}u\\ u^{\prime}\end{pmatrix}+\begin{pmatrix}0\\ \theta_{q}\end{pmatrix}\quad\Rightarrow\quad\begin{pmatrix}u\\ u^{\prime}\end{pmatrix}=(I-M_{t})^{-1}\begin{pmatrix}0\\ \theta_{q}\end{pmatrix} \tag{6.3}\]

We choose \(M_{t}\) as in Eq. 4.120, where the one-turn phase advance is \(\Delta\mu=2\pi\,Q\). We have:

\[(I-M_{t})^{-1}=\begin{pmatrix}1-\cos 2\pi\,Q&-\beta\,\sin 2\pi\,Q \\ \frac{\sin 2\pi\,Q}{\beta}&1-\cos 2\pi\,Q\end{pmatrix}^{-1}=\] \[=\tfrac{1}{2(1-\cos 2\pi\,Q)}\begin{pmatrix}1-\cos 2\pi\,Q& \beta\,\sin 2\pi\,Q\\ -\frac{\sin 2\pi\,Q}{\beta}&1-\cos 2\pi\,Q\end{pmatrix}; \tag{6.4}\] \[\Rightarrow u(s)=\tfrac{\theta\beta(s)}{2}\tfrac{\sin 2\pi\,Q}{1- \cos 2\pi\,Q}=\tfrac{\theta\beta(s)}{2}\tfrac{2\sin\pi\,Q\cos\pi\,Q}{2\sin^{2} \pi\,Q}=\tfrac{\theta\beta(s)\cos\pi\,Q}{2\sin\pi\,Q}\]

Equation 6.4 describes the closed orbit (c.o.) in the presence of a single kick. Since the kick is assumed to be at the same location \(s\) of observation (see Eq. 6.3), we have \(\beta_{1}=\beta_{2}\equiv\beta\). The expression is made more general by considering a series of independent kicks \(q=1\),..., \(N_{q}\), each kick given in correspondence of the betatron function \(\beta(s_{q})\equiv\beta_{q}\). At the generic observation point \(s_{m}\), we have \(\beta(s_{m})\equiv\beta_{m}\), and the betatron phase advance between a kick and the observation point is \(\Delta\mu_{q,m}=\mu(s_{q})-\mu(s_{m})\equiv\mu_{q}-\mu_{m}\). If we now assume that the kicks are all independent, the perturbed closed orbit observed at the \(m\)-th BPM can be written as linear superposition of all individual perturbations:

\[u_{co}(s)=\sum_{q=1}^{N_{q}}u_{co,q}(s)=\sum_{q=1}^{N_{q}}\theta_{q}\,\frac{ \sqrt{\beta_{q}\,\beta_{m}(s)}}{2\sin\pi\,Q}\cos\big{(}\Delta\mu_{q,m}(s)+\pi \,Q\big{)} \tag{6.5}\]

In analogy to the discussion done for the trajectory correction, we draw the following observations.

* When the fractional part of the tune \(\to 0\) or \(1\), the phase advance which minimizes the steerer's strength, i.e., maximizes the orbit sensitivity to steering, is \(\Delta\mu_{q,m}=2n\pi\).
* When the fractional part of the tune \(\approx 0.5\), the minimum strength (maximum sensitivity) is obtained for \(\Delta\mu_{q,m}=(2n+1)\pi/2\).
* \(u_{co}\) oscillates with 2-fold smaller frequency than the betatron oscillation.
* Since \(u_{co}\) was derived in the assumption of pure dipole kicks, it does not modify the strong focusing of the lattice, i.e., the lattice C-S parameters. In other words, \(u_{co}\) is linearly independent from the solutions of Hill's equation. Moreover, since \(u_{co}\) describes the bunch centroid's motion, it corresponds to an identical translation of the orbit for all beam particles. Consequently, the generic particle's position in a synchrotron affected by misaligned quadrupoles and/or dipole field errors, is the linear superposition of the betatron, dispersive and perturbed orbit motion:

\[\left\{\begin{array}{l}u(s)=u_{\beta}(s)+u_{D}(s)+u_{co}(s)\\ \\ u^{\prime}(s)=u^{\prime}_{\beta}(s)+u^{\prime}_{D}(s)+u^{\prime}_{co}(s)\end{array}\right. \tag{6.6}\]

#### Amplification Factor

When the number of independent dipolar kicks from misaligned quadrupoles is \(N_{q}>>1\), and the specific error set \(\Delta u_{q}\) is unknown, the c.o. distortion can be still estimated assuming a distribution function of the misalignment of each quadrupole magnet having standard deviation \(\sigma_{q}\). In most cases, it is reasonable to assume that the misalignment distribution function is the same for all quadrupoles, i.e., \(\sigma_{q}\) is a statistical uncertainty on the alignment of any quadrupole magnet in the accelerator.

The rms c.o. is first evaluated as due to a single kick (\(q\)-th quadrupole magnet) with deviation \(\sigma_{q}\) (this is defined for a large number of errors seeds, \(N_{seed}>>1\)), and observed at a generic position (\(m\)-th monitor):

\[\begin{array}{l}\sigma_{co}^{2}(q,m)=\frac{1}{N_{seed}}\sum_{i=1}^{N_{seed}}u _{co,i}^{2}(s;q,m)=\\ \\ =\frac{1}{N_{seed}}\sum_{i}\Delta u_{q,i}^{2}\frac{(k_{q}l_{q})^{2}\beta_{q} \beta_{m}}{(2\sin\pi Q)^{2}}\cos^{2}(\Delta\mu_{q,m}+\pi\,Q)=\\ \\ =\sigma_{q}^{2}\frac{(k_{q}l_{q})^{2}\beta_{q}\beta_{m}}{(2\sin\pi\,Q)^{2}} \cos^{2}(\Delta\mu_{q,m}+\pi\,Q)\end{array} \tag{6.7}\]

Next, we sum over all quadrupoles' kicks, but we still keep a specific observation point:

\[\sigma_{co}^{2}(m)=\frac{\beta_{m}\sigma_{q}^{2}}{4\sin^{2}\pi\,Q}\sum_{q=1}^{ N_{q}}\,(k_{q}l_{q})^{2}\beta_{q}\cos^{2}(\Delta\mu_{q,m}+\pi\,Q) \tag{6.8}\]

Finally, we average over all \(N_{m}\) observation points. Doing so, we approximate \(\cos^{2}\) to its average value over one betatron period (i.e., we assume many betatron oscillations per turn, or \(Q>>1\)):

\[\sigma_{co}^{2}\approx\frac{1}{N_{m}}\sum_{m=1}^{N_{m}}\beta_{m}\frac{\sigma_ {q}^{2}}{8\sin^{2}\pi\,Q}\sum_{q}\,(k_{q}l_{q})^{2}\beta_{q}=\frac{\sigma_{q}^ {2}(\beta_{m})}{8\sin^{2}\pi\,Q}\sum_{q}\,(k_{q}l_{q})^{2}\beta_{q} \tag{6.9}\]

By introducing the rms value of the integrated quadrupole gradient \(k_{q}l_{q}\) but normalized to the local betatron function \(\sqrt{\beta_{q}}\), it results:

\[\sigma_{kl}^{2}:=\frac{1}{N_{q}}\sum_{q=1}^{N_{q}}(k_{q}l_{q})^{2}\beta_{q}\,, \tag{6.10}\]and we can re-write the rms c.o. distortion as:

\[\sigma_{co}\approx\frac{\sqrt{\langle\beta_{m}\rangle N_{q}}}{2\sqrt{2}\sin\pi Q} \sigma_{kl}\sigma_{q} \tag{6.11}\]

The ratio \(A_{u}=\frac{\sigma_{co,u}}{\sigma_{q,u}}\) is said _amplification factor_ for the \(u\)-transverse plane. Roughly speaking, it describes the expected orbit deviation, averaged over the ring lattice, excited by a characteristic quadrupoles' misalignment \(\sigma_{q}\). It turns out that, for any given \(\sigma_{q}\), \(A_{u}\sim\sqrt{N_{q}}k_{q}l_{q}\,\langle\beta_{u}\rangle\).

### 6.2 Resonances

#### Resonance Order

Equation 6.11 shows that, owing to the presence of dipolar perturbations, the beam motion becomes unstable, i.e. \(u\to\infty\), when the _resonance condition_\(Q=r\) (\(r\) integer) is satisfied [1, 2]. The appearance of tune resonances is an effect intrinsic to strong focusing, and amplified by errors or, as we will see, by nonlinear fields. Indeed, periodicity implies that an error in the magnetic lattice can become a systematic driving force, pushing particles far form the reference orbit, if the particles come back to the error location exactly with the same phase space coordinates. This situation may happen every \(p\,Q=r\) turns, \(p\) integer. For the most general case of errors affecting the motion in both transverse planes, the resonance condition reads:

\[p\,Q_{x}+q\,Q_{y}=r,\quad p,\,q,\,r\in\mathbb{N} \tag{6.12}\]

The resonance coefficients depend from the field order associated to the error. The sum \(n=|p|+|q|\) is the _resonance order_. According to the notation in Eq. 4.47, a systematic field component of the \(m\)-th order gives rise to the (lowest) resonance order \(n=m+1\). Below, we demonstrate that a quadrupole gradient error originates a second order ("half-integer") tune resonance.

The demonstration profits of the graphical representation of the particle's motion in terms of Floquet's coordinates (Eq. 4.109). Let us consider a quadrupole's integrated gradient affected by a small error, \(kl+\delta(kl)\) with \(\delta(kl)<<kl\). For simplicity, we describe the quadrupole in thin lens approximation. Figure 6.1 shows that the particle's position at the location of the quadrupole is not changed, while its divergence is. The error pushes the particle's representative point off the unperturbed orbit of radius \(a\), parallel to the divergence axis \(w^{\prime}\). For small perturbation \(\delta w^{\prime}\), \(\Delta s\) can be approximated to a segment, and the angle contained by \(\Delta s\) and \(\delta w^{\prime}\) with vertex in \(P\) is approximately \(Q\theta\). Consequently, the beam _coherent tune-shift_ evaluated after one turn (\(\theta=2\pi\)) is modified by the small quantity:

\[(\Delta\,Q)_{turn}=\tfrac{1}{2\pi}\tfrac{\Delta s}{a}\approx\tfrac{1}{2\pi} \tfrac{\delta w^{\prime}\cos(Q\theta)}{a} \tag{6.13}\]The tune-shift can also be understood by virtue of Eq. 4.91 as a distortion of the betatron function due to the focusing error.

The perturbation \(\delta w^{\prime}\) is made explicit in terms of the gradient error \(\delta(kl)\) and of the particle's position inside the magnet, in which we also have \(\alpha_{u}=0\):

\[\left\{\begin{array}{ll}\delta u^{\prime}=\delta(kl)u\\ w^{\prime}(\alpha_{u}=0)=\sqrt{\beta_{u}}u^{\prime}\end{array}\right.\Rightarrow \delta w^{\prime}=\sqrt{\beta_{u}}\delta u^{\prime}=\sqrt{\beta_{u}}\delta( kl)u=\sqrt{\beta_{u}}\delta(kl)\sqrt{\beta_{u}}w \tag{6.14}\]

We observe that \(w(P)=a\cos Q\theta\), and by substituting Eq. 6.14 into Eq. 6.13 we obtain:

\[(\Delta Q)_{turn}\approx\tfrac{1}{2\pi}\beta_{u}\delta(kl)\tfrac{a\cos^{2}(Q \theta)}{a}=\tfrac{1}{4\pi}\beta_{u}\delta(kl)\left[1+\cos(2Q\theta)\right] \tag{6.15}\]

The tune-shift induced by a quadrupole gradient error is maximum when \((2Q\theta)_{turn}=4\pi\,Q=2\pi r\), i.e., \(Q=r/2\), \(r\in\mathbb{N}\).

It can be easily inferred that, since higher order gradient errors are proportional to higher powers of the particle's position in Eq. 6.14, they will lead to higher powers of \(\cos(Q\theta)\) in Eq. 6.15. This will eventually generate higher order tune resonances, the resonance order proportional to the magnetic field order. For example, a sextupole magnet generates \(3rd\) order resonances, an octupole magnet \(4th\) order resonances, etc.

Lower order resonances are generally associated to stronger driving terms. Integer and half-integer resonances are systematic, because dipole and quadrupole magnets are essential components of strong focusing lattices.

In practice, resonances up to \(4th\) order are commonly avoided with a suitable choice of the working point. Any residual coherent tune-shift is often minimized by dedicated tune feedback systems through manipulation, for example, of the beam orbit. On top of this, the _incoherent tune-shift_, associated to the individual particle's transverse positions, will tend to limit the oscillation amplitude associated to stable motion. The area in the transverse plane \((x,\,y)\) in which the particles' motion remains

Figure 6.1: Particle’s motion in Floquet’s phase space, in the presence of quadrupole gradient error \(\delta w^{\prime}\). \(Q\) is the betatron tune

stable for a very large number of turns is said _dynamic aperture_. This can either be larger or smaller than physical restrictions imposed by the vacuum chamber, or _physical aperture_. The minimum of the two areas determines the _transverse acceptance_ of the accelerator.

#### Sum and Difference Resonance

Weak betatron coupling can excite \(2nd\) order sum and difference resonance, but they have _not_ the same impact on the stability of the particle's motion [3]. Weak coupling can be modelled in terms of a perturbative driving force in the homogeneous Hill's equations for the two planes, proportional to the particle's coordinate in the opposite plane. This kind of perturbation can be generated, for example, by quadrupole magnets with a small roll error (typically smaller than 1 mrad or so). By adopting the Floquet's normalized coordinates (see Eq. 4.111):

\[\left\{\begin{array}{l}\frac{d^{2}w_{x}}{d\theta^{2}}\,+\,Q_{x}^{2}w_{x}\,= \epsilon\cos(m\theta)y\\ \\ \frac{d^{2}w_{y}}{d\theta^{2}}\,+\,Q_{y}^{2}w_{y}\,=\epsilon\cos(m\theta)x \end{array}\right. \tag{6.16}\]

and \(m\in\dot{\mathbb{N}}\).

For the perturbative amplitude \(\epsilon\ll 1\), the solution of the unperturbed Hill's equation can be substitued in the r.h.s. of Eq. 6.16 to get:

\[\left\{\begin{array}{l}\frac{d^{2}w_{x}}{d\theta^{2}}\,+\,Q_{x}^{2}w_{x}= \epsilon\cos(m\theta)\sqrt{2J_{y}}\cos(Q_{y}\theta)=\frac{\epsilon\sqrt{J_{y}} }{\sqrt{2}}\left[\cos(m+Q_{y})\theta+\cos(m-Q_{y})\theta\right]\\ \\ \frac{d^{2}w_{y}}{d\theta^{2}}\,+\,Q_{y}^{2}w_{y}=\epsilon\cos(m\theta)\sqrt{2J _{x}}\cos(Q_{x}\theta)=\frac{\epsilon\sqrt{J_{x}}}{\sqrt{2}}\left[\cos(m+Q_{x}) \theta+\cos(m-Q_{x})\theta\right]\end{array}\right. \tag{6.17}\]

The driving terms are maximized when the following conditions apply:

\[\left\{\begin{array}{l}2\pi\,(m\pm Q_{x})=2\pi\,r\\ \\ 2\pi\,(m\pm Q_{y})=2\pi\,p\end{array}\right.\Rightarrow\left\{\begin{array}[] {l}Q_{x}\,=\,m-r\\ Q_{x}\,=\,r-m\\ Q_{y}\,=\,m-\,p\\ Q_{y}\,=\,p-m\end{array}\right.\Rightarrow\left\{\begin{array}{l}Q_{x}\,+\,Q_{ y}=n\\ \\ |\,Q_{x}\,-\,Q_{y}|=n\end{array}\right. \tag{6.18}\]

and \(r\), \(p\), \(n\in\mathbb{N}\).

Stability can be investigated through the analysis of the trace of the one-turn \(4\times 4\) matrix perturbed by a quadrupole magnet of focal length \(f\), in the presence of a roll angle \(\phi\ll 1\). The perturbed matrix results:\[\begin{split}&\tilde{M}_{t}=R\,QR^{-1}Q^{-1}M=\\ &=\begin{pmatrix}I_{2}\cos\phi&I_{2}\sin\phi\\ -I_{2}\sin\phi&I_{2}\cos\phi\end{pmatrix}\begin{pmatrix}F_{2}&0\\ 0&D_{2}\end{pmatrix}\begin{pmatrix}I_{2}\cos\phi&-I_{2}\sin\phi\\ I_{2}\sin\phi&I_{2}\cos\phi\end{pmatrix}\begin{pmatrix}D_{2}&0\\ 0&F_{2}\end{pmatrix}M=\\ &\equiv\begin{pmatrix}A&b\\ a&B\end{pmatrix}\end{split} \tag{6.19}\]

The one-turn matrix \(M\) is written in terms of periodic Courant-Snyder parameters, for the simpler case \(\alpha_{x}=\alpha_{y}=0\):

\[M=\begin{pmatrix}M_{2,x}&0\\ 0&M_{2,y}\end{pmatrix}=\begin{pmatrix}\cos\Delta\mu_{x}&\beta_{x}\sin\Delta \mu_{x}&0&0\\ -\frac{1}{\beta_{x}}\sin\Delta\mu_{x}&\cos\Delta\mu_{x}&0&0\\ 0&0&\cos\Delta\mu_{y}&\beta_{y}\sin\Delta\mu_{y}\\ 0&0&-\frac{1}{\beta_{y}}\sin\Delta\mu_{y}&\cos\Delta\mu_{y}\end{pmatrix} \tag{6.20}\]

Then, the components of \(\tilde{M}_{t}\) result from Eq. 6.19:

\[\begin{split} A&=\begin{pmatrix}\cos\Delta\mu_{x}\cos^{2}\phi& \beta_{x}\sin\Delta\mu_{x}\cos^{2}\phi\\ -\frac{1}{\beta_{x}}\sin\Delta\mu_{x}\cos^{2}\phi+\cos\Delta\mu_{x}\,\frac{2 \sin^{2}\phi}{f}\,\cos\Delta\mu_{x}\cos^{2}\phi+\beta_{x}\sin\Delta\mu_{x}\, \frac{2\sin^{2}\phi}{f}\end{pmatrix}\\ &B&=\begin{pmatrix}\cos\Delta\mu_{y}\cos^{2}\phi&\beta_{y}\sin\Delta\mu_{y} \cos^{2}\phi\\ -\frac{1}{\beta_{y}}\sin\Delta\mu_{y}\cos^{2}\phi-\cos\Delta\mu_{y}\,\frac{2 \sin^{2}\phi}{f}\,\cos\Delta\mu_{y}\cos^{2}\phi-\beta_{y}\sin\Delta\mu_{y}\, \frac{2\sin^{2}\phi}{f}\end{pmatrix}\\ &a&=\begin{pmatrix}0&0\\ \cos\Delta\mu_{x}\frac{\sin 2\phi}{f}\,\,\beta_{x}\sin\Delta\mu_{x}\frac{\sin 2 \phi}{f}\end{pmatrix}\\ &b&=\begin{pmatrix}0&0\\ \cos\Delta\mu_{y}\frac{\sin 2\phi}{f}\,\,\beta_{y}\sin\Delta\mu_{y}\frac{\sin 2 \phi}{f}\end{pmatrix}\end{split} \tag{6.21}\]

If \(\lambda_{j}\) and \(\vec{u}_{j}\) are, respectively, eigenvalue and eigenvector of \(\tilde{M}_{t}\), we can write:

\[(\tilde{M}_{t}+\tilde{M}_{t}^{-1})\vec{u}_{j}=(\lambda_{j}+\lambda_{j}^{-1}) \vec{u}_{j}\equiv\kappa_{j}\vec{u}_{j},\ \ j=1,\ldots,4 \tag{6.22}\]

From the calculation of the individual terms in Eq. 6.19 it results:

\[\kappa=\frac{Tr(A+B)}{2}\pm\sqrt{\left(\frac{Tr(A-B)}{2}\right)^{2}+|a+b^{t}|} \tag{6.23}\]The radical is made explicit by means of Eq. 6.21, and by noticing that \(\sin\Delta\mu_{x}=\pm\sin\Delta\mu_{y}\) for the difference and the sum resonance, respectively:

\[\begin{array}{l}\Delta_{\pm}=\left(\frac{Tr(A-B)}{2}\right)^{2}+|a+b^{t}|=\\ =\frac{\sin^{4}\phi}{f^{2}}(\beta_{x}\sin\Delta\mu_{x}-\beta_{y}\sin\Delta\mu_ {y})^{2}+\frac{\beta_{x}\beta_{y}}{f}\sin^{2}2\phi\sin\Delta\mu_{x}\sin\Delta \mu_{y}=\\ =\frac{\sin^{4}\phi}{f^{2}}(\beta_{x}\pm\beta_{y})^{2}\sin^{2}\Delta\mu_{x}\mp \frac{\beta_{x}\beta_{y}}{f}\sin^{2}2\phi\sin^{2}\Delta\mu_{x}=\\ =\frac{\sin^{2}\phi\sin^{2}\Delta\mu_{x}}{f^{2}}\left[(\beta_{x}^{2}+\beta_{y} ^{2})\sin^{2}\phi\pm 2\beta_{x}\beta_{y}(\sin^{2}\phi-2\cos^{2}\phi)\right]=\\ \approxq\mp\frac{4\beta_{x}\beta_{y}\sin^{2}\Delta\mu_{x}}{f^{2}}\phi^{2}+o( \phi^{4})\end{array} \tag{6.24}\]

We conclude that for the difference resonance, \(\Delta_{-}>0\), thereby Eq. 6.23 provides real quantities. Instead, for the sum resonance, \(\Delta_{+}<0\), and Eq. 6.23 provides imaginary quantities. We now remind that the eigenvalues of \(M_{t}\) must satisfy \(\prod_{j=1}^{4}\lambda_{j}=\det M=1\) (see Eq. 4.81). So, by virtue of Floquet's theorem we expect \(\lambda_{1,2}=pe^{\pm i\mu_{j}}\), \(\lambda_{3,4}=\frac{1}{p}e^{\pm i\mu_{j}}\). Equation 6.23 allows us to discriminate two situations:

\[\lambda_{j}+\frac{1}{\lambda_{j}}=\left\{\begin{array}{ll}2\cos\mu_{j}\in \mathbb{R}e&for\ \ \ \Delta_{-}>0&\Leftrightarrow\mu\in\mathbb{R}e\\ 2\cos\mu_{j}\in\mathbb{I}m&for\ \ \ \Delta_{+}<0&\Leftrightarrow\mu\in \mathbb{I}m\end{array}\right. \tag{6.25}\]

The former case for the difference resonance corresponds to real betatron phase advance, thus stable motion. The latter case of sum resonance corresponds to imaginary betatron phase advance, hence hyperbolic, unstable motion.

#### 6.2.3 Sextupole Resonances and Numerology

Sextupole magnets can induce \(3rd\) order resonances. This can be easily inferred by repeating the procedure adopted already to obtain Eq. 6.15, but here considering the quadratic dependence of the magnetic field from the particle's coordinate \(u\). If \(m\) is the normalized sextupole's gradient, then we have:

\[\begin{array}{l}\delta w^{\prime}=\sqrt{\beta_{u}}\delta(ml)u^{2}=\beta_{u }^{3/2}\delta(ml)w^{2}=\beta_{u}^{3/2}\delta(ml)a^{2}\cos^{3}(Q\theta)\\ \Rightarrow(\Delta\,Q)_{turn}\approx\frac{1}{2\pi}\frac{\delta w^{\prime}\cos (Q\theta)}{a}=\frac{1}{8\pi}\beta_{u}^{3/2}\delta(ml)\left[3\cos(Q\theta)+ \cos(3Q\theta)\right]\end{array} \tag{6.26}\]

The tune-shift induced by a sextupole gradient error is maximum when \((3Q\theta)_{turn}=6\pi\,Q=2\pi\,r\), i.e., \(Q=r/3\), \(r\in\mathbb{N}\).

It is now apparent that, by virtue of the aforementioned trigonometric relations, and in the approximation of a weak perturbation to the poles geometry ("systematicerror"), any magnet of _odd (even) multipole order_\(m\) could excite only resonances of _even (odd) resonance order_\(n=m+l\) (according to our notation in Eq. 4.47, \(m\in\mathbb{N}\)).

We also recognize lower order resonances in Eq. 6.26, associated to a feed-down effect of the magnetic field. The feed-down multipole order is \(m_{l}=m-2p\), \(p\in\mathbb{N}\). For example, a decapole magnet (\(m=4\)) can behave as a sextupole and a dipole magnet in the sense that it allows systematic resonances of order \(n_{l}=m_{l}+1\). When \(p=1\), \(m_{l}=4-2=2\) (equivalent sextupole magnet) and therefore \(n_{l}=2+1=3\). For \(p=2\), \(m_{l}=4-4=0\) (equivalent dipole magnet) and therefore \(n_{l}=0+1=1\).

Owing to the finite pole width, higher order components of the multipole magnetic field can be generated, but they are selected by the symmetry of the poles. Consequently, higher order resonances are allowed, so that \(n_{h}=(m+1)(2r+1)\), \(r\in\mathbb{N}\). For example, a dipole magnet (\(m\)=0) can also generate \(3rd\), \(5th\),...resonance orders, equivalently to a sextupole, a decapole, etc.

In general, both lower and higher order resonances correspond to driving terms in the equations of motion, whose amplitude is smaller than that of the main resonance order \(n=m+1\). This is expected because the magnet is primarily conceived to behave as a multipole magnet of order \(m\) (unless combined functions magnets are considered). Random field errors are also commonly present because of manufacturing and assembly inaccuracies. In this case the field or gradient uniformity is perturbed with no symmetry rules, and any resonance order is in principle allowed.

### 6.3 Linear Chromaticity

#### Natural Chromaticity

Linear chromaticity is the linear dependence of strong focusing from the particle's longitudinal momentum [1]. It is represented by the normalized gradient "error" \(k\delta\) in Eqs. 4.53, 4.55. In a synchrotron, it manifests primarily as a tune shift. The tune shift is intended to be a "coherent" effect when the energy deviation is referring to the synchronous particle but off-energy (e.g., in case of injection energy mismatch into a synchrotron, dipoles field error, energy ramping, etc.). It is "incoherent" when the motion of generic off-energy particles inside a bunch is considered.

The tune shift is quantified below through the matrix formalism. \(M_{t}\) is the one-turn transfer matrix (see Eq. 4.120), \(M_{q}\) and \(M_{e}\) the matrix of the thin lens quadrupole magnet (see Eq. 4.75), associated respectively to the nominal focal length \(f^{-1}=kl\) and to the perturbed one, \(\tilde{f}^{-1}=(k+\Delta k)l\). The one-turn perturbed transfer matrix is:\[\begin{array}{l}\tilde{M}_{t}=M_{e}M_{q}^{-1}M_{t}=\begin{pmatrix}1&0\\ (k+\Delta k)l&1\end{pmatrix}\begin{pmatrix}1&0\\ -kl&1\end{pmatrix}\begin{pmatrix}\cos\Delta\mu_{0}&\beta\sin\Delta\mu_{0}\\ \frac{1}{\beta}\sin\Delta\mu_{0}&\cos\Delta\mu_{0}\end{pmatrix}=\\ =\begin{pmatrix}\cos\Delta\mu_{0}&\beta\sin\Delta\mu_{0}\\ \Delta kl\cos\Delta\mu_{0}-\frac{1}{\beta}\sin\Delta\mu_{0}&\cos\Delta\mu_{0} +\beta\Delta kl\sin\Delta\mu_{0}\end{pmatrix}\end{array}\end{array} \tag{6.27}\]

The perturbed phase advance is retrieved from the matrix trace, and evaluated for a small perturbation (\(\Delta\mu-\Delta\mu_{0}<<\Delta\mu_{0}\)):

\[\begin{array}{l}\cos\Delta\mu=\frac{1}{2}Tr(\tilde{M}_{t})=\cos\Delta\mu_{0 }+\frac{1}{2}\beta\Delta kl\sin\Delta\mu_{0};\\ \cos\Delta\mu-\cos\Delta\mu_{0}\approxq-\sin\Delta\mu_{0}\cdot\Delta\mu= \frac{1}{2}\beta\Delta kl\sin\Delta\mu_{0};\\ \Rightarrow\Delta Q_{u}=\frac{\Delta\mu_{u}}{2\pi}=-\frac{1}{4\pi}\beta_{u} \Delta kl\end{array} \tag{6.28}\]

When considering the sum of small independent chromatic perturbations distributed along the lattice, \(\beta_{u}\,\Delta kl\,\rightarrow\,\beta_{u}(s)k(s)\delta ds\), and the linear chromaticity in the \(u\)-plane is:

\[\xi_{u}^{nat}:=\frac{\Delta Q_{u}}{\delta}=-\frac{1}{4\pi}\oint ds\,\beta_{u} (s)k(s) \tag{6.29}\]

\(\xi_{u}^{nat}\) is called _natural chromaticity_. In alternated gradient lattices, \(k(s)\) has opposite sign in the \(x\) and \(y\) plane, at each quadrupole. Nevertheless, with the convention \(k\,>\,0\) for focusing magnets, the total chromaticity is always a negative quantity, in both planes, because transverse stability corresponds to an overall focusing effect.

Natural chromaticity in synchrotrons has typical absolute values in the range \(\sim\)10 \(-\) 100. If not corrected, it can shift the betatron tunes to the proximity of strong resonances. For example, a beam's energy relative mismatch of 0.01% (e.g., 300 keV at the nominal energy of 3 GeV) can lead to a fractional coherent tune shift \(\Delta\,Q\,\sim\,0.005\), and 10-times larger internally to the bunch, where typically \(\delta\,\sim\,0.1\%\). A safe control of the working point commonly tolerates residual variations of the tunes \(\Delta\,Q\,<\,0.001\).

#### Chromaticity Correction

Chromaticity correction cannot be done with additional quadrupole magnets, since they would simply add their contribution to the natural chromaticity further. We then need to recur to higher order field magnets, such as sextupoles, as shown below. Sextupoles act like quadrupoles by means of a feed-down effect, though at the expense of additional nonlinear focusing and therefore higher order resonances.

The magnetic field components in a focusing sextupole magnet are shown in Fig. 6.2, and can be written as follows:

\[\left\{\begin{array}{l}B_{y}=\frac{1}{2}g^{\prime}(x^{2}-y^{2})\\ B_{x}=g^{\prime}xy\end{array}\right.,\;\;g^{\prime}=\frac{\partial^{2}B_{y}}{ \partial x^{2}} \tag{6.30}\]By substituting the solutions of Hill's equations (see Eq. 4.59), we find:

\[\left\{\begin{array}{l}B_{y}=\frac{1}{2}g^{\prime}(x_{\beta}^{2}+D_{x}^{2} \delta^{2}+2x_{\beta}D_{x}\delta-y_{\beta}^{2})\\ \\ B_{x}=g^{\prime}(x_{\beta}y_{\beta}+y_{\beta}D_{x}\delta)\end{array}\right. \tag{6.31}\]

The sextupole field components show a quadratic dependence from the particle's coordinates, \(\sim x_{\beta}^{2}\), \(\sim y_{\beta}^{2}\), \(\sim\delta^{2}\), which leads to linear optics distortion denominated \(2nd\) order _optical aberrations_, geometric and chromatic, respectively. Nevertheless, two additional terms \(\propto x_{\beta}\), \(y_{\beta}\) can be interpreted as the effect of a quadrupole (linear) gradient, but proportional to the dispersion function at the sextupole's location. Hence, in analogy to the normalized integrated quadrupole gradient \(kds\), we can define a normalized integrated sextupole gradient \(m=\frac{eg^{\prime}}{p_{z,s}}\) such that the sextupole's focal length becomes \(\frac{1}{f_{sext}}=mD_{x}\delta ds\), and \(ds\) is the magnet's length.

The one-turn transfer matrix in Eq. 6.27 for the \(u\)-plane, but with the addition of a sextupole magnet in the lattice, becomes:

\[M_{t,cor}=S\tilde{M}_{t}=\begin{pmatrix}1&0\\ mD_{x}\delta ds&1\end{pmatrix}\begin{pmatrix}\cos\Delta\mu_{0}&\beta\sin\Delta \mu_{0}\\ \Delta kds\cos\Delta\mu_{0}+&\cos\Delta\mu_{0}+\\ -\frac{1}{\beta}\sin\Delta\mu_{0}&+\beta\Delta kds\sin\Delta\mu_{0}\end{pmatrix} \tag{6.32}\]

Its trace gives the sextupole-corrected phase advance:

\(\cos\Delta\mu=\frac{1}{2}Tr(\tilde{M}_{t,cor})=\cos\Delta\mu_{0}+\frac{1}{2} \beta ds\sin\Delta\mu_{0}(\Delta k+mD_{x}\delta)\);

\(\cos\Delta\mu-\cos\Delta\mu_{0}\approx-\sin\Delta\mu_{0}\cdot\Delta\mu=\frac {1}{2}\beta ds\sin\Delta\mu_{0}(\Delta k+mD_{x}\delta)\);

\(\Rightarrow\Delta Q_{u}=\frac{\Delta\mu_{u}}{2\pi}=-\frac{1}{4\pi}\beta_{u}( \Delta k+mD_{x}\delta)ds\)

Figure 6.2: Sextupole magnet of the Australian Synchrotron (left, photo reproduced under Creative Commons Zero license) and sketch of its magnetic field lines

The linear chromaticity in the two transverse planes is calculated in the presence of the sextupole magnet, properly taking into account the opposite sign of the quadrupole gradient in the two planes:

\[\Rightarrow\left\{\begin{array}{l}\xi_{x}^{cor}=\frac{\Delta Q_{x}}{\delta}=- \frac{1}{4\pi}\oint\beta_{x}(s)\left[k(s)+m(s)D_{x}(s)\right]ds\\ \\ \xi_{y}^{cor}=\frac{\Delta Q_{y}}{\delta}=-\frac{1}{4\pi}\oint\beta_{y}(s) \left[-k(s)+m(s)D_{x}(s)\right]ds\end{array}\right. \tag{6.33}\]

We draw the following conclusions.

1. To zero horizontal and vertical chromaticity simultaneously, we need at least two sextupole "families", providing opposite sign of their gradient. _Focusing_ and _defocusing sextupoles_ are rotated by \(60^{\circ}\) around the magnetic axis.
2. To minimize the sextupoles' gradient and therefore correct the chromaticities while reducing the strength of higher order optical aberrations and higher order resonances, sextupoles should be installed in regions of large horizontal dispersion (_chromatic sextupoles_).
3. Internally to dispersive regions, the sextupole gradient is further minimized by installing sextupoles in correspondence of a large betatron function (either horizontal or vertical). Since this has local maxima at quadrupole magnets, chromatic sextupoles should be installed in proximity of quadrupoles.

In practice, chromaticity is corrected to positive values \(\sim\)\(1-10\) to counteract single and multi-bunch collective effects (see later). A large number of sextupole families is commonly adopted, not only to minimize the individual sextupoles' strength, but also to mutually cancel optical aberrations. Sextupoles devoted to the minimization of aberrations can be installed in non-dispersive regions (_harmonic sextupoles_).

##### Discussion: Chromaticity in Multi-bend Lattices

Synchrotrons are made of arc cells with a number of dipole magnets per arc commonly in the range \(N_{b}=2-7\). The total number of dipole magnets is usually large enough to ensure that the dipole's bending angle is \(\theta_{b}<<1\). Lattices with \(N_{b}>3\) are most recent designs, denominated "multi-bend lattices". We want to evaluate if, generally speaking, the sextupoles' strengths required for chromaticity correction are stronger in a lattice with large or small number of dipoles.

For any given natural chromaticity, the strength of chromatic sextupoles is minimized by a large dispersion function at the sextupoles' location, see Eq. 6.33. In the approximation of small bending angles, the dispersion is just proportional to the bending angle (Eq. 4.74). This points out that the larger the angle is, the larger the dispersion function excited in the arc will be.

The dipole bending angle depends in turn from the total number of dipoles, given that the total angle has to be \(2\pi\). Thus:

\[m_{sext}\propto\frac{1}{D_{x}}\propto\frac{1}{\theta_{b}}=\frac{N_{b}}{2\pi} \tag{6.34}\]We thereby expect lower sextupole gradients from a lattice with low number of dipoles per arc. In other words, for few dipoles per arc, each dipole has to bend the beam by a larger angle, which excites in turn a larger dispersion function. This makes the correction of chromaticity by sextupoles more efficient.

## References

* [1] J. Rossbach, P. Schmuser, Basic course on accelerator optics, in _Proceedings of CERN Accelerator School: 5th General Accelerator Physics Course_, CERN 94-01, vol. I, ed. by S. Turner (Geneva, Switzerland, 1994), pp. 69-79
* [2] E. Wilson, Non-linearities and resonances, in _Proceedings of CERN Accelerator School: 5th General Accelerator Physics Course_, CERN 94-01, vol. I, ed. by S. Turner (Geneva, Switzerland, 1994), pp. 239-252
* [3] M. Conte, W.W. MacKay, _An Introduction to the Physics of Particle Accelerators_, 2nd edn. (Published by World Scientific, 2008), pp. 127-134, 215-217. ISBN: 978-981-277-960-1

