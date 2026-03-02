## Chapter 13 Equilibrium Particle Distribution*

The wide variety of particle beam applications require often very specific beam characteristics in terms of say cross section, divergence, energy spread or pulse structure. To a large extend such parameters can be adjusted by particular application of focusing and other forces. In this chapter, we will discuss some of these methods of beam optimization and manipulation.

### 13.1 Particle Distribution in Phase Space

The beam emittance of particle beams is primarily defined by characteristic source parameters and source energy. Given perfect matching between different accelerators and beam lines during subsequent acceleration, this source emittance is reduced inversely proportional to the particle momentum by adiabatic damping and stays constant in terms of normalized emittance. This describes accurately the situation for proton and ion beams, for nonrelativistic electrons and electrons in linear accelerators.

The beam emittance for relativistic electrons, however, evolves fundamentally different in circular accelerators. Relativistic electron and positron beams passing through bending magnets emit synchrotron radiation, a process that leads to quantum excitation and damping. As a result, the original beam emittance at the source is completely replaced by an equilibrium emittance that is unrelated to the source characteristics.

This chapter has been made Open Access under a CC BY 4.0 license. For details on rights and licenses please read the Correction [https://doi.org/10.1007/978-3-319-18317-6_28](https://doi.org/10.1007/978-3-319-18317-6_28)

#### Diffusion Coefficient and Synchrotron Radiation

Emission of a photon causes primarily a change of the particle energy but the characteristics of the particle motion is changed as well. Neither position nor the direction of the particle trajectory is changed during the emission of photons. From beam dynamics, however, we know that different reference trajectories exist for particles with different energies. Two particles with energies \(cp_{0}\) and \(cp_{1}\) follow two different reference trajectories separated at the position \(z\) along the beam transport line by a distance

\[\Delta x=\eta(z)\,\frac{cp_{1}-cp_{0}}{cp_{0}}\,, \tag{13.1}\]

where \(\eta(z)\) is the dispersion function and \(cp_{0}\) the reference energy. Although particles in general do not exactly follow these reference trajectories they do perform betatron oscillations about them. The sudden change of the particle energy during the emission of a photon leads to a sudden change in the reference path and thereby to a sudden change in the betatron oscillation amplitude.

Following the discussion of the Fokker-Planck equation in Chap. 12, we may derive a diffusion coefficient from these sudden changes in the coordinates. Using normalized coordinates \(w=x/\sqrt{\beta}\), the change in the betatron amplitude at the moment a photon of energy \(\epsilon_{\gamma}\) is emitted becomes

\[\Delta w=\xi=-\frac{\eta(z)}{\sqrt{\beta_{x}}}\frac{\epsilon_{\gamma}}{E_{0}}. \tag{13.2}\]

Similarly, the conjugate coordinate \(\dot{w}=\sqrt{\beta_{x}}\,x^{\prime}_{\beta}+\alpha_{x}\,x_{\beta}\) changes by

\[\Delta\dot{w}=\pi\,=-\sqrt{\beta_{x}}\eta^{\prime}\frac{\epsilon_{\gamma}}{E_ {0}}-\frac{\alpha_{x}}{\sqrt{\beta_{x}}}\eta\frac{\epsilon_{\gamma}}{E_{0}}. \tag{13.3}\]

The frequency at which these statistical variations occur is the same for \(\xi\) and \(\pi\) and is equal to the number of photons emitted per unit time

\[\mathcal{N}_{\xi}=\mathcal{N}_{\pi}\,=\mathcal{N}. \tag{13.4}\]

From (13.2), (13.3) we get

\[\xi^{2}+\pi^{2}=\left(\frac{\epsilon_{\gamma}}{E_{0}}\right)^{2}\left[\frac{ \eta^{2}}{\beta_{x}}+\left(\sqrt{\beta_{x}}\eta^{\prime}+\frac{\alpha_{x}}{ \sqrt{\beta_{x}}}\eta\right)^{2}\right]=\left(\frac{\epsilon_{\gamma}}{E_{0}} \right)^{2}\mathcal{H}\,, \tag{13.5}\]where we have defined a special lattice function

\[\mathcal{H}=\beta_{x}\eta^{\prime 2}+2\,\alpha_{x}\eta\eta^{\prime}+\gamma_{x} \eta^{2}. \tag{13.6}\]

We are interested in the average value of the total diffusion coefficient (12.101)

\[D=\tfrac{1}{2}\langle\mathcal{N}\,(\xi^{2}+\pi^{2})\rangle_{z}=\frac{1}{2\,E_ {0}^{2}}\langle\mathcal{N}\,\langle\epsilon_{\gamma}^{2}\rangle\,\mathcal{H} \rangle_{z}\,, \tag{13.7}\]

where the average \(\langle\cdots\rangle_{z}\) is to be taken along the whole transport line or the whole circumference of a circular accelerator. Since photon emission does not occur outside of bending magnets, the average is taken only along the length of the bending magnets. To account for the variation in photon energies, we use the rms value of the photon energies \(\langle\epsilon_{\gamma}^{2}\rangle\). The theory of synchrotron radiation is discussed in much detail in Chap. 23 and we take in the following paragraph only relevant results of this theory.

The number of photons emitted per unit time with frequencies between \(\omega\) and \(\omega+\mathrm{d}\omega\) is simply the spectral radiation power at this frequency divided by the photon energy \(\hbar\omega\). Here, we consider only bending magnet radiation and treat radiation from insertion devices as perturbations. Of course, this approach must be modified if a significant part of radiation comes from non-bending magnet radiation. The spectral photon flux from a single electron is from (25.132) with the synchrotron radiation power (24.34)

\[\frac{\mathrm{d}n(\omega)}{\mathrm{d}\omega}=\frac{1}{\hbar\,\omega}\,\frac{ \mathrm{d}P(\omega)}{\mathrm{d}\omega}=\frac{P_{\gamma}}{\hbar\omega_{c}^{2}} \frac{9\sqrt{3}}{8\pi}\int_{\xi}^{\infty}K_{5/3}(x)\,\mathrm{d}x\,, \tag{13.8}\]

where \(\xi=\omega/\omega_{c}\) and the critical photon energy defined in (24.49). The total photon flux is by integration over all frequencies

\[\mathcal{N}=\frac{P_{\gamma}}{\hbar\omega_{c}}\frac{9\sqrt{3}}{8\,\pi}\int_{0 }^{\infty}\int_{\xi}^{\infty}K_{5/3}(x)\,\mathrm{d}x\,\mathrm{d}\xi \tag{13.9}\]

which becomes with GR(6.561.16) and \(\Gamma(1/6)\,\Gamma(1/6)=5\pi/3\) after integration by parts from AS(6.1.17)

\[\mathcal{N}=\frac{P_{\gamma}}{\hbar\omega_{c}}\frac{9\sqrt{3}}{8\,\pi}\int_{0 }^{\infty}K_{5/3}(\xi)\,\mathrm{d}\xi=\frac{15\sqrt{3}}{8}\frac{P_{\gamma}}{ \hbar\omega_{c}}. \tag{13.10}\]

The rms value of the photon energy \(\langle\epsilon_{\gamma}^{2}\rangle\) can be derived in the usual way from the spectral distribution \(n\left(\omega\right)\) by

\[\langle\epsilon_{\gamma}^{2}\rangle=\frac{\hbar^{2}}{\mathcal{N}}\int_{0}^{ \infty}\omega^{2}n(\omega)\,\mathrm{d}\omega=\frac{9\sqrt{3}P_{\gamma}\hbar \omega_{c}}{8\pi\mathcal{N}}\int_{0}^{\infty}\xi^{2}\int_{\xi}^{\infty}K_{5/3 }(x)\,\mathrm{d}x\,\mathrm{d}\xi \tag{13.11}\]and is after integration by parts

\[\langle\,\epsilon_{\gamma}^{2}\,\rangle\,=\,\frac{P_{\gamma}\hbar\omega_{c}}{{ \cal N}}\frac{9\sqrt{3}}{8\pi}\frac{1}{3}\int_{0}^{\infty}\xi^{3}K_{5/3}(\xi)\, \mathrm{d}\xi\,. \tag{13.12}\]

The integral of the modified Bessel's function in (13.12) is from GR[6.561.16] \(4\,\Gamma(2+\frac{5}{6})\,\Gamma(2-\frac{5}{6})\) where we use again AS(6.1.17) for \(\Gamma\left(\frac{5}{6}\right)\,\Gamma(\frac{1}{6})\,=\,2\pi\). Collecting all terms

\[{\cal N}\,\langle\,\epsilon_{\gamma}^{2}\,\rangle\,=\,\frac{55}{24\sqrt{3}}P_ {\gamma}\,\hbar\omega_{c} \tag{13.13}\]

and the diffusion coefficient (13.7) becomes

\[D=\tfrac{1}{2}(\,{\cal N}\,(\xi^{2}+\pi^{2}))_{z}=\frac{55}{48\sqrt{3}}\frac{ \langle P_{\gamma}\,\hbar\omega_{c}{\cal H}\,\rangle_{z}}{E_{0}^{2}}. \tag{13.14}\]

The stationary solution for the Fokker-Planck equation has been derived describing the equilibrium particle distribution in phase space under the influence of quantum excitation and damping. In all six dynamical degrees of freedom the equilibrium distribution is a Gaussian distribution and the standard value of the distribution width is determined by the damping time and the respective diffusion coefficient. In this chapter, we will be able to calculate quantitatively the diffusion coefficients and from that the beam parameters.

##### Quantum Excitation of Beam Emittance

High energy electron or positron beams passing through a curved beam transport line suffer from quantum excitation which is not compensated by damping since there is no acceleration. In Sect. 12.3 we have discussed this effect and found the transverse beam emittance to increase linear with time (12.142) and we get with (13.14)

\[\frac{\mathrm{d}\epsilon_{x}}{c\mathrm{d}t}=\,\frac{\mathrm{d}\epsilon_{x}}{ \mathrm{d}z}=\,\frac{55}{24\sqrt{3}}\frac{r_{\mathrm{e}}\,\hbar c}{mc^{2}} \gamma^{5}\left(\frac{{\cal H}}{\rho^{3}}\right)_{z}\,. \tag{13.15}\]

There is a strong energy dependence of the emittance increase along the beam transport line and the effect becomes significant for high beam energies as proposed for linear collider systems. Since the emittance blow up depends on the lattice function \({\cal H}\), we would choose a very strong focusing lattice to minimize the dilution of the beam emittance. For this reason, the beam transport system for the linear collider at the Stanford Linear Accelerator Center [1] is composed of very strongly focusing combined bending magnets.

Particle distributions become modified each time we inject a beam into a circular accelerator with significant synchrotron radiation. Arbitrary particle distributions can be expected from injection systems before injection into a circular accelerator. If the energy in the circular accelerator is too small to produce significant synchrotron radiation the particular particle distribution is preserved according to Liouville's theorem while all particles merely rotate in phase space as discussed in Sect. 12.1. As the beam energy is increased or if the energy is sufficiently high at injection to generate significant synchrotron radiation, all modes in the representation of the initial particle distribution vanish within a few damping times while only one mode survives or builds up which is the Gaussian distribution with a standard width given by the diffusion constant and the damping time. In general, any deviation from this unique equilibrium solution and be it only a mismatch to the correct orientation of the beam in phase space will persist for a time not longer than a few damping times.

### 13.2 Equilibrium Beam Emittance

In circular electron accelerators, as in electron storage rings, quantum excitation is counteracted by damping. Since quantum excitation is not amplitude dependent but damping is, there is an equilibrium beam emittance when both effects are equally strong. In the presence of quantum fluctuations Liouville's theorem is not applicable strictly anymore. In the case of an electron beam in equilibrium the phase space density for a beam in equilibrium is preserved, although in a different way. While Liouville's theorem is based on Hamiltonian mechanics and demands that no particle should escape its phase space position we allow in the case of an electron beam in equilibrium that a particle may escape its phase space position but be replaced instantly by another particle due to damping.

##### Horizontal Equilibrium Beam Emittance

The horizontal beam size is related to damping and diffusion coefficient from (12.113) like

\[\frac{\sigma_{x}^{2}}{\beta_{x}}=\tau_{x}D_{x}\,. \tag{13.16}\]

Damping times have been derived in Sect. 12.2 and with (13.7) the horizontal beam size \(\sigma_{x}\) at a location where the value of the betatron function is \(\beta_{x}\) becomes

\[\frac{\sigma_{x}^{2}}{\beta_{x}}=\frac{\langle\mathcal{N}\langle\epsilon_{ \gamma}^{2}\rangle\mathcal{H}\rangle_{z}}{2\,E_{0}\,J_{x}\langle P_{\gamma} \rangle_{z}}. \tag{13.17}\]The ratio \(\sigma_{x}^{2}/\beta_{x}\) is consistent with our earlier definition of the beam emittance. For a particle beam which is in equilibrium between quantum excitation and damping, this ratio is defined as the equilibrium beam emittance being equivalent to the beam emittance for all particles within one standard value of the Gaussian distribution. For further simplification, we make use of the expression (13.13) and get with the radiation power (24.34) and the critical frequency (24.49) the horizontal beam emittance equation

\[\epsilon_{x}=C_{\rm q}\gamma^{2}\frac{\langle\mathcal{H}/|\rho^{3}|\rangle_{z }}{J_{x}\,\langle 1/\rho^{2}\rangle_{z}}\,, \tag{13.18}\]

where we adopted Sands' [2] definition of a quantum excitation constant

\[C_{\rm q}=\frac{55}{32\,\sqrt{3}}\frac{\hbar c}{mc^{2}}=3.84\times 10^{-13}\, \text{m}. \tag{13.19}\]

The equilibrium beam emittance scales like the square of the beam energy and depends further only on the bending radius and the lattice function \(\mathcal{H}\). From the definition of \(\mathcal{H}\) the horizontal equilibrium beam emittance depends on the magnitude of the dispersion function and can therefore be adjusted to small or large values depending on the strength of the focusing for the dispersion function.

#### Vertical Equilibrium Beam Emittance

The vertical beam emittance follows from (13.18) considering that the dispersion function and therefore \(\mathcal{H}\) vanishes. Consequently, the equilibrium vertical beam emittance seems to be zero because there is only damping but no quantum excitation. In this case we can no longer ignore the fact that the photons are emitted into a finite although very small angle about the forward direction of particle propagation. Each such emission causes both a loss in the particle energy and a transverse recoil deflecting the particle trajectory. The photons are emitted typically within an angle \(1/\gamma\) generating a transverse kick without changing the betatron oscillation amplitude. With \(\delta y=0\) and \(\delta y^{\prime}=\frac{1}{\gamma}\frac{\epsilon_{y}}{E_{0}}\), we get for the statistical variations

\[\begin{split}\xi^{2}&=0\,,\\ \pi^{2}&=\beta_{y}\frac{1}{\gamma^{2}}\left(\frac{ \epsilon_{y}}{E_{0}}\right)^{2}\,.\end{split} \tag{13.20}\]

Following a derivation similar to that for the horizontal beam emittance, we get for the vertical beam emittance equation

\[\epsilon_{y}=C_{\rm q}\frac{\langle\beta_{y}/|\rho^{3}|\rangle_{z}}{J_{y}(1/ \rho^{2})_{z}}\,. \tag{13.21}\]This is the fundamentally lower limit of the equilibrium beam emittance due to the finite emission angle of synchrotron radiation. For an isomagnetic ring the vertical beam emittance

\[\epsilon_{y}=C_{\mathrm{q}}\frac{\langle\beta_{y}\rangle_{z}}{J_{y}\left|\rho\right|} \tag{13.22}\]

does not depend on the particle energy but only on the bending radius and the average value of the betatron function. In most practical circular accelerator designs, both the bending radius and the betatron function are of similar magnitude and the fundamental emittance limit therefore is of the order of \(C_{\mathrm{q}}=10^{-13}\) radian meter, indeed very small compared to actually achieved beam emittances.

The assumption that the vertical dispersion function vanishes in a flat circular accelerator is true only for an ideal ring. Dipole field errors, quadrupole misalignments and any other source of undesired dipole fields create a vertical closed orbit distortion and an associated vertical dispersion function. This vertical dispersion function, often called spurious dispersion function, is further modified by orbit correction magnets but it is not possible to completely eliminate it because the location of dipole errors are not known.

Since the diffusion coefficient \(D\) is quadratic in the dispersion function (13.7) we get a contribution to the vertical beam emittance from quantum excitation similar to that in the horizontal plane. Indeed, this effect on the vertical beam emittance is much larger than that due to the finite emission angle of photons discussed above and is therefore together with coupling the dominant effect in the definition of the vertical beam emittance.

The contribution to the vertical beam emittance is in analogy to the derivation leading to (13.18)

\[\Delta\epsilon_{y}=C_{\mathrm{q}}\gamma^{2}\frac{\langle\mathcal{H}_{y}/|\rho ^{3}|\rangle_{z}}{J_{y}\langle 1/\rho^{2}\rangle_{z}}\,, \tag{13.23}\]

where

\[\frac{\mathcal{H}_{y}}{\left|\rho\right|^{3}}=\left\langle\frac{\beta_{y}{ \eta_{y}^{\prime}}^{2}+2\alpha_{y}\eta_{y}\eta_{y}^{\prime}+\gamma_{y}\eta_{y} ^{2}}{\left|\rho\right|^{3}}\right\rangle_{z}. \tag{13.24}\]

To minimize this effect, orbit correction schemes must be employed which not only correct the equilibrium orbit but also the perturbation to the dispersion function. Of course, the same effect with similar magnitude occurs also in the horizontal plane but is in general negligible compared to ordinary quantum excitation.

### Equilibrium Energy Spread and Bunch Length

The statistical processes caused by the emission of synchrotron radiation photons affect not only the four transverse dimensions of phase space but also the energy-time phase space. Particles orbiting in a circular accelerator emit photons with a statistical distribution of energies while only the average energy loss is replaced in the accelerating cavities.

##### Equilibrium Beam Energy Spread

This leaves a residual statistical distribution of the individual particle energies which we have derived in Sect. 12.3 to be Gaussian just like the transverse particle distribution with a standard width given by (12.121). The conjugate coordinate is the "time" \(w=\frac{\Omega}{\eta_{c}}\tau\) where \(\tau\) is the deviation in time of a particle from the synchronous particle, and \(\epsilon\) the energy deviation of a particle from the reference energy \(E_{0}\).

The emission of a photon will not change the position of the particle in time and therefore \(\xi=0\). The conjugate coordinate being the particle energy will change due to this event by the magnitude of the photon energy and we have \(\pi=\epsilon_{\gamma}/E_{0}\). Comparing with (13.5), we note that we get the desired result analogous to the transverse phase space by setting \(\mathcal{H}=1\) and using the correct damping time for longitudinal motion. The equilibrium energy spread becomes then from (12.121) in analogy to (13.18)

\[\frac{\sigma_{\epsilon}^{2}}{E_{0}^{2}}=C_{\rm q}\gamma^{2}\frac{\langle|1/ \rho^{3}|\rangle_{z}}{J_{\epsilon}\langle 1/\rho^{2}\rangle_{z}}\,, \tag{13.25}\]

which in a separated function lattice depends only on the particle energy and the bending radius. In a fully or partially combined function lattice, the partition number \(J_{\epsilon}\) can be modified providing a way to vary the energy spread.

##### Equilibrium Bunch Length

There is also a related equilibrium distribution in the longitudinal dimension which defines the length of the particle bunch. This distribution is also Gaussian and the standard bunch length is from (12.123), (12.124)

\[\sigma_{\ell}=c\beta\frac{|\eta_{\rm c}|}{\Omega_{\rm s}}\frac{\sigma_{ \epsilon}}{E_{0}}. \tag{13.26}\]The equilibrium bunch length not only depends on the particle energy and the bending radius but also on the focusing lattice through the momentum compaction factor and the partition number as well as on rf-parameters included in the synchrotron oscillation frequency \(\Omega_{\rm s}\). To exhibit the scaling, we introduce lattice and rf-parameters into (13.26) to get with (13.25) and the definition of the synchrotron frequency (9.32) an expression for the equilibrium bunch length

\[\sigma_{\ell}^{2} = \frac{2\pi\,C_{\rm q}}{(mc^{2})^{2}}\frac{\eta_{c}E_{0}^{3}R^{2} }{J_{\epsilon}he\hat{V}_{0}\cos\psi_{\rm s}}\,\frac{\langle|1/\rho^{3}|\rangle _{z}}{\langle 1/\rho^{2}\rangle_{z}}, \tag{13.27}\]

where \(R\) is the average radius of the ring. The bunch length can be modified through more parameters than any other characteristic beam parameter in the six-dimensional phase space. Lattice design affects the resulting bunch length through the momentum compaction factor and the partition number. Strong focusing results in a small value for the momentum compaction factor and a small bunch length. Independent of the strength of the focusing, the momentum compaction factor can in principle be adjusted to any value including zero and negative values by allowing the dispersion function to change sign along a circular accelerator because the momentum compaction factor is the average of the dispersion function \(\alpha_{\rm c}=\langle\eta/\rho\rangle\). In this degree of approximation the bunch length could therefore be reduced to arbitrarily small values by reducing the momentum compaction factor. However, close to the transition energy phase focusing to stabilize synchrotron oscillations is lost.

Introduction of gradient magnets into the lattice modifies the partition numbers as we have discussed in Sect. 12.2.1. As a consequence, both, the energy spread and bunch length increase or decrease at the expense of the opposite effect on the horizontal beam emittance. The freedom to adjust any of these three beam parameters in this way is therefore limited but nonetheless an important means to make small adjustments if necessary. Obviously, the rf-frequency as well as the rf-voltage have a great influence on the bunch length. The bunch length scales inversely proportional to the square root of the rf-frequency and is shorter for higher frequencies. Generally, no strong reasons exist to choose a particular rf-frequency but might become more important if control of the bunch length is important for the desired use of the accelerator. The bunch length is also determined by the rate of change of the rf-voltage in the accelerating cavities at the synchronous phase

\[\hat{V}(\psi_{\rm s}) = \frac{\rm d}{\rm d\psi}\,\hat{V}\sin\psi\bigg{|}_{\psi=\psi_{\rm s }}=\hat{V}\cos\psi_{\rm s}\,. \tag{13.28}\]

For a single frequency rf-system the bunch length can be shortened when the rf-voltage is increased. To lengthen the bunch the rf-voltage can be reduced up to a point where the rf-voltage would fail to provide a sufficient energy acceptance.

### Phase-Space Manipulation

The distribution of particles in phase space is given either by the injector characteristics and injection process or in the case of electron beams by the equilibrium of quantum excitation due to synchrotron radiation and damping. The result of these processes are not always what is desired and it is therefore useful to discuss some method to modify the particle distribution in phase space within the validity of Liouville's theorem.

##### Exchange of Transverse Phase-Space Parameters

In beam dynamics we are often faced with the desire to change the beam size in one of the six phase-space dimensions. Liouville's theorem tells us that this is not possible with macroscopic fields unless we let another dimension vary as well so as not to change the total volume in six-dimensional phase space.

A very simple example of exchanging phase-space dimensions is the increase or decrease of one transverse dimension at the expense of its conjugate coordinate. A very wide and almost parallel beam, for example, can be focused to a small spot size where, however, the beam divergence has become very large. Obviously, this process can be reversed too and we describe such a process as the rotation of a beam in phase space or as phase-space rotation.

A more complicated but often very desirable exchange of parameters is the reduction of beam emittance in one plane at the expense of the emittance in the other plane. Is it, for example, possible to reduce say the vertical beam emittance to zero at the expense of the horizontal emittance? Although Liouville's theorem would allow such an exchange other conditions in Hamiltonian theory will not allow this kind of exchange in multidimensional phase space. The condition of symplecticity is synonymous with Liouville's theorem only in one dimension. For \(n\) dimensions the symplecticity condition imposes a total of \(n(2n-1)\) conditions on the dynamics of particles in phase space [3]. These conditions impose an important practical limitation on the exchange of phase space between different degrees of freedom. Specifically, it is not possible to reduce the smaller of two phase-space dimensions further at the expense of the larger emittance, or if the phase space is the same in two dimensions neither can be reduced at the expense of the other.

##### 13.4.2 Bunch Compression

Longitudinal phase space can be exchanged also by special application of magnetic and rf-fields. Specifically, we often face the problem to compress the bunch to a very short length at the expense of energy spread.

For linear colliders the following problem exists. Very small transverse beam emittances can be obtained only in storage rings specially designed for low equilibrium beam emittances. Therefore, an electron beam is injected from a conventional source into a damping ring specially designed for low equilibrium beam emittance. After storage for a few damping times the beam is ejected from the damping ring again and transferred to the linear accelerator to be further accelerated. During the damping process in the damping ring, however, the bunch length will also reach its equilibrium value which in practical storage rings is significantly longer than could be accepted in, for example, an S-band or X-band linear accelerator. The bunch length must be shortened.

This is done in a specially designed beam transport line between the damping ring and linear accelerator consisting of a non-isochronous transport line and an accelerating section installed at the beginning of this line (Fig. 13.1).

The accelerating section is phased such that the center of the bunch or the reference particle does not see any field while the particles ahead of the reference particle are accelerated and the particles behind are decelerated. Following this accelerating section, the particles travel through a curved beam transport system with a finite momentum compaction factor \(\alpha_{\mathrm{c}}=\frac{1}{L_{0}}\int_{0}^{L_{0}}\frac{\eta}{\rho}\,\mathrm{d}z\) where \(L_{0}\) is the length of the beam transport line. Early particles within a bunch, having been accelerated, follow a longer path than the reference particles in the center of the bunch while the decelerated particles being late with respect to the bunch center follow a shortcut. All particles are considered highly relativistic and the early particles fall back toward the bunch center while late particles catch up with the bunch center. If the parameters of the beam transport system are chosen correctly the bunch length reaches its minimum value at the desired location at, for example, the entrance of the linear accelerator. From that point on the phase-space rotation is halted because of lack of momentum compaction in a straight line. Liouville's theorem is not violated because the energy spread in the beam has been increased through the phase dependent acceleration in the bunch-compression system.

Figure 13.1: Bunch-compressor system (schematic)

Formulating this bunch compression in more mathematical terms, we start from a particle distribution in longitudinal phase space described by the phase ellipse

\[\hat{\tau}_{0}^{2}\epsilon^{2}+\hat{\epsilon}_{0}^{2}\tau^{2}=\hat{\tau}_{0}^{2} \hat{\epsilon}_{0}^{2}=a^{2}\,, \tag{13.29}\]

where \(a\) is the longitudinal emittance and \(\tau\) is the particle location along the bunch measured from the bunch center such that \(\tau>0\) if the particle trails the bunch center. In the first step of bunch compression, we apply an acceleration

\[\Delta\epsilon=-eV_{0}\sin\omega_{\rm rf}\tau\,\approx-eV_{0}\,\omega_{\rm rf }\tau\,. \tag{13.30}\]

The particle energy is changed according to its position along the bunch. Replacing \(\epsilon\) in (13.29) by \(\epsilon\,+\,\Delta\epsilon\) and sorting we get

\[\hat{\tau}_{0}^{2}\,\epsilon^{2}-2\hat{\tau}_{0}^{2}\,eV_{0}\,\omega_{\rm rf} \,\epsilon\,\tau\,+(\hat{\tau}_{0}^{2}\,e^{2}V_{0}^{2}\,\omega_{\rm rf}^{2}+ \hat{\epsilon}_{0}^{2})\,\tau^{2}=a^{2}\,, \tag{13.31}\]

where the appearance of the cross term indicates the rotation of the ellipse. The second step is the actual bunch compression in a non-isochronous transport line of length \(L\) and momentum compaction \(\Delta z/L=\eta_{c}\,\epsilon/(cp_{0})\). Traveling though this beam line, a particle experiences a shift in time of

\[\Delta\tau\,=\frac{\Delta z}{\beta c}=\frac{\eta_{c}L}{\beta c}\,\frac{ \epsilon}{cp_{0}}\,. \tag{13.32}\]

Again, the time \(\tau\) in (13.31) is replaced by \(\tau\,+\,\Delta\tau\) to obtain the phase ellipse at the end of the bunch compressor of length \(L\). The shortest bunch length occurs when the phase ellipse becomes upright. The coefficient for the cross term must therefore be zero giving a condition for minimum bunch length

\[eV_{0}=-\frac{cp_{0}\,\beta c}{L\eta_{c}\omega_{\rm rf}}\,. \tag{13.33}\]

From the remaining coefficients of \(\epsilon^{2}\) and \(\tau^{2}\), we get the bunch length after compression

\[\hat{\tau}\,=\,\frac{\hat{\epsilon}_{0}}{eV_{\rm rf}\,\omega_{\rm rf}} \tag{13.34}\]

and the energy spread

\[\hat{\epsilon}\,=\,\hat{\tau}_{0}\,\omega_{\rm rf}\,eV_{\rm rf}\,, \tag{13.35}\]

where we used the approximation \(\hat{\tau}_{0}\,eV_{0}\,\omega_{\rm rf}\gg\hat{\epsilon}_{0}\). This is justified because we must accelerate particles at the tip of the bunch by much more than the original energy spread to obtain efficient bunch compression. Liouville's theorem is obviously kept intact since

\[\hat{\epsilon}\;\hat{\tau}=\hat{\epsilon}_{0}\;\hat{\tau}_{0}\,. \tag{13.36}\]

For tight bunch compression, a particle beam with small energy spread is required as well as an accelerating section with a high rf-voltage and frequency. Of course, the same parameters contribute to the increase of the energy spread which can become the limiting factor in bunch compression. If this is the case, one could compress the bunch as much as is acceptable followed by acceleration to higher energies to reduce the energy spread by adiabatic damping, and then go through a bunch compression again.

The momentum compaction factor \(\alpha_{\mathrm{c}}=\frac{1}{L_{0}}\int\frac{\eta}{\rho}\mathrm{d}z\) is often referred to as the \(R_{56}\) of the compression lattice. This designation comes from the TRANSPORT nomenclature where a 6 \(\times\) 6-transformation matrix is defined for the variables \((x,x^{\prime},y,y^{\prime},s,\delta)\). Here \(s\) is the individual particle path length and \(\delta\) the relative energy deviation. The correlation of \(s\) with \(\delta\) is the \(R_{56}\) element and in linear approximation \(s=s_{0}+R_{56}\delta\). Recalling the definition of the momentum compaction factor \(\alpha_{\mathrm{c}}=\frac{\Delta L/L_{0}}{\Delta p/p_{0}}\) we recognize the identity \(\alpha_{\mathrm{c}}=\frac{1}{L_{0}}\int\frac{\eta}{\rho}\mathrm{d}z=\frac{R_{56 }}{L_{0}}\).

#### Alpha Magnet

Bunch compression requires two steps. First, an accelerating system must create a correlation between particle energy and position. Then, we utilize a non-isochronous magnetic transport line to rotate the particle distribution in phase space until the desired bunch length is reached.

The first step can be eliminated in the case of an electron beam generated in an rf-gun. Here the electrons emerge from a cathode which is inserted into an rf-cavity [4]. The electrons are accelerated immediately where the acceleration is a strong function of time because of the rapidly oscillating field. In Fig. 13.2 the result from computer simulations of the particle distribution in phase space [5] is shown for an electron beam from a 3 GHz rf-gun [6, 7] (Fig. 13.3).

For bunch compression we use an alpha magnet which got its name from the alpha like shape of the particle trajectories. This magnet is made from a quadrupole split in half where the other half is simulated by a magnetic mirror plate at the vertical midplane. While ordinarily a particle beam would pass through a quadrupole along the axis or parallel to this axis this is not the case in an alpha magnet. The particle trajectories in an alpha magnet have very unique properties which were first recognized by Enge [8]. Most obvious is the fact that the entrance and exit point can be the same for all particles independent of energy. The same is true also for the total deflection angle. Borland [9] has analyzed the particle dynamics in an alpha magnet in detail and we follow his derivation here. Particlesentering the alpha magnet fall under the influence of the Lorentz force

\[\mathbf{F}_{\rm L}=e\mathbf{E}+e[\mathbf{v}\times\mathbf{B}], \tag{13.37}\]

Figure 13.3: Cross section of a microwave electron gun [6; 7]

Figure 13.2: Particle distribution in phase space for an electron beam from an rf gun with thermionic cathode

Figure 13.3: Cross section of a microwave electron gun [6; 7]

where we ignore the electrical field. Replacing the magnetic field by its gradient \(\mathbf{B}=(g\,u_{3},0,g\,u_{1})\), we get in the coordinate system of Fig. 13.4 the equation of motion,

\[\frac{\mathrm{d}^{2}\mathbf{u}}{\mathrm{d}z^{2}}=-\sigma^{2}\left[\frac{\mathrm{d} \mathbf{u}}{\mathrm{d}z}\times\mathbf{u}\right], \tag{13.38}\]

where the scaling factor

\[\sigma^{2}\left(\mathrm{m}^{-2}\right)=\frac{eg}{mc^{2}\beta\gamma}=5.86674 \times 10^{6}\,\frac{g(\mathrm{T}/\mathrm{m})}{\beta\gamma}\,, \tag{13.39}\]

and the coordinate vector \(\mathbf{u}=(u_{1},u_{2},u_{3})\).

By introducing normalized coordinates \(\mathbf{U}=\sigma\,\mathbf{u}\) and path length \(S=\sigma z\), Eq. (13.38) becomes

\[\frac{\mathrm{d}^{2}\mathbf{U}}{\mathrm{d}S^{2}}=-\left[\frac{\mathrm{d}\mathbf{U}}{ \mathrm{d}S}\times(U_{3},0,U_{1})\right]. \tag{13.40}\]

The remarkable feature of (13.40) is the fact that it does not exhibit any dependence on the particle energy or the magnetic field. One solution for (13.40) is valid for all operating conditions and beam energies. The alpha shaped trajectories are similar to each other and scale with energy and field gradient according to the normalization introduced above.

Equation (13.40) can be integrated numerically and in doing so, Borland obtains for the characteristic parameters of the normalized trajectory in an alpha magnet [9]

\[\begin{array}{ll}\theta_{\alpha}=0.71052&\mathrm{rad}\,,\;\;S_{\alpha}=4.64 210,\\ =40.70991&\mathrm{deg}\,,\;\;\hat{U}_{1}=1.81782,\end{array} \tag{13.41}\]

w

Figure 13.4: Alpha magnet and particle trajectories

where \(\theta_{\alpha}\) is the entrance and exit angle with respect to the magnet face, \(S_{\alpha}\) is the normalized path length and \(\hat{U}_{1}\) is the apex of the trajectory in the alpha magnet. We note specifically that the entrance and exit angle \(\theta_{\alpha}\) is independent of beam energy and magnetic field. It is therefore possible to construct a beam transport line including an alpha magnet.

Upon introducing the scaling factor (13.39), (13.41) becomes equation

\[\begin{array}{rl}s_{\alpha}(\mathrm{m})&=\ \frac{S_{\alpha}}{\sigma}\ =0.19165\sqrt{\frac{\beta\gamma}{g\left( \mathrm{T}/\mathrm{m}\right)}},\\ \hat{u}_{1}(\mathrm{m})&=\ \frac{\hat{U}_{1}}{\sigma}\ =0.07505\sqrt{\frac{ \beta\gamma}{g\left(\mathrm{T}/\mathrm{m}\right)}}.\end{array} \tag{13.42}\]

Bunch compression occurs due to the functional dependence of the path length on the particle energy. Taking the derivative of (13.42) with respect to the particle momentum \(\tilde{p}_{0}=\beta\gamma\), one gets the compression equation

\[\frac{\mathrm{d}s_{\alpha}(\mathrm{m})}{\mathrm{d}\tilde{p}_{0}}\ =\ \frac{0.07505}{\sqrt{2\ g\left( \mathrm{T}/\mathrm{m}\right)\tilde{p}_{0}}}. \tag{13.43}\]

For bunch compression, higher momentum particles must arrive first because they follow a longer path and therefore fall back with respect to later particles. For example, an electron beam with the phase-space distribution from Fig. 13.2 becomes compressed as shown in Fig. 13.5.

Because of the small longitudinal emittance of the beam it is possible to generate very short electron bunches of some 100 f-sec (rms) duration which can be used to produce intense coherent far infrared radiation [10].

Figure 13.5: Particle distribution in longitudinal phase space after compression in an alpha magnet

### Polarization of a Particle Beam

For high energy physics experimentation, it is sometimes important to have beams of transversely or longitudinally polarized particles. It is possible, for example, to create polarized electron beams by photoemission from GaAs cathodes [11]. From a beam dynamics point of view, we are concerned with the transport of polarized beams through a magnet system and the resulting polarization status. The magnetic moment vector of a particle rotates about a magnetic field vector. An electron with a component of a longitudinal polarization traversing a vertical dipole field would experience a rotation of the longitudinal polarization about the vertical axis. On the other hand, the vertical polarization would not be affected while passing through a horizontally bending magnet. This situation is demonstrated in Fig. 13.6.

Similarly, longitudinal polarization is not affected by a solenoid field. In linear collider facilities, specific spin rotators are introduced to manipulate the electron spin in such a way as to preserve beam polarization and obtain the desired spin direction at an arbitrarily located collision point along the beam transport line. For the preservation of beam polarization, it is important to understand and formulate spin dynamics.

Electron and positron beams circulating for a long time in a storage ring can become polarized due to the reaction of continuous emission of transversely polarized synchrotron radiation. The evolution of the polarization has been studied in detail by several researchers [12, 13, 14, 15] and the polarization time is given by [15]

\[\frac{1}{\tau_{\mathrm{pol}}}=\frac{5\sqrt{3}}{8}\frac{r_{\mathrm{c}}c^{2}\hbar \gamma^{5}}{mc^{2}\rho^{3}} \tag{13.44}\]

with a theoretically maximum achievable polarization of 92.38 %. The polarization time is a strong function of beam energy and is very long for low energies. At

Figure 13.6: Precession of the particle spin in a transverse or longitudinal magnetic fieldenergies of several GeV, however, this time becomes short compared to the storage time of an electron beam in a storage ring.

This build up of polarization is counteracted by nonlinear magnetic field errors which cause precession of the spin depending on the betatron amplitude and energy of the particle thus destroying polarization. Again, we must understand spin dynamics to minimize this depolarization. Simple relations exist for the rotation of the spin while the particle passes through a magnetic field. To rotate the spin by a magnetic field, there must be a finite angle between the spin direction and that of the magnetic field. The spin rotation angle about the axis of a transverse field depends on the angle between the spin direction \(\boldsymbol{\sigma}_{\mathrm{s}}\left(\left|\boldsymbol{\sigma}_{\mathrm{s}} \right|=1\right)\) and magnetic field \(\boldsymbol{B}_{\perp}\) and is given by [14]

\[\psi_{\perp}=C_{\perp}\left(1+\frac{1}{\gamma}\right)\left|\boldsymbol{\sigma}_ {\mathrm{s}}\times\boldsymbol{B}_{\perp}\right|\ell\,, \tag{13.45}\]

where

\[\eta_{\mathrm{g}} = \frac{g-2}{2}=0.00115965\,, \tag{13.46}\] \[C_{\perp} = \frac{e\eta_{\mathrm{g}}}{mc^{2}}=0.0068033\,\left(\mathrm{T}^{- 1}\mathrm{m}^{-1}\right) \tag{13.47}\]

\(g\) the gyromagnetic constant and \(\boldsymbol{B}_{\perp}\ell\) the integrated transverse magnetic field strength. Apart from a small term \(1/\gamma\), the spin rotation is independent of the energy. In other words, a spin component normal to the field direction can be rotated by \(90\,^{\circ}\) while passing though a magnetic field of \(2.309\,\mathrm{Tm}\) and it is therefore not important at what energy the spin is rotated.

Equation (13.45) describes the situation in a flat storage ring with horizontal bending magnets only unless the polarization of the incoming beam is strictly vertical. Any horizontal or longitudinal polarization component would precess while the beam circulates in the storage ring. As long as this spin is the same for all particles the polarization would be preserved. Unfortunately, the small energy dependence of the precession angle and the finite energy spread in the beam would wash out the polarization. On the other hand the vertical polarization of a particle beam is preserved in an ideal storage ring. Field errors, however, may introduce a depolarization effect. Horizontal field errors from misalignments of magnets, for example, would rotate the vertical spin. Fortunately, the integral of all horizontal field components in a storage ring is always zero along the closed orbit and the net effect on the vertical polarization is zero. Nonlinear fields, however, do not cancel and must be minimized to preserve the polarization.

A transverse spin can also be rotated about the longitudinal axis of a solenoid field and the rotation angle is

\[\psi_{\parallel}=\frac{e}{E}\left(1+\eta_{\mathrm{g}}\frac{\gamma}{1+\gamma} \right)\left|\boldsymbol{\sigma}_{\mathrm{s}}\times\boldsymbol{B}_{\parallel}\right|\ell \tag{13.48}\]In a solenoid field it is therefore possible to rotate a horizontal polarization into a vertical polarization, or vice versa. Spin rotation in a longitudinal field is energy dependent and such spin rotations should therefore be done at low energies if possible.

The interplay between rotations about a transverse axis and the longitudinal axis is responsible for a spin resonance which destroys whatever beam polarization exists. To show this, we assume a situation where the polarization vector precesses just by \(2\pi\), or an integer multiple \(n\) thereof, while the particle circulates once around the storage ring. In this case \(\psi_{\perp}=n\,2\pi\), \(eB_{\perp}\ell/E=2\pi\), and we get from (13.45)

\[n=\eta_{\rm g}(1+\gamma)\,. \tag{13.49}\]

For \(n=1\), resonance occurs at a beam energy of \(E=440.14\,\)MeV. At this energy any small longitudinal field interacts with the polarization vector at the same phase, eventually destroying any transverse polarization. This resonance occurs at equal energy intervals of

\[E_{n}({\rm MeV})=440.14+\,440.65(n-1) \tag{13.50}\]

and can be used in storage rings as a precise energy calibration by observing the loss of polarization due to spin-resonances at \(E_{n}\) while the beam energy is changed.

In Fig. 13.7 spin dynamics is shown for the case of a linear collider where a longitudinally polarized beam is desired at the collision point. For example, a longitudinally polarized beam is generated at the source and accelerated in a linear accelerator. No rotation of the polarization direction occurs because no magnetic fields are involved yet. At an energy of 1.2 GeV the beam is transferred to a damping ring to reduce the beam emittance. To preserve polarization in the damping ring

Figure 13.7: Spin manipulation during beam transfer from linear accelerator to damping ring and back

the polarization must be vertical. In Fig. 13.7, we assume a longitudinal polarized beam coming out of the linear accelerator. A series of transverse fields amounting to \(5\times 2.309\) Tm creating a total deflection angle of \(5\times 32.8^{\circ}\). The longitudinal spin from the linac is rotated by \(5\times 90^{\circ}\) to a horizontal spin as shown in Fig. 13.7 by the open arrows. A solenoid of \(6.34\) Tm rotates this spin about the beam axis to become a vertical spin which survives the storage time in the damping ring. After emittance reduction in a few damping times the beam is ejected again with vertical spin. Now we have to decide which spin orientation we need at the collision point of the linear collider. there are two cases. In the first case (left side of the ejected beam in Fig. 13.7 a solenoid of \(6.34\) Tm rotates the spin to become transverse followed by a \(32.8^{\circ}\) bending section to make the spin longitudinal. This beam is injected back into the linac for collisions with longitudinal spin. of course, any bending downstream must be carefully implemented to preserve the spin. In the second case we turn the first solenoid after ejection off and the beam with vertical spin arrives at the second solenoid unaffected by the \(32.8^{\circ}\) bend. in the second solenoid the spin is rotated to become a horizontal spin which is then reinjected into the linac. Note in both cases there is some spin rotation after the second solenoid yet in both cases the effect on the spin is just what is desired to have a transverse or longitudinal spin in the linac.

To rotate the longitudinal into a horizontal spin, followed by a solenoid field which rotates the horizontal into a vertical spin, is used in the transport line to the damping ring to obtain the desired vertical spin orientation. This orientation is in line with all magnets in the damping ring and the polarization can be preserved.

To obtain the desired rotation in the beam transport magnets at a given energy, the beam must be deflected by a specific deflection angle which is from (13.45)

\[\theta\,=\,\frac{e}{\beta E}B_{\perp}\ell\,=\,\frac{\psi_{\perp}}{\eta_{\rm g }}\,\frac{1}{1\,+\,\gamma}\,. \tag{13.51}\]

Coming out of the damping ring the beam passes through a combination of two solenoids and two transverse field sections. Depending on which solenoid is turned on, we end up with a longitudinal or transverse polarization at the entrance of the linac. By the use of both solenoids any polarization direction can be realized.

## Problems

### 13.1 (S)

Show that the horizontal damping partition number is negative in a fully combined function FODO lattice as employed in older synchrotron accelerators. Why, if there is horizontal anti-damping in such synchrotrons, is it possible to retain beam stability during acceleration? What happens if we accelerate a beam and keep it orbiting in the synchrotron at some higher energy?

**13.2.** Use the high energy, linear part of the particle distribution in Fig. 13.2 and specify an alpha magnet to achieve best bunch compression at the observation point 2 m downstream from the magnet. By how much is the bunchlength increased if you now also include the variation of velocities. What are the alpha magnet parameters? Sketch the particle distribution at the entrance and exit of the alpha magnet.

**13.3.** Specify relevant parameters for an electron storage ring made of FODO cells with the goal to produce a very short equilibrium bunch length of \(\sigma_{\ell}=1\) mm. Use superconducting cavities for bunch compression with a maximum field of 10 MV/m and a total length of not more than 10 % of the ring circumference.

**13.4.** Describe spin rotations in bending magnets in matrix formulation.

**13.5.** Consider an electron storage ring for an energy of 30 GeV and a bending radius of \(\rho=500\) m and calculate the polarization time. The vertical polarization will be perturbed by spurious dipole fields. Use statistical methods to calculate the rms change of polarization direction per unit time and compare with the polarization time. Determine the alignment tolerances to get a polarized beam.

## Bibliography

* [1] H. Wiedemann, Basic lattice for the slc. Technical Report, Int. Report, Stanford Linear Accelerator Center, Stanford (1979)
* [2] M. Sands, The physics of electron storage rings, an introduction, in _Physics with Intersecting Storage Rings_ (Academic, New York, 1971)
* [3] H. Weyl, _The Classical Groups_ (Princeton University Press, Princeton/New York, 1946)
* [4] S.P. Kapitza, V.N. Melekhin, _The Microtron_ (Harwood Academic, London, 1987)
* [5] H. Wiedemann, Nucl. Instrum. Methods **A266**, 24 (1988)
* [6] E. Tanabe, M. Borland, M.C. Green, R.H. Miller, L.V. Nelson, J.N. Weaver, H. Wiedemann, in _14th Meeting on Linear Accelerators_ (Nara, 1989), p. 140
* [7] M. Borland, M.C. Green, R.H. Miller, L.V. Nelson, E. Tanabe, J.N. Weaver, H. Wiedemann, in _Proceedings of Linear Accelerator Conference_ (Los Alamos National Laboratory, Albuquerque, 1990)
* [8] H.A. Enge, Rev. Sci. Instrum. **34**, 385 (1963)
* [9] M. Borland, _A High-Brightness Thermionic Electron Gun_, Ph.D. thesis, Stanford University, Stanford, 1991
* [10] P.H. Kung, H.C. Linn, D. Borek, H. Wiedemann, Phys. Rev. Lett. **73**, 967 (1994)
* [11] D.C. Schultz, J. Clendenin, J. Frisch, E. Hoyt, L. Klaisner, M. Wood, D. Wright, M. Zoloterev, in _Proceedings of 3rd European Particle Accelerator Conference_, ed. by H. Henke, H. Homeyer, Ch. Petiot-Jean-Genaz (Edition Frontiere, Gif-sur Yvette, 1992), p. 1029
* [12] Ya.S. Derbenev, A.M. Kondratenko, Sov. Phys. JETP **37**, 968 (1973)
* [13] A.A. Sokolov, I.M. Ternov, _Synchrotron Radiation_ (Pergamon, Oxford, 1968)
* [14] V.N. Baier, Radiative polarization of electrons in storage rings, in _Physics with Intersecting Storage Rings_, ed. by B. Touschek (Academic, New York, 1971)
* [15] A.W. Chao, in _Physics of High Energy Particle Accelerators_, ed. by M. Month, M. Dienes, vol. 87 (American Institute of Physics, New York, 1982), p. 395


