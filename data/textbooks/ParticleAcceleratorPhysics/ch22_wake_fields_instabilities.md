## Chapter 22 Wake Fields and Instabilities*

While discussing self fields of a charged particle bunch, we noticed a significant effect from nearby metallic surfaces. The dynamics of individual particles as well as collective dynamics of the whole bunch depends greatly on the electromagnetic interaction with the environment. Such interactions must be discussed in more detail to establish stability criteria for particle beams.

The electric field from a charge in its rest frame extends isotropic from the charge into all directions. In the laboratory frame, this field is Lorentz contracted and assumes for a charge in a uniform beam pipe the form shown in Fig. 22.1a. The contracted field lines spread out longitudinally only within an angle \(\pm 1/\gamma\). This angle is very small for most high energy electron beams and we may describe the single-particle current as well as its image current by a delta function. Some correction must be made to this assumption for lower energy protons and specifically ions for which the angle \(1/\gamma\) may still be significant. In the following discussions, however, we will assume that the particle energy is sufficiently large and \(\gamma\gg 1\).

Electron storage rings are being planned, designed, constructed, and operated for a variety of applications. While in the past such storage rings were optimized mostly as colliding beam facilities for high energy physics, in the future most applications for storage rings seem to be connected with the production of synchrotron radiation. Some of these radiation sources will be designed for higher energy particle beams (few GeV) to produce hard X-rays while others have moderate to low beam energies (\(\gtrsim 100\) MeV) to, for example, produce VUV and soft X-rays or to drive free electron lasers.

The beam in an electron storage ring is composed of bunches which are typically a few centimeters long and are separated by a distance equal to one or more rf-wavelengths. The total number of bunches in a storage ring can range from one bunch to a maximum of \(h\) bunches, where \(h\) is the harmonic number for the storage ring system. The particle beam covers therefore a wide frequency spectrum from the kHz regime of the order of the revolution frequency up to many GHz limited only by the bunch length or size of the vacuum chamber. On the other hand, the vacuumchamber environment constitutes an impedance which can become significant in the same frequency regime and efficient coupling can occur leading to collective effects. The most important impedance in an accelerator is that of the accelerating cavity at the cavity fundamental frequency. Since the particle beam is bunched at the same frequency, we observe a very strong coupling which has been extensively discussed in Sect. 19.4 in connection with beam loading. In this section, we will therefore ignore beam loading effects in resonant cavities at the fundamental frequency and concentrate only on higher-order mode losses and interaction with the general vacuum chamber environment.

Depending on the particular application and experiment conducted, it may be desirable to store only a single bunch with the highest intensity possible. In other cases the maximum total achievable intensity is desired in as many bunches as possible and the particular bunch distribution around the ring does not matter. In either case the ultimate electron beam intensity will most probably be limited by instabilities caused by electromagnetic interaction of the beam current with the environment of the vacuum chamber. We ignore here technical limitations due to, for example, insufficient available rf-power or inability to cool the radiation heating of the vacuum chamber.

Since the radiation intensity produced is directly proportional to the stored electron beam current, it is obvious that the usefulness of such a radiation source depends among other parameters on the maximum electron beam current that can be stored in each bunch or in the storage ring.

### Definitions of Wake Field and Impedance

The image currents of a charge \(q\) travelling along the axis of a uniform and perfectly conducting tube move with the charge without losses and no forces are generated that would act back on the particle. This is different for a resistive wall where the image fields drag a significant distances behind the charge or in case of an obstacle extending into the tube or any other sudden variation of the tube cross section (Fig. 22.1b).

Figure 22: Coupling of a charged particle beam to the environment; uniform chamber cross section (**a**), and obstacle on vacuum chamber surface (**b**)

In any of these cases, wake fields are created which have the ability to pull or push the charge \(q\) or test particles following that charge. Because of causality, no such fields exist ahead of a relativistically moving charge.

Energy losses and gains of a single or collection of particles can cause significant modifications in the dynamics of particle motion. Specifically, we are concerned that such forces may lead to particle or beam instability which must be understood in detail to determine limitations or corrective measures in a particular accelerator design. The interaction of a charged particle beam with its environment can be described in time domain or frequency domain where both have their advantages and disadvantages when it comes to evaluate their effect on particle dynamics.

##### Parasitic Mode Losses and Impedances

In time domain, the interaction is described by wake fields which then act on charges. In frequency domain, vacuum chamber components can be represented as a frequency dependent impedance. We used this picture before while discussing properties of accelerating cavities. Many vacuum chamber components or sudden changes in cross section behave like cavities and represent therefore frequency dependent impedances. Together with the frequency spectrum of the beam, we find strong coupling to the vacuum chamber if the impedance and particle beam have a significant component at the same frequency. The induced voltage \(V(\omega)\) from this interaction is proportional to the collective particle current \(I(\omega)\) and the impedance \(Z(\omega)\) acting as the proportionality factor, describes the actual coupling from the particle beam via the vacuum chamber environment to the test particle. Mathematically, we set

\[V(\omega)=-Z(\omega)\,I(\omega) \tag{22.1}\]

indicating by the minus sign that the induced voltage leads to an energy loss for beam particles. The impedance is in general complex and depends for each piece of vacuum chamber including accelerating cavities or accidental cavities, on its shape, material and on the frequency under consideration. The coupling impedance for a particular vacuum chamber component or system may be narrow band with a quality factor \(Q\gg 1\) like that in an accelerating cavity or broad band with \(Q\approx 1\) due to a sudden change in the vacuum chamber cross section.

Fields induced by the beam in a high \(Q\) structure are restricted to a narrow frequency width and persist for a long time and can act back on subsequent particle bunches or even on the same particles after one or more revolutions. Such narrow-band impedances can be the cause for multi-bunch instabilities but rarely affect single bunch limits. The main source for a narrow-band impedance in a well-designed accelerator comes from accelerating cavities at the fundamental as well as higher-order mode frequencies. There is little we can or want do about the impedance at the fundamental frequency which is made large by design for efficiency. The design of modern accelerator cavities exhibit significantly reduced impedances for higher-order modesor HOMs.

The source for broad-band impedances are discontinuities in cross section or material along the vacuum chamber including accelerating cavities, flanges, kicker magnets with ferrite materials, exit chambers electrostatic plates, beam position monitors, etc. Many higher order modes over a wide frequency range can be excited in such discontinuities by a passing short particle bunch, but all modes decoher very fast. Only for a very short time are these mode fields in phase, adding up to a high field intensity but at the time of arrival of the next particle bunch or the same bunch after one or more revolutions these fields have essentially vanished. Broad-band wake fields are therefore mainly responsible for the appearance of single-bunch beam instabilities.

Due to tight particle bunching by the rf-system to about 5 % of the rf-wavelength, we have large instantaneous currents with significant amplitudes of Fourier components at harmonics of the revolution frequency up to about 20 times the rf-frequency or down to wavelength of a few centimeters. Strong electromagnetic interaction between electron bunches and cavity like structures as part of the vacuum enclosure must therefore be expected. Any but the smallest steps in the cross section of the vacuum chamber constitute cavity like structures. A bunch passing by such a structure deposits electromagnetic energy which in turn causes heating of the structure and can act back on particles in a later segment of the same bunch or in a subsequent bunch. Schematically such fields, also called wake fields, are shown in Fig. 22.1 where the beam passes by a variation in the cross section of the vacuum chamber. We will discuss the nature and the frequency spectrum of these wake fields to determine the effect on the stability of the beam and to develop counter measures to minimize the strength and occurrence of these wake fields.

We distinguish broad band parasitic losses where the quality factor \(Q\) is of the order of unity from narrow band losses with higher \(Q\) values. Fields from broad band losses last only a very short time of the order of one period and are mainly responsible for single bunch instabilities, where the fields generated by electrical charges in the head of the bunch act back on the particles in the tail of the same bunch. Due to the low value of the quality factor (\(Q\approx 1\)) these broad band wake fields decay before the next bunch arrives.

Wake fields can appear as longitudinal or transverse modes and cause correspondingly longitudinal or transverse instabilities. Obviously, a perfect vacuum chamber would have a superconducting surface and be completely uniform around the ring. This is not possible in real accelerators because we need rf-systems which by their nature are not smooth, injection/ejection components, synchrotron light ports, bellows, and beam position monitors. While we cannot avoid such lossy components we are able by proper design to minimize the detrimental effects of less than ideal components.

The loss characteristics of a particular piece of the vacuum chamber for the whole ring is generally expressed in terms of an impedance \(Z\) or in terms of a loss factor \(k\). To illustrate the different nature of wake fields we assume a cavity like change in the cross section of the vacuum chamber as shown in Fig. 22.2.

A bunch passing through such a structure on axis excites in lowest order a longitudinal electrical field and a transverse magnetic field as shown. Such a field pattern will not cause a transverse deflection of the whole beam since the electrical field is strictly longitudinal and the transverse magnetic field is zero on axis and out of phase. For this situation we define a longitudinal impedance \(Z_{\parallel}\) by

\[Z_{\parallel}(\omega)=-\frac{\int\mathbf{E}(\omega)\;\mbox{d}\mathbf{z}}{I(\omega)}, \tag{22.2}\]

where \(\mathbf{E}(\omega)\) is the electric field at the frequency \(\omega\) and \(I(\omega)\) the Fourier transform of the bunched beam current. The r.h.s. of (22.2) is the energy gained per unit charge and is equivalent to an accelerating voltage divided by the current, where the actual frequency dependence as determined by the specific physical shape of the "resonating" structure.

In a similar way we can define a transverse impedance. A beam passing off axis through a "cavity" excites asymmetric fields, as shown in Fig. 22.3, proportional to the moment of the beam current \(I(\omega)\;\Delta x\), where \(\Delta x\) is the displacement of the beam from the axis.

Such an electrical field is connected through Maxwell's equation with a finite transverse magnetic field on axis, as shown in Fig. 22.3,

Figure 22.2: Longitudinal parasitic mode

Figure 22.3: Transverse parasitic mode

deflection of the beam. Consistent with the definition of the longitudinal impedance we define a transverse impedance by

\[Z_{\perp}(\omega)=\mathrm{i}\,\frac{\int\,\left(\mathbf{E}(\omega)+\left[\mathbf{v}\times \mathbf{B}(\omega)\right]\right)|\perp\mathrm{d}\!z}{I(\omega)\;\Delta x}\,, \tag{22.3}\]

where \(\mathbf{v}\) is the velocity of the particle and \(\mathbf{B}(\omega)\) the magnetic field component of the electromagnetic field at frequency \(\omega\). In general the impedances are complex

\[Z(\omega)=Z_{\mathrm{Re}}(\omega)+\mathrm{i}\,Z_{\mathrm{Im}}(\omega)\,. \tag{22.4}\]

The resistive part of the impedance can lead to a shift in the betatron oscillation frequency of the particles while the reactive or imaginary part may cause damping or antidamping.

The impedance is a function of the frequency and its spectrum depends on the specific design of the vacuum chambers in a storage ring. The longitudinal impedance of vacuum chambers has been measured in SPEAR and in other existing storage rings and has been found to follow a general spectrum as a consequence of similar design concepts of storage ring components. SPEAR measurements, as shown in Fig. 19.9, demonstrate the general form of the frequency spectrum of the vacuum chamber impedance [1].

Characteristic for the spectrum is the cutoff frequency \(f_{\mathrm{c}}\) at which the linear impedance function reaches a maximum and above which the fields are able to propagate in the vacuum chamber. This cutoff frequency obviously is determined by the aperture of the vacuum chamber and therefore occurs at different frequencies for different rings with different vacuum chamber apertures. For the longitudinal broad band impedance at high frequencies above the cutoff frequency \(f_{\mathrm{c}}\) we have the simple power law

\[Z_{\parallel}(\omega)=Z_{\mathrm{c}}\omega^{-0.68},\qquad(\omega>\omega_{ \mathrm{c}})\,. \tag{22.5}\]

To simplify comparisons between different storage rings we define a normalized impedance \(Z/n\) as the impedance at the cut off frequency divided by the mode number \(n\) which is the ratio of the cutoff frequency \(f_{\mathrm{c}}\) to the revolution frequency \(f_{\mathrm{rev}}\)

\[\left|\frac{Z}{n}\right|_{\mathrm{c}}=\left|\frac{Z_{\mathrm{c}}}{f_{\mathrm{ c}}/f_{\mathrm{rev}}}\right|. \tag{22.6}\]

This definition of the normalized impedance can be generalized to all frequencies and together with (22.5) the impedance spectrum becomes

\[\left|\frac{Z_{\parallel}}{n}\right|_{\mathrm{eff}}=\left|\frac{Z_{\parallel} }{n}\right|_{\mathrm{c}}\left(\frac{\omega}{\omega_{\mathrm{c}}}\right)^{-1.68}. \tag{22.7}\]Where only one is known, we can make an estimate of the other one through the approximate relation which is correct only for cylindrically symmetric structures [2, 3]

\[Z_{\perp}=\frac{2R}{b^{2}}\frac{Z_{\parallel}}{n}, \tag{22.8}\]

where \(2\pi R\) is the ring circumference and \(b\) the typical vacuum chamber radius. The longitudinal impedance of the whole storage ring vacuum system including rf-cavities can be determined by measuring the energy loss of particles in a high intensity bunch compared to the energy loss for particles in a low intensity bunch. Such loss measurements are performed by observing the shift in synchronous phase for the low and high intensity beam. The parasitic losses of rf-cavities can be calculated very accurately with computer programs or are known from laboratory measurements. From the separate knowledge of cavity and total ring losses we derive the vacuum chamber losses by simple subtraction.

A bunched particle beam of high intensity represents a source of electromagnetic fields, called wake fields [4] in a wide range of wavelengths down to the order of the bunch length. The same is true for a realistic coasting beam where fluctuations in beam current simulate short particle bunches on top of an otherwise uniform beam.

Introducing wake fields and higher-order mode losses, we distinguish two groups, the longitudinal and the transverse wake fields. The longitudinal wake fields being in phase with the beam current cause energy losses to the beam particles, while transverse wakes deflect particle trajectories. There is no field ahead of relativistically moving charge due to causality. From the knowledge of such wake fields in a particular environment we may determine the effect on a test charge moving behind a charge \(q\).

The character of local wake fields depends greatly on the actual geometry and material of the vacuum chamber and we may expect a significant complication in the determination of wake field distributions along a vacuum enclosure of an actual accelerator. It is not practical to evaluate these fields in detail along the beam path and fortunately we do not need to. Since the effects of localized fields are small compared to the energy of the particles, we may integrate the wake fields over a full circumference. As we will see, this integral of the field can be experimentally determined.

#### Longitudinal Wake Fields

One may wonder how the existence of an obstacle in the vacuum chamber, like a disk which is relatively far away from the charge \(q\), can influence a test particle following closely behind the charge \(q\). To illustrate this, we consider the situation shown in Fig. 22.4.

Long before the charge reaches the obstruction, fields start to diverge from the charge toward the obstruction to get scattered there. Some of the scattered fields move again toward the charge and catch up with it due to its slightly faster speed.

The details of this catch up process are, however, of little interest compared to the integrated effect of wake fields on the test particle. Each charge at the position \(z\) creates a wake field for a particle at location \(\vec{z}<z\) and this wake field persists during the whole travel time along an accelerator segment \(L\) assuming that the distance \(\xi=z-\vec{z}\) does not change appreciably along \(L\). We define now a longitudinal wake function by integrating the longitudinal wake fields \(\vec{E}_{\parallel}\) along the interaction length \(L\), which might be the length of a vacuum chamber component, a linear accelerator or the circumference of a circular accelerator, and normalize it to a unit charge. By integrating, which is the same as averaging over the length \(L\), we eliminate the need to calculate everywhere the complicated fields along the vacuum chambers. The wake field at the location of a test particle at \(\vec{z}\) from a charge \(q\) at location \(z\) is then (Fig. 22.4)

\[W_{\parallel}(\xi)=\frac{1}{q}\int_{L}\vec{E}_{\parallel}(z,t-\xi/\beta c)\, \mathrm{d}z\,, \tag{22.9}\]

where \(\xi=z-\vec{z}>0\). The wake function is measured in V/Cb using practical units and is independent of the sign of the charge. To get the full wake field for a test particle, one would integrate the wake function over all particles ahead of the test particle.

The longitudinal wake function allows us to calculate the total energy loss of the whole bunch by integrating over all particles. We consider a test particle with charge \(e\) at position \(\vec{z}\) and calculate the energy loss of this particle due to wake fields from charges further ahead at \(z\geq\vec{z}\). The total induced voltage from a collection of particles with distribution \(\lambda(z)\) on the test charge at \(\vec{z}\) is then determined by the wake potential1

Footnote 1: Expression (22.9) is sometimes called the wake potential. We do not follow this nomenclature because the expression (22.9) does not have the dimension of a potential but (22.10) does.

\[V_{{}_{\mathrm{HOM}}}(\vec{z})=-e\int_{\vec{z}}^{\infty}\lambda(z)W_{\parallel} (z-\vec{z})\,\mathrm{d}z\,, \tag{22.10}\]

Figure 22.4: Catch up of wake fields with test particle

where a negative sign was added to indicate that the wake fields are decelerating. Integrating over all slices \(\mathrm{d}\bar{z}\), the total energy loss of the bunch into HOM fields is

\[\Delta U_{\mathrm{ HOM}}=-\int_{-\infty}^{\infty}e\lambda(\bar{z})\,\mathrm{d} \bar{z}\underbrace{\int_{\bar{z}}^{\infty}e\lambda(z)W_{\parallel}(z-\bar{z}) \,\mathrm{d}z}_{\mathrm{ wake potential at }\bar{z}}\,. \tag{22.11}\]

The linear distribution \(\lambda(z)\) of particles with charge \(e\) is normalized to the total number of particles \(N_{\mathrm{b}}\) in the bunch \(\int\lambda(z)\,\mathrm{d}z=N_{\mathrm{b}}\). It is interesting to perform the integrations in (22.11) for a very short bunch such that the wake function seen by particles in this bunch is approximately constant and equal to \(W_{0}\). In this case, we define the function \(w(\bar{z})=\int_{\bar{z}}^{\infty}e\lambda(z)\,\mathrm{d}z\) and the double integral assumes the form \(-\int_{-\infty}^{\infty}w\,\mathrm{d}w=\frac{1}{2}(eN_{\mathrm{b}})^{2}\) where we have used the normalization \(w(-\infty)=eN_{\mathrm{b}}\). Particles in a bunch see therefore only \(50\,\%\) of the wake fields produced by the same bunch consistent with our earlier formulation of the fundamental theorem of wake fields discussed in Sect. 19.3 in connection with wake fields in rf-cavities. By the same argument, each particle sees only half of its own wake field.

Wake functions describe higher-order mode losses in the time domain. For further discussions, we determine the relationship to the concept of impedance in the frequency domain and replace in (22.10)the charge distribution with the instantaneous current passing by \(\bar{z}\)

\[I(\bar{z},t)=\hat{I}_{0}\,\mathrm{e}^{\mathrm{i}(\bar{z}-\omega t)}\,. \tag{22.12}\]

The beam current generally includes more than one mode \(k\) but for simplicity we consider only one in this discussion. Integrating over all parts of the beam which have passed the location \(\bar{z}\) before, the wake potential (22.10) becomes

\[V_{\mathrm{ HOM}}(\bar{z},t)=-\frac{1}{c\beta}\int_{\bar{z}}^{\infty}I\left( \bar{z},t+\frac{z-\bar{z}}{c\beta}\right)W_{\parallel}(z-\bar{z})\,\mathrm{d}z\,. \tag{22.13}\]

Consistent with a time dependent beam current, the induced voltage depends on location \(\bar{z}\) and time as well. The wake function vanishes due to causality for \(z-\bar{z}<0\) and the integration can therefore be extended over all values of \(z\). With (22.12), \(\xi=z-\bar{z}\) and applying a Fourier transform (22.13) becomes

\[V_{\mathrm{ HOM}}(t,\omega)=-I(t,\omega)\frac{1}{c\beta}\int_{-\infty}^{\infty} \mathrm{e}^{-\mathrm{i}\omega\nicefrac{{c}}{{\beta}}}W_{\parallel}(\xi)\, \mathrm{d}\xi\,. \tag{22.14}\]

From (22.14) we define the longitudinal coupling impedance in the frequency domain

\[Z_{\parallel}(\omega)=\frac{1}{c\beta}\int_{-\infty}^{\infty}\mathrm{e}^{- \mathrm{i}\,\omega\nicefrac{{c}}{{\xi}}/c\beta}W_{\parallel}(\xi)\,\mathrm{d}\xi \tag{22.15}\]which has in practical units the dimension Ohm. The impedance of the environment is the Fourier transform of the wake fields left behind by the beam in this environment. Because the wake function has been defined in (22.9) for the length \(L\) of the accelerator under consideration, the impedance is an integral parameter of the accelerator section \(L\) as well. Conversely, we may express the wake function in terms of the impedance spectrum

\[W_{\parallel}(z)=\frac{1}{2\pi}\int_{-\infty}^{\infty}Z_{\parallel}(\omega)\, \mathrm{e}^{\mathrm{i}\,\omega z/c\beta}\,\mathrm{d}\omega. \tag{22.16}\]

The interrelations between wake functions and impedances allows us to use the most appropriate quantity for the problem at hand. Generally, it depends on whether one wants to work in the frequency or the time domain. For theoretical discussions, the well defined impedance concept allows quantitative predictions for beam stability or instability to be made. In most practical applications, however, the impedance is not quite convenient to use because it is not well known for complicated shapes of the vacuum chamber. In a linear accelerator, for example, we need to observe the stability of particles in the time domain to determine the head-tail interaction. The most convenient quantity depends greatly on the problem to be solved, theoretically or experimentally.

#### Loss Parameter

In a real accelerator, the beam integrates over many different vacuum chamber pieces with widely varying impedances. The interaction of the beam with the vacuum chamber impedance leads to an energy loss which has to be compensated by the rf-system. We are therefore not able to experimentally determine the impedance or wake function of a particular vacuum chamber element. Only the integrated impedance for the whole accelerator can sometimes be probed at specific frequencies by observing specific instabilities as we will discuss later. The most accurate quantity to measure the total resistive impedance for the whole accelerator integrated over all frequencies is the loss factor or loss parameter.

We characterize this loss through the loss factor \(k\) defined by

\[k=\frac{\Delta U}{q^{2}}, \tag{22.17}\]

where \(\Delta U\) is the total energy deposited by the passing bunch and \(q\) is the total electrical charge in this bunch. This definition is a generalization of the energy loss of a single particle passing once through a resonator where \(k=-(\omega/4)(R_{\mathrm{s}}/Q)\) and \(R_{\mathrm{s}}\) is the shunt impedance of this resonator. The loss factor is related to the real part of the impedance by

\[k=\frac{2}{q^{2}}\int_{o}^{\infty}\mathrm{Re}[Z(\omega)]\,I^{2}(\omega)\, \mathrm{d}\omega\]and depends strongly on the bunch length as can be seen from measurements of the loss factor in SPEAR [5] shown in Fig. 22.5. Specifically, we find the loss factor to scale with the bunch length like

\[k(\sigma_{\ell})\sim\sigma_{\ell}^{-1.21}. \tag{22.18}\]

Similar to the definitions of impedances, we also distinguish a longitudinal and a transverse loss factor. The loss factor can be related to the wake function and we get from comparison with (22.11) the relation

\[k_{\mathrm{{}_{HOM}}}=\frac{1}{N_{\mathrm{b}}^{2}}\int_{-\infty}^{\infty} \lambda(\bar{z})\,\mathrm{d}\bar{z}\int_{\bar{z}}^{\infty}\lambda(z)W_{\parallel }(z-\bar{z})\,\mathrm{d}z\,. \tag{22.19}\]

The loss parameter can be defined for the complete circular accelerator or for a specific vacuum chamber component installed in a beam line or accelerator. Knowledge of this loss factor is important to determine possible heating effects which can become significant since the total higher-order mode losses are deposited in the form of heat in the vacuum chamber component. In a circular accelerator, the energy loss rate or heating power of a beam circulating with revolution frequency \(f_{0}\) is

\[P_{\mathrm{{}_{HOM}}}=k_{\mathrm{{}_{HOM}}}\frac{I_{0}^{2}}{f_{0}\,n_{ \mathrm{b}}}\,, \tag{22.20}\]

where \(n_{\mathrm{b}}\) is the number of bunches in the beam and \(I_{\mathrm{o}}=n_{\mathrm{b}}\,qN_{\mathrm{b}}f_{0}\) is the average circulating beam current in the accelerator. As an example, we consider a circulating beam of 1mA in one bunch of the LEP storage ring where the revolution frequency is about \(f_{0}=10\) kHz. The heating losses in a component with loss factor \(k_{\mathrm{{}_{HOM}}}=0.1\) V/pC would be 10 Watts. This might not seem much if the component is large

Figure 22.5: Dependence of the overall loss factor \(k\) in the storage ring SPEAR on the bunch length [5]

and an external cooling fan might be sufficient. On the other hand, if the vacuum component is small and not accessible like a bellows this heating power might be significant and must be prevented by design. The higher-order heating losses scale like the average current, the bunch current and inversely proportional with the revolution frequency. For a given circulating beam current, the losses depend therefore greatly on the number of bunches and the size of the circular accelerator. As the bunch length becomes smaller, higher and higher modes can be excited as demonstrated by the steep increase in loss parameter with decreasing bunch length (Fig. 22.5). Although we try to apply a careful design to all accelerator components to minimize the impedance it is prudent to be aware of this heating effect while developing accelerators that involve significantly reduced bunch length like those in quasi-isochronous storage rings or beams accelerated by laser beams.

The loss parameter can be measured by observing the shift in the synchronous phase. A bunch of particles circulating in an accelerator looses energy due to the resistive impedance of the vacuum chamber. This additional energy loss is compensated by an appropriate shift in the synchronous phase which is given by

\[\Delta U_{\mbox{\tiny HOM}}=eN_{\rm b}V_{\rm rf}|\sin\left(\phi_{\rm s}-\phi _{\rm s0}\right)|\,, \tag{22.21}\]

where \(\phi_{\rm s0}\) is the synchronous phase for a very small beam current and \(V_{\rm rf}\) the peak rf-voltage. The loss factor is then with the number of particles per bunch \(N_{\rm b}\)

\[k_{\mbox{\tiny HOM}}=\frac{\Delta U_{\mbox{\tiny HOM}}}{e^{2}N_{\rm b}^{2}}\,. \tag{22.22}\]

Performing this measurement as a function of rf-voltage one can establish a curve similar to that shown in Fig. 22.5 for the storage ring SPEAR and the dependence of the loss parameter on the bunch length can be used to determine the total resistive impedance of the accelerator as a function of frequency. To do that, we write (22.19) in terms of Fourier transforms

\[k_{\mbox{\tiny I}\mbox{\tiny HOM}}=\frac{\pi}{e^{2}N_{\rm b}^{2}}\int_{- \infty}^{\infty}Z_{\rm res}(\omega)\,|I(\omega)|^{2}\,{\rm d}\omega \tag{22.23}\]

and recall that the bunch or current distribution in a storage ring is Gaussian

\[I(\tau)=\frac{I_{0}}{\sqrt{2\pi}\sigma_{\tau}}{\rm e}^{-\tau^{2}/2\sigma_{ \tau}^{2}}\,. \tag{22.24}\]

The Fourier transform of a Gaussian distribution is

\[I(\omega)=I_{0}\,{\rm e}^{-\frac{1}{2}\omega^{2}\sigma_{\tau}^{2}}\,, \tag{22.25}\]where \(I_{0}\) is the total bunch current and inserting (22.25) into (22.23), we get

\[k_{\parallel_{\rm HOM}}\,=\,\frac{\pi I_{0}}{e^{2}N_{\rm b}^{2}}\int_{-\infty}^{ \infty}Z_{\rm res}(\omega)\,{\rm e}^{-\omega^{2}\sigma_{\rm t}^{2}}\,{\rm d} \omega\,. \tag{22.26}\]

With (22.26) and the measurement \(k_{\parallel_{\rm HOM}}(\sigma_{\ell})\), where \(\sigma_{\ell}=c\sigma_{\rm t}\), one may solve for \(Z_{\rm res}(\omega)\) and determine the resistive-impedance spectrum of the ring.

Unfortunately, it is not possible to attach a resistance meter to an accelerator to determine its impedance and we will have to apply a variety of wake field effects on the particle beams to determine the complex impedance as a function of frequency. No single effect, however, will allow us to measure the whole frequency spectrum of the impedance.

#### Transverse Wake Fields

Similar to the longitudinal case we also observe transverse wake fields with associated impedances. Such fields exert a transverse force on particles generated by either transverse electrical or magnetic wake fields. Generally such fields appear when a charged particle beam passes off center through a nonuniform but cylindrical or through an asymmetric vacuum chamber. Transverse wake fields can be induced only on structures which also exhibit a longitudinal impedance. A beam travelling off center through a round pipe with perfectly conducting walls will not create longitudinal and therefore also no transverse wake fields.

We consider a charge \(q\) passing through a vacuum chamber structure with an offset \(\Delta u=(\Delta x,\,\Delta y)\) in the horizontal or vertical plane as shown in Fig. 22.3.

In analogy to the definition of the longitudinal wake function (22.9), we define a transverse wake function per unit transverse offset by

\[W_{\perp}(\xi,t)\,=\,+\frac{\int_{L}\left\{\mathbf{E}(t-\xi/\beta c)+c[\mathbf{\beta} \times\mathbf{B}(t-\xi/\beta c)]\right\}_{\perp}\,{\rm d}z}{q\,\Delta u} \tag{22.27}\]

which is measured in units of V/Cb/m. Consistent with the definition (22.15) of the longitudinal impedance, the transverse coupling impedance is the Fourier transform of the transverse wake functions defined by

\[Z_{\perp}(\omega)\,=\,{\rm i}\frac{1}{c\beta}\int_{-\infty}^{\infty}{\rm e}^{ -{\rm i}\,\omega\xi/c\beta}\,W_{\perp}(\xi)\,{\rm d}\xi \tag{22.28}\]

adding the factor \({\rm i}\) to indicate that the action of the transverse force is a mere deflection while the particle energy stays constant. This transverse impedance is measured in Ohm/m. The inverse relation is similar to the longitudinal case

\[W_{\perp}(z)\,=\,{\rm i}\frac{1}{2\pi}\int_{-\infty}^{\infty}Z_{\perp}(\omega )\,{\rm e}^{{\rm i}\,\omega z/c\beta}\,{\rm d}\omega\,. \tag{22.29}\]

#### Panofsky-Wenzel Theorem

The general relationship between longitudinal and transverse wake fields is expressed by the Panofsky-Wenzel theorem [6]. Panofsky and Wenzel studied the effect of transverse electromagnetic fields on a particle trajectory and applied general relations of electromagnetic theory to derive a relationship between longitudinal and transverse electromagnetic forces. We will derive the same result in the realm of wake fields. The Lorentz force on a test particle at \(\vec{z}\) due to transverse wake fields from charges at location \(z>\vec{z}\) causes a deflection of the particle trajectory and the change in transverse momentum of the test particle is after integration over all charges at locations \(z<\vec{z}\)

\[c\vec{p}_{\perp}=\frac{e}{\beta}\int_{-\infty}^{\infty}\left[\vec{E}+\left( \vec{v}\times\vec{B}\right)\right]_{\perp}\mathrm{d}z\,. \tag{22.30}\]

Note that the wake fields vanish because of causality for \(\xi<0\). The fields can be expressed by the vector potential \(\vec{E}_{\perp}=-\partial\vec{A}_{\perp}/\partial t\) and \(\vec{B}_{\perp}=(\nabla\times\vec{A})_{\perp}\). The particle velocity has only one nonvanishing component \(\vec{v}=(0,0,v)\) and (22.30) becomes with \(\partial\vec{z}/\partial t=v\)

\[c\vec{p}_{\perp}=-ce\underbrace{\int_{0}^{d}\overbrace{\left(\frac{\partial}{ \partial t}\frac{\partial t}{\partial\vec{z}}+\frac{\partial}{\partial\vec{z}} \right)}^{=\mathrm{d}/\mathrm{d}z}}_{=0}+ce\nabla_{\perp}\int_{0}^{d}A_{ \parallel}\mathrm{d}z\,, \tag{22.31}\]

where we made use of the vector relation for \(\vec{v}\times\left(\vec{\nabla}\times\vec{A}\right)+\underbrace{A\times\left( \nabla\times\vec{v}\right)}_{=0}\) which is equal to \(\nabla_{\perp}(\vec{v}\vec{A})-\left(\vec{v}\nabla\right)\vec{A}_{\perp}- \underbrace{\left(\vec{A}\nabla\right)\vec{v}}_{=0}\) noting that the particle velocity is a constant.

The integrand in the first integral of (22.31) is equal to the total derivative \(\mathrm{d}A_{\perp}/\mathrm{d}z\) and the integral vanishes because the fields vanish for \(\xi=\pm\infty\). After differentiation with respect to the time \(t\) (22.31) becomes

\[\frac{\mathrm{d}\vec{p}_{\perp}}{\mathrm{d}t}=-\,e\nabla_{\perp}\int_{-\infty }^{\infty}E_{\parallel}\mathrm{d}z \tag{22.32}\]

which is in terms of forces

\[\frac{\partial}{\partial z}\vec{F}_{\perp}=-\nabla_{\perp}\vec{F}_{\parallel}\,. \tag{22.33}\]

The longitudinal gradient of the transverse force or electromagnetic field is proportional to the transverse gradient of the longitudinal force or electromagnetic field and knowledge of one allows us to calculate the other.

### 22.2 Impedances in an Accelerator Environment

The vacuum chamber of an accelerator is too complicated in geometry to allow an analytical expression for its impedance. In principle each section of the chamber must be treated individually. By employing two or three-dimensional numerical codes it may be possible to determine the impedance for a particular component and during a careful design process for a new accelerator, this should be done to avoid later surprises. In [7] expressions for many geometries are compiled. Yet, every accelerator is somewhat different from another and will have its own particular overall impedance characteristics. For this reason, we focus in these discussions specifically on such instabilities which can be studied experimentally revealing impedance characteristics of the ring. However, depending on the frequency involved, there are a few classes of impedances which are common to all accelerators and may help understand the appearance and strength of certain instabilities. In this section, we will discuss such impedances to be used in later discussions on stability conditions and growth rate of instabilities.

Consistent with (21.88), (22.10) the longitudinal impedance for a circular accelerator is defined as the ratio of the induced voltage at frequency \(\omega\) to the Fourier transform of the beam current at the same frequency

\[Z_{\parallel}(\omega) = -\frac{\int\boldsymbol{E}_{\parallel}(\omega)\,\mathrm{d} \boldsymbol{z}}{I(\omega)}\] \[= \frac{1}{4\pi\epsilon_{0}}\frac{1}{eN_{\mathrm{b}}}\int_{L} \boldsymbol{E}_{\parallel}(z,t-\zeta/\beta c)\,\mathrm{e}^{-\mathrm{i}\omega \zeta/\beta c}\,\mathrm{d}\boldsymbol{z}\,.\]

Similarly the transverse impedance is from (22.27), (22.28) the ratio of induced transverse voltage to the transverse moment of the beam current

\[Z_{\perp}(\omega)=-\mathrm{i}\frac{\int(\boldsymbol{E}_{\perp}+[\boldsymbol{v }\times\boldsymbol{B}]_{\perp})|_{(z,t-\zeta/\beta c)}\mathrm{e}^{-\mathrm{i} \omega\zeta/\beta c}\,\mathrm{d}z}{I(\omega)\;\Delta u}\,, \tag{22.35}\]

where \(\Delta u\) is the horizontal or vertical offset of the beam from the axis.

#### Space-Charge Impedance

In (21.78) we found an induced voltage leading to an energy gain or loss due to a collection of charged particles. It is customary to express (21.78) in a form exhibiting the impedance of the vacuum chamber. In case of a perfectly conducting vacuum chamber \(E_{z\mathrm{w}}=0\) and (21.78) becomes

\[V_{z}=-Z_{\parallel\mathrm{sc}}\,I_{n}\,\mathrm{e}^{\mathrm{i}(n\theta- \omega_{n}t)}\,, \tag{22.36}\]where the longitudinal space-charge impedance \(Z_{\parallel\mathrm{sc}}\) is defined by [8, Sect. 2.4.5].2

Footnote 2: Note: the factor \(1/\left(\epsilon_{0}c\right)=\sqrt{\mu_{0}/\epsilon_{0}}=Z_{0}=376.73\;\Omega\) is often called the free space impedance. We will not use it because it is not a physical quantity but only a convenient unit scaling factor. A current passing through vacuum will not loose energy into this impedance.

\[\frac{Z_{\parallel\mathrm{sc}}(\omega)}{n}=-\frac{\mathrm{i}}{\epsilon_{0}c} \frac{1}{2\beta\gamma^{2}}\left(1\,+\,2\ln\frac{r_{\mathrm{w}}}{r_{0}}\right), \tag{22.37}\]

where \(n=\omega/\omega_{0}\) and \(\omega_{0}\) is the revolution frequency. This expression is correct for long wavelength below cut off of the vacuum chamber or for \(\omega\,<c/r_{\mathrm{w}}\). The space-charge impedance is purely reactive and, as we will see, capacitive. For a round beam of radius \(r_{0}\) and offset from the axis of a round beam pipe with diameter \(2r_{\mathrm{w}}\) a transverse space-charge impedance can be derived [7]

\[Z_{\perp\mathrm{sc}}(\omega)=-\frac{\mathrm{i}}{\epsilon_{0}c}\frac{\overline {R}}{\beta^{2}\gamma^{2}}\left(\frac{1}{r_{0}^{2}}-\frac{1}{r_{\mathrm{w}}^{2} }\right)\,, \tag{22.38}\]

where \(\overline{R}\) is the average ring radius. The transverse space-charge impedance is inversely proportional to \(\beta^{2}\) and is therefore especially strong for low energy particle beams.

#### Resistive-Wall Impedance

The particle beam induces an image current in the vacuum chamber wall in a thin layer with a depth equal to the skin depth. For less than perfect conductivity of the wall material, we observe resistive losses which exert a pull or decelerating field on the particles. This pull is proportional to the beam current and integrating the fields around a full circumference \(2\pi\overline{R}\) of the accelerator we get the longitudinal resistive wall impedance in a uniform tube of radius \(r_{\mathrm{w}}\) at frequency \(\omega_{n}\) for lowest order monopole oscillations [9]

\[Z_{\parallel}(\omega)\big{|}_{\mathrm{res}}=(1-\mathrm{i})\;\frac{\overline{R }}{r_{\mathrm{w}}\sigma\;\delta_{\mathrm{skin}}}, \tag{22.39}\]

where the skin depth is defined by [10]\(\delta_{\mathrm{skin}}(\omega_{n})=\sqrt{\frac{2}{\mu_{0}\mu_{\mathrm{r}}\, \omega_{\mathrm{h}}\,\sigma}}\). The longitudinal resistive wall impedance decays with increasing frequency and therefore plays an important role only for lower frequencies up to tens of GHz[9]. The transverse resistive wall impedance for a round beam pipe is from the Panofsky-Wenzel theorem [6]

\[Z_{\perp}(\omega)_{\mathrm{res}} = \frac{2c}{\omega\,r_{\mathrm{w}}^{2}}\;Z_{\parallel}(\omega) \big{|}_{\mathrm{res}} \tag{22.40}\]

#### Cavity-Like Structure Impedance

The impedance of accelerating cavities or cavity like objects of the vacuum chamber can be described by the equivalent of a parallel resonant circuit for which the impedance is from (19.11)

\[\left.\frac{1}{Z_{\parallel}(\omega)}\right|_{\mathrm{cy}}=\frac{1}{R_{\mathrm{s }}}\left(1+\mathrm{i}Q\frac{\omega^{2}-\omega_{\mathrm{r}}^{2}}{\omega_{ \mathrm{r}}\,\omega}\right)\,, \tag{22.41}\]

where \(Q\) is the quality factor and \(R_{\mathrm{s}}\) the cavity impedance at the resonance frequency \(\omega_{\mathrm{r}}\) or cavity shunt impedance. Taking the inverse, we get for the normalized impedance

\[\left.\frac{Z_{\parallel}(\omega)}{n}\right|_{\mathrm{cy}}=\left|\frac{Z_{ \parallel}(\omega)}{n}\right|_{0}\frac{1-\mathrm{i}\,Q\frac{\omega^{2}-\omega _{\mathrm{r}}^{2}}{\omega_{\mathrm{r}}\,\omega}}{1+Q^{2}\frac{(\omega^{2}- \omega_{\mathrm{r}}^{2})^{2}}{\omega_{\mathrm{r}}^{2}\,\omega^{2}}}\,, \tag{22.42}\]

where \(\left|\frac{Z}{n}\right|_{0}=R_{\mathrm{s}}\) is purely resistive and \(n=\omega/\omega_{0}\).

Vacuum chamber impedances occur, for example, due to sudden changes of cross section, flanges, beam position monitors, etc., and are collectively described by a cavity like impedance with a quality factor \(Q\approx 1\). This is justified because fields are induced in these impedances at any frequency. From (22.42) the longitudinal broad-band impedance is therefore

\[\left.\frac{Z_{\parallel}}{n}\right|_{\mathrm{bb}}(\omega)=\left|\frac{Z_{ \parallel}}{n}\right|_{0}\frac{1-\mathrm{i}\,\frac{\omega^{2}-\omega_{\mathrm{ r}}^{2}}{\omega_{\mathrm{r}}\,\omega}}{1+\frac{(\omega^{2}-\omega_{\mathrm{r}}^{2})^{2} }{\omega_{\mathrm{r}}^{2}\,\omega^{2}}}\,. \tag{22.43}\]

This broad-band impedance spectrum is shown in Fig. 22.6 and we note that the resistive and reactive part exhibit different spectra.

The resistive broad-band impedance has a symmetric spectrum and scales like \(\omega^{2}\) for low frequencies decaying again for very high frequencies like \(1/\omega^{2}\). At low frequencies, the broad-band impedance (22.43) is almost purely inductive scaling linear with frequency

\[\left.\frac{Z_{\parallel}(\omega)}{n}\right|_{\mathrm{bb}}=\mathrm{i}\,\left| \frac{Z_{\parallel}(\omega)}{n}\right|_{0}\frac{\omega}{\omega_{\mathrm{r}}} \qquad\mathrm{for}\quad\omega\ll\omega_{\mathrm{r}}\,. \tag{22.44}\]

At high frequencies the impedance becomes capacitive

\[\left.\frac{Z_{\parallel}(\omega)}{n}\right|_{\mathrm{bb}}=-\mathrm{i}\, \left|\frac{Z_{\parallel}(\omega)}{n}\right|_{0}\frac{\omega_{\mathrm{r}}}{ \omega}\qquad\mathrm{for}\quad\omega\gg\omega_{\mathrm{r}} \tag{22.45}\]and decaying slower with frequency than the resistive impedance. We note, however, that the reactive broad-band impedance spectrum changes sign and beam stability or instability depend therefore greatly on the actual coupling frequency. At resonance, the broad-band impedance is purely resistive as would be expected.

Sometimes it is convenient to have a simple approximate correlation between longitudinal and transverse impedance in a circular accelerator as shown in (22.40). Although this correlation is valid only for the resistive wall impedance in a round beam pipe, it is often used for approximate estimates utilizing the broad-band impedance.

#### Overall Accelerator Impedance

At this point, we have identified all significant types of impedances we generally encounter in an accelerator which are the space charge, resistive wall, narrow-band impedances in high \(Q\) cavities, and broad-band impedance. In Fig. 22.7 we show qualitatively these resistive as well as reactive impedance components as a function of frequency.

At low frequency the reactive as well as the resistive component of the resistive wall impedance dominates while the space charge impedance is independent of frequency. The narrow-band cavity spectrum includes the high impedances at the fundamental and higher mode frequencies.

Generally, it is not possible to use a uniform vacuum chamber in circular accelerators. Deviations from a uniform chamber occur at flanges, bellows, rf-cavities, injection/ejection elements, electrostatic plates, etc. It is not convenient to consider the special impedance characteristics of every vacuum chamber piece and we may therefore look for an average impedance as seen by the beam. The broad-band impedance spectrum created by chamber components in a ring reaches a maximum at some frequency and then diminishes again like \(1/\omega\). This turn over of the broad-band impedance function depends on the general dimensions of all

Figure 22.6: Resistive and reactive broad-band impedance spectrum

vacuum chamber components of a circular accelerator and has to do with the cut off frequency for travelling waves in tubes.

In Fig. 19.9 the measured impedance spectrum of a storage ring was shown and is typical for complex storage ring vacuum chambers which are generally composed of similar components exhibiting at low frequencies an inductive impedance increasing linearly with frequency and diminishing again at high frequencies. This is also the characteristics of broad-band cavity impedance and therefore expressions for broad-band impedance are useful tools in developing theories for beam instabilities and predicting conditions for beam stability. The induced voltage for the total ring circumference scales like \(L_{\mathrm{w}}\,\dot{I}_{\mathrm{w}}\) where \(L_{\mathrm{w}}\) is the wall inductance and \(\dot{I}_{\mathrm{w}}\) the time derivative of the image current in the wall. The induced voltage is

\[\Delta V_{z0}=-L_{\mathrm{w}}\frac{\mathrm{d}I}{\mathrm{d}t}=\mathrm{i}\omega L _{\mathrm{w}}(\omega)I_{n}\,\mathrm{e}^{\mathrm{i}(n\theta-\omega_{n}t)} \tag{22.46}\]

where the inductive impedance is defined by

\[Z_{\parallel\mathrm{ind}}(\omega)=-\mathrm{i}\omega L_{\mathrm{w}}(\omega). \tag{22.47}\]

The total induced voltage due to space charge, resistive and inductive wall impedance is finally

\[V_{z\mathrm{w}}=-Z_{\parallel}I_{n}\,\mathrm{e}^{\mathrm{i}(n\theta-\omega_{n }t)}, \tag{22.48}\]

where the total longitudinal normalized impedance at frequency \(\omega_{n}\) is from (22.37), (22.39), (22.47)

\[\frac{Z_{\parallel}(\omega_{n})}{n}=\mathrm{i}\frac{1}{2\epsilon_{0}\beta c{ \gamma^{2}}}\left(1+2\ln\frac{r_{\mathrm{w}}}{r_{0}}\right)+\left(1-\mathrm{i }\right)\frac{\bar{R}}{r_{\mathrm{w}}\sigma\delta_{\mathrm{skin}}}-\mathrm{i} \frac{\omega_{0}}{4\pi\epsilon_{0}}L_{\mathrm{w}}(\omega_{n}). \tag{22.49}\]

Figure 22.7: Qualitative spectra of resistive and reactive coupling impedances in a circular accelerator

From the frequency dependence we note that space charge and inductive wall impedance becomes more important at high frequencies while the resistive wall impedance is dominant at low frequencies. The inductive wall impedance derives mostly from vacuum chamber discontinuities like sudden change in the vacuum chamber cross section, bellows, electrostatic plates, cavities, etc. In older accelerators, little effort was made to minimize the impedance and total ring impedances of \(|Z_{\parallel}/n|\approx 20\) to \(30\Omega\) were not uncommon. Modern vacuum design have been developed to greatly reduce the impedance mostly by avoiding changes of vacuum chamber cross section or by introducing gentle transitions and impedances of the order of \(|Z_{\parallel}/n|\lesssim 1\)\(\Omega\) can be achieved whereby most of this remaining impedance comes from accelerating rf-cavities.

From (22.49), we note that the space-charge impedance has the opposite sign of the inductive impedance and is therefore capacitive in nature. In general, we encounter in a realistic vacuum chamber resistive as well as reactive impedances causing both real frequency shifts or imaginary shifts manifesting themselves in the form of damping or instability. In subsequent sections, we will discuss a variety of such phenomena and derive stability criteria, beam-current limits or rise times for instability. At this point, it is noteworthy to mention that we have not made any assumption as to the nature of the particles involved and we may therefore apply the results of this discussion to electron as well as proton and ion beams.

#### Broad-Band Wake Fields in a Linear Accelerator

The structure of a linear accelerator constitutes a large impedance for a charged particle beam, specifically, since particle bunches are very short compared to the periodicity of the accelerator lattice. Every single cell resembles a big sudden change of the vacuum chamber cross section and we expect therefore a large accumulation of wake fields or impedance along the accelerator. The wake fields can be calculated numerically [4] and results for both the longitudinal and transverse wakes from a point charge are shown in Fig. 22.8 as a function of the distance behind this point charge.

Broad-band wake fields for other structures look basically similar to those shown in Fig. 22.8. Specifically, we note the longitudinal wake to be strongest just behind the head of the bunch while the transverse wake builds up over some distance. For an arbitrary particle distribution, one would fold the particle distribution with these wake functions to obtain the wake potential at the location of the test particle.

### Coasting-Beam Instabilities

The space-charge impedance as well as resistive and reactive wall impedances extract energy from a circulating particle beam. As long as the particle distribution is uniform, this energy loss is the same for all particles and requires simple replacement in acceleration cavities. In reality, however, some modulation of the longitudinal particle distribution cannot be avoided and we encounter therefore an uneven energy loss along the coasting particle beam. This can have serious consequences on beam stability and we therefore need to discuss stability criteria for coasting beams.

##### Negative-Mass Instability

Consider a beam in a ring below transition energy. The repulsive electrostatic field from a lump in the charge distribution causes particles ahead of the lump to be accelerated and particles behind the lump to be decelerated. Since accelerated particles will circulate faster and decelerated particles circulate slower, we observe a stabilizing situation and the lumpy particle density becomes smoothed out. Nature demonstrates this in the stability of Saturn's rings.which is equi8valent to this case below transition energy.

At energies above transition energy the situation changes drastically. Now the acceleration of a particle ahead of a lump leads to a slower revolution frequency and it will actually move closer to the lump with every turn. Similarly a particle behind the lump becomes decelerated and circulates therefore faster, again catching up with the lump. We observe an instability leading to a growing concentration of particles wherever a small perturbation started to occur. We call this instability the negative-mass instability [11] because acceleration causes particles to circulate slower similar to the acceleration of a negative mass. The same mechanism can lead to stabilization of oscillations if the forces are attractive rather than repulsive.

We will derive conditions of stability for this effect in a more quantitative way. The stability condition depends on the variation of the revolution frequency for particles close to the small perturbation of an otherwise uniform longitudinal

Figure 22.8: Time dependence of transverse (_left_) and longitudinal (_right_) wake fields from a point charge moving through one 3.3 cm long cell of a SLAC type 3 GHz linear accelerator structure [4]

particle distribution and we therefore investigate the time derivative of the revolution frequency

\[\frac{\mathrm{d}\omega}{\mathrm{d}t}=\frac{\partial\omega}{\partial t}+\frac{ \partial\omega}{\partial\theta}\frac{\partial\theta}{\partial t} \tag{22.50}\]

which can also be expressed in the form

\[\frac{\mathrm{d}\omega}{\mathrm{d}t}=\frac{\mathrm{d}\omega}{\mathrm{d}E}\frac {\mathrm{d}E}{\mathrm{d}t}=\frac{\eta_{\mathrm{c}}\omega_{0}}{\beta^{2}E_{0}} \frac{\mathrm{d}E}{\mathrm{d}t}, \tag{22.51}\]

where \(\eta_{\mathrm{c}}\) is the momentum compaction. The energy change per unit time is for a longitudinal impedance \(Z_{z}\) and \(n\)th harmonic of the beam current

\[\frac{\mathrm{d}E}{\mathrm{d}t}=q\,V_{z0}\frac{\omega_{0}}{2\pi}=-qZ_{z}\,I_{ n}\mathrm{e}^{\mathrm{i}(n\theta-\omega_{0}t)}\frac{\omega_{0}}{2\pi}, \tag{22.52}\]

where \(q=eZ>0\) is the electrical charge of the particle and \(Z\) the charge multiplicity. Collecting all terms for (22.51) we get with

\[\omega=\omega_{0}+\omega_{n}\mathrm{e}^{\mathrm{i}(n\theta-\Omega t)} \tag{22.53}\]

the relation

\[\omega_{n}(\Omega-n\omega_{0})=-\mathrm{i}\,\frac{q\eta_{\mathrm{c}}\omega_{0 }^{2}}{2\pi\beta^{2}}\frac{I_{n}Z_{z}}{E_{0}}. \tag{22.54}\]

This can be further simplified with the continuity equation

\[\frac{\partial\lambda}{\partial t}+\frac{1}{\tilde{R}}\,\frac{\partial}{ \partial\theta}(\beta c\,\lambda)=\frac{\partial\lambda}{\partial t}+\frac{ \partial\lambda}{\partial\theta}\omega_{0}+\frac{\partial\omega}{\partial \theta}\lambda_{0}=0\]

and we get with (21.77), (22.53)

\[(\Omega-n\omega_{0})I_{n}=\omega_{n}nI_{0}\,. \tag{22.55}\]

Replacing \(\omega_{n}\) in (22.54) by the expression (22.55,) we finally get for the perturbation frequency \(\Omega\) with \(I_{0}=\beta c\lambda_{0}\)

\[\Delta\Omega^{2}=(\Omega-n\omega_{0})^{2}=-\mathrm{i}\,\frac{nq\eta_{\mathrm{ c}}\omega_{0}^{2}I_{0}}{2\pi\beta^{2}E_{0}}Z_{\parallel}\,. \tag{22.56}\]

Equation (22.56) determines the evolution of the charge or current perturbation \(\lambda_{n}\) or \(I_{n}\) respectively. With \(\Delta\Omega=\Delta\Omega_{\mathrm{r}}+\mathrm{i}\Delta\Omega_{\mathrm{i}}\), the current perturbation is

\[I_{n}\,\mathrm{e}^{\mathrm{i}(n\theta-n\omega_{0}t-\Delta\Omega_{\mathrm{r}}t- \mathrm{i}\Delta\Omega_{\mathrm{i}}t)}=I_{n}\,\mathrm{e}^{\Delta\Omega_{ \mathrm{i}}t}\,\mathrm{e}^{\mathrm{i}(n\theta-n\omega_{0}t-\Delta\Omega_{ \mathrm{r}}t)} \tag{22.57}\]exhibiting an exponential factor which can cause instability or damping since there is a positive as well as negative solution from (22.56) for the frequency shift \(\Delta\Omega_{\rm i}\). The situation in a particular case will depend on initial conditions describing the actual perturbation of the density distribution, however, different initial perturbations must be expected to be present along a real particle distribution including at least one leading to instability.

Beam stability occurs only if the imaginary part of the frequency shift vanishes. This is never the case if the impedance has a resistive component causing a resistive-wall instability [12]. From (22.56) and the resistive wall impedance (22.39) we may derive a growth rate for the instability

\[\frac{1}{\tau_{\rm res.wall}}={\rm Im}\{\Delta\Omega\}=\frac{\sqrt{2}-1}{\sqrt {2}}\frac{n^{2}q\eta_{\rm c}\omega_{\rm i}^{2}l_{0}\overline{R}}{2\pi c\beta^ {2}E_{0}r_{\rm w}}\sqrt{\frac{2\pi\omega_{0}\mu}{n\sigma}}. \tag{22.58}\]

This result requires some more discussion since we know that circular accelerators exist, work, and have metallic vacuum chambers with a resistive surface. The apparent discrepancy is due to the fact that we have assumed a monochromatic beam which indeed is unstable but also unrealistic. In the following sections, we include a finite momentum spread, find a stabilizing mechanism called Landau damping and derive new stability criteria.

Below transition energy, \(\eta_{\rm c}>0\) will assure stability of a coasting beam as long as we consider only a purely capacitive impedance like the space-charge impedance (22.37) in which case \(\Delta\Omega_{\rm i}=0\). Above transition energy \(\eta_{\rm c}<0\) and the negative-mass instability appears as long as the impedance is capacitive or \(Z_{\rm i}>0\). For an inductive impedance, the stability conditions are exchanged below and above transition energy. In summary, we have the following longitudinal coasting beam stability conditions:

\[{\rm if}\quad Z_{\tau}\neq 0\to\Delta\omega_{\rm i}\neq 0\to\left\{\begin{array} []{l}\mbox{always stable}\\ \mbox{resistive-wall instability}\end{array}\right. \tag{22.59}\]

\[{\rm if}\quad Z_{\tau}=0\,\,\left\{\begin{array}{l}\left\{\begin{array}{l} Z_{i}<0\\ (\mbox{inductive})\end{array}\right.\to\left\{\begin{array}{l}\mbox{ stable for }\gamma\,>\,\gamma_{\rm tr}\,\,\,\mbox{or}\,\,\eta_{\rm c}<0\\ \mbox{unstable for }\gamma\,\leq\gamma_{\rm tr}\,\,\,\mbox{or}\,\,\eta_{\rm c}>0\\ \left\{\begin{array}{l}Z_{i}>0\\ (\mbox{capacitive})\end{array}\right.\to\left\{\begin{array}{l}\mbox{ stable for }\gamma\,<\gamma_{\rm tr}\,\,\,\mbox{or}\,\,\eta_{\rm c}>0\\ \mbox{unstable for }\gamma\,\geq\gamma_{\rm tr}\,\,\,\mbox{or}\,\,\eta_{\rm c}<0\,.\end{array}\right.\end{array}\right. \tag{22.60}\]

It is customary to plot the stability condition (22.56) in a \((Z_{\tau},Z_{\rm i})\)-diagram with \(\Delta\Omega_{\rm i}\) as a parameter. We solve (22.56) for the imaginary impedance \(Z_{\rm i}\) and get

\[Z_{\rm i}={\rm sgn}(\eta_{\rm c})a\left[\left(\frac{Z_{\tau}}{2\Delta\Omega_{ \rm i}}\right)^{2}\mp\left(\frac{\Delta\Omega_{\rm i}}{a}\right)^{2}\right], \tag{22.61}\]where

\[a=\frac{nq|\eta_{\rm c}|\omega_{0}^{2}l_{0}}{2\pi\beta^{2}E_{0}} \tag{22.62}\]

and plot the result in Fig. 22.9. Only the case \(\eta_{\rm c}>0\) is shown in Fig. 22.9 noting that the case \(\eta_{\rm c}<0\) is obtained by a \(180^{\circ}\) rotation of Fig. 22.9 about the \(Z_{\tau}\)-axis. Figure 22.9 demonstrates that beam stability occurs only if \(Z_{\rm r}=0\) and \(Z_{\rm i}>0\). Knowing the complex impedance for a particular accelerator, Fig. 22.9 can be used to determine the rise time \(1/\tau=\Delta\Omega_{\rm i}\) of the instability.

The rise time or growth rate of the negative-mass instability above transition is for a beam circulating within a perfectly conducting vacuum chamber from (22.37) and (22.56)

\[\frac{1}{\tau_{\rm neg.mass}}=\frac{n\omega_{0}}{\beta c\gamma}\sqrt{\frac{q| \eta_{\rm c}|cl_{0}\left(1+2\ln\frac{r_{\rm g}}{r_{0}}\right)}{\beta E_{0}}}. \tag{22.63}\]

In this section, it was implicitly assumed that all particles have the same momentum and therefore, the same revolution frequency \(\omega_{0}\) allowing a change of the revolution frequency only for those particles close to a particle density perturbation. This restriction to a monochromatic beam is not realistic and provides little beam stability for particle beams in a circular accelerator. In the following section, we will discuss the more general case of a beam with a finite momentum spread and review beam stability conditions under more realistic beam parameters.

#### Dispersion Relation

In the previous section, conditions for beam stability were derived based on a monochromatic particle beam. The rise time of the instability depends critically

Figure 22.9: Stability diagram for a coasting monochromatic particle beam

on the revolution frequency and we may assume that the conditions for beam stability may change if we introduce the more realistic case of a beam with a finite momentum spread and therefore a finite spread of revolution frequencies. In Chap. 15, we discussed the mathematical tool of the Vlasov equation to describe collectively the dynamics of a distribution of particles in phase space. We will apply this tool to the collective interaction of a particle beam with its environment.

The canonical variables describing longitudinal motion of particles are the azimuth \(\theta\) and relative momentum error \(\delta=\Delta p/p_{0}\). Neglecting radiation damping, the Vlasov equation is

\[\frac{\partial\Psi}{\partial t}+\dot{\theta}\frac{\partial\Psi}{\partial\theta }+\dot{\delta}\frac{\partial\Psi}{\partial\delta}=0\,, \tag{22.64}\]

where \(\Psi(\delta,\theta,t)\) is the particle distribution. For a coasting beam with a small perturbation

\[\Psi=\Psi_{0}+\Psi_{n}\,\mathrm{e}^{\mathrm{i}(n\theta-\omega_{n}t)} \tag{22.65}\]

we get after insertion in (22.64) and sorting terms the relation

\[\mathrm{i}\left(\omega_{n}-n\omega\right)\Psi_{n}=\frac{\dot{\delta}}{\mathrm{ e}^{\mathrm{i}(n\theta-\omega_{n}t)}}\frac{\partial\Psi_{0}}{\partial\delta}. \tag{22.66}\]

Making use of the correlation between particle momentum and revolution frequency, we get from (22.66) with \(\frac{\partial\Psi_{0}}{\partial\delta}=\frac{\partial\Psi_{0}}{\partial \omega}\frac{\partial\omega}{\partial\delta}=\eta_{c}\omega_{0}\frac{\partial \Psi_{0}}{\partial\omega}\)

\[\Psi_{n}=-\mathrm{i}\,\frac{\eta_{c}\omega_{0}\dot{\delta}}{\mathrm{e}^{ \mathrm{i}(n\theta-\omega_{n}t)}}\frac{\partial\Psi_{0}}{\partial\omega}\frac{ 1}{\omega_{n}-n\omega}. \tag{22.67}\]

Integrating the l.h.s. of (22.67) over all momenta, we get for the perturbation current

\[q\ \frac{\beta c}{\dot{R}}\int_{-\infty}^{\infty}\Psi_{n}(\delta)\,\mathrm{d} \delta=I_{n}\,.\]

At this point, it is convenient to transform from the variable \(\delta\) to the frequency \(\omega\) and obtain the particle distribution in these new variables

\[\Psi(\delta,\theta)=\eta_{c}\omega_{0}\Phi(\omega,\theta). \tag{22.68}\]

Performing the same integration on the r.h.s. of (22.67), we get with (22.52) and \(\dot{\delta}=(\mathrm{d}E/\mathrm{d}t)/(\beta^{2}E_{0})\) the dispersion relation [13]

\[1=-\,\mathrm{i}\,\frac{q^{2}\omega_{0}^{3}\eta_{c}Z_{z}}{2\pi\beta^{2}E_{0}} \int\frac{\partial\Phi_{0}/\partial\omega}{\omega_{n}-n\omega}\,\mathrm{d} \omega\,. \tag{22.69}\]The integration is to be taken over the upper or lower complex plane where we assume that the distribution function \(\Phi\) vanishes sufficiently fast at infinity. Trying to establish beam stability for a particular particle distribution, we solve the dispersion relation for the frequency \(\omega_{n}\) or frequency shift \(\Delta\omega_{n}=\omega_{n}-n\omega\) which is in general complex. The real part causes a shift in the frequency while the imaginary part determines the state of stability or instability for the collective motion.

For example, it is interesting to apply this result to the case of a coasting beam of monochromatic particles as discussed in the previous section. Let the particle distribution be uniform in \(\theta\) and a delta function in energy. In the dispersion relation, we need to take the derivative with respect to the revolution frequency and set therefore

\[\frac{\partial\Phi_{0}}{\partial\omega}\,=\,\frac{N_{\rm p}}{2\pi}\,\frac{ \partial}{\partial\omega}\delta(\omega-\omega_{0}). \tag{22.70}\]

Insertion into (22.69) and integration by parts results in

\[\int_{-\infty}^{\infty}\frac{\partial\Phi_{0}/\partial\omega}{\omega_{n}-n \omega}\,{\rm d}\omega\,=\,\frac{N_{\rm b}}{2\pi}\,\frac{n}{(\omega_{n}-n \omega_{0})^{2}} \tag{22.71}\]

which is identical to the earlier result (22.56) in the previous section. Application of the Vlasov equation therefore gives the same result as the direct derivation of the negative-mass instability conditions as it should be.

We may now apply this formalism to a beam with finite momentum spread. In preparation to do that, we note that the integrand in (22.69) has a singularity at \(\omega=\omega_{n}/n\) which we take care of by applying Cauchy's residue theorem for

\[\int\,\frac{\partial\Phi_{0}/\partial\omega}{\omega_{n}-n\omega}\,{\rm d} \omega\,=\,\mbox{P.V.}\,\int\limits_{n\omega\neq\omega_{n}}\frac{\partial \Phi_{0}/\partial\omega}{\omega_{n}-n\omega}\,{\rm d}\omega\,-\,{\rm i}\,\pi \,\left.\frac{\partial\Phi_{0}}{\partial\omega}\right|_{\omega_{n}/n}. \tag{22.72}\]

The dispersion relation (22.69) then assumes the form

\[1\,=\,{\rm i}\,\frac{q^{2}\omega_{0}^{3}\,\eta_{\rm c}\,Z_{z}}{2\pi\beta^{2}E _{0}}\left[\,{\rm i}\,\frac{\pi}{n}\frac{\partial\Phi_{0}}{\partial\omega} \right|_{\omega=\frac{n\omega}{n}}\,-\,\mbox{P.V.}\int\,\frac{\partial\Phi_{ 0}/\partial\omega}{\omega_{n}-n\omega}\,{\rm d}\omega\right], \tag{22.73}\]

where P.V. indicates that only the principal value of the integral be taken.

The solutions of the dispersion function depend greatly on the particle distribution in momentum or revolution-frequency space. To simplify the expressions, we replace the revolution frequency by its deviation from the reference value [14]. With \(2S\) being the full-width half maximum of the particle momentum distribution (Fig. 22.10), we define the new variables

\[x=\,\frac{\omega-\omega_{0}}{S},\qquad\mbox{and}\qquad x_{1}\,=\,\frac{ \Delta\omega_{n}}{nS}\,=\,\frac{\Omega-n\omega_{0}}{nS}. \tag{22.74}\]In these variables the particle distribution becomes

\[f(x)=\left.\frac{2\pi S}{N_{\rm b}}\,\Phi(\omega)\right. \tag{22.75}\]

which is normalized to \(f(\pm 1)=\frac{1}{2}f(0)\) and \(\int\!f(x){\rm d}x=1\). The full momentum spread at half maximum intensity is

\[\frac{\Delta p}{p_{0}}=\frac{2S}{|\eta_{\rm c}|\omega_{0}} \tag{22.76}\]

and (22.73) becomes with this

\[1=-\,{\rm i}\frac{2qZ_{\rm c}I_{0}}{\pi\beta^{2}E_{0}n\eta_{\rm c}\left(\frac {\Delta p}{p_{0}}\right)\,^{2}}\left[\,{\rm P.V.}\int_{-\infty}^{\infty}\frac {\partial f_{0}(x)/\partial x}{x_{1}-x}\,{\rm d}x-\,{\rm i}\,\pi\,\frac{ \partial f_{0}}{\partial x}\bigg{|}_{x_{1}}\right]. \tag{22.77}\]

It is customary to define parameters \(U,V\) by

\[V+{\rm i}\,U=\left.\frac{2q\,I_{0}}{\pi\beta^{2}E_{0}\eta_{\rm c}\left(\frac{ \Delta p}{p_{0}}\right)^{2}}\frac{(Z_{\tau}+{\rm i}\,Z_{\rm i})_{z}}{n}\right. \tag{22.78}\]

and the dispersion relation becomes finally with this

\[1=-(V+{\rm i}\,U)I, \tag{22.79}\]

where the integral

\[I=\left[\,{\rm P.V.}\int_{-\infty}^{\infty}\frac{\partial f_{0}(x)/\partial x }{x_{1}-x}\,{\rm d}x-\,{\rm i}\pi\,\frac{\partial f_{0}}{\partial x}\bigg{|}_ {x_{1}}\right]. \tag{22.80}\]

For a particular accelerator all parameters in (22.79) are known, at least in principle, and we may determine the status of stability or instability for a desired beam current \(I_{0}\) by solving for the generally complex frequency shift \(\Delta\omega\). The specific

Figure 22.10: Particle distribution \(f\left(x\right)\)

boundary of stability depends on the actual particle distribution in momentum. Unfortunately, (22.79) cannot be solved analytically for an arbitrary momentum distribution and we will have to either restrict our analytical discussion to simple solvable distributions or to numerical evaluation.

For reasonable representations of real particle distributions in an accelerator a central region of stability can be identified for small complex impedances and finite spread in momentum. Regions of stability have been determined for a number of particle distributions and the interested reader is referred for more detailed information on such calculations to references [15, 16, 17, 18, 19].

As an example, we use a simple particle distribution (Fig. 22.11)

\[f(x)=\frac{1}{\pi}\,\frac{1}{1+x^{2}} \tag{22.81}\]

and evaluate the dispersion relation (22.79). The integral in (22.80) becomes now after integration by parts

\[I=\mathrm{P.V.}\int_{-\infty}^{\infty}\frac{1}{(1+x^{2})(x_{1}-x)^{2}}\,\mathrm{ d}x \tag{22.82}\]

exhibiting a new singularity at \(x=i\) while the integration path still excludes the other singularity at \(x=x_{1}\). Applying the residue theorem

\[\int\frac{f\,(z)\mathrm{d}z}{z-z_{0}}=\mathrm{i}\,2\pi\,\mathrm{Res}[f(z),z_{0 }]=\mathrm{i}\,2\pi\lim_{z\to z_{0}}(z-z_{0})f\,(z) \tag{22.83}\]

we get

\[\mathrm{P.V.}\int_{-\infty}^{\infty}\frac{1}{(1+x^{2})(x_{1}-x)}\,\mathrm{d}x =\frac{1}{(x_{1}-\mathrm{i})^{2}}. \tag{22.84}\]

The second term in (22.80) is

\[-\left.\mathrm{i}\,\pi\,\frac{\partial f_{0}}{\partial x}\right|_{x_{1}}= \mathrm{i}\frac{2x_{1}}{(1+x_{1}^{2})^{2}} \tag{22.85}\]

Figure 22.11: Particle distribution in momentum space

and the dispersion relation (22.79) becomes

\[1=-{\rm i}(V+{\rm i}U)\left(\frac{1}{(x_{1}-{\rm i})^{2}}+{\rm i}\frac{2x_{1}}{(1 +x_{1}^{2})^{2}}\right). \tag{22.86}\]

We solve this for \((x_{1}-i)^{2}\) and get

\[x_{1}={\rm i}\pm\sqrt{-{\rm i}\left(V+{\rm i}U\right)\left(1+{\rm i}\frac{2x_{ 1}}{({\rm i}+x_{1})^{2}}\right)}. \tag{22.87}\]

For a small beam current \(i_{0}\), we get \(x_{1}\approx i\) and the second term in the square bracket becomes approximately \(1/2\). Recalling the definition (22.74) for \(x_{1}\), we get from (22.87)

\[\Delta\Omega={\rm i}\,nS\pm\sqrt{\tfrac{3}{2}n^{2}S^{2}(U-{\rm i}\,V)}, \tag{22.88}\]

where from (22.76) \(S=\tfrac{1}{2}|\eta_{c}|\omega_{0}\Delta p/p_{0}\). The significant result in (22.88) is the fact that the first term on the right-hand side has a definite positive sign and provides therefore damping which is called Landau damping [20].

Recalling the conditions for the negative-mass instability of a monochromatic beam, we did not obtain beam stability for any beam current if \(Z_{\rm r}\propto V=0\) and the reactive impedance was inductive or \(Z_{\rm i}\propto U<0\). Now with a finite momentum spread in the beam we get in the same case

\[\Delta\Omega_{\rm neg.mass}={\rm i}\,nS\pm{\rm i}\sqrt{\tfrac{3}{2}n^{2}S^{2}|U |}, \tag{22.89}\]

where \(S^{2}|U|\) is independent of the momentum spread. We note that it takes a finite beam current (\(U\propto I_{0}\)) to overcome Landau damping and cause instability. Of course Landau damping is proportional to the momentum spread \(S\) and does not occur for a monochromatic beam. Equation (22.88) serves as a stability criterion for longitudinal coasting-beam instabilities and we will try to derive a general expression by writing (22.88) in the form

\[\Delta\Omega={\rm i}\,n\,S\pm\sqrt{a-{\rm i}b} \tag{22.90}\]

and get after evaluating the square root

\[\Delta\Omega={\rm i}\,nS\pm\left(\sqrt{\frac{r+a}{2}}-{\rm i}\sqrt{\frac{r-a} {2}}\right), \tag{22.91}\]

where \(r=\sqrt{a^{2}+b^{2}}\). Beam stability occurs for \({\rm Im}\{\Delta\Omega\}>0\) or

\[n^{2}S^{2}=\frac{r-a}{2} \tag{22.92}\]which is in more practical quantities recalling the definition (22.76) for \(S\)

\[\left(\frac{\Delta p}{p_{0}}\right)^{2}\geq\frac{3}{2\pi}\,\frac{q\,I_{0}}{\beta ^{2}E_{0}|\eta_{\rm c}|}\left(\frac{|Z_{z}|}{n}-\frac{Z_{\rm i}}{n}\right). \tag{22.93}\]

We may solve (22.93) for the impedance and get an equation of the form

\[Z_{\rm i}=AZ_{\rm r}^{2}-\frac{1}{4A} \tag{22.94}\]

which is shown in Fig. 22.12.

Any combination of actual resistive and reactive impedances below this curve cause beam instability for the particle distribution (22.81). We note the significant difference to Fig. 22.9 where the impedance had to be purely positive and reactive to obtain beam stability.

Other momentum distributions like \(f(x)\propto(1-x^{2})^{m}\) lead to similar results [15] although the stability curves allow less resistive impedance than the distribution (22.81). As a safe stability criterion which is true for many such distributions including a Gaussian distribution we define the area of stability by a circle with a radius \(R=Z_{\rm i}|_{Z_{\rm r}=0}\,=\,1/(4A)\). With this assumption, the stability criterion for the longitudinal microwave instability is

\[\frac{|Z_{z}|}{n}\leq F\frac{\beta^{2}E_{0}|\eta_{\rm c}|}{ql_{0}}\left(\frac{ \Delta p}{p_{0}}\right)^{2}, \tag{22.95}\]

Figure 22.12: Stability diagram for the particle distribution (22.93)

where the form factor \(F=\pi/3\) for the distribution (22.81) and is generally of the order of unity for other bell shaped distributions. The criterion (22.95) has been derived by Keil and Schnell [21] and is known as the Keil-Schnell stability criterion. For a desired beam current and an allowable momentum spread an upper limit for the normalized impedance can be derived.

The impedance seen by the particle beam obviously should be minimized to achieve the highest beam-beam currents. A large value of the momentum compaction is desirable here to increase the mixing of the revolution frequencies as well as a large momentum spread to achieve high beam currents. A finite momentum spread increases beam stability where there was none for a monochromatic coasting beam as discussed earlier. This stabilization effect of a finite momentum spread is called Landau damping.

##### Landau Damping

In previous sections, we repeatedly encountered a damping phenomenon associated with the effect of collective fields on individual particle stability. Common to the situations encountered is the existence of a set of oscillators or particles, under the influence of an external driving force. Particularly, we are interested in the dynamics when the external excitation is caused by the beam itself. Landau [20] studied this effect first and later Hereward [22] formulated the theory for application to particle accelerators.

We consider a bunch of particles where each particle oscillates at a different frequency \(\Omega\), albeit within a small range of frequencies. The equation of motion for each oscillator under the influence of the force \(F\,e^{-\mathrm{i}\omega t}\) is

\[\ddot{u}\,+\,\Omega^{2}\,u=F\,\mathrm{e}^{-\mathrm{i}\omega t} \tag{22.96}\]

and the solution

\[u=F\frac{\mathrm{e}^{-\mathrm{i}\omega t}}{2\omega}\,\left(\frac{1}{\Omega- \omega}-\frac{1}{\Omega\,+\,\omega}\right). \tag{22.97}\]

Folding this solution with the distribution function of particles in frequency space

\[\psi(\omega)=\frac{1}{N_{\mathrm{b}}}\frac{\mathrm{d}N_{\mathrm{b}}}{\mathrm{ d}\Omega} \tag{22.98}\]

one obtains the center of mass amplitude of the bunch

\[\bar{u}=F\frac{\mathrm{e}^{-\mathrm{i}\omega t}}{2\omega}\int_{-\infty}^{ \infty}\left[\frac{\psi(\Omega)}{\Omega-\omega}-\frac{\psi(\Omega)}{\Omega+ \omega}\right]\,\mathrm{d}\Omega \tag{22.99}\]or with \(\psi(\Omega)=\psi(-\Omega)\) and \(\int_{-\infty}^{\infty}\frac{\psi(\Omega)}{\Omega-\omega}\,\mathrm{d}\Omega=- \int_{-\infty}^{\infty}\frac{\psi(\Omega)}{\Omega+\omega}\,\mathrm{d}\Omega\)

\[\tilde{u}=F\,\frac{\mathrm{e}^{-\mathrm{i}\omega t}}{\omega}\int_{-\infty}^{ \infty}\frac{\psi(\Omega)}{\Omega-\omega}\,\mathrm{d}\Omega. \tag{22.100}\]

Here we apply again Cauchy's residue theorem and get

\[\tilde{u}=F\frac{\mathrm{e}^{-\mathrm{i}\omega t}}{\omega}\left[+\mathrm{i} \pi\,\psi(\omega)+\mathrm{P.V.}\int_{-\infty}^{\infty}\frac{\psi(\Omega)}{ \Omega-\omega}\,\mathrm{d}\Omega\right]. \tag{22.101}\]

The derivation up to here appears quite abstract and we pause a moment to reflect on the physics involved here. We know that driving an oscillator at resonance leads to infinitely large amplitudes and that is what the mathematical formulation above expresses. However, we also know that infinite amplitudes take an infinite time to build up and the solutions gained above describe only the state after a long time. The same result can be obtained in a physical more realistic way if we apply the excitation at time \(t=0\) and look for the solution at \(t\to\infty\) as has been shown by Hofmann [23]. As an added result of this time evolution derivation, we obtain the correct sign for the residue which we have tacitly assumed to be negative, but mathematically could be of either sign.

To understand the damping effect, we calculate the velocity \(\tilde{u}\) and get from (22.101)

\[\tilde{u} =-\mathrm{i}\omega\,\tilde{u}\] \[=F\,\mathrm{e}^{-\mathrm{i}\omega t}\left[+\pi\,\psi(\omega)- \mathrm{i}\,\mathrm{P.V.}\int_{-\infty}^{\infty}\frac{\psi(\Omega)}{\Omega- \omega}\,\mathrm{d}\Omega\right]. \tag{22.102}\]

The bunch velocity is in phase with the external force for the residue term allowing extraction of energy from the external force. The principal value term, on the other hand, is out of phase and no work is done. If, for example, the external force is due to a transverse wake field generated by a bunch performing coherent betatron oscillations, the described mechanism would extract energy from the wake field thus damping the coherent betatron oscillation. The question is where does the energy go?

For this, we study the time evolution of the solution for the inhomogeneous differential equation (22.96) in the form

\[u=a\sin\Omega t+\frac{F}{\Omega^{2}-\omega^{2}}\sin\omega t. \tag{22.103}\]

At time \(t=0\) we require that the amplitude and velocity of the bunch motion be zero \(u(t=0)=0\) and \(\dot{u}(t=0)=0\). The oscillation amplitude

\[a=-\frac{\omega}{\Omega}\,\frac{F}{\Omega^{2}-\omega^{2}} \tag{22.104}\]and the final expression for the solution to (22.96) is for \(\Omega\neq\omega\)

\[u_{\Omega\neq\omega}(t)=\frac{F}{\Omega^{2}-\omega^{2}}\left(\sin\omega t-\frac{ \omega}{\Omega}\sin\Omega t\right). \tag{22.105}\]

Close to or at resonance \(\Omega=\omega+\Delta\) and (22.105) becomes

\[u_{\Omega\approx\omega}(t)=-\frac{F}{2\omega}\left(t\cos\omega t-\frac{\sin \omega t}{\omega}\right). \tag{22.106}\]

The oscillation amplitude of particles at resonance grows continuously with time while the width of the resonance shrinks like \(1/t\) thus absorbing energy linear in time. This Landau damping depends critically on the resistive interaction with the wake fields or external forces and is mathematically expressed by the residue term. This residue term, however, depends on the particle density at the excitation frequency \(\omega\) and is zero if the particle distribution in frequency space does not overlap with the frequency \(\omega\). For effective Landau damping to occur such an overlap is essential.

##### Transverse Coasting-Beam Instability

Particle beams travelling off center through vacuum chamber sections can induce transverse fields which act back on the beam. We express the strength of this interaction by the transverse coupling impedance. In terms of the transverse coupling impedance, the force is

\[F_{\perp}=\mathrm{i}\,\frac{qZ_{\perp}I_{0}\,u}{2\pi\bar{R}}, \tag{22.107}\]

where \(I_{0}\) is the beam current, \(u\) the transverse beam displacement, \(Z_{\perp}/(2\pi\bar{R})\) the average transverse coupling impedance and \(2\pi\bar{R}\) the ring circumference. The equation of motion is then

\[\ddot{u}+v_{0}^{2}\omega_{0}^{2}\,u=-\mathrm{i}\frac{qZ_{\perp}I_{0}}{2\pi \bar{R}m\gamma}(u+\bar{u}) \tag{22.108}\]

with \(u\) the betatron oscillation amplitude of an individual particle and \(\bar{u}\) the amplitude of the coherent bunch oscillation. Since the perturbation is linear in the amplitudes, we expect tune shifts from the perturbations. The incoherent tune shift due to individual particle motion will be incorporated on the l.h.s. as a small tune shift

\[\delta v_{0}=\mathrm{i}\frac{cqZ_{\perp}I_{0}}{4\pi\,v_{0}\omega_{0}E_{0}}. \tag{22.109}\]The transverse impedance is generally complex and we get therefore from the real part of the coupling impedance a real tune shift while the imaginary part leads to damping or antidamping depending on the sign of the impedance. The imaginary frequency shift is equal to the growth rate of the instability and is given by

\[\frac{1}{\tau}=\mathrm{Im}\{\omega\}=\frac{q\,\mathrm{Re}\{Z_{\perp}\}\,I_{0}}{ 4\pi\,\overline{R}m\gamma\omega_{\beta_{0}}}\,. \tag{22.110}\]

For a resistive transverse impedance, we observe therefore always instability known as the transverse resistive-wall instability.

Similar to the case of a longitudinal coasting beam, we find instability for any finite beam current just due to the reactive space-charge impedance alone, and again we have to rely on Landau damping to obtain beam stability for a finite beam intensity. To derive transverse stability criteria including Landau damping, we consider the coherent tune shift caused by the coherent motion of the whole bunch for which the equation of motion is

\[\tilde{u}+\omega_{\beta 0}^{2}\,u=2v_{0}\omega_{0}[U+(1+\mathrm{i})V]\tilde{u}\,, \tag{22.111}\]

where

\[U+(1+\mathrm{i})V=-\mathrm{i}\frac{cqZ_{\perp}I_{0}}{4\pi\,v_{0}E_{0}}, \tag{22.112}\]

The coherent beam oscillation must be periodic with the circumference of the ring and is of the form \(\tilde{u}=\hat{u}\,e^{\mathrm{i}(n\theta-\omega t)}\). As can be verified by back insertion the solution of (22.111) is

\[y=[U+(1+\mathrm{i})V]\frac{2v_{0}\omega_{0}}{v_{1}^{2}\omega_{0}^{2}-(n\omega _{0}-\omega)^{2}}\tilde{u}\,. \tag{22.113}\]

Now we must fold (22.113) with the distribution in the spread of the betatron oscillation frequency. This spread is mainly related to a momentum spread via the chromaticity and the momentum compaction. The distribution \(\psi(\delta)\) where \(\delta=\Delta p/p_{0}\), is normalized to unity \(\int\psi(\delta)d\delta=1\) and the average particle position is \(\tilde{u}=\int u\psi(\delta)\mathrm{d}\delta\). The dispersion relation is then with this from (22.113)

\[1=[U+(1+\mathrm{i})V]\int_{-\infty}^{\infty}\frac{2v_{0}\omega_{0}\psi(\delta) \mathrm{d}\delta}{v_{1}^{2}\omega_{0}^{2}-(n\omega_{0}-\omega)^{2}}. \tag{22.114}\]

or simplified by setting \(v_{1}\approx v_{0}\) and ignoring the fast wave \((n+v)\omega_{0}\)[24]

\[1=[U+(1+\mathrm{i})V]\int_{-\infty}^{\infty}\frac{\psi(\delta)\mathrm{d}\delta }{\omega-(n-v_{0})\omega_{0}}. \tag{22.115}\]This is the dispersion relation for transverse motion and can be evaluated for stability based on a particular particle distribution in momentum. As mentioned before, the momentum spread transforms to a betatron oscillation frequency spread by virtue of the momentum compaction

\[\Delta v_{\beta}=v_{\beta 0}\Delta\omega_{0}=v_{\beta 0}\eta_{c}\delta\omega_{0} \tag{22.116}\]

and by virtue of the chromaticity

\[\Delta v_{\beta}=\xi_{u}\delta\,. \tag{22.117}\]

Landau damping provides again beam stability for a finite beam current and finite coupling impedances, and the region of stability depends on the actual particle distribution in momentum.

### 22.4 Longitudinal Single-Bunch Effects

The dynamics in bunched particle beams is similar to that of a coasting beam with the addition of synchrotron oscillations. The frequency spectrum of the circulating beam current contains now many harmonics of the revolution frequency with sidebands due to betatron and synchrotron oscillations. The bunch length depends greatly on the interaction with the rf-field in the accelerating cavities but also with any other field encountered within the ring. It is therefore reasonable to expect that wake fields may have an influence on the bunch length which is know as potential well distortion.

#### Potential-Well Distortion

From the discussions on longitudinal phase space motion in circular accelerators, it is known that the particle distribution or bunch length depends on the variation in time of the rf-field interacting with the beam in the accelerating cavities. Analogous, we would expect that wake fields may have an impact on the longitudinal particle distribution. Pellegrini and Sessler [25] For a particular wake field, we have studied this effect in Chap. 15 recognizing that a bunch passing through an rf-cavity causes beam loading by exciting fields at the fundamental frequency in the cavity. These fields then cause a modification of the bunch length. In this section, we will expand on this thought and study the effect due to higher-order mode wake fields.

To study this in more detail, we ignore the transverse particle distribution. The rate of change in the particle momentum can be derived from the integral of all longitudinal forces encountered along the circumference and we set with \(\delta=dp/p_{0}\)

\[\frac{\mathrm{d}\delta}{\mathrm{d}t}=\frac{qF(\tau)}{\beta^{2}E_{0}T_{0}}, \tag{22.118}\]where \(qF(\tau)\) is the sum total of all acceleration and energy losses of a particle at a position \(z=\beta c\tau\) from the bunch center or reference point over the course of one revolution and \(T_{0}\) is the revolution time. The change of \(\tau\) per unit time depends on the momentum compaction of the lattice and the momentum deviation

\[\frac{\mathrm{d}\tau}{\mathrm{d}t}=-\eta_{\mathrm{c}}\delta\,. \tag{22.119}\]

Both equations can be derived from the Hamiltonian

\[\mathcal{H}=-\tfrac{1}{2}\eta_{\mathrm{c}}\delta^{2}-\int_{0}^{\tau}\frac{qF( \bar{\tau})}{\beta^{2}E_{0}T_{0}}\,\mathrm{d}\bar{\tau}\,. \tag{22.120}\]

For an electron ring and small oscillation amplitudes, we have

\[qF(\tau)=qV_{\mathrm{rf}}(\tau_{\mathrm{s}}+\tau)-U(E)+qV_{\mathrm{w}}(\tau)=q \left.\frac{\partial V_{\mathrm{rf}}}{\partial\tau}\right|_{\tau_{\mathrm{s}}} \tau+qV_{\mathrm{w}}(\tau)\,, \tag{22.121}\]

where we ignored radiation damping and where \(V_{\mathrm{w}}(\tau)\) describes the wake field. In the last form, the equation is also true for protons and ions if we set the synchronous time \(\tau_{\mathrm{s}}=0\). Inserting (22.121) into (22.120) and using the definition of the synchrotron oscillation frequency (9.35) we get the new Hamiltonian

\[\mathcal{H}=-\tfrac{1}{2}\eta_{\mathrm{c}}\delta^{2}-\tfrac{1}{2}\frac{ \Omega_{s0}^{2}}{\eta_{\mathrm{c}}}\tau^{2}-\int_{0}^{\tau}\frac{qV_{\mathrm{ w}}(\bar{\tau})}{\beta^{2}E_{0}T_{0}}\,\mathrm{d}\bar{\tau}\,. \tag{22.122}\]

#### Synchrotron Oscillation Tune Shift

First we use the Hamiltonian to formulate the equation of motion and determine the effect of wake fields on the dynamics of the synchrotron motion. The equation of motion is from (22.122)

\[\ddot{\tau}+\Omega_{s0}^{2}\tau=\mathrm{sign}(\eta_{\mathrm{c}})\frac{2\pi \,\Omega_{s0}^{2}V_{\mathrm{w}}}{\omega_{0}hV_{\mathrm{rf}}|\cos\psi_{s0}|}, \tag{22.123}\]

where we have made use of the definition of the unperturbed synchrotron oscillation frequency \(\Omega_{s0}\). We express the wake field in terms of impedance and beam spectrum

\[V_{\mathrm{w}}(t)=-\int_{-\infty}^{\infty}Z_{\parallel}(\omega)\,I(t,\omega) \,\mathrm{e}^{\mathrm{i}\omega t}\,\mathrm{d}\omega\,, \tag{22.124}\]

and use (21.85) for

\[V_{\mathrm{w}}(t)=-I_{\mathrm{b}}\sum_{p=-\infty}^{\infty}Z_{\parallel}(p)\, \Psi(p)\,\mathrm{e}^{-\mathrm{i}p\omega_{0}\tau}, \tag{22.125}\]where \(I_{\rm b}\) is the bunch current and

\[\Psi(p)=\int_{-\infty}^{+\infty}J_{0}(p\omega_{0}\hat{\tau})\,\Phi(t,\hat{\tau}) \,{\rm d}\hat{\tau}\;.\]

The maximum excursion \(\hat{\tau}\) during phase oscillation is much smaller than the revolution time and the exponential factor

\[{\rm e}^{{\rm i}p\omega_{0}\tau}\approx 1+{\rm i}p\omega_{0}\tau-{{ 1\over 2}}p^{2}\omega_{0}^{2}\tau^{2}+{\cal O}(3) \tag{22.126}\]

can be expanded. After insertion of (22.120), (22.121) into (22.123) the equation of motion is

\[\vec{\tau}\,+\,{\cal Q}_{s0}^{2}\,\tau \approx \tag{22.127}\] \[-{\rm sign}(\eta_{\rm c}){2\pi I_{\rm b}{\cal Q}_{s0}^{2}\over \omega_{0}hV_{\rm rf}|\cos\psi_{s0}|}\sum_{p=-\infty}^{\infty}Z_{\parallel}(p) \Psi(p)\left(1-{\rm i}p\omega_{0}\tau-{{ 1\over 2}}p^{2}\omega_{0}^{2}\tau^{2}\right).\]

The first term in the factor \(\left(1-{\rm i}p\omega_{0}\tau-{{ 1\over 2}}p^{2}\omega_{0}^{2}\tau^{2}\right)\) is independent of \(\tau\) and causes a synchronous phase shift due to resistive losses

\[\Delta\psi_{\rm s}={\rm sgn}(\eta_{\rm c}){2\pi\,I_{\rm b}\over V_{\rm rf}| \cos\psi_{s0}|}\sum_{p=-\infty}^{\infty}{\rm Re}\{Z_{\parallel}(p)\}\,\Psi(p)\,. \tag{22.128}\]

For a resistive positive impedance, for example, the phase shift is negative above transition indicating that the beam requires more energy from the rf-cavity. By measuring the shift in the synchronous phase of a circulating bunch as a function of bunch current, it is possible to determine the resistive part of the longitudinal impedance of the accelerator. To do this one may fill a small amount of beam in the bucket upstream from the high intensity bunch and use the signal from the small bunch as the time reference against which the big bunch will shift with increasing current.

The second term in (22.127) is proportional to \(\tau\) and therefore acts like a focusing force shifting the incoherent synchrotron oscillation frequency by

\[\Delta\Omega_{\rm s}=-\,{\rm sign}(\eta_{\rm c}){\pi\,I_{\rm b}{\cal Q}_{s0} \over h\,V_{\rm rf}|\cos\psi_{s0}|}\sum_{p=-\infty}^{\infty}{\rm Im}\{Z_{ \parallel}(p)\}\,p\,\Psi(p). \tag{22.129}\]

The real part of the impedance is symmetric in \(p\) and therefore cancels in the summation over \(p\) which leaves only the imaginary part consistent with the expectation that the tune shift be real. At this point, it becomes necessary to introduce a particular particle distribution and an assumption for the impedance spectrum. For long bunches, the frequencies involved are low and one might use for the impedance the space charge and broad-band impedance which both are constant for low frequencies. In this case, the impedance can be extracted from the sum in (22.129) and the remaining arguments in the sum depend only on the particle distribution.

For a parabolic particle distribution, for example, (22.129) reduces to [26]

\[\Delta\Omega_{\rm s}=-\,{\rm sgn}(\eta_{\rm c})\frac{16\,I_{\rm b}}{\pi^{3}B^{3 }h\,V_{\rm rf}|\cos\psi_{0}|}\,{\rm Im}\left\{\frac{Z_{\parallel}(p)}{p} \right\}\;, \tag{22.130}\]

where \(B\) is the bunching factor \(B=\ell/(2\pi\widetilde{R})\) with \(\ell\) the effective bunch length.

A measurement of the incoherent synchrotron tune shift as a function of bunch current allows the determination of the reactive impedance of the accelerator for a given particle distribution. This tune shift is derived from a measurement of the unperturbed synchrotron frequency \(\Omega_{\rm s0}\) for a very small beam current combined with the observation of the quadrupole mode frequency \(\Omega_{\rm 2s}\) as a function of bunch current. The incoherent tune shift is then

\[\Delta\Omega_{\rm s,incoh}=\mu\,(\Omega_{\rm 2s}-2\Omega_{\rm s0})\;, \tag{22.131}\]

where \(\mu\) is a distribution dependent form factor of order 2 for a parabolic distribution [27].

The third and subsequent terms in (22.127) contribute nonlinear effects making the synchrotron oscillation frequency amplitude dependent similar to the effects of nonlinear fields in transverse beam dynamics.

##### Bunch Lengthening

A synchrotron frequency shift is the consequence of a change in the longitudinal focusing and one might expect therefore also a change in the bunch length. In first approximation, one could derive expressions for the new bunch length by scaling with the synchrotron tune shift. Keeping the phase space area constant in the proton and ion case or keeping only the energy spread constant in the electron case, a rough estimate for bunch lengthening can be obtained for a specific particle distribution. Since the electron bunch length scales inversely proportional to the synchrotron frequency, we have

\[\frac{\sigma_{\ell}}{\sigma_{\ell 0}}=\frac{\Omega_{\rm s}}{\Omega_{\rm s0}}=1 +\,\frac{\Delta\Omega_{\rm s}}{\Omega_{\rm s0}}. \tag{22.132}\]

From (22.132), one can determine for an electron beam the potential-well bunch lengthening or shortening, depending on the sign of the reactive impedance. For a proton or ion beam, the scaling is somewhat different because of the preservation of phase space.

This approach to understanding potential-well bunch lengthening assumes that the particle distribution does not change which is an approximate but not correct assumption. The deformation of the potential well is nonlinear and can create significant variations of the particle distribution specifically, for large amplitudes.

In this discussion, we determine the stationary particle distribution under the influence of wake fields by solving the Vlasov equation

(22.133)

For a stationary solution, and therefore any function of the Hamiltonian is a solution of the Vlasov equation. Since the Hamiltonian does not exhibit explicitly the time variable, any such function could be the stationary solution which we are looking for and we set therefore. The local particle density is then after integrating over all momenta

(22.134)

where is the number of particles per bunch or with (22.122)

(22.135)

Without wake fields, the distribution of an electron beam is Gaussian and the introduction of wake fields does not change that for the energy distribution. We make therefore the ansatz

(22.136)

where and are normalization factors for the respective distributions. Integrating over all momenta, the longitudinal particle distribution is finally

(22.137)

where we used from (13.26). A self-consistent solution of this equation will determine the longitudinal particle distribution under the influence of wake fields. Obviously, this distribution is consistent with our earlier results for an electron beam in a storage ring, in the limit of no wake fields. The nature of the wake fields will then determine the distortion from the Gaussian distribution.

As an example, we assume a combination of an inductive (\(L\)) and a resistive (\(R\)) wake field

\[V_{\rm w}=-L\frac{{\rm d}I}{{\rm d}t}-RI_{\rm b}. \tag{22.138}\]

Such a combination actually resembles rather well the real average impedance in circular accelerators at least at lower frequencies as evidenced in the impedance spectrum of the SPEAR storage ring shown in Fig. 19.9. Inserting (22.138) into (22.137) while setting for a moment the resistance to zero (\(R=0\)) we get after integration the transcendental equation

\[\lambda(\tau)=N_{\rm b}\,A_{\lambda}\exp\left[-\tfrac{1}{2}\frac{\tau^{2}}{ \sigma_{\tau}^{2}}-\frac{q^{2}LN_{\rm b}\,\lambda(\tau)}{\eta_{\rm c}\beta^{2 }E_{0}T_{0}\sigma_{\delta}^{2}}\right] \tag{22.139}\]

which must be solved numerically to get the particle distribution \(\lambda(\tau)\). We note that the inductive wake does not change the symmetry of the particle distribution in \(\tau\). For large values of \(\tau\), the particle distribution must approach zero to meet the normalization requirement (\(\lim_{\tau\to\infty}\lambda(\tau)=0\)) and the particle distribution is always Gaussian for large amplitudes. The effect of the inductive wake field is mainly concentrated to the core of the particle bunch.

Evaluating numerically (22.139), we distinguish between an electron beam and a proton or ion beam. The momentum spread \(\sigma_{\delta}\) in case of an electron beam is determined by quantum effects related to the emission of synchrotron radiation and is thereby for this discussion a constant. Not so for proton and ion beams which are subject to Liouville's theorem demanding a strong correlation between bunch length and momentum spread such that the longitudinal phase space of the beam remains constant. Equation (22.139) has the form

\[f(t)=K\,\exp\left[-\tfrac{1}{2}t^{2}-f(t)\right] \tag{22.140}\]

or after differentiation with respect to \(t\)

\[\frac{{\rm d}f\left(t\right)}{{\rm d}t}=-\frac{tf(t)}{1+f\left(t\right)}. \tag{22.141}\]

For strong wake fields \(f\left(t\right)\gg 1\) and (22.141) can be integrated for

\[f\left(t\right)=f_{0}-\tfrac{1}{2}t^{2}. \tag{22.142}\]

The particle distribution in the bunch center assumes more and more the shape of a parabolic distribution as the wake fields increase. Figure 22.13 shows the particle distribution for different strengths of the wake field.

Now we add the resistive wake field component. This field actually extracts energy from the bunch and therefore one expects that the whole bunch is shiftedsuch as to compensate this extra loss by moving to a higher field in the accelerating cavities. Inserting the full wake field (22.138) into (22.137) results in the distribution

\[\lambda(\tau)=N_{\rm b}\,A_{\lambda}\exp\left[-\tfrac{\tau^{2}}{2}\frac{\tau^{2 }}{\sigma_{\tau}^{2}}-aLN_{\rm b}\lambda(\tau)-aRN_{\rm b}\int_{0}^{\tau} \lambda(\bar{\tau})\,{\rm d}\bar{\tau}\right]\,, \tag{22.143}\]

where

\[a=\frac{q^{2}}{\eta_{c}\beta E_{0}T_{0}\sigma_{\delta}^{2}}\,. \tag{22.144}\]

Looking for a shift of the tip of the particle distribution, we get from \({\rm d}\lambda/{\rm d}\tau=0\) the location of the distribution maximum

\[\tau_{\rm max}\propto N_{\rm b}\lambda(\tau_{\rm max})\,. \tag{22.145}\]

The maximum of the particle distribution is therefore shifted proportional to the bunch intensity and the general distortion is shown in Fig. 22.13b for a resistive wake much larger than generally encountered in an accelerator. The distortion of the particle distribution leads to a deviation from a Gaussian distribution and a variation of the bunch length. In the limit of a strong and inductive wake field, for example, the full-width half maximum value of the bunch length increases like

\[\tau_{\rm fwhm}=\sigma_{\tau}\sqrt{f_{0}}=\frac{q\sigma_{\tau}}{\beta\sigma_{ \delta}}\sqrt{\frac{\beta LN_{\rm b}\,\lambda(\tau)}{\eta_{c}E_{0}T_{0}}}\,. \tag{22.146}\]

The bunch length changes as the bunch intensity is increased while the sign and rate of change is dependent on the actual ring impedance spectrum on hand. We have

Figure 22.13: Potential-well distortion of Gaussian particle distributions (**a**) for an inductive wake field, and (**b**) for a combination of an inductive and a resistive wake field

used an induction as an example for the reactive impedance in a ring because it most appropriately represents the real impedance for lower frequencies or longer bunch length. In general, this potential-well bunch lengthening may be used to determine experimentally the nature and quantity of the ring impedance by measuring the bunch length as a function of bunch current.

### Turbulent Bunch Lengthening

At higher bunch currents the bunch lengthening deviates significantly from the scaling of potential well distortion and actually proceeds in the direction of true lengthening. Associated with this lengthening is also an increase in the particle momentum spread. The nature of this instability is similar to the microwave instability for coasting beams.

Considering long bunches, a strong instability with a rise time shorter than the synchrotron oscillation period and high frequencies with wavelength short compared to the bunch length, we expect basically the same dynamics as was discussed earlier for a coasting beam. This was recognized by Boussard [28] who suggested a modification of the Keil-Schnell criterion by replacing the coasting-beam particle density by the bunch density. For a Gaussian particle distribution, the peak bunch current is

\[\hat{I}=I_{0}\,\frac{2\pi\overline{R}}{\sqrt{2\pi}\sigma_{\ell}}\,, \tag{22.147}\]

where \(I_{0}\) is the average circulating beam current per bunch, and the bunch length is related to the energy spread by

\[\sigma_{\ell}\,=\,\frac{\beta c|\eta_{\rm c}|}{\Omega_{\rm s0}}\frac{\sigma_{ \epsilon}}{E_{0}}\,. \tag{22.148}\]

With these modifications, the Boussard criterion is

\[\left|\frac{Z_{z}}{n}\right|\leq F\,\frac{\beta^{3}E_{0}|\eta_{\rm c}|^{2}}{qI _{0}\sqrt{2\pi}\,v_{\rm s0}}\left(\frac{\sigma_{\epsilon}}{E_{0}}\right)^{3}\,, \tag{22.149}\]

where the form factor \(F\) is still of the order unity.

As a consequence of this turbulent bunch lengthening we observe an increase of the energy spread as well as an increase of the bunch length. The instability does not necessarily lead to a beam loss but rather to an adjustment of energy spread and bunch length such that the Boussard criterion is met. For very low beam currents the stability criterion is always met up to a threshold where the r.h.s. of (22.149) becomes smaller than the l.h.s. Upon further increase of the beam current beyond the threshold current the energy spread and consequently the bunch length increases to avoid the bunched beam microwave instability.

### Transverse Single-Bunch Instabilities

Transverse wake fields can also greatly modify the stability of a single bunch. Specifically at high frequencies, we note an effect of transverse wake fields generated by the head of a particle bunch on particles in the tail of the same bunch. Such interaction occurs for broad-band impedances where the bunch generates a short wake including a broad spectrum of frequencies. In the first moment all these fields add up being able to act back coherently on particles in the tail but they quickly decoher and vanish before the next bunch arrives. This effect is therefore a true single-bunch effect. In order to affect other bunches passing by later, the fields would have to persist a longer time which implies a higher Q value of the impedance structure which we ignore here.

#### Beam Break-Up in Linear Accelerators

A simple example of a transverse microwave instability is the phenomenon of beam break-up in linear accelerators. We noted repeatedly that the impedance of vacuum chambers originates mainly from sudden changes in cross section which therefore must be avoided to minimize impedance and microwave instabilities. This, however, is not possible in accelerating cavities of which there are particularly many in a linear accelerator. Whatever single-pass microwave instabilities exist they should become apparent in a linear accelerator. We have already discussed the effect of longitudinal wake fields whereby the fields from the head of a bunch act back as a decelerating field on particles in the tail. In the absence of corrective measures we therefore expect the particles in the tail to gain less energy than particles in the head of an intense bunch.

Transverse motion of particles is confined to the vicinity of the linac axis by quadrupole focusing in the form of betatron oscillations while travelling along the linear accelerator. However, coherent transverse betatron oscillations can create strong transverse wake fields at high bunch intensities. Such fields may act back on subsequent bunches causing bunch to bunch instabilities if the fields persist long enough. Here we are more interested in the effect on the same bunch. For example, the wake fields of the head of a bunch can act back on particles in the tail of the bunch. This interaction is effected by broad-band impedances like sudden discontinuities in the vacuum chamber which are abundant in a linear accelerator structure. The interaction between particles in the head of a bunch on particles in the tail of the same bunch can be described by a two macro particle model resembling the head and the tail.

Transverse wake fields are proportional to the transverse oscillation amplitude of the head and we describe the dynamics of the head and tail of a bunch in a two particle model where each particle represents half the charge of the whole bunch as shown in Fig. 22.14.

The head particle with charge \(\frac{1}{2}qN_{\rm b}\) performs free betatron oscillations while the tail particle responds like a driven oscillator. Since all particles travel practically at the speed of light, the longitudinal distribution of particles remains fixed along the whole length of the linear accelerator. The equations of motion in smooth approximation where \(k_{\beta}=1/(v_{0}\,\overline{\beta}_{u})\) and \(\overline{\beta}_{u}\) is the average value of the betatron function in the plane \(u\), are for both macroparticles

\[\begin{array}{l}x_{\rm h}^{\prime\prime}+k_{\beta}^{2}\,x_{\rm h}=0\,,\\ x_{\rm t}^{\prime\prime}+k_{\beta}^{2}\,x_{\rm t}=r_{\rm c}\frac{x_{\rm h}}{ \gamma}\,\int_{z}^{\infty}\lambda(z)\widetilde{W}_{\perp}(z-\widetilde{z})\, \mathrm{d}z=\frac{r_{\rm c}N_{\rm b}\widetilde{W}_{\perp}}{2\gamma}x_{\rm h}\,, \end{array} \tag{22.150}\]

where we use the indices h and t for the head and tail particles respectively and introduce the average wake field per unit length

\[\widetilde{W}_{\perp}=\frac{W_{\perp}}{L_{\rm acc}}\,. \tag{22.151}\]

For simplicity, it was assumed in (22.150) that the beam is just coasting along the linear accelerator to demonstrate the dynamics of the instability. If the beam is accelerated the adiabatic damping effect through the increase of the energy must be included.

Because of causality only the tail particle is under the influence of a wake field. The transverse wake field \(W_{\perp}(2\sigma_{z})\), for example, which is shown in Fig. 22.8, is to be taken at a distance \(2\sigma_{z}\) behind the head particle. Inserting the solution \(x_{\rm h}(z)=\hat{x}_{\rm h}\,\cos k_{\beta}z\) into the second equation, we obtain the solution for the betatron oscillation of the tail particle in the form

\[x_{\rm t}(z)=\hat{x}_{\rm h}\cos k_{\beta}z+\hat{x}_{\rm h}\frac{r_{\rm c}N_{ \rm b}\widetilde{W}_{\perp}}{4\gamma k_{\beta}}z\sin k_{\beta}z\,. \tag{22.152}\]

The second term in this expression increases without bound leading to particle loss or beam break-up as soon as the amplitude reaches the edge of the aperture. If the bunch does reach the end of the linear accelerator of length \(L_{\rm acc}\), the betatron oscillation amplitude of the tail has grown by a factor

\[F_{\rm bb}=\frac{\hat{x}_{\rm t}}{\hat{x}_{\rm h}}=\frac{r_{\rm c}N_{\rm b} \widetilde{W}_{\perp}L_{\rm acc}}{4\gamma k_{\beta}}\,. \tag{22.153}\]

Figure 22.14: Head-tail dynamics of a particle bunch represented by two macroparticles

One consequence of this instability is an apparent increase in beam emittance long before beam loss occurs. A straight bunch with a small cross section becomes bent first like a banana and later like a snake and the transverse distribution of all particles in the bunch occupies a larger cross-sectional area than before. This increase in apparent beam size has a big detrimental effect on the attainable luminosity in linear colliders and therefore must be minimized as much as possible. The two particle model adopted here is insufficient to determine a more detailed structure than that of a banana. However, setting up equations similar to (22.150) for more than two macroparticles will start to reveal the oscillatory nature of the transverse bunch perturbation.

One scheme to greatly reduce the beam break-up effect is called BNS damping in reference to its inventors Balakin et al. [29] and has been successfully implemented into the Stanford Linear Collider [30]. The technique utilizes the fact that the betatron oscillation frequency depends by virtue of the chromaticity on the energy of the particles. By accelerating the bunch behind the crest of the accelerating field the tail gains less energy than the head. Therefore the tail is focused more by the quadrupoles than the head. Since the transverse wake field introduces defocusing this additional chromatic focusing can be used for compensation.

Of course this method of damping the beam break-up by accelerating ahead of the crest is counter productive to compensating for the energy loss of tail particles due to longitudinal wake fields. In practice, BNS damping is applied only at lower energies where the instability is strongest and in that regime the energy reducing effect of the longitudinal wake field actually helps to maximize BNS damping. Toward the end of the linear accelerator at high beam energies, the beam break up effect becomes small (\(\propto 1/\gamma\)) and the bunch is now moved ahead of the crest to reduce the energy spread in the beam.

##### Fast Head-Tail Effect

Transverse bunch perturbations due to broad-band impedances are not restricted to linear accelerators but occur also in circular accelerators. In a circular proton accelerator, for example, the "length" is for all practical purposes infinite, there is no radiation damping and therefore even weak transverse wake fields can in principle lead to transverse bunch blow up and beam loss. This instability is known as the fast head-tail instability or strong head-tail instability and has been first discussed and analyzed by Kohaupt [31]. The dynamics in a circular accelerator is, however, different from that in a linear accelerator because particles in the head of a bunch will not stay there but rather oscillate between head and tail in the course of synchrotron oscillations. These synchrotron oscillations disturb the coherence between head and tail and the instability becomes much weaker.

On the other hand, particles in circular accelerators and especially in storage rings are expected to circulate for a long time and even a much reduced growth rate of the transverse bunch blow up may still be too strong. The dynamics of interaction is similar to that in a linear accelerator at least during about half a synchrotron oscillation period \(\left(\frac{1}{2}t_{\rm s}\right)\), but during the next half period the roles are interchanged for individual particles. Particles travelling for one half period in the head of the bunch find themselves close to the tail for the next half period only to reach the head again and so forth. To understand the dynamics over many oscillations, we set up equations of motion for two macroparticles resembling the head and tail of a particle bunch similar to (22.150), but we now use the time as the independent variable. The distance \(\xi\) between head and tail particle varies between 0 and the maximum distance of the two macro particles \(2\ell\) during the course of a synchrotron oscillation period and since the transverse wake field increases linearly with \(\xi\), we set \(W_{\perp}(\xi)=W_{\perp}\left(2\sigma_{\ell}\right)\sin\Omega_{\rm s}t\). With this the equations of motion are for \(0\leq t\leq t_{\rm s}/2\)

\[\begin{array}{l}\ddot{x}_{1}+\omega_{\beta}^{2}\,x_{1}=0\,,\\ \ddot{x}_{2}+\omega_{\beta}^{2}\,x_{2}=\frac{r_{\rm s}\,\beta^{2}c^{2}N_{\rm b }\,\bar{W}_{\perp}\left(2\sigma_{\ell}\right)\,\sin\Omega_{\rm s}t}{2\,\gamma }\,x_{1}\,,\end{array} \tag{22.154}\]

where \(\bar{W}_{\perp}=W_{\perp}/(2\pi\bar{R})\) is the wake function per unit length. For the next half period \(t_{\rm s}/2\leq t\leq t_{\rm s}\)

\[\begin{array}{l}\ddot{x}_{1}+\omega_{\beta}^{2}\,x_{1}=\frac{r_{\rm c}\beta^ {2}c^{2}N_{\rm b}\bar{W}_{\perp}\left(2\ell\right)\sin\Omega_{\rm s}t}{2\gamma }x_{2}\,,\\ \ddot{x}_{2}+\omega_{\beta}^{2}\,x_{2}=0\,.\end{array} \tag{22.155}\]

For further discussions we consider solutions to (22.154), (22.155) in the form of phasors defined by

\[\boldsymbol{x}(t)=\boldsymbol{x}(0)\,{\rm e}^{{\rm i}\omega_{\beta}t}=x-{\rm i }\frac{\dot{x}}{\omega_{\beta}}\,. \tag{22.156}\]

The first Eq. (22.154) can be solved immediately for

\[\boldsymbol{x}_{1}(t)=\boldsymbol{x}_{1}(0)\,{\rm e}^{{\rm i}\omega_{\beta}t} \tag{22.157}\]

and the second Eq. (22.154) becomes with (22.157)

\[\ddot{\boldsymbol{x}}_{2}+\omega_{\beta}^{2}\boldsymbol{x}_{2}=A\sin\Omega_{ \rm s}t\,{\rm e}^{{\rm i}\omega_{\beta}t}\boldsymbol{x}_{1}(0)\,, \tag{22.158}\]

where

\[A=\frac{r_{\rm c}\beta^{2}c^{2}N_{\rm b}\bar{W}_{\perp}(2\ell)}{2\gamma}\,. \tag{22.159}\]The synchrotron oscillation frequency is generally much smaller than the betatron oscillation frequency \(\left(\Omega_{\rm s}\ll\omega_{\beta}\right)\) and the solution of (22.159) becomes with this approximation

\[\boldsymbol{x}_{2}(t)=\boldsymbol{x}_{2}(0)\,{\rm e}^{{\rm i}\omega_{\beta}t}+ \,\frac{1}{\omega_{\beta}}\int_{0}^{t}[A\,\boldsymbol{x}_{1}(0)\sin\Omega_{\rm s }t^{\prime}\,{\rm e}^{{\rm i}\omega_{\beta}t^{\prime}}]\sin\omega_{\beta}(t-t ^{\prime})\,{\rm d}t^{\prime}\]

or after some manipulation

\[\boldsymbol{x}_{2}(t)=\boldsymbol{x}_{2}(0)\,{\rm e}^{{\rm i}\omega_{\beta}t}-{ \rm i}\,\boldsymbol{x}_{1}(0)\tfrac{1}{2}a(1-\cos\Omega_{\rm s}t)\,{\rm e}^{{ \rm i}\omega_{\beta}t}\,, \tag{22.160}\]

where \(a=A/(\omega_{\beta}\,\Omega_{\rm s})\). During the second half synchrotron oscillation period, the roles of both macroparticles are exchanged. We may formulate the transformation through half a synchrotron oscillation period in matrix form and get with \(1-\cos\left(\Omega_{\rm s}\tfrac{1}{2}t_{\rm s}\right)=2\) since \(\Omega_{\rm s}\tfrac{1}{2}t_{\rm s}\approx\pi\) for the first half period

\[\begin{pmatrix}\boldsymbol{x}_{1}(t_{\rm s}/2)\\ \boldsymbol{x}_{2}(t_{\rm s}/2)\end{pmatrix}={\rm e}^{{\rm i}\omega_{\beta}t_{ \rm s}/2}\begin{pmatrix}1&0\\ -{\rm i}a&1\end{pmatrix}\begin{pmatrix}\boldsymbol{x}_{1}(0)\\ \boldsymbol{x}_{2}(0)\end{pmatrix} \tag{22.161}\]

and for the second half period

\[\begin{pmatrix}\boldsymbol{x}_{1}(t_{\rm s})\\ \boldsymbol{x}_{2}(t_{\rm s})\end{pmatrix}={\rm e}^{{\rm i}\omega_{\beta}t_{ \rm s}/2}\begin{pmatrix}1&-{\rm i}a\\ 0&1\end{pmatrix}\begin{pmatrix}\boldsymbol{x}_{1}(t_{\rm s}/2)\\ \boldsymbol{x}_{2}(t_{\rm s}/2)\end{pmatrix} \tag{22.162}\]

Combining both half periods one gets finally for a full synchrotron oscillation period

\[\begin{pmatrix}\boldsymbol{x}_{1}(t_{\rm s})\\ \boldsymbol{x}_{2}(t_{\rm s})\end{pmatrix}={\rm e}^{{\rm i}\omega_{\beta}t_{ \rm s}}\begin{pmatrix}1-a^{2}&-{\rm i}a\\ -{\rm i}a&1\end{pmatrix}\begin{pmatrix}\boldsymbol{x}_{1}(0)\\ \boldsymbol{x}_{2}(0)\end{pmatrix}\,. \tag{22.163}\]

The stability of the motion after many periods can be extracted from (22.163) by solving the eigenvalue equation

\[\begin{pmatrix}1-a^{2}&-{\rm i}a\\ -{\rm i}a&1\end{pmatrix}\begin{pmatrix}\boldsymbol{x}_{1}\\ \boldsymbol{x}_{2}\end{pmatrix}=\lambda\begin{pmatrix}\boldsymbol{x}_{1}\\ \boldsymbol{x}_{2}\end{pmatrix}\,. \tag{22.164}\]

The characteristic equation

\[\lambda^{2}-(2-a^{2})\,\lambda+1=0 \tag{22.165}\]

has the solution

\[\lambda_{1,2}=(1-\tfrac{1}{2}a^{2})\pm\sqrt{(1-\tfrac{1}{2}a^{2})^{2}-1} \tag{22.166}\]and the eigenvalues can be expressed by

\[\lambda={\rm e}^{\pm{\rm i}\phi}, \tag{22.167}\]

where \((1-\frac{1}{2}a^{2})=\cos\Phi\) for \(|a|\,\leq 2\) or

\[|a|=\frac{r_{\rm c}\beta^{2}c^{2}N_{\rm b}\tilde{W}_{\perp}(2\ell)}{2\gamma \omega_{\beta}\Omega_{\rm s}}\leq 2\,. \tag{22.168}\]

The motion remains stable since no element of the transformation matrix increases unbounded as the number of periods increases to infinity. In the form of a stability criterion, the single-bunch current \(I_{\rm b}=qN_{\rm b}f_{\rm rev}\) must not exceed the limit

\[I_{\rm b}\leq\frac{4\,q\gamma\omega_{0}^{2}v_{\beta}v_{\rm s}}{r_{\rm c}\, \beta cW_{\perp}(2\ell)}\,, \tag{22.169}\]

where \(q\) is the charge of the particles and (\(v_{\beta}\), \(v_{\rm s}\)) the betatron and synchrotron tune, respectively. In a storage ring, it is more convenient to use impedance rather than wake fields. Had we set up the equations of motion (22.150), (22.151) expressing the perturbing force in terms of impedance we would get the same results but replacing the wake field by

\[W_{\perp}(2\ell)=\frac{\omega_{0}}{\pi}{\rm Im}\{Z_{\perp}\} \tag{22.170}\]

and the threshold beam current for the fast head-tail instability becomes

\[I_{\rm b}\leq\frac{4\pi q\gamma\omega_{0}v_{\beta}v_{\rm s}}{r_{\rm c}\beta c{ \rm Im}\{\frac{Z_{\perp}}{n}\}}\,. \tag{22.171}\]

The bunch current \(I_{\rm b}\) is a threshold current which prevents us from filling more current into a single bunch. Exceeding this limit during the process of filling a bunch in a circular accelerator leads to an almost immediate loss of the excess current. This microwave instability is presently the most severe limitation on single-bunch currents in storage rings and special care must be employed during the design to minimize as much as possible the transverse impedance of the vacuum chamber system.

The strength of the instability becomes more evident when we calculate the growth time for a beam current just by an increment \(\epsilon\) above the threshold. For \(|a|>2\) we have \((1-\frac{1}{2}a^{2})=-\cosh\mu\) and the eigenvalue is \(\lambda=e^{\pm\mu}\). The phase \(\mu=0\) at threshold and \(\cosh\mu\approx 1+\frac{1}{2}\mu^{2}\) for \(a=2+\epsilon\) and we get

\[\mu=2\sqrt{\epsilon}\,. \tag{22.172}\]In each synchrotron oscillation period the eigenvalues increase by the factor \(e^{\mu}\) or at a growth rate of \(\frac{1}{\tau}=\frac{\mu}{t_{\rm s}}=\frac{2\sqrt{\epsilon}}{t_{\rm s}}\). If, for example, the beam current exceeds the threshold by \(10\,\%\), we have \(\epsilon=0.2\) and the rise time would be \(\tau/t_{\rm s}=0.89\) or the oscillation amplitudes increase by more than a factor of two during a single synchrotron oscillation period. This is technically very difficult to counteract by a feedback system.

We have assumed that transverse wake fields are evenly distributed around the accelerator circumference. In a well designed accelerator vacuum chamber, however, most of the transverse wake field occur in the accelerating cavities and therefore only the transverse betatron oscillation amplitude in the cavities are relevant. In this case, one recalls the relation \(v_{\beta}\approx\bar{R}/\beta_{u}\) and we replace in (22.171) the average value of the betatron function by the value in the cavities for

\[I_{\rm b}\leq\frac{4q\nu\Omega_{\rm s}}{r_{\rm c}\beta_{u,cy}W_{\perp,cy}(2 \ell)}. \tag{22.173}\]

This result suggest that the betatron function in the plane \(u=x\) or \(u=y\) at the location of cavities should be kept small and the synchrotron oscillation frequency should be large. The exchange of head and tail during synchrotron oscillation slows down considerably the growth rate of the instability. The result (22.173) is the same as the amplification factor (22.153) if we consider that in a linear accelerator the synchrotron oscillation period is infinite.

As we approach the threshold current, the beam signals the appearance of the head-tail instability on a spectrum analyzer with a satellite below the betatron frequency. The threshold for instability is reached when the satellite frequency reaches a value \(\omega_{\rm sat}=\omega_{\beta}-\frac{1}{2}\,\Omega_{\rm s}\). This becomes apparent when replacing the transformation matrix in (22.163) by the eigenvalue

\[\begin{pmatrix}\mathbf{x}_{1}(t_{\rm s})\\ \mathbf{x}_{2}(t_{\rm s})\end{pmatrix}=\mathrm{e}^{\mathrm{i}\alpha_{\beta}t_{\rm s }}\,\mathrm{e}^{\mathrm{i}\phi}\,\,\begin{pmatrix}\mathbf{x}_{1}(0)\\ \mathbf{x}_{2}(0)\end{pmatrix}. \tag{22.174}\]

The phase reaches a value of \(\varPhi=\pi\) at the stability limit and (22.174) becomes at this limit

\[\begin{pmatrix}\mathbf{x}_{1}(t_{\rm s})\\ \mathbf{x}_{2}(t_{\rm s})\end{pmatrix}=\mathrm{e}^{\mathrm{i}\left(\alpha_{\beta} -\frac{1}{2}\,\Omega_{\rm s}\right)t_{\rm s}}\begin{pmatrix}\mathbf{x}_{1}(0)\\ \mathbf{x}_{2}(0)\end{pmatrix}. \tag{22.175}\]

At this point, it should be noted, however, that the shift of the betatron frequency to \(\frac{1}{2}\Omega_{\rm s}\) is a feature of the two macro particle model. In reality there is a distribution of particles along the bunch and while increasing the beam current the betatron frequency decreases and the satellite \(v_{\beta}-v_{\rm s}\) moves until both frequencies merge and become imaginary. This is the point of onset for the instability. It is this feature of merging frequencies which is sometimes called mode mixing or mode coupling.

#### Measurement of the Broad-Band Impedance

As shown by Chao [9] the combined motion of head and tail represents a coherent transverse betatron motion which can be picked up by beam position monitors. The betatron frequency changes with increasing beam intensity and the initial slope can be used to determine the broad band impedance. From the lowest order instability mode we take the derivative of the betatron oscillation frequency and get

\[\frac{\mathrm{d}\omega_{\beta}}{\mathrm{d}N_{\mathrm{b}}}\,=-\frac{\omega_{ \mathrm{s}}}{2\pi}\left.\left(\frac{\mathrm{d}\Phi}{\mathrm{d}N_{\mathrm{b}}} \right)\right|_{N_{\mathrm{b}}=0}\,=-\frac{r_{\mathrm{e}}c^{2}W_{0}}{16\pi\, \gamma\,\bar{R}\omega_{\beta}} \tag{22.176}\]

where \(N_{\mathrm{b}}\) is the number of electrons per bunch. Measuring the slope \(\mathrm{d}\omega_{\beta}/\mathrm{d}N_{\mathrm{b}}\) at low beam intensities allows to determine the wake function \(W_{0}\). From this wake the transverse impedance is [9]

\[Z_{\perp}\,=\,\frac{\bar{R}}{\beta_{z}v_{\beta}}\frac{b}{c}\,W_{0} \tag{22.177}\]

and from (22.8) the relation to the longitudinal impedance is \(Z_{\perp}\,=\,\frac{2\bar{R}}{b^{2}}\,\frac{Z_{\parallel}}{n}\), where \(\bar{R}\) is the average radius of the accelerator, \(b\) the typical vacuum chamber radius, \(v_{\beta}\) the betatron frequency and \(\beta_{z}\) the average value of the betatron function around the ring. the mode number \(n\) is the frequency in units of the revolution frequency \(n=\omega/\omega_{0}\).

##### Head-Tail Instability

Discussing the fast head-tail instability we considered the effect of transverse wake fields generated by the head of a particle bunch on the transverse betatron motion of the tail. We assumed a constant betatron oscillation frequency which is only an approximation since the betatron frequency depends on the particle energy. On the other hand, there is a distinct relationship between particle energy and particle motion within the bunch, and it is therefore likely that the dynamics of the head-tail instability becomes modified by considering the energy dependence of the betatron oscillation frequency.

Like in the previous section, we represent the particle bunch by two macroparticles which perform synchrotron oscillations being \(180^{\circ}\) apart in phase. The wake fields of the head particle act on the tail particle while the reverse is not true due to causality. However, during each half synchrotron oscillation period the roles become reversed.

In (22.160), we obtained an expression which includes the perturbation term and consider the variation of this term due to chromatic oscillations of the betatronfrequency. The perturbation term is proportional to \(\mathrm{e}^{\mathrm{i}\omega_{\beta}t}\) and we set therefore with \(\delta=\Delta p/p_{0}\)

\[\omega_{\beta}=\omega_{\beta}(\delta)=\omega_{\beta 0}+\,\frac{\partial\omega_{ \beta}}{\partial\delta}\delta+\mathcal{O}(\delta^{2})\,. \tag{22.178}\]

The chromaticity is defined by the betatron tune shift per unit relative momentum deviation

\[\xi_{\beta}=\frac{\Delta v_{\beta}}{\delta} \tag{22.179}\]

and (22.178) becomes with \(\omega_{\beta}=v_{\beta}\,\omega_{0}\)

\[\omega_{\beta}=\omega_{\beta 0}+\xi_{\beta}\delta\omega_{0}\,. \tag{22.180}\]

The momentum deviation is oscillating at the synchrotron frequency and is correlated with the longitudinal motion by

\[\delta=-\,\frac{\Omega_{s}\ell}{\beta c|\eta_{c}|}\sin\Omega_{s}t\,, \tag{22.181}\]

where \(2\ell\) is the maximum longitudinal distance between the two macroparticles. Combining (22.180), (22.181) we get

\[\omega_{\beta}=\omega_{\beta 0}-\frac{\Omega_{s}\,\ell\,\xi_{\beta}}{v_{ \beta}\bar{R}|\eta_{c}|}\sin\Omega_{s}t\,, \tag{22.182}\]

where the second term is much smaller than unity so that we may expand the exponential function of this term to get

\[\mathrm{e}^{\mathrm{i}\omega_{\beta}t}\approx\mathrm{e}^{\mathrm{i}\omega_{ \beta 0}t}\left[1-\mathrm{i}\,\frac{\Omega_{s}\ell\xi_{\beta}}{v_{\beta}\bar{R}| \eta_{c}|}t\sin(\Omega_{s}t)\right]\,. \tag{22.183}\]

The expression in the square bracket is the variation of the scaling factor \(a\) in (22.160) and we note specifically, the appearance of the imaginary term which gives rise to an instability. The phase \(\Phi\) in the eigenvalue equation (22.167) becomes for small beam currents \(\Phi\approx a\) and with (22.183)

\[\Phi=a\left[1-\mathrm{i}\,\frac{\Omega_{s}\ell\xi_{\beta}}{\pi v_{\beta}\bar{ R}|\eta_{c}|}t_{s}\,\right]\,, \tag{22.184}\]

where we have set \(t=\frac{1}{2}t_{s}\) and \(\langle\sin\Omega_{s}t\rangle\approx 2/\pi\). The first term represents the fast head tail instability with its threshold characteristics discussed in the previous section. The second term is an outright damping or antidamping effect being effective at any beam current. This instability is called the head-tail effect discovered and analyzed by Pellegrini [32] and Sands [33] at the storage ring ADONE.

Considering only the imaginary term in (22.184), we note an exponential growth of the head-tail instability with a growth rate of

\[\frac{1}{\tau}=\frac{\Omega_{\rm s}a\ell\dot{\xi}_{\beta}}{\pi\,v_{\beta}\bar{ R}|\eta_{\rm c}|}=\frac{\ell\dot{\xi}_{\beta}r_{\rm c}\beta cN_{\rm b}\bar{W}_{ \perp}(2\ell)}{2\pi\,\gamma|\eta_{\rm c}|v_{\beta}^{2}}. \tag{22.185}\]

Instability may occur either in the vertical or the horizontal plane depending on the magnitude of the transverse wake function in both planes. There are two modes, one stable and one unstable depending on the sign of the chromaticity and momentum compaction. Above transition \(\eta_{\rm c}<0\) and the beam is unstable for negative chromaticity. This instability is the main reason for the insertion of sextupole magnets into circular accelerators to compensate for the naturally negative chromaticity. Below transition, the situation is reversed and no correction of chromaticity by sextupoles is required. From (22.185), we would conclude that we need to correct the chromaticity exactly to zero to avoid instability by one or the other mode. In reality, this is not the case because a two particle model overestimates the strength of the negative mode. Following a more detailed discussion including Vlasov's equation [9] it becomes apparent that the negative mode is much weaker to the point where, at least in electron accelerators, it can be ignored in the presence of radiation damping.

Observation of the head-tail damping for positive chromaticities or measuring the risetime as a function of chromaticity can be used to determine the transverse wake function or impedance of the accelerator [34; 35]. Measurements of head-tail damping rates have been performed in SPEAR [34] as a function of chromaticity and are reproduced in Figs. 22.15 and 22.16.

Figure 22.15: Measurement of the head-tail damping rate in SPEAR as a function of chromaticity (**a**) and beam energy (**b**) [34]

We clearly note the linear increase of the damping rate with chromaticity. The scaling with energy and beam current is less linear due to a simultaneous change in bunch length. Specifically the bunch length increases with beam intensity causing the wake fields to drop for a smaller damping rate.

### Multi-Bunch Instabilities

Single-bunch dynamics is susceptible to all kinds of impedances or wake fields whether it be narrow or broad-band impedances. This is different for multi-bunch instabilities or coupled-bunch instabilities [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]. In order for wake fields to generate an effect on more than one bunch it must persist at least until the next bunch comes by the location of the impedance. We expect therefore multi-bunch instabilities only due to high \(Q\) or narrow-band impedances like those encountered in accelerating cavities. Higher-order modes in such cavities persist some time after excitation and actually reach a finite amplitude in a circular accelerator where the orbiting beam may periodically excite one or the other mode. Because these modes have a high quality factor they are also confined to a narrow frequency spread. The impedance spectrum we need to be concerned with in the study of multi-bunch instabilities is therefore a line spectrum composed of many cavity modes.

To study the effect of these modes on the circulating beam, we must fold the beam current spectrum with the mode spectrum and derive from this interaction conditions for beam stability. We will do this for the case of the two lowest order mode oscillations only where all bunches oscillate in synchronism at the same phase or are 90\({}^{\circ}\) out of phase from bunch to bunch respectively. Of course in a real accelerator higher-order modes can be present too and must be taken into account. Here we must limit ourself, however, to the discussion of the physical effect only and

Figure 22: Measurement of the head-tail damping rate in SPEAR as a function of beam current [34]

direct the interested reader to more detailed discussions on this subject in references [3; 9; 19; 26].

We consider the dynamics of rigid coupled bunches ignoring the internal motion of particles within a single bunch. The beam spectrum is then from (21.84) with \(q\) the bunch charge and observing at \(\varphi=0\) for simplicity

\[I_{\parallel}(\omega,\varphi)=\frac{qn_{\rm b}\omega_{0}}{2\pi}\sum_{p=-\infty }^{+\infty}\sum_{n=-\infty}^{+\infty}{\rm i}^{-n}J_{n}(pn_{\rm b}\omega_{0} \hat{\tau})\,\delta(\omega-\Omega_{n})\,, \tag{22.186}\]

where now \(\Omega_{n}=(pn_{\rm b}+nm+nv_{\rm s})\,\omega_{0}\) and where we have replaced the synchrotron frequency by the synchrotron tune and the phase \(\zeta_{i}\) for individual particles by the mode of the bunch distribution setting \(\zeta_{i}=m\omega_{0}t\) with \(0\leq m\leq n_{\rm b}\). A beam of \(n_{\rm b}\) equidistant bunches can oscillate in \(n_{\rm b}\) different modes. Two bunches, for example, can oscillate in phase or \(180^{\circ}\) out of phase; four bunches can oscillate with a phase difference of \(0^{\circ}\), \(90^{\circ}\), \(180^{\circ}\), and \(270^{\circ}\) between consecutive bunches. In general the order of the mode \(m\) defines the phase difference of consecutive bunches by

\[\Delta\phi=m\frac{360^{\circ}}{n_{\rm b}}\,. \tag{22.187}\]

To determine the multi-bunch dynamics we calculate first the induced voltage \(V(t)\) by the beam current in the impedance \(Z(\omega)\) and then fold the voltage function with the beam function to calculate the energy loss per turn by each particle. Knowing this, we will be able to formulate the equation of motion for the synchrotron oscillation. Specifically, we will be able to formulate frequency shifts and damping or antidamping due to the interaction of the multi-bunch beam with its environment to identify conditions for beam stability.

For simplicity we assume small phase oscillations (\(\hat{\tau}\ll 1\)) and consider only the fundamental beam frequency and the first satellite \(n=0,1\). With this (22.186) becomes

\[I_{\parallel}(\omega)=\frac{qn_{\rm b}\omega_{0}}{2\pi}\sum_{p=-\infty}^{+ \infty}J_{0}(\,pn_{\rm b}\omega_{0}\hat{\tau})\,\delta(\omega-\Omega_{0})-{ \rm i}\,J_{1}(\,pn_{\rm b}\omega_{0}\hat{\tau})\,\delta(\omega-\Omega_{1})\,, \tag{22.188}\]

where \(\Omega_{0}=p\,n_{\rm b}\omega_{0}\), \(\Omega_{1}=(\,pn_{\rm b}+m+v_{\rm s})\,\omega_{0}\), and \(J_{i}\) are Bessel's functions. The induced voltage spectrum is \(V(\omega)=Z(\omega)\,I(\omega)\) and its Fourier transform \(V(t)=\int V(\omega)e^{i\omega t}\,{\rm d}\omega\) or

\[V_{\parallel}(t) =\frac{qn_{\rm b}\omega_{0}}{2\pi}\sum_{p=-\infty}^{+\infty} \left[J_{0}(\hat{\tau}\,\Omega_{0})\,Z(\Omega_{0})\,{\rm e}^{{\rm i}\Omega_{0}t}\right. \tag{22.189}\] \[\left.-{\rm i}\,J_{1}(\hat{\tau}\,\Omega_{0})\,Z(\Omega_{1})\,{ \rm e}^{{\rm i}\Omega_{1}t}\right]\,.\]The energy loss per particle is then defined by integrating in time the product of voltage function and single-bunch current function

\[U=\frac{1}{N_{\rm b}}\int V_{\parallel}(t)\,\frac{I_{\parallel}(t+\tau)}{n_{\rm b }}\,{\rm d}t\,, \tag{22.190}\]

\(N_{\rm b}\) is the number of particles per bunch and \(T_{\rm b}=T_{0}/n_{\rm b}\) the time between passage of consecutive bunches. The bunch current can be expanded for \(\tau\ll 1\)

\[I_{\parallel}(t+\tau)\approx I_{\parallel}(t)+\tau\,\frac{{\rm d}}{{\rm d}t}I_ {\parallel}(t)\,. \tag{22.191}\]

The Fourier transforms of both current and its derivative with respect to time are correlated by

\[\frac{{\rm d}}{{\rm d}t}I_{\parallel}(\omega)={\rm i}\omega\,I_{\parallel}( \omega) \tag{22.192}\]

and (22.191) becomes in frequency domain with (22.188)

\[I_{\parallel}(t+\tau)=\frac{qn_{\rm b}\omega_{0}}{(2\pi)^{2}}\int\sum_{p=- \infty}^{+\infty}(1+{\rm i}\omega\tau)\,(J_{0}\,\delta_{0}-{\rm i}J_{1}\, \delta_{1})\,{\rm e}^{{\rm i}\omega t}\,{\rm d}t\,, \tag{22.193}\]

where we have used some abbreviations which become obvious by comparison with (22.188). Inserting (22.193) and (22.189) into (22.190), we get

\[U=\frac{(q\omega_{0})^{2}n_{\rm b}}{(2\pi)^{2}\,N_{\rm b}}\int _{t}\int_{\omega}\sum_{p}\left(J_{0}\,Z_{0}{\rm e}^{{\rm i}\Omega_{0}t}-{\rm i }\,J_{1}Z_{1}{\rm e}^{{\rm i}\Omega_{1}t}\right) \tag{22.194}\] \[\times(1+{\rm i}\omega\tau)\sum_{r}(J_{0}\,\delta_{0r}-iJ_{1}\, \delta_{1r})\,{\rm e}^{{\rm i}\omega t}\,{\rm d}\omega\,{\rm d}t\,.\]

For abbreviation we have set \(\delta_{i}=\delta(\Omega_{i})\), \(Z_{i}=Z(\Omega_{i})\), \(J_{0}=J_{0}(\hat{\tau}\Omega_{0})\), and \(J_{1}=J_{1}(\hat{\tau}\Omega_{0})\). An additional index has been added to indicate whether the quantity is part of the summation over \(p\) or \(r\). Before we perform the time integration we reverse the first summation by replacing \(p\to-p\) and get terms like \(\int e^{-{\rm i}(2\Omega_{0}-\omega)t}\,{\rm d}t=2\pi\delta_{0}\) etc. and (22.194) becomes

\[U =\frac{(q\omega_{0})^{2}n_{\rm b}}{2\pi\,N_{\rm b}}\int_{\omega} \sum_{p}(J_{0}\,Z_{0}\delta_{0r}+{\rm i}\,J_{1}Z_{1}\delta_{1r}) \tag{22.195}\] \[\times(1+{\rm i}\omega\tau)\sum_{r}(J_{0}\,\delta_{0}-{\rm i}\,J _{1}\delta_{1})\,{\rm d}\omega\,.\]The integration over \(\omega\) will eliminate many components. Specifically, we note that all cross terms \(\delta_{0}\delta_{1}\) vanish after integration. We also note that the terms \(\delta_{0p}\delta_{0r}\) vanish unless \(r=p\). With this in mind we get from (22.195)

\[U=\,\frac{(q\omega_{0})^{2}n_{\rm b}}{2\pi\;N_{\rm b}}\,\sum_{p}(J_{0}^{2}\,Z_{ 0}+{\rm i}\Omega_{0}\tau\,J_{0}^{2}\,Z_{0}+J_{1}^{2}\,Z_{1}+{\rm i}\Omega_{1} \tau J_{1}^{2}\,Z_{1})\,. \tag{22.196}\]

Finally the summation over \(p\) leads to a number of cancellations considering that the resistive impedance is an even and the reactive impedance an odd function. With \(Z_{0}=Z_{\rm r0}+iZ_{\rm i0}\), \(Z_{\rm r0}(\omega)=Z_{\rm r0}(-\omega)\), and \(Z_{\rm i0}(\omega)=-Z_{\rm i0}(-\omega)\) (22.196) becomes

\[U =\,\frac{(q\,\omega_{0})^{2}n_{\rm b}}{2\pi\;N_{\rm b}}\,\sum_{p=- \infty}^{+\infty}\left[J_{0}^{2}(\hat{\tau}\;\Omega_{0})\,Z_{\rm r}(\Omega_{0 })+J_{1}^{2}(\hat{\tau}\;\Omega_{0})\,Z_{\rm r}(\Omega_{1})\right. \tag{22.197}\] \[\left.+{\rm i}\;\tau\;\Omega_{1}J_{1}^{2}(\hat{\tau}\;\Omega_{1}) \,Z_{\rm r}(\Omega_{1})-\tau\;\Omega_{1}J_{1}^{2}(\hat{\tau}\;\Omega_{1})\,Z_ {\rm i}(\Omega_{1})\right].\]

The first and second term are the resistive energy losses of the circulating beam and synchrotron oscillations respectively while the third and fourth term are responsible for the stability of the multi bunch beam.

The equation of motion for synchrotron oscillations has been derived in Chap. 9 and we found that frequency and damping is determined by the accelerating rf-field and energy losses. We expect therefore that the energy loss derived for coupled bunch oscillations will also lead to a frequency shift and damping or anti damping. Specifically, we have for the equation of motion from (9.25)

\[\ddot{\varphi}+\left.\omega_{0}^{2}\frac{h\,\eta_{\rm c}}{2\pi\beta c\,cp_{0 }}\;e\,\frac{{\rm d}V}{{\rm d}\psi}\right|_{\psi_{\rm s}}\varphi-\frac{1}{T_{ 0}}\;\left.\frac{{\rm d}U}{{\rm d}E}\right|_{E_{0}}\dot{\varphi}=0\,, \tag{22.198}\]

where we notice the phase proportional term which determines the unperturbed synchrotron frequency

\[\Omega_{\rm s0}^{2}=\left.\omega_{0}^{2}\frac{h\,\eta_{\rm c}}{2\pi\beta cp_{ 0}}e\,\left.\frac{{\rm d}V}{{\rm d}\psi}\right|_{\psi_{\rm s}}=\omega_{0}^{2} \,\frac{h\,\eta_{\rm c}\,e\hat{V}_{0}\,\cos\psi_{\rm s}}{2\pi\;\beta\;cp_{0}}\,. \tag{22.199}\]

The term proportional to \(\dot{\varphi}\) gave rise to the damping decrement

\[\alpha_{\rm s0}=-\frac{1}{2T_{0}}\;\left.\frac{{\rm d}U}{{\rm d}E}\right|_{E_ {0}}\,. \tag{22.200}\]

The modification of the synchrotron frequency is with \(=\tau=\varphi/h\omega_{0}\) from (22.93)-(22.95) similar to the derivation of the unperturbed frequency

\[\Omega_{\rm s}^{2}=\Omega_{\rm s0}^{2}+\omega_{0}^{2}\frac{h\eta_{\rm c}n_{ \rm b}}{\beta cp_{0}N_{\rm b}}\sum_{p=-\infty}^{+\infty}\tau\,\Omega_{1}[qf_{0 }\,J_{1}(\hat{\tau}\,\Omega_{1})]^{2}Z_{\rm i}(\Omega_{1})\,, \tag{22.201}\]where \(f_{0}=\omega_{0}/2\pi\) is the revolution frequency. Note that \(\eta_{\rm c}<0\) above transition and the additional damping or energy loss due to narrow-band impedances reduces the frequency as one would expect.

Similarly we derive the modification of the damping decrement from the imaginary term in (22.197) noting that the solution of the synchrotron oscillation gives \(\hat{t}=-{\rm i}\Omega_{\rm s}\tau\) with \(\varphi=h\omega_{0}\tau\) and the damping decrement for a multi-bunch beam is

\[\alpha_{\rm s}=\alpha_{\rm s0}-\frac{\omega_{0}\eta_{\rm c}n_{\rm b}}{2cp_{0} \,N_{\rm b}}\sum_{p=-\infty}^{+\infty}\frac{\Omega_{1}}{\Omega_{\rm s}}[qf_{0} \,J_{1}(\hat{\tau}\,\Omega_{1})]^{2}Z_{\tau}(\Omega_{1})\,. \tag{22.202}\]

For proton and ion beams we would set \(\alpha_{\rm s0}=0\) because there is no radiation damping and the interaction of a multi-bunch beam with narrow-band impedances would provide damping or antidamping depending on the sign of the damping decrement for each term. If, however, only one term is antidamped the beam would be unstable and get lost as was observed first at the storage ring DORIS [38]. It is therefore important to avoid the overlap of any line of the beam spectrum with a narrow-band impedance in the ring.

Since this is very difficult to achieve and to control, it is more convenient to minimize higher-order narrow-band impedances in the ring by design as much as possible to increase the rise time of the instabilities. In electron storage rings the situation is similar, but now the instability rise time must exceed the radiation damping time. Even though, modern storage rings are designed for high beam currents and great efforts are being undertaken to reduce the impedance of higher cavity modes by designing monochromatic cavities where the higher-order modes are greatly suppressed [39, 40, 41, 42].

We have discussed here only the dipole mode of the longitudinal coupled-bunch instability. Of course, there are more modes and a similar set of instabilities in the transverse dimensions. A more detailed discussion of all aspects of multi-bunch instabilities would exceed the scope of this text and the interested reader is referred to the specific literature, specifically to [3, 9, 19, 26].

### Problems

#### 22.1 (S)

Consider a storage ring with 250 m circumference and a stored beam current of 50 mA in 1 bunch. Assume that the bunch length is about 5 % of the bunch spacing. A typical loss parameter for a BPM assembly is \(k_{\parallel}=3.35\cdot 10^{-2}\) V/pC and for bellows its \(k_{\parallel}=6.12\cdot 10^{-2}\) V/pC. Calculate the induced power in both BPM with 50 \(\Omega\) termination and bellows. What is the power at the 50 \(\Omega\) termination?

#### 22.2 (S)

Show that (22.94) is the same as (22.93) and show that the constant \(A\) in (22.94) is given by \(A=\frac{3}{4\pi}\frac{q!_{0}}{\beta^{2}E_{0}|\eta_{\rm c}|\delta^{2}}\), where \(\delta=\Delta p/p_{0}\).

**22.3.** Specify a damping ring at an energy of 1.5 GeV and an emittance of \(10^{-10}\) m-rad. The rf-frequency be 500 MHz and \(10^{11}\) electrons are to be stored into a single bunch at full coupling. Calculate the Touschek lifetime and the coherent and incoherent space-charge tune shift. Would the beam survive in case a tune shift of \(\Delta Q_{y}\leq 0.05\) were permissible?

**22.4.** Use the wake field for the SLAC linear accelerator structure (Fig. 22.8) and calculate the energy loss of a particle in the tail of a 1 mm long bunch of \(10^{11}\) electrons for the whole SLAC linear accelerator of 3 km length. This energy droop along a bunch is mostly compensated by accelerating the bunch ahead of the crest of the accelerating wave. This way the particles in the head of the bunch gain less energy than the particles in the tail of the bunch. The extra energy gain of the tail particles is then lost again due to wake field losses. How far off the crest must the bunch be accelerated for this compensation?

**22.5.** Consider the phenomenon of beam break-up in a linear accelerator and split the bunch into a head, center and tail part with a particle distribution \(N_{\rm b}/4\) to \(N_{\rm b}/2\) to \(N_{\rm b}/4\). Set up the equations of motion for all three particles including wake fields and solve the equations. Show the exponential build up of oscillation amplitudes of the tail particle. Perform the same derivation including BNS damping where each macroparticle has a different betatron oscillation frequency. Determine the condition for optimum BNS damping.

**22.6.** Determine the perturbation of a Gaussian particle distribution under the influence of a capacitive wake field. In particular, derive expressions for the perturbation of the distribution (if any) and the change in the fwhm bunch width as a function of \(\sigma_{\tau}\) in the limit of small wakes. If there is a shift in the distribution what physical effects cause it? Hint: think of a loss mechanism for a purely capacitive wake field?

**22.7.** During the discussion of the dispersion relation we observed the stabilizing effect of Landau damping and found the stability criterion (22.95) stating that the threshold current can be increased proportional to the square of the momentum spread in the beam. How does this stability criterion in terms of a momentum spread relate to the conclusion in the section on Landau damping that the beam should have a frequency overlap with the excitation frequency? Why is a larger momentum spread better than a smaller spread?

**22.8.** Determine stability conditions for the fast head-tail instability in a storage ring of your choice assuming that all transverse wake fields come from accelerating cavities. Use realistic parameters for the rf-system and the number of cells appropriate for your ring. What is the maximum permissible transverse impedance for a bunch current of 100 mA? Is this consistent with the transverse impedance of pill box cavities? If not how would you increase the current limit?

**22.9.** Calculate the real and imaginary impedance for the first longitudinal and transverse higher-order mode in a pill box cavity and apply these to determine themulti-bunch beam limit for a storage ring of your choice assuming that the beam spectrum includes the HOM frequency. Calculate also the frequency shift at the limit.

## Bibliography

* [1] A.W. Chao, J. Gareyte, Technical Report, Int Note-197, SLAC, Stanford (1976)
* [2] W. Schnell, Technical Report, CERN ISR-RF/70-7, CERN, Geneva (1970)
* [3] F. Sacherer, in _9th International Conference on High Energy Accelerators_ (Stanford Linear Accelerator Center, Stanford, 1974), p. 347
* [4] P.B. Wilson, Introduction to wake fields and wake functions. Technical Report, SLAC PUB-4547, SLAC (1989)
* [5] P.B. Wilson, R. Servranckx, A.P. Sabersky, J. Gareyte, G.E. Fischer, A.W. Chao, IEEE Trans. Nucl. Sci. **24**, 1211 (1977)
* [6] W.K.H. Panofsky, W.A. Wenzel, Rev. Sci. Instrum. **27**, 967 (1956)
* [7] A. Chao, K.H. Mess, M. Tigner, F. Zimmermann, (eds.), _Handbook of Accelerator Physics and Engineering_, 2nd edn. (World Scientific, Singapore, 2013)
* [8] A. Chao, K. Mess, M. Tigner, F. Zimmermann, (eds.), _Handbook of Accelerator Physics and Engineering_, 2nd edn. (World Scientific, Singapore, 2013)
* [9] A. Chao, _Physics of Collective Beam Instabilities in High Energy Accelerators_ (Wiley, New York, 1993)
* [10] J.D. Jackson, _Classical Electrodynamics_, 2 edn. (Wiley, New York, 1975)
* [11] C.E. Nielsen, A.M. Sessler, K.R. Symon, in _International Conference on High Energy Accelerators_ (CERN, Geneva, 1959), p. 239
* [12] V.K. Neil, A.M. Sessler, Rev. Sci. Instrum. **36**, 429 (1965)
* [13] L.J. Laslett, K.V. Neil, A.M. Sessler, Rev. Sci. Instrum. **32**, 279 (1961)
* [14] K. Hubner, A.G. Ruggiero, V.G. Vaccaro, in _8th International Conference on High Energy Accelerators_ (Yerevan Physics Institute, Yerevan, 1969), p. 296
* [15] A.G. Ruggiero, V.G. Vaccaro, Technical Report, CERN isr-th/68-33, CERN, Geneva (1968)
* [16] K. Hubner, A.G. Ruggiero, V.G. Vaccaro, Technical Report, CERN ISR-TH/69-23, CERN, Geneva (1969)
* [17] K. Hubner, V.G. Vaccaro, Technical Report, CERN ISR-TH/70-44, CERN, Geneva (1970)
* [18] K. Hubner, P. Strolin, V.G. Vaccaro, B. Zotter, Technical Report, CERN ISR-TH/70-2, CERN, Geneva (1970)
* [19] B. Zotter, F. Sacherer, Technical Report, CERN 77-13, CERN, Geneva (1977)
* [20] J. Landau, J. Phys. **10**, 25 (1946)
* [21] E. Keil, W. Schnell, Technical Report, CERN ISR-TH-RF/69-48, CERN, Geneva (1969)
* [22] H.G. Hereward, Technical Report, CERN 65-20, CERN, Geneva (1965)
* [23] A. Hofmann, Lecture Notes of Physics, vol. 296 (Springer, Berlin/Heidelberg, 1986), p. 112
* [24] L.R. Evans, _Physics of Particle Accelerators_, vol. 127 (American Institute of Physics, New York, 1985), p. 243
* [25] C. Pellegrini, A.M. Sessler, Nuovo Cimento **3A**, 116 (1971)
* [26] J.L. Laclare, Technical Report, CERN 87-03, CERN, Geneva (1987)
* [27] F. Sacherer, Technical Report, CERN SI-BR/72-5, CERN, Geneva (1972)
* [28] D. Boussard, Technical Report, CERN Rept. Lab II/RF/Int 75-2, CERN, Geneva (1975)
* [29] V. Balakin, A. Novokhatsky, V. Smirnov, in _12th International Conference on High Energy Accelerators_ (Fermi Laboratory, FNAL, Chicago, 1983), p. 119
* [30] J.T. Seeman, Lecture Notes of Physics, vol. 400 (Springer, Berlin/Heidelberg, 1986), p. 255
* [31] R.D. Kohaupt, Excitation of a transverse instability by parasitic cavity modes. Technical Report, Int. Note DESY H1-74/2, DESY, Hamburg (1972)* [32] C. Pellegrini, Nuovo Cimento **A64**, 447 (1969)
* [33] M. Sands, Technical Report, TN-69-8, TN-69-10, SLAC, Stanford (1969)
* [34] J.M. Paterson, B. Richter, A.P. Sabersky, H. Wiedemann, P.B. Wilson, M.A. Allen, J.E. Augustin, G.E. Fischer, R.H. Helm, M.J. Lee, M. Matera, P.L. Morton, in _9th International Conference on High Energy Accelerators_ (Stanford Linear Accelerator Center, Stanford, 1974), p. 338
* [35] J. Gareyte, F. Sacherer, in _9th International Conference on High Energy Accelerators_ (Stanford Linear Accelerator Center, Stanford, 1974), p. 341
* [36] J. Haissinski, Nuovo Cimento B **18**, 72 (1973)
* [37] D. Kohaupt, IEEE Trans. Nucl. Sci. **26**, 3480 (1979)
* [38] J. LeDuff, J. Maidment, E. Dasskovski, D. Degele, H.D. Dehne, H. Gerke, D. Heins, K. Hoffmann, K. Holm, E. Jandt, R.D. Kohaupt, J. Kouptsidis, F. Krafft, N. Lehhart, G. Mulhaupt, H. Nesemann, S. Paeztold, H. Pingel, A. Piwinski, R. Rossmanith, H.D. Schulz, K.G. Steffen, H. Wiedemann, K. Wille, A. Wrulich, in _9th International Conference on High Energy Accelerators_ (Stanford Linear Accelerator Center, Stanford, 1974), p. 43
* [39] F. Voelker, G. Lambertson, R. Rimmer, in _Proceedings of 1991 IEEE Particle Accelerator Conference_, San Francisco, IEEE Cat. No. 91CH3038-7 (1991), p. 687
* [40] R. Rimmer, F. Voelker, G. Lambertson, M. Allen, J. Hodgeson, K. Ko, R. Pendelton, H. Schwartz, in _Proceedings of 1991 IEEE Particle Accelerator Conference_, San Francisco, IEEE Cat. No. 91CH3038-7 (1991), p. 2688
* [41] S. Bartalucci, R. Boni, A. Gallo, L. Palumbo, R. Parodi, M. Sergio, B. Spataro, G. Vignola, in _Proceedings of 3th European Conference on Particle Accelerator_, ed. by H. Henke, H. Homeyer, Ch. Petit-Jean-Genaz (Berlin, 1992)
* [42] M. Svandrlik, G. D'Auria, A. Fabris, A. masserotti, C. Passotti, C. Rossi, A pill-box resonator with very strong suppression of the hom spectrums. Technical Report, ST/M-92/14, Sincrotrone Trieste, Trieste (1992)

