## Chapter 27 Free Electron Lasers

Synchrotron radiation is emitted when electromagnetic fields exert a force on a charged particle. This opens the possibility to apply external fields with specific properties for the stimulation of electrons to emit more radiation. Of course, not just any external electromagnetic field would be useful. Fields at some arbitrary frequency would not work because particles interacting with such fields would in general be periodically accelerated and decelerated without any net energy transfer. The external field must have a frequency and phase such that a particle may continuously lose energy into synchrotron radiation. Generally, it is most convenient to recycle and use spontaneous radiation emitted previously by the same emission process. In this part, we will discuss in some detail the process of stimulation as it applies to a free electron laser.

In a free electron laser (FEL) quasi-monochromatic, spontaneous radiation emitted from an undulator is recycled in an optical cavity to interact with the electron beam causing accelerations which are periodic with the frequency of the undulator radiation. In order to couple the particle motion to the strictly transverse electromagnetic radiation field, the path of the electrons is modulated by periodic deflections in a magnetic field to generate transverse velocity components. In a realistic setup, this magnetic field is provided in an undulator magnet serving both as the source of radiation and the means to couple to the electric field. The transverse motion of the particle results in a gain or loss of energy from/to the electromagnetic field depending on the location of the particle with respect to the phase of the external radiation field. The principle components of a FEL are shown in Fig. 27.1.

An electron beam is guided by a bending magnet unto the axis of an undulator. Upon exiting the undulator, the beam is again deflected away from the axis by a second bending magnet, both deflections to protect the mirrors of the optical cavity. Radiation that is emitted by the electron beam while travelling through the undulator is reflected by a mirror, travels to the mirror on the opposite side of the undulator and is reflected there again. Just as this radiation pulse enters the undulator again, another electron bunch joins to establish the emission of stimulated radiation. Theelectron beam pulse consists of a long train of many bunches, much longer than the length of the optical cavity such that many beam-radiation interactions can be established.

### Small Gain Regime

We may follow this process in great detail observing an electron as it travels through say the positive half period of its oscillatory trajectory. During this phase, the electron experiences a negative acceleration from the undulator magnet field which is in phase with the oscillation amplitude. Acceleration causes a perturbation of the electric fields of the electron as was discussed in detail in Chap. 12. This perturbation travels away from the source at the speed of light, which is what we call electromagnetic radiation. For an electron, the electric radiation field points in the direction of the acceleration. As the electron travels through the positive half wave, it emits a radiation field made of half a wave. Simultaneously, this radiation field, being faster than the electron, travels ahead of the electron by precisely half a wavelength. This process tells us that the radiation wavelength is closely related to the electron motion and that it is quasi-monochromatic. Of course, for a strong undulator the sinusoidal motion becomes perturbed and higher harmonics appear, but the principle arguments made here are still true. Now, the electron starts performing the negative half of its oscillation and, experiencing a positive acceleration, emits the second halfwave of the radiation field matching perfectly the first halfwave. This happens in every period of the undulator and when the electron reaches the end of the last period a radiation wave composed of \(N_{\rm p}\) oscillations exists ahead of the electron. This process describes the spontaneous radiation emission from an electron in an undulator magnet.

The radiation pulse just created is recycled in the optical cavity to reenter the undulator again at a later time. The length of the optical cavity must be adjusted very precisely to an integer multiple of both the radiation wavelength and the distance between electron bunches. Under these conditions, electron bunches and radiation pulses enter the undulator synchronously. A complication arises from the fact that the electrons are contained in a bunch which is much longer than the wavelength of the radiation. The electrons are distributed for all practical purposes uniformly over many wavelengths. For the moment, we ignore this complication and note that there is an electron available whenever needed.

Figure 27: Free electron laser setup (schematic)

We pick now an electron starting to travel through a positive half wave of its oscillation exactly at the same time and location as the radiation wave starts its positive field halfperiod. The electron, experiences then a downward acceleration from the radiation field. During its motion the electron is continuously accelerated until it has completed its travel through the positive half oscillation. At the same time, the full positive have wave of the radiation field has moved over the electron. At this moment the electron and the radiation field are about to start their negative half periods. Continuing its motion now through the negative half period, the electron still keeps loosing energy because it now faces a negative radiation field. The fact that the radiation field "slides" over the electron just one wavelength per undulator period ensures a continuous energy transfer from electron to the radiation field. The electron emits radiation which is now exactly in synchronism with the existing radiation field and the new radiation intensity is proportional to the acceleration or the external radiation field. Multiple recycling and interaction of radiation field with electron bunches results therefore in an exponential increase in radiation intensity.

At this point, we must consider all electrons, not just the one for which the stimulation works as just described. This process does not work that perfect for all particles. An electron just half a wavelength behind the one discussed above would continuously gain energy from the radiation field and any other electron would loose or gain energy depending on its phase with respect to the radiation. It is not difficult to convince oneself that on average there may not be any net energy transfer one way or another and therefore no stimulation or acceleration. To get actual stimulation, some kind of asymmetry must be introduced.

To see this, we recollect the electron motion in a storage ring in the presence of the rf-field in the accelerating cavity. In Sect. 9.2.1 we discussed the phase space motion of particles under the influence of a radiation field. The radiation field of a FEL acts exactly the same although at a much shorter wavelength. The electron beam extends over many buckets as shown in Fig.27.2 and it is obvious that in its interaction with the field half of the electrons gain and the other half loose energy from/to the radiation field. The effect of the asymmetry required to make the FEL work is demonstrated in Fig. 27.3. Choosing an electron beam energy to be off

Figure 27.2: Interaction of an electron beam (on-resonance energy) with the radiation field of a FEL. The _arrows_ in the first bucket indicate the direction of particle motion in its interaction with the electromagnetic field

resonance by a small amount, the energy gain and losses for all electrons within a bucket becomes unbalanced and we can choose a case where all electrons on average loose energy into (FEL) or gain energy (particle acceleration by a radiation field) from the radiation field. The arrows in the first bucket of Fig. 27.3 show clearly the imbalance of energy gain or loss. What it means to choose an electron beam energy off-resonance will be discussed in more detail in the next section, where we formulate quantitatively the processes discussed so far only qualitatively.

We concentrate on the case where only a small fraction of the particle energy is extracted such that we can neglect effects on particle parameters. This regime is called the "small-gain" regime. Specifically, we ignore changes in the particle energy and redistribution in space as a consequence of the periodic energy modulation.

#### Energy Transfer

Transfer of energy between a charged particle and an electromagnetic wave is effected by the electric field term of the Lorentz force equation and the amount of transferred energy is

\[\Delta W=e\int\mathbf{E}_{\mathrm{L}}\,\mathrm{d}\mathbf{z}=e\int_{\mathrm{L}}\mathbf{Ev} \,\mathrm{d}t\,, \tag{27.1}\]

where \(\mathbf{E}_{\mathrm{L}}\) is the external field or the Laser field in the optical cavity and \(\mathbf{v}\) the particle velocity. In free space \(\mathbf{v}\perp\mathbf{E}_{\mathrm{L}}\) and therefore there is no energy transfer possible (\(\Delta W\equiv 0\)). Generating some transverse velocity \(\mathbf{v}_{\perp}\) through periodic deflection in an undulator, we get from (26.3)

\[v_{x}=+\,\beta c\frac{K}{\gamma}\sin\left(k_{\mathrm{P}}z\right)\,, \tag{27.2}\]

where \(k_{\mathrm{P}}=2\pi/\lambda_{\mathrm{P}}\). The external radiation field can be expressed by

\[\mathbf{E}_{\mathrm{L}}=\mathbf{E}_{\mathrm{0L}}\cos\left(\omega_{\mathrm{L}}t-k_{ \mathrm{L}}z+\varphi_{0}\right) \tag{27.3}\]

Figure 27.3: Interaction of an electron beam (off-resonance energy) with the radiation field of a FEL

and the energy transfer is

\[\begin{split}\Delta W&=e\int\mathbf{v}E_{\rm L}{\rm d}t=e \int v_{x}E_{\rm L}{\rm d}t\\ &=e\beta c\frac{K}{\gamma}E_{0{\rm L}}\int\cos\left(\omega_{\rm L }t-k_{\rm L}z+\varphi_{0}\right)\sin\left(k_{\rm P}z\right)\,{\rm d}t\\ &=\tfrac{1}{2}e\beta c\frac{K}{\gamma}E_{0{\rm L}}\int\left(\sin \Psi^{+}-\sin\Psi^{-}\right)\,{\rm d}t\,,\end{split} \tag{27.4}\]

where

\[\Psi^{\pm}=\omega_{\rm L}t-\left(k_{\rm L}\pm k_{\rm P}\right)z+\varphi_{0}. \tag{27.5}\]

The energy transfer appears to be oscillatory, but continuous energy transfer can be obtained if either \(\Psi^{+}\)= const. or \(\Psi^{-}\)= const. In this case

\[\frac{{\rm d}\Psi^{\pm}}{{\rm d}t}=\omega_{\rm L}-\left(k_{\rm L}\pm k_{\rm P} \right)\dot{z}=0 \tag{27.6}\]

and we must derive conditions for this to be true. The velocity \(\dot{z}\) is from (26.3)

\[\dot{z}=\bar{\beta}c+\beta c\frac{K^{2}}{4\gamma^{2}}\cos\left(2k_{\rm P}z \right), \tag{27.7}\]

where the average drift velocity \(\bar{\beta}c\) is defined by

\[\frac{{\rm d}\bar{z}}{{\rm d}t}=\bar{\beta}c=\beta c\left(1-\frac{K^{2}}{4 \gamma^{2}}\right). \tag{27.8}\]

We modify slightly the condition (27.6) and require that it be true only on average

\[\frac{{\rm d}\Psi^{\pm}}{{\rm d}t}=\omega_{\rm L}-\left(k_{\rm L}\pm k_{\rm P} \right)\frac{{\rm d}\bar{z}}{{\rm d}t}=0, \tag{27.9}\]

or

\[\left(k_{\rm L}\pm k_{\rm P}\right)\beta\left(1-\frac{K^{2}}{4\gamma^{2}} \right)-k_{\rm L}=0. \tag{27.10}\]

With \(\beta\approx 1-1/2\gamma^{2}\) and \(k_{\rm P}\ll k_{\rm L}\), (27.10) becomes

\[k_{\rm L}\left[\left(1-\frac{1}{2\gamma^{2}}\right)\left(1-\frac{K^{2}}{4 \gamma^{2}}\right)-1\right]\pm k_{\rm P}\approx 0, \tag{27.11}\]or for \(\gamma\gg 1\)

\[-\frac{k_{\rm L}}{2\gamma^{2}}\left(1+\tfrac{1}{2}K^{2}\right)\pm k_{\rm p}=0. \tag{27.12}\]

Equation (27.12) can be met only for the \(+\)sign or for

\[k_{\rm p}=\frac{k_{\rm L}}{2\gamma^{2}}\left(1+\tfrac{1}{2}K^{2}\right), \tag{27.13}\]

which is identical to the definition of the fundamental undulator radiation wavelength

\[\lambda_{\rm L}=\frac{\lambda_{\rm p}}{2\gamma^{2}}\left(1+\tfrac{1}{2}K^{2} \right). \tag{27.14}\]

Radiation at the fundamental wavelength of undulator radiation guarantees a continuous energy transfer from the particles to the electromagnetic wave or stimulation of radiation emission by an external field. For this reason, it is most convenient to use spontaneous undulator radiation as the external field to start the build-up of the free electron laser.

##### Equation of Motion

The energy gain \(\mathrm{d}W\) of the electromagnetic field is related to the energy change \(\mathrm{d}\gamma\) of the electron by

\[\frac{\mathrm{d}\gamma}{\mathrm{d}z}=-\frac{1}{mc^{2}}\frac{\mathrm{d}W}{ \beta\mathrm{c}\mathrm{d}t} \tag{27.15}\]

or with (27.4)

\[\frac{\mathrm{d}\gamma}{\mathrm{d}z}=-\frac{eKE_{0\rm L}}{2\gamma mc^{2}} \left(\sin\Psi^{+}-\sin\Psi^{-}\right). \tag{27.16}\]

With the substitution \(\sin x=-\mathrm{Re}\,\left(\mathrm{i}\,e^{\mathrm{i}x}\right)\)

\[\frac{\mathrm{d}\gamma}{\mathrm{d}z}=\frac{eKE_{0\rm L}}{2\gamma mc^{2}} \mathrm{Re}\,\left(\mathrm{i}\mathrm{e}^{\mathrm{i}\Psi^{+}}-\mathrm{i} \mathrm{e}^{\mathrm{i}\Psi^{-}}\right). \tag{27.17}\]

In \(\Psi^{\pm}=\omega_{\rm L}t-\left(k_{\rm L}\pm k_{\rm p}\right)\,z\left(t \right)+\varphi_{0}\), we replace the location function \(z(t)\) by its expression (26.5)

\[z\left(t\right)=\underbrace{\tilde{\beta}ct}_{=\tilde{z}}+\underbrace{\frac{K ^{2}}{8\gamma^{2}k_{\rm p}}\sin\left(2k_{\rm p}\tilde{\beta}ct\right)}_{ \ll\tilde{\beta}ct}, \tag{27.18}\]composed of an average position \(\bar{z}\) and an oscillatory term. With \(k_{\rm p}\ll k_{\rm L}\)

\[\frac{{\rm d}\gamma}{{\rm d}z} = \frac{e\beta KE_{0{\rm L}}}{2\gamma mc^{2}}{\rm Re}\ \left\{{\rm i}\exp\left[{\rm i}\,\frac{k_{\rm L}K^{2}}{8 \gamma^{2}k_{\rm p}}\sin\left(2k_{\rm p}\bar{z}\right)\right]\left[{\rm e}^{{ \rm i}\bar{\Psi}^{+}}-{\rm e}^{{\rm i}\bar{\Psi}^{-}}\right]\right\} \tag{27.19}\]

and the phase

\[\bar{\Psi}^{\pm} = \omega_{\rm L}\,t-\left(k_{\rm L}\pm k_{\rm p}\right)\,\bar{z}+ \varphi_{0}\,. \tag{27.20}\]

With the definition \(\exp\left({\rm i}x\sin\phi\right)=\sum_{n=-\infty}^{n=+\infty}J_{n}\left(x \right){\rm e}^{{\rm i}n\phi}\) we get finally

\[\frac{{\rm d}\gamma}{{\rm d}z} = \frac{e\beta K\,E_{0{\rm L}}}{2\gamma mc^{2}}{\rm Re}\left[{\rm i }\,\sum\nolimits_{n=-\infty}^{n=+\infty}J_{n}\left(\frac{k_{\rm L}K^{2}}{8 \gamma^{2}k_{\rm p}}\right)\,{\rm e}^{{\rm i}2nk_{\rm p}\bar{z}}\left({\rm e}^ {{\rm i}\bar{\Psi}^{+}}-{\rm e}^{{\rm i}\bar{\Psi}^{-}}\right)\right]\,. \tag{27.21}\]

The infinite sum reflects the fact that the condition for continuous energy transfer can be met not only at one wavenumber but also at all harmonics of that frequency. Combining the exponential terms and sorting for equal wavenumbers \(hk_{\rm p}\), where \(h\) is an integer, we redefine the summation index by setting

\[2nk_{\rm p}+k_{\rm p} = hk_{\rm p}\quad\longrightarrow\quad n=\frac{h-1}{2} \tag{27.22a}\] \[2nk_{\rm p}-k_{\rm p} = hk_{\rm p}\quad\longrightarrow\quad n=\frac{h+1}{2} \tag{27.22b}\]

and get

\[\frac{{\rm d}\gamma}{{\rm d}z} = \frac{e\beta K\,E_{0{\rm L}}}{2\gamma mc^{2}}\sum_{h=1}^{\infty} \left[J_{\frac{h-1}{2}}\left(x\right)-J_{\frac{h+1}{2}}\left(x\right)\right] \underbrace{{\rm Re}\ \left\{{\rm i}\,{\rm e}^{{\rm i}\left[\left(k_{\rm L}+h\,k_{\rm p} \right)\bar{z}-\omega_{\rm L}\,t+\varphi_{0}\right]}\right\}}_{=-\sin\left[ \left(k_{\rm L}+h\,k_{\rm p}\right)\bar{z}-\omega_{\rm L}\,t+\varphi_{0} \right]}, \tag{27.23}\]

where \(x=\frac{K^{2}}{4+2K^{2}}\). Using the \(JJ\)-function (26.60) the energy transfer is

\[\frac{{\rm d}\gamma}{{\rm d}z} = -\frac{e\beta K\,E_{0{\rm L}}}{2\gamma mc^{2}}\sum_{h=1}^{\infty} \left[JJ\right]\sin\Psi. \tag{27.24}\]

For maximum continuous energy transfer \(\sin\Psi=\) const. or

\[\frac{{\rm d}\Psi}{{\rm d}t} = \left(k_{\rm L}+h\,k_{\rm p}\right)\frac{{\rm d}\bar{z}}{{\rm d}t}- \omega_{\rm L}\] \[= \left(k_{\rm L}+h\,k_{\rm p}\right)\beta c\left(1-\frac{K^{2}}{4 \gamma^{2}}\right)-\omega_{\rm L}\]\[= \left(k_{\rm L}+h\,k_{\rm p}\right)\,\left(1-\frac{1}{2\gamma^{2}} \right)c\left(1-\frac{K^{2}}{4\gamma^{2}}\right)-ck_{\rm L}\] \[\frac{{\rm d}\Psi}{{\rm d}t} = -\frac{ck_{\rm L}}{2\gamma_{\rm r}^{2}}\left(1+\tfrac{1}{2}K^{2} \right)+h\,k_{\rm p}c=0\,,\]

where we assumed that \(k_{\rm L}\gg h\,k_{\rm p}\), which is true since \(\lambda_{\rm p}\gg\lambda_{\rm L}\) and the harmonic number of interest is generally unity or a single digit number. This condition confirms our earlier finding (27.14) and extends the synchronicity condition to multiples \(h\) of the fundamental radiation frequency

\[\lambda_{\rm L}=\frac{\lambda_{\rm p}}{2\gamma^{2}h}\left(1+\tfrac{1}{2}K^{2} \right). \tag{27.26}\]

The integer \(h\) therefore identifies the harmonic of the radiation frequency with respect to the fundamental radiation.

In a real particle beam with a finite energy spread we may not assume that all particles exactly meet the synchronicity condition. It is therefore useful to evaluate the tolerance for meeting this condition. To do this, we define a resonance energy

\[\gamma_{\rm r}^{2}=\frac{k_{\rm L}}{2hk_{\rm p}}\left(1+\tfrac{1}{2}K^{2} \right), \tag{27.27}\]

which is the energy at which the synchronicity condition is met exactly. For any other particle energy \(\gamma=\gamma_{\rm r}+\delta\gamma\) we get from (27.25) and (27.27)

\[\frac{{\rm d}\Psi}{{\rm d}z}=2hk_{\rm p}\frac{\delta\gamma}{\gamma_{\rm r}}\,. \tag{27.28}\]

With the variation of the energy deviation \(\frac{{\rm d}}{{\rm d}z}\delta\gamma=\left.\frac{{\rm d}\gamma}{{\rm d}z} \right|_{\gamma_{\rm r}}-\frac{{\rm d}\gamma_{\rm r}}{{\rm d}z}=\left.\frac{{ \rm d}\gamma}{{\rm d}z}\right|_{\gamma_{\rm r}}\) and (27.24) we get from (27.28) after differentiating with respect to \(z\)

\[\frac{{\rm d}^{2}\Psi}{{\rm d}z^{2}}=2hk_{\rm p}\frac{{\rm d}}{{\rm d}z}\frac{ \delta\gamma}{\gamma_{\rm r}}=-\frac{ehk_{\rm p}KE_{\rm 0L}}{\gamma_{\rm r}^{2} mc^{2}}[JJ]\sin\Psi(z)\,, \tag{27.29}\]

where, for simplicity, we use only one harmonic \(h\). This equation can be written in the form

\[\frac{{\rm d}^{2}\Psi}{{\rm d}z^{2}}+\Omega_{\rm L}^{2}\sin\Psi=0 \tag{27.30}\]

exhibiting the dynamics of a harmonic oscillator. Equation (27.30) is known as the Pendulum equation [1] with the frequency

\[\Omega_{\rm L}^{2}=\frac{ehk_{\rm p}KE_{\rm 0L}}{\gamma_{\rm r}^{2}mc^{2}}\left| JJ\right|\,. \tag{27.31}\]While interacting with the external radiation field, the particles perform harmonic oscillations in a potential generated by this field. This situation is very similar to the synchrotron oscillation of particles in a storage ring interacting with the field of the rf-cavities as was discussed in Sect. 9.2.1. In phase space, the electron perform synchrotron oscillations at the frequency \(\Omega_{\rm L}\) while exchanging energy with the radiation field.

##### FEL-Gain

Having established the possibility of energy transfer from an electron to a radiation field, we may evaluate the magnitude of this energy transfer or the gain in field energy per interaction process or per pass. One pass is defined by the interaction of an electron bunch with the radiation field while passing through the entire length of the undulator. The gain in the laser field \(\Delta W_{\rm L}=-mc^{2}n_{\rm e}\Delta\gamma\), where \(\Delta\gamma\) is the energy loss per electron and pass to the radiation field and \(n_{\rm e}\) the number of electrons per bunch. The energy in the laser field

\[W_{\rm L}\,=\,\tfrac{1}{4}\epsilon_{0}E_{0\rm L}^{2}V\,, \tag{27.32}\]

where \(V\) is the volume of the radiation field. With this, we may define the average FEL-gain for the \(h\)th harmonic by

\[G_{h}\,=\,\frac{\left\langle\Delta W_{\rm L}\right\rangle}{W_{\rm L}}\,=-\frac {mc^{2}n_{\rm e}\left\langle\Delta\gamma\right\rangle_{n_{\rm e}}}{\tfrac{ \epsilon_{0}}{4}E_{0\rm L}^{2}V}\,=\,-\frac{2mc^{2}\gamma_{\rm r}n_{\rm e}}{ \epsilon_{0}hk_{\rm p}E_{0\rm L}^{2}V}\left\langle\Delta\Psi^{\prime}\right \rangle_{n_{\rm e}}\,, \tag{27.33}\]

making use of (27.28) \(\left\langle\Delta\Psi^{\prime}\right\rangle_{n_{\rm e}}\) is the average value of \(\Delta\Psi^{\prime}\)=\(\Psi_{\rm f}^{\prime}\)-\(\Psi_{0}^{\prime}\) for all electrons per bunch, where \(\Psi_{0}^{\prime}\) is defined at the beginning of the undulator and \(\Psi_{\rm f}^{\prime}\) at the end of the undulator. To further simplify this expression, we use (27.31), solve for the laser field

\[E_{0\rm L}\,=\,\frac{mc^{2}\gamma_{\rm r}^{2}\Omega_{\rm L}^{2}}{ehKk_{\rm p}[ JJ]}, \tag{27.34}\]

and define the electron density \(n_{\rm b}=n_{\rm e}/V\). Here we have tacitly assumed that the volume of the radiation field perfectly overlaps the volume of the electron beam. This is not automatically the case and must be achieved by carefully matching the electron beam to the diffraction dominated radiation field. If this cannot be done, the volume \(V\) is the overlap volume, or the larger of both. With this the FEL-gain becomes

\[G=-\frac{8\pi\,e^{2}n_{\rm b}hK^{2}k_{\rm p}[JJ]^{2}}{mc^{2}\gamma_{\rm r}^{3} \Omega_{\rm L}^{4}}\left\langle\Delta\Psi^{\prime}\right\rangle_{n_{\rm e}} \tag{27.35}\]Numerical evaluation of \(\left(\Delta\Psi^{\prime}\right)_{n_{\rm e}}\) can be performed with the pendulum equation. Multiplying the pendulum equation \(2\Psi^{\prime}\) and integrating we get

\[\Psi^{\prime 2}-2\Omega_{\rm L}^{2}\cos\Psi\,={\rm const.} \tag{27.36}\]

Evaluating this at the beginning of the undulator

\[\Psi^{\prime 2}-\Psi_{0}^{\prime 2}=2\Omega_{\rm L}^{2}\left(\cos\Psi-\cos\Psi_{0 }\right)\,, \tag{27.37}\]

which becomes with \(\Psi_{0}^{\prime}=2h\,k_{\rm p}\frac{\gamma_{0}-\gamma_{\rm r}}{\gamma_{\rm r}}\)

\[\Psi^{\prime 2}=\left(2hk_{\rm p}\frac{\gamma_{0}-\gamma_{\rm r}}{\gamma_{\rm r }}\right)^{2}+2\Omega_{\rm L}^{2}\left(\cos\Psi-\cos\Psi_{0}\right) \tag{27.38}\]

Finally,

\[\Psi^{\prime}\left(z\right)=2\,h\,k_{\rm p}\frac{\gamma-\gamma_{\rm r}}{ \gamma_{\rm r}}\sqrt{1+\frac{\Omega_{\rm L}^{2}}{2k^{2}\,k_{\rm p}^{2}}\frac{ \gamma_{\rm r}^{2}}{\left(\gamma-\gamma_{\rm r}\right)^{2}}\left[\cos\Psi \left(z\right)-\cos\Psi_{0}\right]}, \tag{27.39}\]

or with

\[w=h\,k_{\rm p}L_{\rm u}\frac{\gamma-\gamma_{\rm r}}{\gamma_{\rm r}}\,, \tag{27.40}\]

where \(L_{\rm u}=N_{\rm p}\lambda_{\rm p}\) is the undulator length,

\[\Psi^{\prime}\left(z\right)=\frac{2w}{L_{\rm u}}\sqrt{1+\frac{L_{\rm u}^{2}Q_ {\rm L}^{2}}{2w^{2}}\left[\cos\Psi\left(z\right)-\cos\Psi_{0}\right]}\,. \tag{27.41}\]

We solve this by expansion and iteration. For a low gain FEL, the field \(E_{0\rm L}\) is weak and does not influence the particle motion. Therefore \(\Omega_{\rm L}\ll 1\) and (27.41) becomes

\[\Psi^{\prime} \approx \frac{2w}{L}\left[1+\frac{1}{2}\frac{L^{2}\Omega_{\rm L}^{2}}{2w^ {2}}\left(\cos\Psi-\cos\Psi_{0}\right)\right. \tag{27.42}\] \[\left.-\frac{1}{8}\frac{L^{4}\Omega_{\rm L}^{4}}{4w^{4}}\left( \cos\Psi-\cos\Psi_{0}\right)^{2}+\ldots\right].\]

In the lowest order of iteration \(\Psi^{\prime}=\Psi_{0}^{\prime}=\frac{2w}{L}\) and \(\Delta\Psi_{(0)}^{\prime}=0\) for all particles, which means there is no energy transfer. For first order approximation, we integrate \(\Psi_{0}^{\prime}\left(z\right)=\frac{2w}{L_{\rm u}}\) to get \(\Psi_{(1)}(z)=\frac{2w}{L_{\rm u}}z+\Psi_{0}\) and

\[\Delta\Psi_{(1)}^{\prime}=\Psi^{\prime}(L_{\rm u})-\Psi_{1}^{\prime}\left(0 \right)=\frac{L\,\Omega_{\rm L}^{2}}{2w}\left[\cos\left(2w+\Psi_{0}\right)- \cos\Psi_{0}\right]+{\cal O}(2) \tag{27.43}\]from (27.42). Averaging over all initial phases occupied by electrons \(0\leq\Psi_{0}\leq 2\pi\)

\[\left\langle\Delta\Psi_{1}^{\prime}\right\rangle=\frac{L\,\Omega_{\rm L}^{2}}{2w} \frac{1}{2\pi}\int_{0}^{2\pi}\left[\cos\left(2w+\Psi_{0}\right)-\cos\Psi_{0} \right]\,{\rm d}\Psi_{0}=0\,. \tag{27.44}\]

No energy transfer to the laser field occurs in this approximation either. We need a still higher order approximation. The higher order correction to \(\Psi_{1}^{\prime}\left(s\right)=\Psi_{0}^{\prime}\left(s\right)+\delta\Psi_{1} ^{\prime}\left(s\right)\,\) is from (27.42)

\[\delta\Psi_{(1)}^{\prime}=\frac{L\Omega_{\rm L}^{2}}{2w}\left[\cos\Psi-\cos \Psi_{0}\right]\,, \tag{27.45}\]

and the correction to \(\Psi_{1}\left(s\right)\) is

\[\delta\Psi_{(1)} = \frac{L\Omega_{\rm L}^{2}}{2w}\int_{0}^{L}\left[\cos\left(\frac{ 2w}{L}z+\Psi_{0}\right)-\cos\Psi_{0}\right]\,{\rm d}s \tag{27.46}\] \[= \frac{L^{2}\Omega_{\rm L}^{2}}{4w^{2}}\left[\sin\left(2w+\Psi_{ 0}\right)-\sin\Psi_{0}-2w\cos\Psi_{0}\right].\]

The second order approximation to the phase is then \(\Psi_{1}(z)=\frac{2w}{L_{\rm u}}z+\Psi_{0}+\delta\Psi_{(1)}\) and using (27.42) in second order as well we get

\[\Delta\Psi_{(2)}^{\prime} = \frac{L\,\Omega_{\rm L}^{2}}{2w}\left[\cos\left(2w+\Psi_{0}+ \delta\Psi_{(1)}\right)-\cos\Psi_{0}\right] \tag{27.47}\] \[- \frac{L^{3}\Omega_{\rm L}^{4}}{4w^{2}}\left[\cos\left(2w+\Psi_{0} \right)-\cos\Psi_{0}\right]^{2}+\ldots\,,\]

where in the second order term only the first order phase \(\Psi_{1}(z)=\frac{2w}{L_{\rm u}}z+\Psi_{0}\,\) is used. The first term becomes with \(\delta\Psi_{(1)}\ll\Psi_{0}+2w\)

\[\cos\left(2w+\Psi_{0}+\delta\Psi_{1}\right)-\cos\Psi_{0} \tag{27.48}\] \[\approx \cos\left(2w+\Psi_{0}\right)-\delta\Psi_{1}\sin\left(2w+\Psi_{0} \right)-\cos\Psi_{0} \tag{27.49}\]

and

\[\Delta\Psi_{2}^{\prime} = \frac{L_{\rm u}^{3}\,\Omega_{\rm L}^{4}}{16w^{3}}\left\{\frac{8w^ {2}}{L_{\rm u}^{2}\Omega^{2}}\left[\cos\left(2w+\Psi_{0}\right)-\cos\Psi_{0} \right]\right. \tag{27.50}\] \[\left.-\left[\cos\left(2w+\Psi_{0}\right)-\cos\Psi_{0}\right]^{ 2}+\ldots\right\}\,.\]Now, we average over all initial phases assuming a uniform distribution of particles in \(z\) or in phase. The individual terms become then

\[\left\langle\cos\left(2w+\Psi_{0}\right)-\cos\Psi_{0}\right\rangle =0\] \[\left\langle\sin^{2}\left(2w+\Psi_{0}\right)\right\rangle =\tfrac{1}{2}\] \[\left\langle\sin\left(2w+\Psi_{0}\right)\sin\Psi_{0}\right\rangle =\tfrac{1}{2}\cos\left(2w\right) \tag{27.51a}\] \[\left\langle\sin\left(2w+\Psi_{0}\right)\cos\Psi_{0}\right\rangle =\tfrac{1}{2}\sin\left(2w\right)\] \[\left\langle\cos\left(2w+\Psi_{0}\right)\cos\Psi_{0}\right\rangle =\tfrac{1}{2}\cos\left(2w\right)\,.\]

With this

\[\left\langle\Delta\Psi_{2}^{\prime}\right\rangle=-\tfrac{I_{\mathrm{u}}^{3} \,Q_{\mathrm{L}}^{4}}{16w^{3}}\left[1-\cos\left(2w\right)-w\sin\left(2w\right)\right] \tag{27.52}\]

and finally with \(\left[1-\cos\left(2w\right)-w\sin\left(2w\right)\right]/w^{3}=-\tfrac{\mathrm{d }}{\mathrm{d}w}\left(\tfrac{\sin w}{w}\right)^{2}\)

\[\left\langle\Delta\Psi_{2}^{\prime}\right\rangle=\frac{L_{\mathrm{u}}^{3}\,Q _{\mathrm{L}}^{4}}{8}\frac{\mathrm{d}}{\mathrm{d}w}\left(\frac{\sin w}{w} \right)^{2}\,. \tag{27.53}\]

The FEL-gain is finally from (27.35)

\[G_{h} = -\frac{\pi r_{\mathrm{c}}n_{\mathrm{b}}h\,K^{2}L_{\mathrm{u}}^{3} k_{\mathrm{p}}}{\gamma_{\mathrm{r}}^{3}}\,[JJ]^{2}\frac{\mathrm{d}}{\mathrm{d}w} \left(\frac{\sin w}{w}\right)^{2}, \tag{27.54}\]

where we may express the particle density \(n_{\mathrm{b}}\) by beam parameters as obtained from the electron beam source

\[n_{\mathrm{b}} = \frac{n_{\mathrm{e}}}{V}=\frac{n_{\mathrm{e}}}{2\pi\sigma^{2} \ell}, \tag{27.55}\]

where \(\sigma\) is the radius of a round beam. With these definitions, and \(\hat{I}=\mathit{cen}_{\mathrm{e}}/\ell\)  the electron peak current, the gain per pass becomes

\[G_{h} = -\frac{2^{2/3}\pi\,r_{\mathrm{c}}h\,\lambda^{3/2}L_{\mathrm{u}}^{ 3}}{c\sigma^{2}\lambda_{\mathrm{p}}^{5/2}}\frac{\hat{I}}{e}\frac{K^{2}[JJ]^{2} }{\left(1+\tfrac{1}{2}K^{2}\right)^{3/2}}\frac{\mathrm{d}}{\mathrm{d}w}\left( \frac{\sin w}{w}\right)^{2}. \tag{27.56}\]

The gain depends very much on the choice of the electron beam energy through the function (27.40), which is expressed by the gain curve as shown in Fig. 27.4.

There is no gain if the beam energy is equal to the resonance energy (\(\gamma=\gamma_{\mathrm{r}}\)). As has been discussed in the introduction to this chapter, we must introduce an asymmetry to gain stimulation of radiation or gain and this asymmetry is generated by a shift in energy. For a monochromatic electron beam maximum gain can be reached for \(w\approx\,1.2\). A realistic beam, however, is not monochromatic and the narrow gain curve indicates that a beam with too large an energy spread may not produce any overall gain. There is no precise upper limit for the allowable energy spread but from Fig. 27.4 we see that gain is all but gone when \(|w|\gtrsim 5\). We use this with (27.40) and (27.27) to formulate a condition for the maximum allowable energy spread

\[\left|\frac{\delta\gamma}{\gamma}\right|\ll\frac{2\gamma_{\rm r}^{2}\lambda_{ \rm L}}{1+\frac{1}{2}K^{2}}. \tag{27.57}\]

For efficient gain the geometric size of the electron beam and the radiation field must be matched. In (27.55) we have introduced a volume for the electron bunch. Actually, this volume is the overlap volume of radiation field and electron bunch. Ideally, one would try to get a perfect overlap by forming both beams to be equal. This is in fact possible and we will discuss the conditions for this to happen. First, we assume that the electron beam size varies symmetrically about the center of the undulator. The beam size develops like

\[\sigma^{2}\left(z\right)=\sigma_{0}^{2}+\left(\frac{\epsilon}{\sigma_{0}} \right)^{2}z^{2} \tag{27.58}\]

with distance \(z\) from the beam waist. To maximize gain we look for the minimum average beam size within an undulator. This minimum demands a symmetric solution about the undulator center. Furthermore, we may select the optimum beam size at the center by looking for the minimum value of the maximum beam size within the undulator. From \({\rm d}\sigma^{2}/{\rm d}\sigma_{0}^{2}=0\), the optimum solution is obtained for \(z=\frac{1}{2}L_{\rm u}=\sigma_{0}^{2}/\epsilon=\beta_{0}\). For \(\beta_{0}=\frac{1}{2}L_{\rm u}\) the beam cross section grows from a value of \(\sigma_{0}^{2}\) in the middle of the undulator to a maximum value of \(2\sigma_{0}^{2}\) at either end.

The radiation field is governed by diffraction. Starting at a beam waist, the growth of the radiation field cross section due to diffraction is quantified by the Rayleigh length

\[z_{\rm R}=\pi\,\frac{w_{0}^{2}}{\lambda}, \tag{27.59}\]where \(w_{0}\) is the beam size at the waist and \(\lambda\) the wavelength. This length is defined as the distance from the radiation source (waist) to the point at which the cross section of the radiation beam has grown by a factor of two. For a Gaussian beam, we have for the beam size at a distance \(z\) from the waist

\[w^{2}(z)=w_{0}^{2}+\Theta^{2}z^{2}, \tag{27.60}\]

where \(\Theta=\frac{\lambda}{\pi w_{0}}\) is the divergence angle of the radiation field. This is exactly the same condition as we have just discussed for the electron beam assuming the center of the undulator as the source of radiation.

### High Gain Free Electron Laser

We have discussed the interaction of an electron beam with an external electromagnetic field and found that repeated recycling of the photon beam by reflecting mirrors this photon beam intensity can be made to grow until it is big enough to modulate the electron beam into microbunches at a distance equal to the radiation wavelength. This interaction works only at wavelength where good reflectors are available. This is, for example, not possible for UV and x-rays. The question arises what would happen if an electron beam would travel through a very long undulator instead of being reflected many times. This is the principle of self-amplified-spontaneous-emission or SASE.

The goal is to look for electron dynamics which leads to micro bunching at the wavelength of interest. Any bunch radiates coherently at wavelengths equal or longer than the bunch length as was discussed in Sect. 24.7. This coherent radiation scales like the square of the number of particles per bunch \(n_{\mathrm{b}}^{2}\) rather than linear with \(n_{\mathrm{b}}\) as is the case of incoherent radiation emitted at wavelength shorter than the bunch length. Since the number of electrons per bunch can be very large we gain a large increase in the photon intensity. Actually this is the highest photon intensity one can extract from a bunch of electrons. Unfortunately, it is technically not possible to generate bunches at visible or shorter wavelength and preserve such bunches along a beam line. The way out is to possibly generate microbunches at the place of the radiation source. This was possible in the FEL and we will now discuss this possibility in the realm of SASE.

#### Electron Dynamics in a SASE FEL

In this section we aim at producing coherent radiation at any wavelength specifically at very short wavelength like x-rays without the support of reflecting mirrors. The electron beam appears in bunches which are long compared to the desired wavelength. Although the electron distribution is assumed to be uniform, there will be statistical fluctuations due to the finite number of electrons per bunch. This beam is travelling through an undulator of as yet undetermined length. An x-ray single pass FEL consists basically of a high brightness electron source, a linac followed by a long undulator, both with parameters to produce the desired photon radiation wavelength as the fundamental wavelength of the undulator.

In a perfect uniform beam every electron radiates in the undulator at an arbitrary phase resulting in in-coherent radiation. The radiation travels faster than the electrons and therefore the radiation field will interact with them. This interaction however is incoherent and will not lead to anything. Now we assume that along the bunch there is a density fluctuation or whisker which is very short of the order of the desired wavelength. That whisker radiates coherently at wavelengths equal to the temporal length of the whisker. The coherent radiation, although very small at first, interacts with the electron beam. However, only the fundamental wavelength as determined by electron beam energy and undulator properties will constructively grow from undulator period to period. As this fundamental radiation travels over the bunch ahead of the whisker it interacts coherently with the electrons and modulates their energy periodically at the fundamental wavelength. This energy modulation together with the deflection in the undulator leads to a density modulation at the desired wavelength. This process occurs because electrons which have been decelerated by the photon field get deflected more in the undulator field while electrons being accelerated by the photon field get deflected less. Both effects lead to a density modulation.

Such whiskers can and do occur at any place along the bunch. Therefore a number of coherent fields will be created and grow along the undulator. Eventually though there will be a strongest radiation field and all others, being spread over statistical phases, will decoher and vanish in the one largest spike. This spike keeps growing along the undulator and reaches a point from which on the photon field is strong enough to microbunch the electron beam at which point the intensity does not grow anymore. The SASE-FEL has reached its saturation. At the same time the energy change introduced by the photon field is big enough to spoil the SASE principle leading also to saturation. The theory of SASE-FEL has been first developed by [2] as a 1-D theory. Later this was extended to a 3-D theory [3, 4]. We will however restrict ourselves in this text to the 1-D approximation which is well met for a high quality electron beam. Where ideal parameters are not available some degradation of the photon beam parameters must be accepted and the actual characteristics are mostly determined by numerical simulations.

The increase of photon intensity along the undulator is exponential because the bunching depends on the photon intensity itself and is given by

\[I_{\rm ph}\propto I_{0}\exp\left(\frac{z}{L_{\rm G}}\right), \tag{27.61}\]where \(L_{\rm G}\) the power gain length and \(I_{0}\) the initial spontaneous coherent intensity for an undulator of length \(L_{\rm G}\) (26.66). The gain length is defined by

\[L_{\rm G}=\frac{\lambda_{\rm u}}{4\sqrt{3}\pi\rho} \tag{27.62}\]

where \(\rho\) is the FEL-parameter

\[\rho=\left(\frac{K\cdot JJ}{4\sqrt{2}}\frac{\lambda_{\rm u}\Omega_{\rm p}}{2\pi c \gamma}\right)^{2/3}. \tag{27.63}\]

Here the \(JJ\)-function is defined by (26.60) with the argument \(x=\frac{K^{2}}{4+2K^{2}}\), \(\Omega_{\rm p}=\sqrt{4\pi\,c^{2}r_{\rm e}n_{\rm e}\frac{1}{\gamma}}\) is the plasma frequency and \(n_{\rm e}\) the electron density. Tacitly we have assumed a planar undulator which could also generate third harmonic radiation albeit at a lower intensity (\(\sim 1\) %) while a helical undulator would only produce the fundamental harmonic. Numerical simulations indicate that for presently achievable electron beam parameters about 20 gain lengths are required to reach saturation while the FEL-parameter is of the order of \(10^{-3}\).

The peak photon pulse power at saturation is expected [2] to be

\[P_{\rm peak}=\frac{\rho_{\rm eff}N_{\rm e}E}{\sqrt{2\pi\,\tau_{\rm b,rms}}}= \rho_{\rm eff}\,I_{\rm peak}\frac{E}{e}, \tag{27.64}\]

which is about 14 GW for SLAC-LCLS parameters [5] (\(\rho_{\rm eff}=2.9\cdot 10^{-4}\), \(\tau_{\rm b,rms}=77\) fs, \(I_{\rm peak}=3,400\) A, \(E=14.35\) GeV). Simulations give a somewhat lower power of about 8 GW by taking all inefficiencies like increase of beam emittance along linac and undulator into account.

A high photon intensity (27.61) demand a short gain length for a given length of the facility while the gain length (27.62) itself is only related to the undulator period length and FEL-period. The period length is limited to a minimum of a few cm by technical considerations and the available linear accelerator energy and desired radiation wavelength. The FEL-parameter (27.63) is greatly determined by the electron density, e.g. by the electron beam emittance and bunch length. Therefore a small beam emittance and bunch length is of paramount importance. In addition the electron beam emittance must be close to the photon emittance for the desired wavelength to get maximum overlap of both beams. Theoretical considerations also require that the beam energy spread should be less than the FEL-parameter \(\left(\frac{\sigma_{E}}{E}<\rho\right)\).

To make SASE work well, a very high quality electron beam must be produced and preserved along the linac and undulator. In the following sections we will shortly discuss the requirements and the solutions employed in the first few x-ray facilities. The development in this newest accelerator system is still flowing and experimental experience from the first facilities contribute to a vigorous development especially toward more compact solutions.

Different from storage ring which can provide radiation to many users simultaneously an x-ray laser can do so only for one user at one wavelength at a time. This is acceptable because of the extraordinary properties of the radiation in terms of photon intensity, brightness, coherence and femto-second pulse length. A high desired pulse repetition rate obviously pushes the facility designs more and more to superconducting technology.

#### Electron Source

The electron source determines the ultimate performance of the x-ray laser. For maximum radiation intensity the number of electrons per microbunch should be as large as possible. This strongly points to a laser gun where it is possible to generate a large electron charge within a pulse of less than 100 fs. Not to loose spatial coherence the beam emittance must be very small of the order of less than \(10^{-10}\) m at the end of the linac. At source energies of \(\gamma=1\) the lowest possible normalized emittance is \(1-2\)\(10^{-6}\) m at high electron intensity of about 1 pC per bunch. Lower emittances are possible for lower charges. The design requirements are determined by many detailed simulations with specially developed numerical programs to find solutions close to desired performance.

##### Beam Dynamics

Along the linac and undulator the beam should be focused as much as possible to maximize the electron density. However, if the beam size is too small diffraction effects will appear. Therefore there is an optimum beam size which can be realized by quadrupole focusing in a FODO channel. Numerical simulations are required to determine the optimum beam size for the parametrization of the FODO channel. The focusing requirements must also include the effect of beam steering which is stronger in strong focusing FODO channels.

To reach a realistic gain length a high peak current or short bunch length in the fs regime must be achieved. That is not possible with present day technology and bunch compression schemes must be included in the beam dynamics design.

In the low energy section of the linac (up to 200-300 MeV) the electron bunch is accelerated "off-crest" to obtain a mostly linear correlation of energy with phase along the electron bunch in preparation for the bunch compression system. There is a small non-linearity left from the sinusoidal variation of the acceleration field. Simulations show that part of this non-linearity can be eliminated by deceleration in a higher harmonic accelerating section. If the main linac operates at 3 GHz then this linearizing section could operate in the X-band or about 12 GHz where suitable power sources exist [6]. The decelerating in the X-band section is small and has no detrimental effect on the overall beam dynamics. After passing through the X-band section the beam travels through a four-bend bunch compressor.

Following Liouville's theorem the bunch compression is obtained in exchange with energy spread. In order not to increase the energy spread too much two bunch compression systems must be employed to reach the desired short bunch length. After the first bunch compression the beam is further accelerated (still "off-crest") to a higher energy and the energy spread is reduced due to adiabatic damping. At some intermediate energy a second four-bend bunch compressor is installed. The choice of the intermediate energy should be chosen such that the remaining acceleration is enough to reduce the beam energy spread by adiabatic damping to the final value of \(\frac{\Delta E}{E}\sim\rho\) for optimum SASE. No beam heater is necessary here because the bunch length is already much shorter and the non-linearity is very small. Both bunch compressors must be designed such as not to perturb beam parameters like beam emittance too much. The \(R_{56}\)-term is therefore chosen to be about the same in both. Final distribution of compression is determined though by numerical simulations.

The resulting beam after the second bunch compressor includes now a very high peak current which can drive a micro-bunching instability [7; 8] thus possibly ruining emittance and energy spread. Therefore a "beam heater" is installed just before the second bunch compressor. This beam heater is a short and strong wiggler magnet which by emission of synchrotron radiation increases the incoherent energy spread. While this seems to be the wrong method only a very small insignificant increase of the energy spread is required to suppress the instability. After the second bunch compressor the beam is ready to be accelerated "on-crest" to the final energy.

A significant problem arises if the bunch length is reduced too much such that coherent radiation (CSR) can be emitted with detrimental effect on the beam emittance and energy spread. Other problems arise from the interaction of the beam with surface resistance of the vacuum chamber (resistive wall effect). As a consequence vacuum chamber materials with low wall resistance should be used, e.g. aluminum rather than steel or copper-plated steel. In addition the surface must be polished to reduce the roughness which can cause beam instabilities. Satisfactory polishing to a roughness of well below 100 nm must be followed.

The specific techniques described here are not the only way to solve problems. The interested reader is therefore encouraged to review the design reports of various X-FEL facilities. The performance is determined by simulation of the electron beam propagating through linac and undulator as well as the simulation of the photon built-up in the undulator. While it is possible to calculate order of magnitude parameters theoretically, small effects from actual particle distribution from source to end can significantly affect the outcome. Therefore the whole process must be simulated and any undesirable effect be studied and possibly eliminated or corrected.

#### Undulator

The undulator parameters determine together with the electron beam energy the wavelength of the photon beam. This is the fundamental wavelength of the undulator and at reduced intensity one can contemplate the third-harmonic. To reduce the undulator length for a desired wavelength a short period length is desired. There are, however, technical limits for period length below 2-3 cm. When the gap aperture becomes close to the period length the field drops off rapidly. The period length in the SLAC-LCLS is 3 cm and the desired undulator strength \(K=3.71\) which requires an electron beam energy of 14.35 GeV to reach a fundamental wavelength of 1.5 A. This high undulator strength requires a very small aperture of 6 mm which is acceptable for a linac beam because no Gaussian tails must be preserved for lifetime.

The built-up of photon intensity occurs exponentially from noise and therefore many gain length are needed to get the intensity into desired values. In other words, the undulator must be very long. In the SLAC-LCLS case the undulator length is 120 m long of which theoretically 91 m are required to reach saturation. Such a long undulator cannot be built in one piece and is therefore broken down into shorter pieces of, in this case, 3.42 m. This breakup allows some space for beam monitoring and beam control.

## Problems

### 27.1 (S)

Consider an electron travelling through an undulator producing radiation. Show, that the radiation front moves ahead of the electron by one fundamental radiation wavelength per undulator period.

### 27.2 (S)

Why does a helical undulator not produce higher harmonics?

### 27.3 (S)

From the peak power at saturation derive the number of x-ray photons (\(\varepsilon_{\mathrm{x}}\sim 10^{4}\) eV) per electron. For the SLAC-LCLS \(K=3.711\), \(\lambda_{\mathrm{p}}=3\) cm and \(N_{\mathrm{p}}=3,070\). Compare this with incoherent radiation. For the band width use \(\frac{\Delta\omega}{\omega}=\frac{1}{N_{\mathrm{p}}}\).

## Bibliography

* [1] W. Colson, Phys. Rev. **64A**, 190 (1977)
* [2] R. Bonifacio, C. Pellegrini, L. Narducci, Opt. Commun. **50**, 373 (1984)
* [3] K.-J. Kim, Three-dimensional analysis of coherent amplification and self-amplified spontaneous emission in free electron lasers. Phys. Rev. Lett. **57**, 1871 (1986)
* [4] Y.H. Chin, K.-J. Kim, M. Xie, Three-dimensional free electron laser theory including betatron oscillations. Phys. Rev. **A46**, 6662 (1992)* [5] Linac Coherent Light Source (LCLS), Conceptual Design Report, Technical Report SLAC-R-593, SLAC, Stanford (2002)
* [6] P. Emma, Technical Report LCLS TN-01-01, SLAC, Stanford (2001)
* [7] S. Heifets, S. Krinsky, G. Stupakov, Csr instability in a bunch compressor. Technical Report SLAC-PUB-9165, SLAC, Stanford (2002)
* [8] E.L. Saldin, E.A. Schneidmiller, M.V. Yurkov, Longitudinal phase space distortions in magnetic bunch compressors. Technical Report DESY 01-129, DESY, Hamburg (2001)

