## Chapter 9 Longitudinal Beam Dynamics

In previous chapters we have concentrated the discussion on the interaction of transverse electrical and magnetic fields with charged particles and have derived appropriate formalisms to apply this interaction to the design of beam transport systems. The characteristics of these transverse fields is that they allow to guide charged particles along a prescribed path but do not contribute directly to the energy of the particles through acceleration. For particle acceleration we must generate fields with nonvanishing force components in the direction of the desired acceleration. Such fields are called longitudinal fields or accelerating fields. In a very general way we describe in this section the interaction of longitudinal electric fields with charged particles to derive the process of particle acceleration, its scaling laws, and its stability limits.

The usefulness and application of electric fields to accelerate charged particles depends greatly on the temporal variations of these fields. Accelerating fields can be static or pulsed or they may be electromagnetic fields oscillating at high frequencies. Conceptually, the most simple way to accelerate charged particles is through a static field applied to two electrodes as shown in Fig. 9.1. In this case, the total kinetic energy a particle can gain while traveling from one electrode to the other is equal to the product of the particle charge and the voltage between the electrodes.

Electric breakdown phenomena, however, limit the maximum applicable voltage and thereby the maximum energy gain. Nonetheless, this method is intriguingly simple and efficient compared to other accelerating methods and therefore still plays a significant role among modern particle accelerators, for example, in particle sources. Electrostatic acceleration schemes are specifically useful for low energy particles for which other methods of acceleration would be inefficient. Higher voltages and particle energies can be reached if the electric fields are applied in the form of very short pulses. Application of electro-static high voltages to accelerate particles is limited to some 10 million volts due to high voltage breakdown.

For higher particle energies different acceleration methods must be used. The most common and efficient way to accelerate charged particles to high energies is to use high frequency electromagnetic fields in specially designed accelerating structures. Acceleration to high energies occurs while charged particles either pass once through many or many times through one or few accelerating structures each excited to electric field levels below the break down threshold. In this section, we concentrate the discussion on charged particle acceleration by electromagnetic radio frequency fields.

### Longitudinal Particle Motion

Application of radio frequency in short rf-fields has become exceptionally effective for the acceleration of charged particles. Both, fields and particle motion can be synchronized in an effective way to allow the acceleration of charged particles in principle to arbitrary large energies were it not for other limitations.

The first idea and experiment for particle acceleration with radio frequency fields has been published by Ising [1] although he did not actually succeed to accelerate particles due to an inefficient approach to rf-technology. Later Wideroe [2] introduced the concept of generating the accelerating fields in resonating rf-cavities and was able to accelerate heavy ions. Original papers describing these and other early developments of particle acceleration by rf-fields are collected in a monogram edited by Livingston [3].

To study the interaction of electromagnetic rf-fields with charged particles, we assume a plane electromagnetic wave of frequency \(\omega\) propagating in the \(z\)-direction. A free electromagnetic wave does not have a longitudinal electric field component and therefore a special physical environment, called the accelerating structure, must be provided to generate accelerating field components in the direction of propagation. As will be discussed later in Sect. 18.1 this is achieved by proper choice of boundary conditions. To study particle dynamics in longitudinal fields,

Figure 9.1: Principle of electrostatic accelerators

we assume that we were able to generate rf-fields with an electric field component along the path of the particles expressed by \(v\,\approx\,c\)

\[\mathbf{E}(z,t)=\mathbf{E}_{0}\,\mathrm{e}^{\mathrm{i}(\omega t-kz)}=\mathbf{E}_{0}\,\mathrm{ e}^{\mathrm{i}\psi}\, \tag{9.1}\]

where the phase \(\psi\,=\,\omega t-kz\). The particle momentum changes at a rate equal to the electric force exerted on the particle by the rf-field

\[\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t}=e\,\mathbf{E}(\psi)\,=\,\frac{\mathrm{d}}{ \mathrm{d}t}(\gamma mc\mathbf{\beta}). \tag{9.2}\]

Multiplying this with the particle velocity we get the rate of change of the kinetic energy, \(\mathrm{d}E_{\mathrm{kin}}=c\beta\,\mathrm{d}p\). Integration of (9.2) with respect to the time becomes unnecessarily complicated for general fields because of the simultaneous variation of the electric field and particle velocity with time. We therefore integrate (9.2) with respect to the longitudinal coordinate and obtain instead of the momentum gain the increase in the kinetic or total energy for the complete accelerating structure

\[\Delta E=(\gamma-\gamma_{0})\,mc^{2}=e\int\mathbf{E}(\psi)\,\mathrm{d}z\, \tag{9.3}\]

where \(\gamma_{0}\,mc^{2}\) is the energy of the particle before acceleration. Of course, the trick to integrate the electric field through the accelerating section rather than over time following the particle is only a conceptual simplification and the time integration will have to be executed at some point. Generally this is done when the particular accelerating section, the fields, and the synchronization is known.

Travelling electromagnetic waves are used in linear accelerators and the accelerating structure is designed such that the phase velocity of the wave is equal to the velocity of the particles to be accelerated. In this case, the particle travels along the structure in synchronism with the wave and is therefore accelerated or decelerated at a constant rate. Maximum acceleration is obtained if the particles ride on the crest of the wave.

In a standing wave accelerating section the electric field has the form

\[\mathbf{E}(z,t)=\mathbf{E}_{0}(z)\,\mathrm{e}^{\mathrm{i}\omega t+\delta}\, \tag{9.4}\]

where \(\delta\) is the phase at the moment the particle enters the accelerating section at \(t=0\). When we refer to an accelerating voltage \(V\) in a standing wave cavity we mean to say a particle traveling close to the speed of light through the cavity will gain a maximum kinetic energy of \(eV\) while passing the cavity center at the moment the field reaches its crest. Such a particle would enter the cavity some time before the field reaches a maximum and will exit when the field is decaying again. For slower particles the energy gain would be lower because of the longer transit time.

#### Longitudinal Phase Space Dynamics

Successful particle acceleration depends on stable and predictable interaction of charged particles and electromagnetic fields. Because oscillating rf-fields are used, special criteria must be met to assure systematic particle acceleration rather than random interaction with rf-fields producing little or no acceleration. The constructive interaction of particles and waves have been investigated in 1945 independently by Veksler [4] and McMillan [5] leading to the discovery of the fundamental principle of phase focusing. In this subsection, we will derive the physics of phase focusing and apply it to the design of particle accelerators.

The degree of acceleration depends on the momentary phase \(\psi\) of the field as seen by the particle while travelling through or with an electromagnetic field. Straight superposition of an electromagnetic wave and charged particle motion will not necessarily lead to a net acceleration. In general, the particles are either too slow or too fast with respect to the phase velocity of the wave and the particle will, during the course of interaction with the electromagnetic wave, integrate over a range of phases and may gain little or no net energy from the electric fields. Therefore, special boundary conditions for the accelerating rf-wave must be met such that maximum or at least net acceleration can be achieved. This can be done by exciting and guiding the electromagnetic waves in specially designed accelerating structures designed such that the phase velocity of the electromagnetic wave is equal to the particle velocity. Only then can we choose a specific phase and integration of (9.3) becomes straightforward for particles travelling in the direction of propagation of the electromagnetic waves.

For practical reasons, specifically in circular accelerators, particle acceleration occurs in short, straight accelerating sections placed along the particle path. In this case no direct traveling wave exists between adjacent accelerating sections and specific synchronicity conditions must be met for the fields in different accelerating sections to contribute to particle acceleration as desired. For the purpose of developing a theory of stable particle acceleration we may imagine an rf-wave traveling along the path of the particle with a phase velocity equal to the particle velocity and an amplitude which is zero everywhere except in discrete accelerating cavities.

To ensure proper synchronization one could assume that every rf-cavity is powered by its own microwave source. This is done often in high power rf-cavities, but is, for example, impractical in linear accelerators. For the case of individual power sources the phase of the rf-field can be chosen in each cavity such that its voltage reaches the desired value at the moment the particles pass through. The synchronisation for many cavities fed by one power source is more complicated and we will discuss in the following paragraphs how to accomplish this.

We consider a number of rf-cavities powered by a single microwave source. To derive the synchronicity conditions, we consider first two accelerating sections separated by the distance \(L\) as shown in Fig. 9.2. Once the proper operating conditions are known for two sections a third section may be added by applyingthe same synchronicity condition between each pair of cavities. The successive accelerating sections need not necessarily be a physically different sections but could be the same section or the same sections passed through by the particles at periodic time intervals. For example, the distance between successive accelerating sections may be equal to the circumference of a circular accelerator.

For systematic acceleration the phase of the rf-fields in each of the accelerating sections must reach specific values at the moment the particles arrive. If the phase of the fields in each of accelerating sections is adjusted to be the same at the time of arrival of the particles, the total acceleration is times the acceleration in each individual section. This phase is called the synchronous phase defined by

(9.5)

where is the oscillating frequency of the electromagnetic field. The time derivative of (9.5) vanishes and the synchronicity condition is

(9.6)

since. This condition can be met if we set

(9.7)

and the frequency of the electromagnetic field is then from (9.6)

(9.8)

where is the lowest frequency satisfying the synchronicity condition and is the time needed for particles with velocity to travel the distance. This equation relates the time of travel between successive accelerating sections with the frequency of the accelerating rf-fields in a conditional way to assure systematic

Figure 9.2: Discrete accelerating sections

particle acceleration and the relation (9.8) is therefore called the synchronicity condition.

However, any integer multiple of the frequency \(\omega_{1}\) satisfies the synchronicity condition as well and we may instead of (9.8) define permissible frequencies of the accelerating rf-fields by

\[\omega_{h}=h\,\omega_{1}=k_{h}\,\beta c=\frac{2\pi}{L}\,h\,\beta c=\,\frac{2\pi }{\Delta T}\,h\,, \tag{9.9}\]

where \(h\) is an integer called the harmonic number with \(k_{h}=h\,k_{1}\).

The synchronicity condition must be fulfilled for any spatial arrangement of the accelerating structures which are powered by a single microwave source to get the maximum acceleration. To illuminate the principle, we assume here, for example, a series of short, equidistant accelerating gaps or accelerating sections along the path of a particle. Let each of these gaps be excited by its own power source to produce an accelerating rf-field at some random phase. The synchronicity condition (9.8) is fulfilled if the rf-frequency is the same in each of these gaps, which are separated by an integer multiple of rf-wavelength. However, it does not require each accelerating gap to have the same rf-phase at the arrival time of the particles. Each cavity in a set of accelerating cavities oscillating at the same frequency may be tuned to an arbitrary rf-phase and the synchronicity condition still would be met. From a practical point of view, however, it is inefficient to choose arbitrary phases and it is more reasonable to adjust the phase in each cavity to the optimum phase desired for maximum acceleration.

The assumption that the rf-frequency of all cavities be the same is unnecessarily restrictive considering that any harmonic of the fundamental frequency is acceptable. Therefore, a set of accelerating cavities in a circular accelerator, for example, may include cavities resonating at any harmonic of \(\omega_{1}\). This is sometimes done to achieve specific effects (e.g. bunch lengthening), but in the absence of such requirements makes only complicates the Rf-system.

A straightforward application of the synchronicity condition can be found in the design of the Wideroe linear accelerator structure [2] as shown in Fig. 9.3. Here the fields are generated by an external rf-source and applied to a series of metallic drift tubes. Accelerating fields build up at gaps between the tubes while the tubes themselves serve as a field screens for particles during the time the electric fields is changing sign and would be decelerating. The length of the field free drift tubes is determined by the velocity of the particles and is \(L=c\,\beta\,T_{\rm rf}\) where \(T_{\rm rf}\) is the period of the rf-field. As the particle energy increases so does the velocity \(c\,\beta\) and the length \(L\) of the tube must increase too. Only when the particles become highly relativistic will the distance between field free drift sections become a constant together with the velocity of the particles. Structures with varying drift lengths are generally found in low energy proton or ion accelerators based on the Alvarez structure [6], which is a technically more efficient version of the Wideroe structure.

For electrons it is much easier to reach relativistic energies where the velocity is sufficiently constant such that in general no longitudinal variation of the accelerating structure is needed. In circular accelerators, we cannot adjust the distance between cavities or the circumference as the particle velocity \(\beta\) increases. The synchronicity condition therefore must be applied differently. From (9.9) we find the rf-frequency to be related to the particle velocity and distances between cavities. Consequently we have the relation

\[\beta\lambda_{\mathrm{rf}}h=L, \tag{9.10}\]

which requires that the distance between any pair of accelerating cavities be an integer multiple of \(\beta\lambda_{\mathrm{rf}}\). Since \(L\) and \(h\) are constants, this condition requires that the rf-frequency be changed during acceleration proportional to the particle velocity \(\beta\). Only for particles reaching relativistic energies, when \(\beta\approx 1\), will the distance between cavities approach an integer multiple of the rf-wave length and the circumference \(C\) must then meet the condition

\[C=\beta h\lambda_{\mathrm{rf}}. \tag{9.11}\]

### Equation of Motion in Phase Space

So far, we have assumed that both the particle velocity \(\beta\) and the wave number \(k\) are constant. This is not a valid general assumption. For example, we cannot assume that the time of flight from one gap to the next is the same for all particles. For low energy particles we have a variation of the time of flight due to the variation of the particle velocities for different particle momenta. The wave number \(k\) or the distance between accelerating sections need not be the same

Figure 9.3: Wideroe linac structure

for all particles either. A momentum dependent path length between accelerating sections exists if the lattice between such sections includes bending magnets. As a consequence, the synchronicity condition must be modified to account for such chromatic effects.

Removing the restriction of a constant wave number \(k\), we obtain by a variation of (9.6)

\[\Delta\dot{\psi}=\dot{\psi}-\dot{\psi}_{\rm s}=-\Delta(k\beta c)=-ck\Delta \beta-\beta c\frac{\partial k}{\partial p}\frac{\partial p}{\partial t}\Delta t\,, \tag{9.12}\]

where

\[k=k_{\rm h}=h\,\frac{2\pi}{L_{0}}=\frac{2\pi}{\lambda_{\rm rf}}=h\,\frac{ \omega}{\beta c}, \tag{9.13}\]

and \(L_{0}\) is the distance between accelerating gaps along the ideal path. The synchronous phase is kept constant \(\psi_{\rm s}=\rm const\) or \(\dot{\psi}_{\rm s}=0\) and serves as the reference phase against which all deviations are measured.

The momentum dependence of the wave number comes from the fact that the path length \(L\) between accelerating gaps may be different from \(L_{0}\) for off momentum particles. The variation of the wave number with particle momentum is therefore

\[\frac{\partial k}{\partial p}\bigg{|}_{0}=\left.\frac{\partial k}{\partial L }\frac{\partial L}{\partial p}\right|_{0}=-\frac{k_{\rm h}}{L_{0}}\left.\frac{ \partial L}{\partial p}\right|_{0}=-\frac{k_{\rm h}}{p_{0}}\alpha_{\rm c}\,, \tag{9.14}\]

where \(\alpha_{\rm c}\) is the momentum compaction factor. We evaluate the momentum compaction factor starting from the path length \(L=\int_{0}^{L_{0}}(1+\frac{x}{\rho})\,\mathrm{d}z\). For transverse particle motion \(x=x_{\beta}+\eta\,(\Delta p/p_{0})\) and employing average values of the integrands the integral becomes

\[L=L_{0}+\left\langle\frac{x_{\beta}}{\rho}\right\rangle L_{0}+\left\langle \frac{\eta}{\rho}\right\rangle\frac{\Delta p}{p_{0}}L_{0}. \tag{9.15}\]

Because of the oscillatory character of the betatron motion \(\langle\kappa_{x}\,x_{\beta}\rangle=0\). The relative path length variation is \(\frac{\Delta L}{L_{0}}=\left\langle\frac{\eta}{\rho}\right\rangle\frac{\Delta p }{p_{0}}=\alpha_{\rm c}\frac{\Delta p}{p_{0}}\) and the momentum compaction factor becomes

\[\alpha_{\rm c}=\left\langle\frac{\eta}{\rho}\right\rangle. \tag{9.16}\]

The momentum compaction factor increases only in curved sections where \(\rho\neq 0\) and the path length is longer or shorter for higher energy particles depending on the dispersion function being positive or negative, respectively. For a linear accelerator the momentum compaction factor vanishes since the length of a straight line does not depend on the momentum. With \((\partial p/\partial t)\Delta t=\Delta p\) and \(mc\gamma^{3}\Delta\beta=\Delta p\) we get finally for (9.12) with (9.14) and after some manipulation

\[\dot{\psi} = -\beta ck_{\rm h}(\gamma^{-2}-\alpha_{\rm c})\,\frac{\Delta cp}{cp_ {0}}. \tag{9.17}\]

The term \(\gamma^{-2}\) in (9.17) appears together with the momentum compaction factor \(\alpha_{\rm c}\) and therefore has the same physical relevance. This term represents the variation of the particle velocity with energy. Therefore, even in a linear accelerator where \(\alpha_{\rm c}=0\), the time of flight between accelerating gaps is energy dependent as long as particles are still nonrelativistic.

After differentiation of (9.17) with respect to the time, we get the equation of motion in the longitudinal direction describing the variation of the phase with respect to the synchronous phase \(\psi_{\rm s}\) for particles with a total momentum deviation \(\Delta p\)

\[\ddot{\psi} + \frac{\partial}{\partial t}\left(\beta ck_{\rm h}\eta_{\rm c} \frac{\Delta cp}{cp_{0}}\right)=0. \tag{9.18}\]

In most practical applications, parameters like the particle velocity \(\beta\) or the energy vary only slowly during acceleration compared to the rate of change of the phase and we consider them for the time being as constants. The slow variation of these parameters constitutes an adiabatic variation of external parameters for which Ehrenfest's theorem holds. The equation of motion in the potential of the rf-field becomes in this approximation

\[\ddot{\psi} + \frac{\beta ck_{\rm h}\eta_{\rm c}}{cp_{0}}\frac{\partial}{ \partial t}\Delta cp=0. \tag{9.19}\]

Integration of the electrical fields along the accelerating sections returns the kinetic energy gain per turn

\[e\int_{L}\mathbf{E}(\psi){\rm d}\mathbf{z} = eV(\psi), \tag{9.20}\]

where \(V(\psi)\) is the total particle accelerating voltage seen by particles along the distance \(L\). For particles with the ideal energy and following the ideal orbit the acceleration is \(eV(\psi_{\rm s})\) where \(\psi_{\rm s}\) is the synchronous phase.

Acceleration, however, is not the only source for energy change of particles. There are also gains or losses from, for example, interaction with the vacuum chamber environment, external fields like free electron lasers, synchrotron radiation or anything else exerting longitudinal forces on the particle other than accelerating fields. We may separate all longitudinal forces into two classes, one for which the energy change depends only on the phase of the accelerating fields \(V(\psi)\) and the other where the energy change depends only on the energy of the particle \(U(E)\)itself. The total energy gain \(\Delta E\) per unit time or per turn is the composition of both types of external effects

\[\Delta E=eV(\psi)-U(E), \tag{9.21}\]

where \(U(E)\) is the energy dependent loss per turn due, for example, to synchrotron radiation.

#### Small Oscillation Amplitudes

For arbitrary variations of the accelerating voltage with time we cannot further evaluate the equation of motion unless the discussion is restricted to small variations in the vicinity of the synchronous phase. While the ideal particle arrives at the accelerating cavities exactly at the synchronous phase \(\psi_{\rm s}\), most other particles in a real beam arrive at slightly different phases. For small deviations \(\varphi\) from the synchronous phase,

\[\varphi=\psi-\psi_{\rm s}\,, \tag{9.22}\]

we can expand the accelerating voltage into a Taylor series at \(\psi=\psi_{\rm s}\) and get for the average rate of change of the particle energy with respect to the energy of the synchronous particle from (9.20)

\[\frac{\rm d}{\rm d\,t}\,\Delta E=\left.\frac{1}{T_{0}}\left[eV(\psi_{\rm s})+e \,\,\frac{\rm d\,V}{\rm d\psi}\right|_{\psi_{\rm s}}\varphi-U(E_{0})-\left. \frac{\rm d\,U}{\rm d\,E}\right|_{E_{0}}\Delta E\right], \tag{9.23}\]

where the particle energy \(E=E_{0}+\Delta E\) and \(T_{0}\) is the time of flight between adjacent cavities for the reference particle

\[T_{0}=\frac{L_{0}}{\beta c}. \tag{9.24}\]

At equilibrium \(eV(\psi_{\rm s})=U(E_{0})\), and since \(\beta\,\,\Delta cp=\Delta E\), we get with (9.23) and \(\vec{\varphi}=\vec{\psi}\,\) from (9.19) the equation of motion or phase equation

\[\vec{\varphi}\,+\left.\frac{c\,k_{\rm h}\,\eta_{\rm c}}{cp_{0}\,T_{0}}\,e\,\, \left.\frac{\rm d\,V}{\rm d\psi}\right|_{\psi_{\rm s}}\varphi\,+\left.\frac{1} {T_{0}}\,\,\frac{\rm d\,U}{\rm d\,E}\right|_{E_{0}}\frac{\Delta cp}{cp_{0}}=0\,. \tag{9.25}\]

With (9.17) and \(\psi\,=\,\psi_{\rm s}\,+\,\varphi\) Eq. (9.25) becomes the differential equation of motion for small phase oscillations

\[\vec{\varphi}\,+\,2\alpha_{z}\dot{\varphi}+\Omega^{2}\varphi=0\,, \tag{9.26}\]where the damping decrement \(\alpha_{z}\) is defined by

\[\alpha_{z}=-\frac{1}{2T_{0}}\left.\frac{\mathrm{d}U}{\mathrm{d}E}\right|_{E_{0}} \tag{9.27}\]

and the synchrotron frequency by

\[\Omega^{2}=\left.\frac{c\,k_{\mathrm{n}}\,\eta_{c}}{cp_{0}\,T_{0}}e\ \frac{\mathrm{d}V}{\mathrm{d}\psi}\right|_{\psi_{\mathrm{s}}}. \tag{9.28}\]

Particles orbiting in a circular accelerator perform longitudinal oscillations with the frequency \(\Omega\). These phase oscillations are damped or antidamped depending on the sign of the damping decrement. Damping occurs only if there is an energy loss which depends on the particle energy itself as in the case of synchrotron radiation. In most cases of accelerator physics we find the damping time to be much longer than the phase oscillation period and we may therefore discuss the phase equation while ignoring damping terms. Whenever damping becomes of interest, we will include this term again.

This phase equation is valid only for small oscillation amplitudes because only the linear term has been used in the expansion for the rf-voltage. For larger amplitudes this approximation cannot be made anymore and direct integration of the differential equation is necessary. The small amplitude approximation, however, is accurate to describe most of the fundamental features of phase oscillations. At large amplitudes, the nonlinear terms will introduce a change in the phase oscillation frequency and finally a limit to stable oscillations to be discussed later in this chapter.

The phase equation has the form of the equation of motion for a damped harmonic oscillator and we will look for conditions leading to a positive frequency and stable phase oscillations. Because the phase equation was derived first for synchrotron accelerators the oscillations are also called synchrotron oscillations and are of fundamental importance for beam stability in all circular accelerators based on rf-acceleration. For real values of the oscillation frequency we find that particles which deviate from the synchronous phase are subjected to a restoring force leading to harmonic oscillations about the equilibrium or synchronous phase. From the equation of motion (9.25) it becomes clear that phase focusing is proportional to the derivative of the accelerating voltage rather than to the accelerating voltage itself and is also proportional to the momentum compaction \(\eta_{\mathrm{c}}\).

To gain further insight into the phase equation and determine stability criteria, we must make an assumption for the waveform of the accelerating voltage. In most cases, the rf-accelerating fields are created in resonant cavities and therefore the accelerating voltage can be expressed by a sinusoidal waveform

\[V(\psi)=\hat{V}_{0}\sin\psi \tag{9.29}\]and expanded about the synchronous phase to get with \(\psi=\psi_{\rm s}+\varphi\)

\[V(\psi_{\rm s}+\varphi)=\hat{V}_{0}\,(\sin\psi_{\rm s}\cos\varphi+\sin\varphi \cos\psi_{\rm s}). \tag{9.30}\]

Keeping only linear terms in \(\varphi\) the phase equation is

\[\vec{\varphi}+\Omega^{2}\varphi=0, \tag{9.31}\]

where the synchrotron oscillation frequency becomes now

\[\Omega^{2}=\frac{ck_{\rm h}\eta_{\rm c}}{cp_{0}T_{0}}e\hat{V}_{0}\cos\psi_{\rm s}. \tag{9.32}\]

A particle passing periodically through localized and synchronized accelerating fields along its path performs synchrotron oscillations with the frequency \(\Omega\) about the synchronous phase.

In circular accelerators we have frequently the situation that several rf-cavities are employed to provide the desired acceleration. The reference time \(T_{0}\) is most conveniently taken as the revolution time and the rf-voltage \(\hat{V}_{0}\) is the total accelerating voltage seen by the particle while orbiting around the ring once. The rf-frequency is an integer multiple of the revolution frequency,

\[f_{\rm rf}=hf_{\rm rev}, \tag{9.33}\]

where the integer \(h\) is the harmonic number and the revolution frequency is with the circumference \(C\)

\[f_{\rm rev}=\frac{1}{T_{0}}=\frac{C}{c\beta}. \tag{9.34}\]

From (9.32) the synchrotron frequency is in more practical units

\[\Omega^{2}=\omega_{\rm rev}^{2}\frac{h\eta_{\rm c}e\hat{V}_{0}\cos\psi_{\rm s} }{2\pi\ \beta cp_{0}}. \tag{9.35}\]

Similar to the betatron oscillation tunes, we define the synchrotron oscillation tune or short the synchrotron tune as the ratio

\[v_{\rm s}=\frac{\Omega}{\omega_{\rm rev}}. \tag{9.36}\]

For real values of the synchrotron oscillation frequency the phase equation assumes the simple form

\[\varphi=\hat{\varphi}\,\cos\,(\Omega\,t+\chi_{\rm i}), \tag{9.37}\]where \(\chi_{i}\) is an arbitrary phase function for the particle \(i\) at time \(t=0\). With \(\dot{\psi}=\dot{\phi}\) we find from (9.17), (9.32) the relation between the momentum and phase deviation for real values of the synchrotron oscillation frequency\(v_{s}\approx 0.001-0.01\)

\[\delta=\frac{\Delta cp}{cp_{0}}=-\frac{\dot{\phi}}{h\omega_{\rm rev}\eta_{c}}= \frac{\Omega\hat{\phi}}{h\omega_{\rm rev}\eta_{c}}\sin{(\Omega t+\chi_{i})}. \tag{9.38}\]

The particle momentum deviation, being the conjugate variable to the phase, also oscillates with the synchrotron frequency about the ideal momentum. Both, the phase and momentum oscillations describe the particle motion in longitudinal phase spaces shown in Fig. 9.4 for stable and unstable synchrotron oscillations, respectively. At the time \(t_{0}\) when in (9.38) the phase \(\Omega t_{0}+\chi_{i}=0\) and we expect the momentum deviation to be zero while the phase reaches the maximum value \(\hat{\phi}\). Thus both oscillations are \(90^{\circ}\) out of phase. Particles with a negative momentum compaction \(\eta_{\rm c}<0\) move clockwise in phase space about the reference point while a positive momentum compaction causes the particles to rotate counter clockwise.

The same process that has led to phase focusing will also provide the focusing of the particle momentum. Any particle with a momentum different from the ideal momentum will undergo oscillations at the synchrotron frequency which are described by \(\delta=-\hat{\delta}\sin{(\Omega t+\chi_{i})}\), where the maximum momentum deviation is related to the maximum phase excursion \(\hat{\phi}\) by

\[\hat{\delta}=\left|\frac{\Omega}{h\omega_{\rm rev}\eta_{\rm c}}\right|\hat{ \phi}. \tag{9.39}\]

By inverse deduction we may express the momentum equation similar to the phase equation (9.31) and get with \(\Delta p/p_{0}=\delta\) the differential equation for the momentum deviation

\[\frac{\mathrm{d}^{2}\delta}{\mathrm{d}t^{2}}+\Omega^{2}\delta=0. \tag{9.40}\]

Figure 9.4: Synchrotron oscillations in phase space for stable motion (\(\Omega^{2}>0\)) (_left_) and for unstable motion (\(\Omega^{2}<0\)) (_right_)

Similar to the transverse particle motion, we eliminate from (9.37), (9.38) the argument of the trigonometric functions to obtain an invariant of the form

\[\frac{\hat{\delta}^{2}}{\hat{\delta}^{2}}\pm\frac{\varphi^{2}}{\hat{\varphi}^{2}}= 1\qquad\text{with}\qquad\hat{\delta}=\frac{\Omega}{h\,\omega_{\text{rev}}}\hat{ \varphi}, \tag{9.41}\]

where the sign is chosen to indicate stable or unstable motion depending on whether the synchrotron oscillation frequency \(\Omega\) is real or imaginary respectively. The trajectories for both cases are shown in Fig. 9.4. Clearly, the case of imaginary values of the synchrotron oscillation frequency leads to exponential growth in the oscillation amplitude.

#### Phase Stability

The synchrotron oscillation frequency must be real and the right-hand side of (9.32) must therefore be positive to obtain stable solutions for phase oscillations. All parameters in (9.32) are positive quantities except for the momentum compaction \(\eta_{\text{c}}\) and the phase factor \(\cos\psi_{\text{s}}\). For low particle energies the momentum compaction is in general positive because \(\gamma^{-2}>\alpha_{\text{c}}\) but becomes negative for higher particle energies. The energy at which the momentum compaction changes sign is called the transition energydefined by

\[\gamma_{\text{tr}}=\frac{1}{\sqrt{\alpha_{\text{c}}}}. \tag{9.42}\]

Since the momentum compaction factor for circular accelerators is approximately equal to the inverse horizontal tune \(\alpha_{\text{c}}\approx v_{x}^{-2}\), we conclude that the transition energy \(\gamma_{\text{tr}}\) is of the order of the tune and therefore in general a small number reaching up to the order of a hundred for very large accelerators. For electrons, the transition energy is of the order of a few MeV and for protons in the GeV regime. In circular electron accelerators the injection energy always is selected to be well above the transition energy and no stability problems occur during acceleration since the transition energy is not crossed. Not so for protons. Proton linear accelerators with an energy of the order of 10 GeV or higher are very costly and therefore protons and ions in general must be injected into a circular accelerator below transition energy.

The synchronous rf-phase must be selected depending on the particle energy being below or above the transition energy. Stable phase focusing can be obtained in either case if the rf-synchronous phase is chosen as follows

\[\begin{array}{ccc}0<\psi_{\text{s}}<\frac{\pi}{2}&\text{for}&\gamma<\gamma_{ \text{tr}},\\ \frac{\pi}{2}<\psi_{\text{s}}<\pi&\text{for}&\gamma>\gamma_{\text{tr}}.\end{array} \tag{9.43}\]In a proton accelerator with an injection energy below transition energy the rf-phase must be changed very quickly when the transition energy is being crossed. Often the technical difficulty of this sudden change in the rf-phase is ameliorated by the use of pulsed quadrupoles [7, 8], which is an efficient way of varying momentarily the momentum compaction factor by perturbing the dispersion function. A sudden change of a quadrupole strength can lower the transition energy below the actual energy of the particle. This helpful "perturbation" lasts for a small fraction of a second while the particles are still being accelerated and the rf-phase is changed. By the time the quadrupole pulse terminates, the rf-phase has been readjusted and the particle energy is now above the unperturbed transition energy.

In general, we find that a stable phase oscillation for particles under the influence of accelerating fields can be obtained by properly selecting the synchronous phase \(\psi_{\rm s}\) in conjunction with the sign of the momentum compaction such that

\[\Omega^{2}>0. \tag{9.44}\]

This is the principle of phase focusing [5] and is a fundamental process to obtain stable particle beams in circular high-energy accelerators. An oscillating accelerating voltage together with a finite momentum compaction produces a stabilizing focusing force in the longitudinal degree of freedom just as transverse magnetic or electric fields can produce focusing forces for the two transverse degrees of freedom. With the focusing of transverse amplitudes we found a simultaneous focusing of its conjugate variable, the transverse momentum. The same occurs in the longitudinal phase where the particle energy or the energy deviation from the ideal energy is the conjugate variable to the time or phase of a particle. Both variables are related by (9.17) and a focusing force not only exists for the phase or longitudinal particle motion but also for the energy keeping the particle energy close to the ideal energy.

Focusing conditions have been derived for all six degrees of freedom where the source of focusing originates either from the magnet lattice for transverse motion or from a combination of accelerating fields and a magnetic lattice property for the energy and phase coordinate. The phase stability can be seen more clearly by observing the particle trajectories in phase space. Equation (9.31) describes the motion of a pendulum with the frequency \(\Omega\) which, for small amplitudes \(\sin\varphi\approx\varphi\) becomes equal to the equation of motion for a linear harmonic oscillator and can be derived from the Hamiltonian

\[\mathcal{H}=\tfrac{1}{2}\dot{\varphi}^{2}+\tfrac{1}{2}\Omega^{2}\varphi^{2}. \tag{9.45}\]

Small amplitude oscillations in phase space are shown in Fig. 9.4 and we note the confinement of the trajectories to the vicinity of the reference point. In case of unstable motion the trajectories quickly lead to unbound amplitudes in energy and phase (Fig. 9.4 right).

### Large Oscillation Amplitudes

For larger oscillation amplitudes we cannot anymore approximate the trigonometric function \(\sin\varphi\,\approx\,\varphi\) by its argument. Following the previous derivation for the equation of motion (9.31) we get now

\[\ddot{\varphi}=-\Omega^{2}\sin\varphi, \tag{9.46}\]

which can be derived from the Hamiltonian

\[\mathcal{H}=\tfrac{1}{2}\dot{\varphi}^{2}-\Omega^{2}\cos\varphi \tag{9.47}\]

being identical to that of a mechanical pendulum. As a consequence of our ability to describe synchrotron motion by a Hamiltonian and canonical variables, we expect the validity of the Poincare integral

\[J_{1}=\int_{z}\mathrm{d}\dot{\varphi}\mathrm{d}\varphi=\mathrm{const} \tag{9.48}\]

under canonical transformations. Since the motion of particles during synchrotron oscillations can be described as a series of canonical transformations [9], we find the particle density in the \((\,\varphi,\dot{\varphi}\,)\) phase space to be a constant of motion. The same result has been used in transverse phase space and the area occupied by this beam in phase space has been called the beam emittance. Similarly, we define an emittance for the longitudinal phase space. Different choices of canonical variables can be defined as required to emphasize the physics under discussion. Specifically we find it often convenient to use the particle momentum instead of \(\dot{\varphi}\) utilizing the relation (9.17).

Particle trajectories in phase space can be derived directly from the Hamiltonian by plotting solutions of (9.47) for different values of the "energy" \(\mathcal{H}\) of the system. These trajectories, well known from the theory of harmonic oscillators, are shown in Fig. 9.5 for the case of a synchronous phase \(\psi_{\mathrm{s}}=\pi\).

The trajectories in Fig. 9.5 are of two distinct types. In one type the trajectories are completely local and describe oscillations about equilibrium points separated by \(2\pi\) along the abscissa. For the other type the trajectories are not limited to a particular area in phase and the particle motion assumes the characteristics of libration. This phenomenon is similar to the two cases of possible motion of a mechanical pendulum or a swing. At small amplitudes we have periodic motion about the resting point of the swing. For increasing amplitudes, however, that oscillatory motion could become a libration when the swing continues to go over the top. The lines separating the regime of libration from the regime of oscillation are called separatrices.

Particle motion is stable inside the separatrices due to the focusing properties of the potential well which in this representation is just the \(\cos\varphi\)-term in (9.47). The area within separatrices is commonly called an rf-bucket describing a place whereparticles are in stable motion. In Fig. 9.6 the Hamiltonian (9.47) is shown in a three-dimensional representation with contour lines representing the equipotential lines. The stable potential wells, within the separatrices, keeping the particles focused toward the equilibrium position, are clearly visible.

Inside the separatrices the average energy gain vanishes due to oscillatory phase motion of the particles. This is obvious from (9.30) which becomes for \(\psi_{\rm s}=\pi\)

\[V(\psi)=\hat{V}_{0}\sin\psi=\hat{V}_{0}\sin(\psi_{\rm s}+\varphi)=\hat{V}_{0} \sin\varphi \tag{9.49}\]

averaging to zero since the average phase \(\langle\varphi\rangle=0\).

The area within such separatrices is called a stationary rf-bucket. Such buckets, while not useful for particle acceleration, provide the necessary potential well to produce stable bunched particle beams in facilities where the particle energy need not be changed as for example in a proton or ion storage ring where bunched

Figure 9.6: Potential well for stationary rf buckets, \(\psi_{\rm s}=\pi\)

Figure 9.5: Phase space diagrams for a synchronous phase \(\psi_{\rm s}=\pi\)

beams are desired. Whenever particles must receive energy from accelerating fields, may it be for straight acceleration or merely to compensate for energy losses like synchrotron radiation, the synchronous phase must be different from zero. As a matter of fact, due to the principle of phase focusing, particles within the regime of stability automatically oscillate about the appropriate synchronous phase independent of their initial parameters.

In the discussion of large amplitude oscillations we have tacitly assumed that the synchrotron oscillation frequency remains constant and equal to (9.32) yet, we also note that the frequency is proportional to the variation of the rf-voltage with phase and we have included in the definition of the synchrotron frequency only linear terms so far. Specifically, we note in Fig. 9.5 that the trajectories in phase space are elliptical only for small amplitudes but are periodically distorted for larger amplitudes. This distortion leads to a spread of the synchrotron oscillation frequency.

#### Acceleration of Charged Particles

In the preceding paragraph we have arbitrarily assumed that the synchronous phase be zero \(\psi_{\rm s}=0\) and as a result of this choice we obtained stationary, non-accelerating rf-buckets. No particle acceleration occurs since the particles pass through the cavities when the fields are zero. Whenever particle acceleration is required a finite synchronous phase must be chosen. The average energy gain per revolution is then

\[\Delta E=V(\psi_{\rm s})=\hat{V}_{0}\,\sin\psi_{\rm s}. \tag{9.50}\]

Beam dynamics and stability becomes much different for \(\psi_{\rm s}\neq 0\). From (9.19), we get with (9.21), (9.30), (9.32) a phase equation more general than (9.46)

\[\vec{\varphi}+\frac{\Omega^{2}}{\cos\psi_{\rm s}}[\sin(\psi_{\rm s}+\varphi)- \sin\psi_{\rm s}]=0, \tag{9.51}\]

or after expanding the trigonometric term into its components

\[\vec{\varphi}+\frac{\Omega^{2}}{\cos\psi_{\rm s}}(\sin\psi_{\rm s}\cos\varphi +\sin\varphi\cos\psi_{\rm s}-\sin\psi_{\rm s})=0. \tag{9.52}\]

This equation can also be derived directly from the Hamiltonian for the dynamics of phase motion

\[\tfrac{1}{2}\dot{\varphi}^{2}-\frac{\Omega^{2}}{\cos\psi_{\rm s}}[\cos(\psi_{ \rm s}+\varphi)-\cos\psi_{\rm s}+\varphi\sin\psi_{\rm s}]=\mathcal{H}. \tag{9.53}\]The phase space trajectories or diagrams differ now considerably from those in Fig. 9.5 depending on the value of the synchronous phase \(\psi_{\rm s}\). In Fig. 9.7 phase space diagrams are shown for different values of the synchronous phase and a negative value for the momentum compaction \(\eta_{\rm c}\).

We note clearly the reduction in stable phase space area as the synchronous phase is increased or as the particle acceleration is increased. Outside the phase stable areas the particles follow unstable trajectories leading to continuous energy loss or gain depending on the sign of the momentum compaction. Equation (9.53) describes the particle motion in phase space for arbitrary values of the synchronous phase and we note that this equation reduces to (9.45) if we set \(\psi_{\rm s}=\pi\). The energy gain for the synchronous particle at \(\psi=\psi_{\rm s}\) becomes from (9.18)

\[\Delta E=e\int\mathbf{E}(\psi_{\rm s}){\rm d}\mathbf{z}\,. \tag{9.54}\]

We obtain a finite energy gain or loss whenever the synchronous phase in accelerating sections is different from an integer multiple of \(180^{\circ}\) assuming that all accelerating sections obey the synchronicity condition. The form of (9.54) actually is more general insofar as it integrates over all fields encountered along the path of the particle. In case some accelerating sections are not synchronized, the integral collects all contributions as determined by the phase of the rf-wave at the time the particle arrives at a particular section whether it be accelerating or decelerating. The synchronicity condition merely assures that the acceleration in all accelerating sections is the same for each turn.

Particle trajectories in phase space are determined by the Hamiltonian (9.53), which is similar to (9.47) except for the linear term in \(\varphi\). Due to this term, the potential well is now tilted (Fig. 9.8) compared to the stationary case (Fig. 9.6). We still have quadratic minima in the potential well function to provide stable phase oscillations, but particles escaping over the maxima of the potential well will be lost because they continuously loose or gain energy as can be seen by following such trajectories in Fig. 9.9. This is different from the case of stationary buckets where such a particle would just wander from bucket to bucket while staying close to the ideal energy at the center of the buckets. Phase stable regions in case of finite values of the synchronous phase are called moving rf-buckets.

The situation is best demonstrated by the three diagrams in Fig. 9.9 showing the accelerating field, the potential, and the phase space diagram as a function of the phase for different synchronous phases.

In this particular case we have assumed that the particle energy is above transition energy and that the synchronous phase is such that \(\cos\psi_{\rm s}<0\) to obtain stable synchrotron oscillations. The center of the bucket is located at the synchronous phase \(\psi_{\rm s}\) and the longitudinal stability range is limited by the phases \(\psi_{1}\) and \(\psi_{2}\). In the next section we will derive analytical expressions for the longitudinal stability limit and use the results to determine the momentum acceptance of the bucket as well.

While both phases, \(\psi_{\rm s}\) as well as \(\pi-\psi_{\rm s}\), would supply the desired energy gain only one phase provides stability for the particles. The stable phase is easily chosen by noting that the synchrotron oscillation frequency \(\Omega\) must be real and therefore \(\eta_{\rm c}\,\cos\psi_{\rm s}\,>\,0\). Depending on such operating conditions the rf-bucket has different orientations as shown in Fig. 9.10.

We still can choose whether the electric field should accelerate or decelerate the beam by choosing the sign of the field. For the decelerating case which, for example, is of interest for free electron lasers, the "fish" like buckets in the phase space diagram are mirror imaged.

Figure 9.9: Phase space focusing for moving rf buckets displaying the phase relationship of accelerating field, potential, and rf bucket

### Longitudinal Phase Space Parameters

We will here investigate in more detail specific properties and parameters of longitudinal phase space motion. From these parameters it will be possible to define stability criteria.

#### Separatrix Parameters

During the discussions of particle dynamics in longitudinal phase space we found specific trajectories in phase space, called separatrices which separate the phase stable region from the region where particles follow unstable trajectories leading away from the synchronous phase and from the ideal momentum. Within the phase stable region particles perform oscillations about the synchronous phase and the ideal momentum. This "focal point" in the phase diagram is called a stable fixed point (sfp). The unstable fixed point (ufp) is located where the two branches of the separatrix cross. The location of fixed points can be derived from the two conditions:

\[\frac{\partial\mathcal{H}}{\partial\dot{\psi}}=0\qquad\text{and}\qquad\frac{ \partial\mathcal{H}}{\partial\psi}=0. \tag{9.55}\]

From the first condition, we find with (9.53) that \(\dot{\psi}_{\text{f}}=0\) independent of any other parameter. All fixed points are therefore located along the \(\psi\)-axis of the phase diagram as shown in Fig. 9.11.

The second condition leads to the actual location of the fixed points \(\psi_{\text{f}}\) on the \(\psi\)-axis and is with \(\psi=\psi_{\text{s}}+\varphi\)

\[\sin\psi_{\text{f}}-\sin\psi_{\text{s}}=0. \tag{9.56}\]

Figure 9.10: Relationship between rf phase and orientation of moving rf buckets for accelerating as well as decelerating fieldsThis equation can be solved for \(\psi_{\rm f}=\psi_{\rm s}\) or \(\psi_{\rm f}=\pi-\psi_{\rm s}\) and the coordinates of the fixed points are

\[\begin{array}{ll}(\psi_{\rm sf},\dot{\psi}_{\rm sf})=(\psi_{\rm s},0)&\mbox{ for the stable fixed point, $\rm sf$p$, and}\\ (\psi_{\rm uf},\dot{\psi}_{\rm uf})=(\pi-\psi_{\rm s},0)&\mbox{for the unstable fixed point, $\rm ufp$.}\end{array} \tag{9.57}\]

The distinction between a stable and unstable fixed point is made through the existence of a minimum or maximum in the potential at these points respectively. In Fig. 9.9, this distinction becomes obvious where we note the stable fixed points in the center of the potential minima and the unstable fixed points at the saddle points. The maximum stable phase elongation or bunch length is limited by the separatrix and the two extreme points \(\psi_{1}\) and \(\psi_{2}\) which we will determine in Sect. 9.3.3.

#### Momentum Acceptance

Particles on trajectories just inside the separatrix reach maximum deviations in phase and momentum from the ideal values in the course of performing synchrotron oscillations. A characteristic property of the separatrix therefore is the definition of the maximum phase or momentum deviation a particle may have and still undergo stable synchrotron oscillations. The value of the maximum momentum deviation is called the momentum acceptance of the accelerator. To determine the numerical value of the momentum acceptance, we use the coordinates of the unstable fixed point (9.57) and calculate the value of the Hamiltonian for the separatrix which is from (9.53) with \(\psi_{\rm uf}=\psi_{\rm s}+\varphi_{\rm uf}=\pi-\psi_{\rm s}\) and \(\dot{\psi}_{\rm uf}=0\)

\[{\cal H}_{f}=\frac{\Omega^{2}}{\cos\psi_{\rm s}}\left[2\cos\psi_{\rm s}-(\pi-2 \psi_{\rm s})\sin\psi_{\rm s}\right]. \tag{9.58}\]

Following the separatrix from this unstable fixed point, we eventually reach the location of maximum distance from the ideal momentum. Since \(\dot{\varphi}\) is proportional to \(\Delta p/p_{0}\), the location of the maximum momentum acceptance can be obtained through a differentiation of (9.53) with respect to \(\varphi\)

\[\dot{\varphi}\,\frac{\partial\dot{\varphi}}{\partial\varphi}-\Omega^{2}\frac{ \sin\psi_{\rm s}-\sin(\psi_{\rm s}+\varphi)}{\cos\psi_{\rm s}}=0. \tag{9.59}\]

At the extreme points where the momentum reaches a maximum or minimum, \(\partial\dot{\varphi}/\partial\varphi=0\) which occurs at the phase

\[\sin(\psi_{\rm s}+\varphi)=\sin\psi_{\rm s}\qquad\mbox{or}\quad\varphi=0. \tag{9.60}\]

This is exactly the condition we found in (9.56) for the location of the stable fixed points and is independent of the value of the Hamiltonian. The maximum momentum deviation or momentum acceptance \(\dot{\varphi}_{\rm acc}\) occurs therefore for all trajectories at the phase of the stable fixed points \(\psi=\psi_{\rm s}\). We equate at this phase (9.58) with (9.53) to derive an expression for the maximum momentum acceptance

\[\tfrac{1}{2}\dot{\varphi}_{\rm acc}^{2}=\Omega^{2}[2-(\pi-2\psi_{\rm s})\tan \psi_{\rm s}]. \tag{9.61}\]

In accelerator physics it is customary to define an over voltage factor. This factor is equal to the ratio of the maximum rf-voltage in the cavities to the desired energy gain in the cavity \(U_{0}\)

\[q=\frac{eV_{0}}{U_{0}}=\frac{1}{\sin\psi_{\rm s}} \tag{9.62}\]

and can be used to replace trigonometric functions of the synchronous phase. To solve (9.61), we use the expression

\[\tfrac{1}{2}\pi-\psi_{\rm s}=\arccos\frac{1}{q} \tag{9.63}\]

derived from the identity \(\cos\left(\tfrac{1}{2}\pi-\psi_{\rm s}\right)=\sin\psi_{\rm s}\), replace the synchrotron oscillation frequency \(\Omega\) by its representation (9.35) and get with (9.17) the momentum acceptance for a moving bucket

\[\left(\frac{\Delta p}{p_{0}}\right)_{\rm acc}^{2}=\frac{eV_{0}\sin\psi_{\rm s }}{\pi h|\eta_{\rm c}|cp_{0}}2\left(\sqrt{q^{2}-1}-\arccos\frac{1}{q}\right). \tag{9.64}\]

The function

\[F(q)=2\left(\sqrt{q^{2}-1}-\arccos\frac{1}{q}\right) \tag{9.65}\]

is shown in Fig. 9.12 as a function of the over voltage factor \(q\).

The synchronous phase is always different from zero or \(\pi\) when charged particles are to be accelerated. In circular electron and very high-energy proton accelerators the synchronous phase must be nonzero even without acceleration to compensate for synchrotron radiation losses. In low and medium energy circular proton or heavy ion storage rings no noticeable synchrotron radiation occurs and the synchronous phase is either \(\psi_{\rm s}=0\) or \(\pi\) depending on the energy being below or above the transition energy. In either case \(\sin\psi_{\rm s}=0\) which, however, does not necessarily lead to a vanishing momentum acceptance since the function \(F(q)\) approaches the value \(2q\) and the factor \(\sin\psi_{\rm s}\,F(q)\to 2\) in (9.64) while \(q\to\infty\). Therefore stable buckets for protons and heavy ions can be produced with a finite energy acceptance. The maximum momentum acceptance for such stationary buckets is from (9.64)

\[\left(\frac{\Delta p}{p_{0}}\right)^{2}_{\rm max,stat.}=\frac{2\,eV_{0}}{\pi h |\eta_{\rm c}|cp_{0}}. \tag{9.66}\]

Note that this expression for the maximum momentum acceptance appears to be numerically inconsistent with (9.39) for \(\hat{\phi}=\pi\), because (9.39) has been derived for small oscillations only (\(\hat{\phi}\ll\pi\)). From Fig. 9.11, we note that the aspect ratios of phase space ellipses change while going from bucket center towards the separatrices. The linear proportionality between maximum momentum deviation and maximum phase of (9.39) becomes distorted for large values of \(\hat{\phi}\) such that the acceptance of the rf-bucket is reduced by the factor \(2/\pi\) from the value of (9.39).

The momentum acceptance is further reduced for moving buckets as the synchronous phase increases. In circular accelerators, where the required energy gain for acceleration or compensation of synchrotron radiation losses per turn is \(U_{0}\), the momentum acceptance is

\[\left(\frac{\Delta p}{p_{0}}\right)^{2}_{\rm max,moving}=\frac{U_{0}}{\pi h| \eta_{\rm c}|cp_{0}}F(q)=\frac{F(q)}{2\,q}\left(\frac{\Delta p}{p_{0}}\right) ^{2}_{\rm max,static}. \tag{9.67}\]

The reduction \(F(q)/2q\) in momentum acceptance is solely a function of the synchronous phase and is shown in Fig. 9.13 for the case \(\gamma>\gamma_{\rm tr}\).

Figure 9.12: Over voltage function \(F\left(q\right)\)

Overall, the momentum acceptance depends on lattice and rf-parameters and scales proportional to the square root of the rf-voltage in the accelerating cavities. Strong transverse focusing decreases the momentum compaction thereby increasing the momentum acceptance while high rf-frequencies diminish the momentum acceptance. Very high frequency accelerating systems based, for example, on high intensity lasers to produce high accelerating fields are expected to have a rather small momentum acceptance and work therefore best with quasi-monoenergetic beams.

It is often customary to use other parameters than the momentum as the coordinates in longitudinal phase space. The most common parameter is the particle energy deviation \(\Delta E/\omega_{\rm rf}\) together with the phase. In these units, we get for the stationary bucket instead of (9.66)

\[\left.\frac{\Delta E}{\omega_{\rm rf}}\right|_{\rm max,stat.}=\sqrt{\frac{2\, eV_{0}E_{0}\beta}{\pi h|\eta_{\rm c}|\omega_{\rm rf}^{2}}}, \tag{9.68}\]

which is measured in eV-sec. Independent of the conjugate coordinates used, the momentum acceptance for moving rf-buckets can be measured in units of a stationary rf-bucket, where the proportionality factor depends only on the synchronous phase.

#### Bunch Length

During the course of synchrotron oscillations, particles oscillate between extreme values in momentum and phase with respect to the reference point and both modes of oscillation are out of phase by 90\({}^{\circ}\). All particles of a beam perform incoherent phase oscillations about a common reference point and generate thereby the appearance of a steady longitudinal distribution of particles, which we call a

Figure 9.13: Reduction factor of the momentum acceptance \(F(q)/2q\) as a function of the synchronous phase

particle bunch. The total bunch length is twice the maximum longitudinal excursion of particles from the bunch center defined by

\[\frac{\ell}{2}=\pm\frac{c}{h\,\omega_{\rm rev}}\hat{\phi}=\pm\,\frac{\lambda_{\rm rf }}{2\pi}\hat{\phi}, \tag{9.69}\]

where \(\hat{\phi}\) is the maximum phase deviation.

In circular electron accelerators the rf-parameters are generally chosen to generate a bucket which is much larger than the core of the beam. Statistical emission of synchrotron radiation photons generates a Gaussian particle distribution in phase space and therefore the rf-acceptance is adjusted to provide stability far into the tails of this distribution. To characterize the beam, however, only the core (one standard deviation) is used. In the case of bunch length or energy deviation we consider therefore only the situation for small oscillation amplitudes. In this approximation the bunch length becomes with (9.39)

\[\frac{\ell}{2}=\pm\left.\frac{c|\eta_{\rm c}|}{\Omega}\,\,\frac{\Delta p}{p_{0 }}\right|_{\rm max} \tag{9.70}\]

or with (9.35)

\[\frac{\ell}{2}=\pm\left.\frac{c\,\sqrt{2\pi}}{\omega_{\rm rev}}\,\sqrt{\frac{ \eta_{\rm c}cp_{0}}{h\,e\hat{V}\cos\psi_{\rm s}}}\,\,\frac{\Delta p}{p_{0}} \right|_{\rm max}. \tag{9.71}\]

The bunch length in a circular electron accelerator depends on a variety of rf-and lattice parameters. It is inversely proportional to the square root of the rf-voltage and frequency. A high frequency and rf-voltage can be used to reduce the bunch length of which only the rf-voltage remains a variable once the system is installed. Practical considerations, however, limit the range of bunch length adjustment this way. The momentum compaction is a lattice function and theoretically allows the bunch length to adjust to any small value. For high-energy electron rings \(\eta_{\rm c}\approx-\alpha_{\rm c}\) and by arranging the focusing such that the dispersion functions changes sign, the momentum compaction factor of a ring can become zero or even negative. Rings for which \(\eta_{\rm c}=0\) are called isochronous rings [10]. By adjusting the momentum compaction to zero, phase focusing is lost similar to the situation going through transition in proton accelerators and total beam loss may occur. In this case, however, nonlinear, higher order effects become dominant which must be taken into consideration. If on the other hand the momentum compaction is adjusted to very small values, beam instability may be avoidable. [11] The benefit of an isochronous or quasi-isochronous ring would be that the bunch length in an electron storage ring could be made very small. This is important, for example, to either create short synchrotron radiation pulses or maximize the efficiency of a free electron laser by preserving the micro bunching at the laser wavelength as the electron beam orbits in the storage ring.

In a circular proton or ion accelerator we need not be concerned with the preservation of Gaussian tails and therefore the whole rf-bucket could be filled with the beam proper at high density. In this case, the bunch length is limited by the extreme phases \(\psi_{1}\) and \(\psi_{2}\) of the separatrix. Because the longitudinal extend of the separatrix depends on the synchronous phase, we expect the bunch length to depend also on the synchronous phase. One limit is given by the unstable fixed point at \(\psi_{1}=\pi-\psi_{\rm s}\). The other limit must be derived from (9.53), where we replace \(\mathcal{H}\) by the potential of the separatrix from (9.58). Setting \(\dot{\psi}=0\), we get for the second limit of stable phases the transcendental equation

\[\cos\psi_{1,2}+\psi_{1,2}\sin\psi_{\rm s}=(\pi-\psi_{\rm s})\sin\psi_{\rm s}- \cos\psi_{\rm s}. \tag{9.72}\]

This equation has two solutions \(\mathrm{mod}(2\pi)\) of which \(\psi_{1}\) is one solution and the other is \(\psi_{2}\). Both solutions and their difference are shown in Fig. 9.14 as functions of the synchronous phase.

The bunch length of proton beams is therefore determined only by

\[\ell_{p}=\frac{\lambda_{\rm rf}}{2\pi}(\psi_{2}-\psi_{1}). \tag{9.73}\]

Different from the electron case, we find the proton bunch length to be directly proportional to the rf-wavelength. On the other hand, there is no direct way of compressing a proton bunch by raising or lowering the rf-voltage. This difference stems from the fact that electrons radiate and adjust by damping to a changed rf-bucket while non-radiating particles do not have this property. However, applying adiabatic rf-voltage variation we may modify the bunch length as will be discussed in Sect. 9.3.5.

#### Longitudinal Beam Enittance

Separatrices distinguish between unstable and stable regions in the longitudinal phase space. The area of stable phase space in analogy to transverse phase space

Figure 9.14: Maximum phases limiting the extend of moving buckets

is called the longitudinal beam emittance; however, it should be noted that the definition of longitudinal emittance as used in the accelerator physics community often includes the factor \(\pi\) in the numerical value of the emittance and is therefore equal to the real phase space area. To calculate the longitudinal emittance, we evaluate the integral \(\oint p\,\mathrm{d}q\) where \(p\) and \(q\) are the conjugate variables describing the synchrotron oscillation.

Similar to transverse beam dynamics we distinguish again between beam acceptance and beam emittance. The acceptance is the maximum value for the beam emittance to be able to pass through a transport line or accelerator components. In the longitudinal phase space the acceptance is the area enclosed by the separatrices. Of course, we ignore here other possible acceptance limitations which are not related to the parameters of the accelerating system. The equation for the separatrix can be derived by equating (9.53) with (9.58) which gives with (9.17) and (9.35)

\[\left(\frac{\Delta cp}{cp_{0}}\right)^{2}=\frac{eV_{0}}{\pi h|\eta_{\mathrm{c }}|cp_{0}}\left[\cos\varphi+1+(2\psi_{\mathrm{s}}+\varphi-\pi)\sin\psi_{ \mathrm{s}}\right]. \tag{9.74}\]

We define a longitudinal beam emittance by

\[\epsilon_{\varphi}=\int_{S}\frac{\Delta E}{\omega_{\mathrm{rf}}}\mathrm{d} \varphi\, \tag{9.75}\]

where the integral is to be taken along a path \(S\) tightly enclosing the beam in phase space. Only for \(\psi_{\mathrm{s}}=n\,\pi\) can this integral be solved analytically. The maximum value of the beam emittance so defined is the acceptance of the system. Numerically, the acceptance of a stationary bucket can be calculated by inserting (9.74) into (9.75) and integration along the enclosing separatrices resulting in

\[\epsilon_{\varphi,\mathrm{acc}}=8\sqrt{\frac{2\,eV_{0}\,E_{0}\,\beta}{\pi\,h \,|\eta_{\mathrm{c}}|\,\omega_{\mathrm{rf}}^{2}}}. \tag{9.76}\]

Comparison with the momentum acceptance (9.75) shows the simple relation that the longitudinal acceptance is eight times the energy acceptance

\[\epsilon_{\varphi,\mathrm{acc}}=8\,\left.\frac{\Delta E}{\omega_{\mathrm{rf}} }\right|_{\mathrm{max,stat}}. \tag{9.77}\]

For moving rf-buckets, the integration (9.75) must be performed numerically between the limiting phases \(\psi_{1}\) and \(\psi_{2}\). The resulting acceptance in percentage of the acceptance for the stationary rf-bucket is shown in Fig. 9.15 as a function of the synchronous phase angle.

The acceptance for \(\psi_{\mathrm{s}}<180^{\circ}\) is significantly reduced imposing some practical limits on the maximum rate of acceleration for a given maximum rf-voltage. During the acceleration cycle, the magnetic fields in the latticonsistent with the available maximum rf-voltage and by virtue of the principle of phase focusing the particles will keep close to the synchronous phase whenever the rate of energy increase is slow compared to the synchrotron oscillation frequency which is always the case. In high-energy electron synchrotrons or storage rings the required "acceleration" is no more a free parameter but is mainly determined by the energy loss due to synchrotron radiation and a stable beam can be obtained only if sufficient rf-voltage is supplied to provide the necessary acceptance.

#### Phase Space Matching

In transverse phase space a need for matching exists while transferring a beam from one accelerator to another accelerator. Such matching conditions exist also in longitudinal phase space. In the absence of matching part of the beam may be lost due to lack of overlap with the rf-bucket or severe phase space dilution may occur if a beam is injected unmatched into a too large rf-bucket. In the case of electrons a mismatch generally has no detrimental effect on the beam unless part or all of the beam exceeds rf-bucket limitations. Because of synchrotron radiation and concomitant damping, electrons always assume a Gaussian distribution about the reference phase and ideal momentum. The only matching then requires that the rf-bucket is large enough to enclose the Gaussian distribution far into the tails of 7-10 standard deviations.

In proton and heavy ion accelerators such damping is absent and careful phase space matching during the transfer of particle beams from one accelerator to another is required to preserve beam stability and phase space density. A continuous monochromatic beam, for example, being injected into an accelerator with too large an rf-bucket as shown in Fig. 16 will lead to a greatly diluted emittance.

This is due to the fact that the synchrotron oscillation is to some extend nonlinear and the frequency changes with oscillation amplitude with the effect that for all practical purposes the beam eventually occupies all available phase space. This does not conflict with Liouville's theorem, since the microscopic phase space is preserved albeit fragmented and spread through filamentation over the whole bucket.

Figure 15: Acceptance of moving rf buckets in units of the acceptance of a stationary rf bucket

The situation is greatly altered if the rf-voltage is reduced and adjusted to just cover the energy spread in the beam. Not all particles will be accepted, specifically those in the vicinity of the unstable fixed points, but all particles that are injected inside the rf-bucket remain there and the phase space density is not diluted. The acceptance efficiency is equal to the bucket overlap on the beam in phase space. A more sophisticated capturing method allows the capture of almost all particles in a uniform longitudinal distribution by turning on the rf-voltage very slowly [12], a procedure which is also called adiabatic capture.

Other matching problems occur when the injected beam is not continuous. A beam from a booster synchrotron or linear accelerator may be already bunched but may have a bunch length which is shorter than the rf-wavelength or we may want to convert a bunched beam with a significant momentum spread into an unbunched beam with small momentum spread. Whatever the desired modification of the distribution of the beam in phase space may be, there are procedures to allow the change of particular distributions while keeping the overall emittance constant.

For example, to accept a bunched beam with a bunch length shorter than the rf-wavelength in the same way as a continuous beam by matching only the momentum acceptance would cause phase space filamentation as shown in Fig. 9.17. In a proper matching procedure the rf-voltage would be adjusted such that a phase space trajectory surrounds closely the injected beam (Fig. 9.17 left). In mathematical terms, we would determine the bunch length \(\hat{\phi}\) of the injected beam and following (9.70) would adjust the rf-voltage such that the corresponding momentum acceptance \(\hat{\delta}=(\Delta p/p_{0})_{\text{max}}\) matches the momentum spread in the incoming beam. If no correct matching is done and the beam is injected like shown in (Fig. 9.17 right), then the variation of synchrotron oscillation frequency with amplitude would cause filamentation and dilution of beam phase space. Effectively, this simulates in real space a larger effective emittance.

Equation (9.70) represents a relation between the maximum momentum deviation and phase deviation for small amplitude phase space trajectories which allows us to calculate the bunch length as a function of external parameters. Methods have been discussed in transverse particle dynamics which allow the manipulation of conjugate beam parameters in phase space while keeping the beam emittance constant. Specifically, within the limits of constant phase space we were able to

Figure 9.16: Phase space filamentation after a few synchrotron oscillations

exchange beam size and transverse momentum or beam divergence by appropriate focusing arrangements to produce,for example, a wide parallel beam or a small beam focus.

Similarly, we are able to manipulate within the limits of a constant longitudinal beam emittance the bunch length and momentum spread. The focusing device in this case is the voltage in accelerating cavities. Assume, for example, a particle bunch with a very small momentum spread but a long bunch length as shown in Fig. 9.18 left. To transform such a bunch into a short bunch we would suddenly increase the rf-voltage in a time short compared to the synchrotron oscillation period. The whole bunch then starts to rotate within the new bucket (Fig. 9.18 middle) exchanging bunch length for momentum spread. After a quarter synchrotron oscillation period, the bunch length has reached its shortest value and starts to increase again through further rotation of the bunch unless the rf-voltage is suddenly increased a second time to stop the phase space rotation of the bunch (Fig. 9.18 right). The rf-voltage therefore must be increased to such a value that all particles on the bunch boundary follow the same phase space trajectory.

This phase space manipulation can be conveniently expressed by repeated application of (9.39). The maximum momentum deviation \((\widehat{\Delta p}/p_{0})_{0}\) and the maximum phase deviation \(\hat{\phi}_{0}\) for the starting situation in Fig. 9.18 (left) are related by

\[\left.\frac{\widehat{\Delta p}}{p_{0}}\right|_{0}=\frac{\Omega_{0}}{h\omega_{ \mathrm{rev}}|\eta_{\mathrm{c}}|}\,\hat{\phi}_{0}, \tag{9.78}\]

where \(\Omega_{0}\) is the starting synchrotron oscillation frequency for the rf-voltage \(V_{0}\). To start bunch rotation the rf-voltage is increased to \(V_{1}\) (Fig. 9.18 middle) and after

Figure 9.17: Mismatch for a bunched beam (_right_). Proper match for a bunched beam (_left_)

a quarter synchrotron oscillation period at the frequency \(\Omega_{1}\propto\sqrt{V_{1}}\) the phase deviation \(\hat{\phi}_{0}\) has transformed into the momentum deviation

\[\left.\frac{\widehat{\Delta p}}{p_{0}}\right|_{1}=\frac{\Omega_{1}}{h\omega_{ \mathrm{rev}}|\eta_{\mathrm{c}}|}\,\hat{\phi}_{0}. \tag{9.79}\]

At the same time the original momentum error \(\widehat{\Delta p}/p_{0}|_{0}\) has become a phase error \(\hat{\phi}_{1}\) given by

\[\left.\frac{\widehat{\Delta p}}{p_{0}}\right|_{0}=\frac{\Omega_{1}}{h\omega_{ \mathrm{rev}}|\eta_{\mathrm{c}}|}\,\hat{\phi}_{1}. \tag{9.80}\]

Now we need to stop further phase space rotation of the whole bunch. This can be accomplished by increasing a second time the rf-voltage during a time short compared to the synchrotron oscillation period in such a way that the new bunch length or \(\hat{\phi}\) is on the same phase space trajectory as the new momentum spread \(\widehat{\Delta p}/p_{0}|_{1}\) (Fig. 9.18 right). The required rf-voltage is then determined by

\[\left.\frac{\widehat{\Delta p}}{p_{0}}\right|_{1}=\frac{\Omega_{2}}{h\,\omega_ {\mathrm{rev}}\,|\eta_{\mathrm{c}}|}\hat{\phi}_{1} \tag{9.81}\]

while solving \(\Omega_{2}\) for the voltage \(V_{2}\). We take the ratio of (9.77) and (9.80) to get

\[\frac{\hat{\phi}_{1}\,\Omega_{2}}{\hat{\phi}_{0}\,\Omega_{0}}=\frac{\widehat{ \Delta p}/p_{0}|_{1}}{\widehat{\Delta p}/p_{0}|_{0}} \tag{9.82}\]

Figure 9.18: Phase space rotation

and replace the ratio of the momentum spreads by the ratio of (9.78) and (9.79). With \(\Omega_{i}\propto\sqrt{V_{i}}\) and \(\ell\propto\hat{\phi}\) we get finally the scaling law for the reduction of the bunch length

\[\frac{\ell_{1}}{\ell_{0}}=\left(\frac{V_{0}}{V_{2}}\right)^{\frac{1}{4}}\,. \tag{9.83}\]

In other words the bunch length can be reduced by increasing the rf-voltage in a two step process and the bunch length reduction scales like the fourth power of the rf-voltage. This phase space manipulation is symmetric in the sense that a beam with a large momentum spread and a short bunch length can be converted into a bunch with a smaller momentum spread at the expense of the bunch length by reducing the rf-voltage in two steps.

The bunch length manipulation described here is correct and applicable only for non-radiating particles. For radiating particles like electrons, the bunch manipulation is easier due to damping effects. Equation (9.39) still holds, but the momentum spread is independently determined by synchrotron radiation and the bunch length therefore scales simply proportional to the square root of the rf-voltage.

### Higher-Order Phase Focusing

The principle of phase focusing is fundamental for beam stability in circular accelerators and we find the momentum compaction factor to be a controlling quantity. Since the specific value of the momentum compaction determines critically the beam stability, it is interesting to investigate the consequences to beam stability as the momentum compaction factor varies. Specifically, we will discuss the situation where the linear momentum compaction factor is reduced to very small values and higher-order terms become significant. This is, for example, of interest in proton or ion accelerators going through transition energy during acceleration, or as we try to increase the quadrupole focusing in electron storage rings to obtain a small beam emittance, or when we intentionally reduce the momentum compaction to reduce the bunch length. In extreme cases, the momentum compaction factor becomes zero at transition energy or in an isochronous storage ring where the revolution time is made the same for all particles independent of the momentum. The linear theory of phase focusing would predict beam loss in such cases due to lack of phase stability. To accurately describe beam stability when the momentum compaction factor is small or vanishes, we cannot completely ignore higher-order terms. Some of the higher-order effects on phase focusing will be discussed here. There are two main contributions to the higher-order momentum compaction factor, one from the dispersion function and the other from the momentum dependent path length. First, we derive the higher-order contributions to the dispersion function, and then apply the results to the principle of phase focusing to determine the perturbation on the beam stability.

#### Dispersion Function in Higher Order

The first-order change in the reference path for off energy particles is proportional to the relative momentum error. The proportionality factor is a function of the position and is called the dispersion function. This result is true only in linear beam dynamics. We will now derive chromatic effects on the reference path in higher order to allow a more detailed determination of the chromatic stability criteria. The linear differential equation for the normalized dispersion function is

(9.84)

where is the betatron phase,, the betatron function and the undisturbed dispersion function. The periodic solution of (9.84) is called the normalized dispersion function, and

(9.85)

This linear solution includes only the lowest-order chromatic error term from the bending magnets and we must therefore include higher-order chromatic terms into the differential equation of motion. To do that we use the general differential equation of motion while ignoring all coupling terms

(9.86)

where. We are only interested in the chromatic solution with vanishing betatron oscillation amplitudes and insert for the particle position therefore

(9.87)

Due to the principle of linear superposition separate differential equations exist for each component by collecting on the right-hand side terms of equal power in. For the terms linear in, we find the well-known differential equation for the dispersion function

(9.88)where we also express the perturbations by its Fourier expansion. The terms quadratic in \(\delta\) form the differential equation

\[\eta_{1}^{\prime\prime}+K(z)\,\eta_{1} = -\sum_{n}F_{0n}\,\mathrm{e}^{\mathrm{i}n\varphi}\] \[\quad-\tfrac{1}{2}m\eta_{0}^{2}-(\kappa^{3}+2\kappa k)\eta_{0}^{2} +\tfrac{1}{2}\kappa\eta_{0}^{\prime 2}+\kappa^{\prime}\eta_{0}\eta_{0}^{\prime}+(2 \kappa^{2}+k)\eta_{0}\] \[= -\sum_{n}F_{0n}\,\mathrm{e}^{\mathrm{i}n\varphi}+\sum_{n}F_{1n} \,\mathrm{e}^{\mathrm{i}n\varphi},\]

and terms cubic in \(\delta\) are determined by

\[\eta_{2}^{\prime\prime}+K(z)\,\eta_{2} = +\sum_{n}F_{0n}\,\mathrm{e}^{\mathrm{i}n\varphi}-\sum_{n}F_{1n} \,\mathrm{e}^{\mathrm{i}n\varphi}\] \[\quad-m\eta_{0}\eta_{1}-2\,(\kappa^{3}+2\kappa k)\,\eta_{0}\eta _{1}+(2\kappa^{2}+k)\,\eta_{1}\] \[\quad+\kappa\,\eta_{0}^{\prime}\eta_{1}^{\prime}+\kappa^{\prime }(\eta_{0}\eta_{1}^{\prime}+\eta_{0}^{\prime}\eta_{1})+\kappa^{\prime}\,\eta_ {0}\eta_{0}^{\prime}\] \[= +\sum_{n}F_{0n}\,\mathrm{e}^{\mathrm{i}n\varphi}-\sum_{n}F_{1n} \,\mathrm{e}^{\mathrm{i}n\varphi}+\sum_{n}F_{2n}\,\mathrm{e}^{\mathrm{i}n \varphi}\;.\]

We note that the higher-order dispersion functions are composed of the negative lower-order solutions plus an additional perturbation. After transformation of these differential equations into normalized variables, \(w=\eta/\sqrt{\beta}\), etc., we get with \(j=0,1,2\) differential equations of the form

\[\ddot{w}_{j}(\varphi)+v_{0}^{2}w_{j}(\varphi)=v_{0}^{2}\beta^{3/2}F(z)=v_{0}^{ 2}\sum_{m=0}^{m=j}\sum_{n=-\infty}^{n=\infty}(-1)^{m+j}\beta^{3/2}F_{mn}\, \mathrm{e}^{\mathrm{i}n\varphi}, \tag{9.91}\]

where we have expressed the periodic perturbation on the r.h.s. by an expanded Fourier series. Noting that the dispersion functions \(w_{j}(\varphi)\) are periodic, we try the ansatz

\[w_{j}(\varphi)=\sum_{n}w_{jn}\,\mathrm{e}^{\mathrm{i}n\varphi}, \tag{9.92}\]

and insertion into (9.91) allows to solve for the individual Fourier coefficients \(w_{jn}\) by virtue of the orthogonality of the exponential functions \(\mathrm{e}^{\mathrm{i}n\varphi}\). We get for the dispersion functions up to second order and reverting to the ordinary \(\eta\)-function

\[\eta_{0}(\varphi)=+\beta^{2}\,(\varphi)\sum_{n}\frac{F_{0n}\,\mathrm{e}^{ \mathrm{i}n\varphi}}{v^{2}-n^{2}}, \tag{9.93a}\]\[\eta_{1}(\varphi) =-\beta^{2}\left(\varphi\right)\sum_{n}\frac{F_{0n}\,\mathrm{e}^{ \mathrm{i}n\varphi}}{v^{2}-n^{2}}+\beta^{2}\left(\varphi\right)\sum_{n}\frac{F_{1 n}\,\mathrm{e}^{\mathrm{i}n\varphi}}{v^{2}-n^{2}}, \tag{9.93b}\] \[\eta_{2}(\varphi) =+\beta^{2}\left(\varphi\right)\sum_{n}\frac{F_{0n}\,\mathrm{e}^{ \mathrm{i}n\varphi}}{v^{2}-n^{2}}-\beta^{2}\left(\varphi\right)\sum_{n}\frac{F_ {1n}\,\mathrm{e}^{\mathrm{i}n\varphi}}{v^{2}-n^{2}}+\beta^{2}\left(\varphi \right)\sum_{n}\frac{F_{2n}\,\mathrm{e}^{\mathrm{i}n\varphi}}{v^{2}-n^{2}}. \tag{9.93c}\]

The solutions of the higher-order differential equations have the same integer-resonance behavior as the linear solution for the dispersion function. The higher-order corrections will become important for lattices where strong sextupoles are required in which cases the sextupole terms may be the major perturbations to be considered. Other perturbation terms depend mostly on the curvature \(\kappa\) in the bending magnets and, therefore, maybe small for large rings or beam-transport lines with weak bending magnets.

#### Path Length in Higher Order

The path length together with the velocity of particles governs the time of arrival at the accelerating cavities from turn to turn and therefore defines the stability of a particle beam. Generally, only the linear dependence of the path length on particle momentum is considered. We find, however, higher-order chromatic contributions of the dispersion function to the path length as well as momentum independent contributions due to the finite angle of the trajectory with respect to the ideal orbit during betatron oscillations.

The path length for a particular trajectory from point \(z_{0}\) to point \(z\) in our curvilinear coordinate system can be derived from the integral \(L=\oint_{0}^{z}\mathrm{d}s\), where \(s\) is the coordinate along the particular trajectory. This integral can be expressed by

\[L=\oint\sqrt{\left(1+\kappa x\right)^{2}+x^{\prime 2}+y^{\prime 2}}\mathrm{d}z\,, \tag{9.94}\]

where the first term of the integrand represents the contribution to the path length due to curvature generated by bending magnets while the second and third term are contributions due to finite horizontal and vertical betatron oscillations. For simplicity, we ignore vertical bending magnets. Where this simplification cannot be made, it is straight forward to extend the derivation of the path length in higher order to include bending and betatron oscillations in the vertical plane as well. We expand (9.94) up to second order and get for the path length variation \(\Delta L=L-L_{0}\)

\[\Delta L=\oint\left(\kappa x+\tfrac{1}{2}\kappa^{2}x^{2}+\tfrac{1}{2}x^{ \prime 2}+\tfrac{1}{2}y^{\prime 2}\right)\mathrm{d}z\,+\mathcal{O}(3). \tag{9.95}\]


over the entire beam transport line of length \(L_{0}\) and using average values for the integrands, the path-length variation is

\[\begin{split}\frac{\Delta L}{L_{0}}&=\,\tfrac{1}{4} \left(\epsilon_{x}\left\langle\gamma_{x}\right\rangle+\epsilon_{y}\left\langle \gamma_{y}\right\rangle+\epsilon_{x}\left\langle\kappa^{2}\beta_{x}\right\rangle \right)\\ &+\tfrac{1}{2}\left\langle{x_{0}^{\prime}}^{2}\right\rangle+ \tfrac{1}{2}\left\langle{y_{0}^{\prime}}^{2}\right\rangle+\tfrac{1}{2}\left\langle \kappa^{2}x_{0}^{2}\right\rangle\\ &+\alpha_{c}\delta+\left(\left\langle\kappa\eta_{1}\right\rangle +\tfrac{1}{2}\left\langle\kappa^{2}\eta_{0}^{2}\right\rangle+\tfrac{1}{2} \left\langle{\eta_{0}^{\prime}}^{2}\right\rangle\right)\delta^{2}+\mathcal{O} (3).\end{split} \tag{9.99}\]

In this expression for the path-length variation we find separate contributions due to betatron oscillations, orbit distortion and higher-order chromatic effects. We have used the emittance \(\epsilon\) as the amplitude factor for betatron oscillation and get therefore a path length spread within the beam due to the finite beam emittance \(\epsilon\). Note specifically that for an electron beam this emittance scales by the factor \(n_{\sigma}^{2}\) to include Gaussian tails, where \(n_{\sigma}\) is the oscillation amplitude in units of the standard amplitude \(\sigma\). For whole beam stability a total emittance of \(\epsilon_{\text{tot}}=7^{2}\epsilon-10^{2}\epsilon\) should be considered. For stable machine conditions, the contribution of the orbit distortion is the same for all particles and can therefore be corrected by an adjustment of the rf-frequency. We include these terms here, however, to allow the estimation of allowable tolerances for dynamic orbit changes.

#### Higher Order Momentum Compaction Factor

The longitudinal phase stability in a circular accelerator depends on the value of the momentum compaction \(\eta_{c}\), which actually regulates the phase focusing to obtain stable particle motion. This parameter is not a quantity that can be chosen freely in the design of a circular accelerator without jeopardizing other desirable design goals. If, for example, a small beam emittance is desired in an electron storage ring, or if for some reason it is desirable to have an isochronous ring where the revolution time for all particles is the same, the momentum compaction should be made to become very small. This in itself does not cause instability unless the momentum compaction approaches zero and higher-order chromatic terms modify phase focusing to the extent that the particle motion becomes unstable. To derive conditions for the loss of phase stability, we evaluate the path length variation (9.99) with momentum in higher order

\[\frac{\Delta L}{L_{0}}=\alpha_{c}\delta+\alpha_{1}\,\delta^{2}+\xi+\mathcal{O }(3), \tag{9.100}\]

where \(\xi\) represents the momentum independent term

\[\xi=\tfrac{1}{4}\left(\epsilon_{x}\left\langle\gamma_{x}\right\rangle+\epsilon _{y}\left\langle\gamma_{y}\right\rangle+\epsilon_{x}\left\langle\kappa^{2} \beta_{x}\right\rangle\right) \tag{9.101}\]\[\alpha_{1}=\left\langle\kappa\,\eta_{1}\right\rangle+\,\tfrac{1}{2}\left\langle \kappa^{2}\eta_{0}^{2}\right\rangle+\,\tfrac{1}{2}\left\langle\eta_{0}^{\prime 2 }\right\rangle \tag{9.102}\]

is the non-linear momentum compaction factor.

From the higher order dispersion and path length we may now derive the value of the higher order momentum compaction factor. First we note that we are not interested in oscillatory terms. Therefore (9.93b) reduces to

\[\eta_{1}(\varphi)=-\frac{\beta^{2}(\varphi)}{\nu^{2}}F_{00}\,+\,\frac{\beta^{ 2}(\varphi)}{\nu^{2}}F_{10}, \tag{9.103}\]

where

\[F_{00} = \left\langle\kappa\right\rangle\quad\text{and}\] \[F_{10} = \left\langle-\frac{1}{2}m\eta_{0}^{2}-\left(\kappa^{3}+2\kappa k \right)\eta_{0}^{2}+\frac{1}{2}\kappa\eta_{0}^{\prime 2}+\kappa^{\prime} \eta_{0}\eta_{0}^{\prime}+\left(2\kappa^{2}+k\right)\eta_{0}\right\rangle.\]

Furthermore

\[\left\langle\kappa\,\eta_{1}\right\rangle=\left\langle\kappa\,\frac{\beta^{ 2}(\varphi)}{\nu^{2}}\right\rangle\left(-F_{00}\,+\,F_{10}\right), \tag{9.104}\]

where the average is to be taken over one superperiod of the accelerator. The other terms in (9.102) and (9.101) are straight forward. With the knowledge of the higher order momentum compaction factor we are now able to consider higher order phase motion.

#### Higher-Order Phase Space Motion

Following the derivation of the linear phase equation, we note that it is the variation of the revolution time with momentum rather than the path-length variation that affects the synchronicity condition. With the expanded momentum compaction \(\eta_{\text{c}}=\frac{1}{\nu^{2}}-\alpha_{\text{c}}\)we get the differential equation for the phase oscillation to second order

\[\frac{\partial\psi}{\partial t}=-\omega_{\text{rf}}\left(\eta_{\text{c}}\delta -\alpha_{1}\delta^{2}-\xi\right) \tag{9.105}\]

and for the momentum oscillation

\[\frac{\partial\delta}{\partial t}=\frac{eV_{\text{rf}}}{T_{0}cp_{0}}\,\left( \sin\psi\,-\sin\psi_{\text{s}}\right). \tag{9.106}\]In linear approximation, where \(\alpha_{1}=0\) and \(\xi=0\), a single pair of fixed points and separatrices exist in phase space. These fixed points can be found from the condition that \(\dot{\psi}=0\) and \(\dot{\delta}=0\) and they lie on the abscissa for \(\delta=0\). The stable fixed point is located at \((\psi_{\rm sf},\delta_{\rm sf})=(\psi_{\rm s},0)\) defining the center of the rf-bucket where stable phase oscillations occur. The unstable fixed point at \((\psi_{\rm uf},\delta_{\rm uf})=(\pi-\psi_{\rm s},0)\) defines the crossing point of the separatrices separating the trajectories of oscillations from those of librations.

Considering also higher-order terms in the theory of phase focusing leads to a more complicated pattern of phase space trajectories. Setting (9.106) equal to zero we note that the abscissae of the fixed points are at the same location as for the linear case

\[\psi_{\rm 1f}=\psi_{\rm s}\qquad\mbox{and}\qquad\psi_{\rm 2f}=\pi-\psi_{\rm s}. \tag{9.107}\]

The energy coordinates of the fixed points, however, are determined by solving (9.105) for \(\dot{\psi}=0\) or

\[\eta_{\rm c}\delta-\alpha_{1}\delta^{2}-\dot{\xi}=0 \tag{9.108}\]

with the solutions

\[\delta_{\rm f}=+\frac{\eta_{\rm c}}{2\alpha_{1}}\left(1\pm\sqrt{1-\Gamma} \right), \tag{9.109}\]where

\[\Gamma=\frac{4\dot{\xi}\alpha_{1}}{\eta_{\rm c}^{2}}. \tag{9.110}\]

Due to the quadratic character of (9.108), we get now two layers of fixed points with associated areas of oscillation and libration. In Figs. 9.19, 9.20, the phase diagrams are shown for increasing values of \(\alpha_{1}\) while for now we set the momentum independent perturbation \(\xi=0\). Numerically, the contour lines have been calculated from the Hamiltonian (9.114) with \(\Delta/2\eta_{\rm c}=0.005\), where \(\Delta\) is defined in (26199). The appearance of the second layer of stable islands and the increasing perturbation of the original rf-buckets is obvious. There is actually a point [Fig. 9.20 (top)] where the separatrices of both island layers merge. We will use this merging of the separatrices later to define a tolerance limit for the perturbation on the momentum acceptance.

The coordinates of the fixed points in the phase diagram are determined from (9.116), (9.117) and are for the linear fixed points in the first layer

\[\begin{array}{ll}\mbox{point A:}&\psi_{\rm A}=\psi_{\rm s},\qquad\delta_{ \rm A}=\frac{\eta_{\rm c}}{2\alpha_{1}}\left(1-\sqrt{1-\Gamma}\right),\\ \mbox{point B:}&\psi_{\rm B}=\pi-\psi_{\rm s},\ \ \delta_{\rm B}=\frac{\eta_{\rm c }}{2\alpha_{1}}\left(1-\sqrt{1-\Gamma}\right).\end{array} \tag{9.111}\]The momenta of these fixed points are at \(\delta=0\) for \(\Gamma=0\) consistent with earlier discussions. As orbit distortions and betatron oscillations increase, however, we note a displacement of the equilibrium momentum as \(\Gamma\) increases.

The fixed points of the second layer of islands or rf-buckets are displaced both in phase and in momentum with respect to the linear fixed points such that the stable and unstable fixed points are interchanged. The locations of the second layer of fixed points are

\[\begin{array}{ll}\mbox{point C:}&\psi_{\rm C}=\psi_{\rm s},\hskip 28.452756pt \delta_{\rm C}=\frac{\eta_{\rm c}}{2\alpha_{1}}\left(1+\sqrt{1-\Gamma}\right), \\ \mbox{point D:}&\psi_{\rm D}=\pi-\psi_{\rm s},\hskip 28.452756pt\delta_{\rm D}= \frac{\eta_{\rm s}}{2\alpha_{1}}\left(1+\sqrt{1-\Gamma}\right).\end{array} \tag{9.112}\]

The dependence of the coordinates for the fixed points on orbit distortions and the amplitude of betatron oscillations becomes evident from (9.121), (9.124). Specifically, we note a shift in the reference momentum of the beam by \(\xi/\eta_{\rm c}\) as the orbit distortion increases as demonstrated in the examples shown in Figs. 9.21, 9.22, 9.23c, d. Betatron oscillations, on the other hand, cause a spread of the beam momentum in the vicinity of the fixed points. This readjustment of the beam momentum is a direct consequence of the principle of phase focusing whereby the particle follows a path such that the synchronicity condition is met. The phase space diagram of Fig. 9.19 is repeated in Fig. 9.21 with a parameter \(2\xi/\eta_{\rm c}=-0.125\) and in Fig. 9.22 with the further addition of a finite synchronous phase of \(\psi_{\rm s}=0.7\) rad. In addition to the shift of the reference momentum a significant reduction in the momentum acceptance compared to the regular rf-buckets is evident in both diagrams.

As long as the perturbation is small and \(|\alpha_{1}|\ll|\eta_{\rm c}|\), the new fixed points are located far away from the reference momentum and their effect on the particle dynamics can be ignored. The situation becomes very different whenever the linear momentum compaction becomes very small or even zero due to strong

Figure 9.21: Second-order longitudinal phase space for the same parameters as Fig. 9.20 (top), but now with \(2\xi/\eta_{\rm c}=-0.125\)

quadrupole focusing during momentum ramping through transition or in the case of the deliberate design of a low \(\alpha\)-lattice for a quasi isochronous storage ring. In these cases higher order perturbations become significant and cannot be ignored. We cannot assume anymore that the perturbation term \(\alpha_{1}\) is negligibly small and the phase dynamics may very well become dominated by perturbations.

The perturbation \(\alpha_{1}\) of the momentum compaction factor depends on the perturbation of the dispersion function and is therefore also dependent on the sextupole distribution in the storage ring. Given sufficient sextupole families it is at least in principle possible to adjust the parameter \(\alpha_{1}\) to zero or a small value by a proper distribution of sextupoles.

#### Stability Criteria

Stability criteria for phase oscillations under the influence of higher order momentum compaction terms can be derived from the Hamiltonian. The nonlinear equations of motion (9.105), (9.106) can be derived from the Hamiltonian

\[H=\frac{eV_{\rm rf}}{T_{0}cp_{0}}\left[\cos\psi\,-\cos\psi_{\rm s}+(\psi-\psi _{\rm s})\sin\psi_{\rm s}\right]+\omega_{\rm rf}\left(\xi\delta-\tfrac{1}{2} \eta_{\rm c}\delta^{2}+\tfrac{1}{3}\alpha_{1}\delta^{3}\right). \tag{9.113}\]

To eliminate inconsequential factors for the calculation of phase space trajectories, we simplify (9.113) to

\[\tilde{H}=\Delta\left[\cos\psi-\cos\psi_{\rm s}+(\psi-\psi_{\rm s})\sin\psi_{ \rm s}\right]+2\frac{\xi}{\eta_{\rm c}}\delta-\delta^{2}+\tfrac{2}{3}\frac{ \alpha_{1}}{\eta_{\rm c}}\delta^{3}, \tag{9.114}\]where

\[\Delta=\frac{2eV_{\rm rf}}{T_{0}c\rho_{0}\omega_{\rm rf}\eta_{\rm c}}. \tag{9.115}\]

We may use (9.114) to calculate phase space trajectories and derive stability conditions for various combinations of the parameters \(\Delta\), the perturbation of the momentum compaction \(\alpha_{1}\), and the synchronous phase \(\psi_{\rm s}\) (Figs. 9.19, 9.20, 9.21, and 9.22). In Fig. 9.23, the phase diagrams of Figs. 9.19, 9.20, 9.21, and 9.22 are displayed now as three-dimensional surfaces plots with the same parameters. Starting from the linear approximation where only regular rf-buckets appear along the \(\psi\)-axis, we let the ratio \(\alpha_{1}/\eta_{\rm c}\) increase and find the second set of rf-buckets to move in from large relative momentum errors \(\delta_{\rm f}\) toward the main rf-buckets. A significant modification of the phase diagrams occurs when the perturbation reaches such values that the separatrices of both sets of buckets merge as shown in Fig. 9.20 (top). A further increase of the perturbation quickly reduces the momentum

Figure 9.23: Three dimensional rendition of Figs. 9.19(a), 9.20(b), 9.21(c) and 9.22(d)


From this criterion we note that the momentum independent perturbation \(\Gamma\) can further limit the momentum acceptance until there is for \(\Gamma\geq 1\) no finite momentum acceptance left at all.

The momentum shift and the momentum acceptance as well as stability limits can be calculated analytically as a function of \(\alpha_{1}\) and the momentum independent term \(\Gamma\). As long as the perturbation is small and (9.121) is fulfilled we calculate the momentum acceptance for the linear rf-buckets from the value of the Hamiltonian (9.114). For stronger perturbations, where the separatrices of both layers of rf-buckets have merged and are actually exchanged (Fig. 9.20), a different value of the Hamiltonian must be chosen. The maximum stable synchrotron oscillation in this case is not anymore defined by the separatrix through fixed point B but rather by the separatrix through fixed point C. In the course of synchrotron oscillations a particle reaches maximum momentum deviations from the reference momentum at the phase \(\psi\ =\ \psi_{s}\). We have two extreme momentum deviations, one at the fixed point (C), and the other half a synchrotron oscillation away. Both points have the same value of the Hamiltonian (9.114) and are related by

\[2\frac{\xi}{\eta_{c}}\hat{\delta}-\hat{\delta}^{2}+\tfrac{2}{3}\frac{\alpha_{1 }}{\eta_{c}}\hat{\delta}^{3}=2\frac{\xi}{\eta_{c}}\delta_{\text{C}}-\delta_{ \text{C}}^{2}+\tfrac{2}{3}\frac{\alpha_{1}}{\eta_{c}}\delta_{\text{C}}^{3}. \tag{9.122}\]

We replace \(\delta_{\text{C}}\) from (9.112) and obtain a third-order equation for the maximum momentum acceptance \(\hat{\delta}\)

\[2\frac{\xi}{\eta_{c}}\hat{\delta}-\hat{\delta}^{2}+\tfrac{2}{3}\frac{\alpha_{ 1}}{\eta_{c}}\hat{\delta}^{3}=-\frac{\eta_{c}}{6\alpha_{1}^{2}}\left[1+(1- \Gamma)^{3/2}-\tfrac{3}{2}\Gamma\right]. \tag{9.123}\]

This third-order equation can be solved analytically and has the solutions

\[\begin{array}{l}\hat{\delta}_{1}=\frac{\eta_{c}}{2\alpha_{1}}\left(1-2\sqrt {1-\Gamma}\right),\\ \hat{\delta}_{2,3}=\frac{\eta_{c}}{2\alpha_{1}}\left(1+\sqrt{1-\Gamma}\right). \end{array} \tag{9.124}\]

Two of the three solutions are the same and define the momentum at the crossing of the separatrix at the fixed point (C) while the other solution determines the momentum deviation half a synchrotron oscillation away from the fixed point (C). We plot these solutions in Fig. 9.24 together with the momentum shift of the reference momentum at the fixed point (A). As long as there is no momentum independent perturbation (\(\Gamma=0\)) the momentum acceptance is given by

\[-2<-\frac{2\alpha_{1}}{\eta_{c}}\delta_{i}<1. \tag{9.125}\]

The asymmetry of the momentum acceptance obviously reflects the asymmetry of the separatrix. For \(\alpha_{1}\to 0\) the momentum acceptance in (9.120) diverges, which is a reminder that we consider here only the case where the perturbation \(\alpha_{1}\) exceedsthe limit (9.121). In reality the momentum acceptance does not increase indefinitely but is limited by other criteria, for example, by the maximum rf-voltage available. The momentum acceptance limits of (9.124) are further reduced by a finite beam emittance when \(\Gamma\neq 0\) causing a spread in the revolution time. All beam stability is lost as \(\Gamma\) approaches unity and the stability criterion for stable synchrotron motion in the presence of betatron oscillations is defined by

\[\frac{4\dot{\xi}\alpha_{1}}{\eta_{\rm c}^{2}}<1, \tag{9.126}\]

where the parameter \(\dot{\xi}\) is defined by (9.101).

In evaluating the numerical value of \(\dot{\xi}\) we must consider the emittances \(\epsilon_{x,y}\) as amplitude factors. In case of a Gaussian electron beam in a storage ring, for example, a long quantum lifetime can be obtained only if particles with betatron oscillation amplitudes up to at least seven standard values are stable. For such particles the emittance is \(\epsilon=7^{2}\epsilon_{\sigma}\), where \(\epsilon_{\sigma}\) is the beam emittance for one standard deviation. Similarly, the momentum acceptance must be large enough to include a momentum deviation of \(\delta_{\rm max}\geq 7\sigma_{\varepsilon}/E_{0}\).

In general, the stability criteria can be met especially if sextupole magnets are adjusted that the linear perturbation \(\alpha_{1}\) of the momentum compaction is made small. In this case, however, we must consider dynamic stability of the beam and storage ring to prevent \(\alpha_{1}\) to vary more than the stability criteria allow. Any dynamic variation \(\Delta\alpha_{1}\) must meet the condition

\[\Delta\alpha_{1}<\frac{\eta_{\rm c}^{2}}{4\dot{\xi}}. \tag{9.127}\]

Even if the quadratic term \(\alpha_{1}\) is made to approach zero we still must consider the momentum shift due to non-chromatic terms when \(\dot{\xi}\neq 0\). From (9.111) we have for the momentum shift \(\delta_{0}\) of the stable fixed point A

\[\delta_{0}=\frac{\eta_{\rm c}}{2\alpha_{1}}\left(1-\sqrt{1-\Gamma}\right), \tag{9.128}\]

Figure 9.24: Higher-order momentum acceptance

where \(\Gamma\) is small when \(\alpha_{1}\to 0\) and the square root can be expanded. In this limit the momentum shift becomes

\[\delta_{{}_{0}}\to\frac{\xi}{\eta_{\rm c}}\qquad\mbox{for}\qquad\alpha_{1}\to 0. \tag{9.129}\]

To achieve low values of the momentum compaction, it is therefore also necessary to reduce the particle beam emittance. Case studies of isochronous lattices show, however, that this might be very difficult because the need to generate both positive and negative values for the dispersion function generates large values for the slopes of the dispersion leading to rather large beam emittances.

Adjusting the quadratic term \(\alpha_{1}\) to zero finally brings us back to the situation created when the linear momentum compaction was reduced to small values. One cannot ignore higher-order terms anymore. In this case we would expect that the quadratic and cubic perturbations of the momentum compaction will start to play a significant role since \(\eta_{\rm c}\,\approx\,0\) and \(\alpha_{1}\,\approx\,0\). The quadratic term \(\alpha_{3}\) will introduce a spread of the momentum compaction due to the momentum spread in the beam while the cubic term \(\alpha_{4}\) introduces a similar spread to the linear term \(\alpha_{1}\).

## Problems

### 9.1 (S)

A 500 MHz rf-system is supposed to be used in a Wideroe type linac to accelerate protons from a 1 MeV Van de Graaf accelerator. Determine the length of the first three drift tubes for an accelerating voltage at the gaps of 0.5 MeV while assuming that the length of the tubes shall not be less than 15 cm. Describe the operating conditions from an rf-frequency point of view.

### 9.2 (S)

A proton beam with a finite energy spread is injected at an energy of 200 MeV into a storage ring in \(n_{\rm b}\) equidistant short bunches while the rf-system in the storage ring is turned off. Derive an expression for the debunching time or the time it takes for the bunched proton beam to spread out completely.

### 9.3 (S)

The momentum acceptance in a synchrotron is reduced as the synchronous phase is increased. Derive a relationship between the maximum acceleration rate and momentum acceptance. How does this relationship differ for protons and radiating electrons?

### 9.4 (S)

Derive an expression for and plot the synchrotron frequency as a function of oscillation amplitude within the separatrices. What is the synchrotron frequency at the separatrices?

### 9.5 (S)

Sometimes it is desirable to produce short bunches, even only temporary in a storage ring either to produce short X-ray pulses or for quick ejection from a damping ring into a linear collider. By a sudden change of the rf-voltage the bunch can be made to rotate in phase space. Determine analytically the shortest possible bunch length as a function of the rf-voltage increase considering a finite energy spread. For how many turns would the short bunch remain within 50 % of its shortest value?

**9.6.** Calculate the synchrotron oscillation frequency for a 9 GeV proton booster. The maximum momentum is \(cp_{\rm max}=8.9\) GeV the harmonic number \(h=84\), the rf-voltage \(V_{\rm rf}=200\) kV, transition energy \(\gamma_{\rm tr}=\)5.4 and rf-frequency at maximum momentum \(f_{\rm rf}=52.8\) MHz. Calculate and plot the rf and synchrotron oscillation frequency as a function of momentum from an injection momentum of 400 MeV to a maximum momentum of 8.9 GeV while the synchronous phase is \(\psi_{\rm s}=45^{\circ}\). What is the momentum acceptance at injection and at maximum energy? How long does the acceleration last?

**9.7.** Specify a synchrotron of your choice made up of FODO cells for the acceleration of relativistic particles. Assume an rf-system to provide an accelerating voltage equal to \(10^{-4}\) of the maximum particle energy in the synchrotron. During acceleration the synchrotron oscillation tune shall remain less than \(v_{\rm s}<0.02\). What are the numerical values for the rf-frequency, harmonic number, rf-voltage, synchronous phase angle and acceleration time in your synchrotron? In case of a proton synchrotron determine the change in the bunch length during acceleration.

## Bibliography

* [1] G. Ising, Arkiv for Matematik, Astronomi och Fysik **18**, 1 (1924)
* [2] R. Wideroe, Archive fur Elektrotechnik **21**, 387 (1928)
* [3] M.S. Livingston (ed.), _The Development of High-Energy Accelerators_ (Dover, New York, 1966)
* [4] V.I. Veksler, DAN(USSR) **44**, 393 (1944)
* [5] E.M. McMillan, Phys. Rev. **68**, 143 (1945)
* [6] L.W. Alvarez, Phys. Rev. **70**, 799 (1946)
* [7] K. Johnsen, in _CERN Symposium on High Energy Accelerators_ (CERN, Geneva, 1956), p. 295
* [8] G.K. Green, in _CERN Symposium on High Energy Accelerators_ (CERN, Geneva, 1956), p. 103
* [9] H. Goldstein, _Classical Mechanics_ (Addison-Wesley, Reading, 1950)
* [10] D. Deacon, _Theory of the Isochronous Storage Ring Laser_. PhD thesis, Stanford University (1979)
* [11] C. Pellegrini, D. Robin, Nucl. Instrum. Methods **A301**, 27 (1991)
* [12] C.G. Lilliequist, K.R. Symon, Technical Report MURA-491, MURA, Chicago (1959)

