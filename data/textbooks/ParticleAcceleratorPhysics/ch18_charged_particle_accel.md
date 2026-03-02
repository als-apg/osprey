## Chapter 18 Charged Particle Acceleration

Particle acceleration by rf-fields has been discussed, for example, in considerable detail in [1, 2] where relationships between longitudinal phase oscillation and beam stability are derived and discussed. The accelerating fields were assumed to be available in resonant cavities, but we ignored conditions that must be met to generate such fields and ensure positive energy transfer to the particle beam. In this chapter, we will discuss relevant characteristics of rf-cavities and study the interaction of the rf-generator with accelerating cavity and beam.

It is not the intention here to develop a general microwave theory but we will restrict ourselves rather to such aspects which are of importance for particle accelerator physics. Considerable performance limits occur in accelerators by technical limitations in various accelerator systems as, for example, the rf-system and it is therefore useful for the accelerator designer to have a basic knowledge of such limits.

### 18.1 Rf-Waveguides and Cavities

Commonly, high frequency rf-fields are used to accelerate charged particles and the interaction of such electromagnetic waves with charged particles has been discussed earlier together with the derivation of synchronization conditions to obtain continuous particle acceleration. In doing so plane rf-waves have been used ignoring the fact that such fields do not have electrical field components in the direction of particle and wave propagation. Although this assumption has not made the results obtained so far obsolete, a satisfactory description of the wave-particle interaction must include the establishment of appropriate field configurations.

Electromagnetic waves useful for particle acceleration must exhibit field components in the direction of particle propagation which in our coordinate system is the \(z\)-direction. The synchronization condition can be achieved in two ways. First, anelectromagnetic wave travels along the direction of the desired particle acceleration with a phase velocity which is equal to the velocity of the particle. In this case, a particle starting, say, at the crest of the wave where the field strength is largest, would be continuously accelerated at the maximum rate as it moves along with the wave. Another way of particle acceleration occurs from electromagnetic fields created in rf-cavities placed at particular locations along the particle path. In this case, the phase velocity of the wave is irrelevant. For positive particle acceleration the phase of the electromagnetic field must be adjusted such that the integrated acceleration is positive, while the particle passes through the cavity. Obviously, if the velocity of the particle or the length of the cavity is such that it takes several oscillation periods for a particle to traverse the cavity no efficient acceleration is possible.

##### Wave Equation

To generate electromagnetic field components in the direction of wave propagation we cannot use free plane waves, but must apply specific boundary conditions by properly placing conducting metallic surfaces to modify the electromagnetic wave into the desired form. The theory of electromagnetic waves, waveguides and modes is well established and we repeat here only those aspects relevant to particle acceleration. For more detailed reading consult, for example, [3, 4]. Maxwell's equations for our application in a charge free environment are

\[\begin{array}{ll}\nabla(\epsilon\mathbf{E})=0,&\nabla\times\mathbf{E}=-\frac{ \mathrm{d}\mathbf{B}}{\mathrm{d}t},\\ \nabla\mathbf{B}=0,&c^{2}\nabla\times\mathbf{B}=\epsilon\mu\frac{\mathrm{d}\mathbf{E}}{ \mathrm{d}t},\end{array} \tag{18.1}\]

and we look for solutions in the form of rf-fields oscillating with frequency \(\omega\) and \(\mathbf{U=U_{0}\mathrm{e}^{\mathrm{i}\omega t}}\) where \(\mathbf{U=E}\) or \(\mathbf{B}\). A uniform medium is assumed which need not be a vacuum but may have a dielectric constant \(\epsilon\) and a magnetic permeability \(\mu\). Maxwell's curl equations become then

\[\begin{array}{ll}\nabla\times\mathbf{E}=-\mathrm{i}\omega\mathbf{B},\\ c^{2}\nabla\times\mathbf{B}=\mathrm{i}\,\nu\omega\mathbf{E}.\end{array} \tag{18.2}\]

Eliminating the magnetic or electric field strength from both equations and using the vector relation \(\nabla\times(\nabla\times\mathbf{a})=\nabla\)\((\nabla\mathbf{a})-\nabla^{2}\mathbf{a}\) we get the respective wave equations

\[\begin{array}{ll}\nabla^{2}\mathbf{E}+k^{2}\mathbf{E}=0,\\ \nabla^{2}\mathbf{B}+k^{2}\mathbf{B}=0,\end{array} \tag{18.3}\]

where

\[k=\epsilon\mu\frac{\omega^{2}}{c^{2}}. \tag{18.4}\]In the case of a plane wave propagating along the \(z\)-axis the transverse partial derivatives vanish \(\frac{\partial}{\partial x}=\frac{\partial}{\partial y}=0\), since field parameters of a plane wave do not vary transverse to the direction of propagation. The differential equation (18.3) for the electrical field component then becomes \(\left(\frac{\partial^{2}}{\partial z^{2}}+k^{2}\right)\mathbf{E}=0\) and the solution is

\[\mathbf{E}=\mathbf{E}_{0}\mathrm{e}^{\mathrm{i}\left(\omega t-kz\right)}. \tag{18.5}\]

For real values of the wave number \(k\) the solutions of (18.3) describe waves propagating with the phase velocity

\[v_{\mathrm{ph}}=\frac{z}{t}=\frac{c}{\sqrt{\epsilon\mu}}\leq c. \tag{18.6}\]

An imaginary component of \(k\), on the other hand, would lead to an exponential damping term for the fields, a situation that occurs, for example, in a conducting surface layer where the fields decay exponentially over a distance of the skin depth. Between conducting boundaries, the wave number is real and describes propagating waves of arbitrary frequencies. As has been noted before, however, such plane waves lack electrical field components in the direction of propagation. In the following section, we will therefore derive conditions to obtain from (18.3) waves with longitudinal field components.

##### Rectangular Waveguide Modes

Significant modification of wave patterns can be obtained from the proximity of metallic boundaries. To demonstrate this, we evaluate the electromagnetic field of a wave propagating along the axis of a rectangular metallic pipe or rectangular waveguide as shown in Fig. 18.1. Since we are interested in getting a finite value for the \(z\)-component of the electrical field we try the ansatz

\[E_{z}=\psi_{x}(x)\psi_{y}(y)\psi_{z}(z) \tag{18.7}\]

Figure 18.1: Rectangular waveguide

and look for boundary conditions that are required to obtain nonvanishing longitudinal fields. Insertion into (18.3) gives

\[\frac{\psi_{x}^{\prime\prime}(x)}{\psi_{x}(x)}\,+\,\frac{\psi_{y}^{\prime\prime} (y)}{\psi_{y}(y)}\,+\,\frac{\psi_{z}^{\prime\prime}(z)}{\psi_{z}(z)}\,=-\epsilon \mu\frac{\omega^{2}}{c^{2}}\,=-k^{2}, \tag{18.8}\]

where the r.h.s. is a constant while the functions \(\psi_{u}(u)\) are functions of the variable \(u=x,y,\) or \(z\). In order that this equation be true for all values of the coordinates, the ratios \(\frac{\psi_{u}^{\prime\prime}(u)}{\psi_{u}(u)}\) must be constant and we may write (18.8) in the form

\[k_{x}^{2}+k_{y}^{2}+k_{z}^{2}=k^{2}. \tag{18.9}\]

Differentiating (18.7) twice with respect to \(z\) results in the differential equation for the \(z\)-component of the electrical field

\[\frac{\mathrm{d}^{2}E_{z}}{\mathrm{d}z^{2}}=-k_{z}^{2}E_{z}\,, \tag{18.10}\]

which can be solved readily. The wavenumber \(k_{z}\) must be real for propagating waves and with the definition

\[k_{\mathrm{c}}^{2}=k_{x}^{2}+k_{y}^{2}\,, \tag{18.11}\]

we get with (18.9)

\[k_{z}^{2}=k^{2}-k_{\mathrm{c}}^{2}. \tag{18.12}\]

The solution (18.7) of the wave equation for the \(z\)-component of the electrical field is then finally

\[E_{z}=E_{0z}\psi_{x}(x)\psi_{y}(y)\,\mathrm{e}^{\mathrm{i}\,(\omega t-k_{z}z)}\,. \tag{18.13}\]

The nature of the parameters in this equation will determine if the wave fields are useful for acceleration of charged particles. The phase velocity is given by

\[v_{\mathrm{ph}}=\frac{\omega}{k_{z}}=\frac{\omega}{\sqrt{k^{2}-k_{\mathrm{c}} ^{2}}}\,. \tag{18.14}\]

An electromagnetic wave in a rectangular metallic pipe is propagating only if the phase velocity is real or \(k>k_{\mathrm{c}}\) and the quantity \(k_{\mathrm{c}}\) is therefore called the cutoff wave number. For frequencies with a wave number less than the cutoff value the phase velocity becomes imaginary and the wave decays exponentially like \(\exp\left(-\sqrt{|k^{2}-k_{\mathrm{c}}^{2}|z}\right)\).

Conducting boundaries modify electromagnetic waves in such a way that finite longitudinal electric field components can be produced which, at least in principle, can be used for particle acceleration. Although we have found solutions seemingly suitable for particle acceleration, we cannot use such an electromagnetic wave propagating in a smooth rectangular pipe to accelerate particles. Inserting (18.9) into (18.14) the phase velocity of a traveling waveguide mode in a rectangular pipe becomes

\[v_{\text{ph}}=\frac{c}{\sqrt{\epsilon\mu}\sqrt{1-(k_{\text{c}}/k)^{2}}} \tag{18.15}\]

and is with \(k>k_{\text{c}}\) in vacuum or air (\(\epsilon\approx\mu\approx 1\)) larger than the velocity of light. There can be no net acceleration since the wave rolls over the particles, which cannot move faster than the speed of light. This problem occurs in a smooth pipe of any cross section. We must therefore seek for modifications of a smooth pipe in such a way that the phase velocity is reduced or to allow a standing wave pattern, in which case the phase velocity does not matter anymore. The former situation occurs for traveling wave linac structures, while the latter is specially suited for single accelerating cavities.

For a standing wave pattern \(k_{z}=0\) or \(k=k_{\text{c}}\) and with (18.9), (18.11) the cutoff-frequency is

\[\omega_{\text{c}}=\frac{ck_{\text{c}}}{\sqrt{\epsilon\mu}}. \tag{18.16}\]

To complete the solution (18.13) for transverse dimensions, we apply boundary conditions to the amplitude functions \(\psi_{x}\) and \(\psi_{y}\). The rectangular waveguide with a width \(a\) in the \(x\)-direction and a height \(b\) in the \(y\)-direction (Fig. 18.1) be aligned along the \(z\)-axis. Since the tangential component of the electrical field must vanish at conducting surfaces, the boundary conditions are

\[\begin{array}{ll}\psi_{x}(x)&=0\quad\text{for}\quad x=\pm\frac{1}{2}a\,,\\ \psi_{y}(y)&=0\quad\text{for}\quad y=\pm\frac{1}{2}b\,.\end{array} \tag{18.17}\]

The solutions must be cosine functions to meet the boundary conditions and the complete solution (18.13) for the longitudinal electric field can be expressed by

\[E_{z}=E_{0}\cos\frac{m\pi x}{a}\cos\frac{n\pi y}{b}\,\text{e}^{\text{i}\,(ot- k_{z}z)}, \tag{18.18}\]

where \(m\geq 1\) and \(n\geq 1\) are integers defining transverse field modes. The trigonometric functions are eigenfunctions of the differential equation (18.10) with boundary conditions (18.17) and the integers \(m\) and \(n\) are eigenvalues. In a similar way we get an expression for the \(z\)-component of the magnetic field strength \(B_{z}\). The boundary conditions require that the tangential magnetic field component at a conducting surface is the same inside and outside the conductor which is equivalent to the requirement that

\[\left.\frac{\partial B_{z}}{\partial x}\right|_{x=\pm\frac{1}{2}a}=0\quad\text{ and}\qquad\left.\frac{\partial B_{z}}{\partial y}\right|_{y=\pm\frac{1}{2}b}=0\,. \tag{18.19}\]

These boundary conditions can be met by sine functions and the \(z\)-component of the magnetic field strength is therefore in analogy to (18.18) given by

\[B_{z}=B_{0}\sin\frac{m\pi x}{a}\sin\frac{n\pi y}{b}\,\mathrm{e}^{\mathrm{i}\,( out-k_{z}z)}\,. \tag{18.20}\]

The cutoff frequency is the same for both the electrical and magnetic field component and is closely related to the dimension of the wave guide. With the definition (18.11) the cutoff frequency can be determined from

\[k_{\mathrm{c}}^{2}=+k_{x}^{2}+k_{y}^{2}=\left(\frac{m\pi}{a}\right)^{2}+\left( \frac{n\pi}{b}\right)^{2}. \tag{18.21}\]

All information necessary to complete the determination of field components have been collected. Using (18.3), (18.18) the component equations are with \(\frac{\partial}{\partial z}=-\mathrm{i}k_{z}\)

\[\begin{array}{lcl}-\mathrm{i}\omega B_{x}&=&\frac{\partial E_{z}}{\partial y }+\mathrm{i}k_{z}E_{y}\,,&\mathrm{i}\epsilon\mu\frac{\omega}{c}E_{x}=c\frac{ \partial B_{z}}{\partial y}+\mathrm{i}k_{z}cB_{y}\,,\\ -\mathrm{i}\omega B_{y}&=&-\mathrm{i}k_{z}E_{x}-\frac{\partial E_{x}}{\partial x }\,,&\mathrm{i}\epsilon\mu\frac{\omega}{c}E_{y}=-\mathrm{i}k_{z}cB_{x}-c\frac {\partial B_{z}}{\partial x}\,,\\ -\mathrm{i}\omega B_{z}&=&\frac{\partial E_{y}}{\partial x}-\frac{\partial E_{ x}}{\partial y}\,,&\mathrm{i}\epsilon\mu\frac{\omega}{c}E_{z}=c\frac{ \partial B_{y}}{\partial x}-c\frac{\partial B_{x}}{\partial y}.\end{array} \tag{18.22}\]

From the first four equations we may extract expressions for the transverse field components \(E_{x},E_{y},B_{x},B_{y}\) as functions of the known \(z\)-components

\[\begin{array}{lcl}E_{x}&=&-\mathrm{i}\frac{1}{k_{z}^{2}}\left(k_{z}\frac{ \partial E_{z}}{\partial x}+\frac{\omega}{c}c\frac{\partial B_{z}}{\partial y }\right)\,\\ E_{y}&=&\mathrm{i}\frac{1}{k_{z}^{2}}\left(-k_{z}\frac{\partial E_{z}}{ \partial y}+\frac{\omega}{c}c\frac{\partial B_{z}}{\partial x}\right)\,\\ cB_{x}&=&\mathrm{i}\frac{1}{k_{z}^{2}}\left(k_{z}c\frac{\partial B_{z}}{ \partial x}+\epsilon\mu\frac{\omega}{c}\frac{\partial E_{z}}{\partial y} \right)\,\\ cB_{y}&=&-\mathrm{i}\frac{1}{k_{z}^{2}}\left(k_{z}c\frac{\partial B_{z}}{ \partial y}+\epsilon\mu\frac{\omega}{c}\frac{\partial E_{z}}{\partial x} \right)\,\end{array} \tag{18.23}\]

where

\[k_{z}^{2}=k^{2}-\left(\frac{m\pi}{a}\right)^{2}-\left(\frac{n\pi}{b}\right)^ {2} \tag{18.24}\]

and \(k^{2}=\epsilon\mu\omega^{2}/c^{2}\).

By application of proper boundary conditions at the conducting surfaces of a rectangular waveguide we have derived expressions for the \(z\)-component of the electromagnetic fields and are able to formulate the remaining field components in terms of the \(z\)-component. Two fundamentally different field configurations can be distinguished depending on whether we choose \(E_{z}\) or \(B_{z}\) to vanish. All field configurations, for which \(E_{z}=0\), form the class of transverse electrical modes or short TE-modes. Similarly, all fields for which \(B_{z}=0\) form the class of transverse magnetic modes or short TM-modes. Each class of modes consists of all modes obtained by varying the indices \(m\) and \(n\). The particular choice of these mode integers is commonly included in the mode nomenclature and we speak therefore of TM\({}_{nm}\) or TE\({}_{mn}\)-modes. For the remainder of this chapter we will concentrate only on the transverse magnetic or TM-modes, since TE-modes are useless for particle acceleration. The lowest order TM-mode is the TM\({}_{11}\)-mode producing the \(z\)-component of the electrical field, which is maximum along the \(z\)-axis of the rectangular waveguide and falls off from there like a cosine function to reach zero at the metallic surfaces. Such a mode would be useful for particle acceleration if it were not for the phase velocity being larger than the speed of light. In the next subsection we will see how this mode may be used anyway. The next higher mode, the TM\({}_{21}\)-mode would have a similar distribution in the vertical plane but exhibits a node along the \(x\)-axis.

Before we continue the discussion on field configurations we note that electromagnetic waves with frequencies above cutoff frequency (\(k>k_{\rm c}\)) propagate along the axis of the rectangular waveguide. A waveguide wavelength can be defined by

\[\lambda_{z}=\frac{2\pi}{\sqrt{k^{2}-k_{\rm c}^{2}}}>\lambda\,, \tag{18.25}\]

which is always longer than the free space wavelength \(\lambda=2\pi/k\) and

\[\frac{1}{\lambda^{2}}=\frac{1}{\lambda_{z}^{2}}+\frac{1}{\lambda_{\rm c}^{2}}, \tag{18.26}\]

where \(\lambda_{\rm c}=2\pi/k_{\rm c}\).

The frequency of this traveling electromagnetic wave is from (18.25) with (18.9), (18.16)

\[\omega=\omega_{\rm c}\sqrt{1+\frac{k_{z}^{2}}{k_{\rm c}^{2}}}. \tag{18.27}\]

Electromagnetic energy travels along the waveguide with a velocity known as the group velocity defined by

\[v_{\rm g}=\frac{{\rm d}\omega}{{\rm d}k_{z}}=\frac{c}{\sqrt{\mu\epsilon}} \sqrt{1-\frac{k_{\rm c}^{2}}{k^{2}}}<\frac{c}{\sqrt{\mu\epsilon}}<c. \tag{18.28}\]In contrast to the phase velocity, the group velocity is always less than the speed of light as it should be. Rectangular waveguides are mostly used to transport high frequency microwaves from the generator to the accelerating cavity. The bandwidth of waveguides is rather broad and the mechanical tolerances relaxed. Small variation in dimension due to pressurization or evacuation to eliminate field breakdown do generally not matter.

#### Cylindrical Waveguide Modes

For accelerating cavities we try to reach the highest fields possible at a well defined wavelength. Furthermore, accelerating cavities must be operated under vacuum. These requirements result in very tight mechanical tolerances which can be met much easier in round rf-cavities. Analogous to the rectangular case we derive therefore field configurations in cylindrical cavities (Fig. 18.2). The derivation of the field configuration is similar to that for rectangular waveguides although now the wave equation (18.3) is expressed in cylindrical coordinates \((r,\varphi,z)\) and we get for the \(z\)-component of the electrical field

\[\frac{\partial^{2}E_{z}}{\partial r^{2}}+\frac{1}{r}\frac{\partial E_{z}}{ \partial r}+\frac{1}{r^{2}}\frac{\partial^{2}E_{z}}{\partial\varphi^{2}}+ \frac{\partial^{2}E_{z}}{\partial z^{2}}+k^{2}E_{z}=0. \tag{18.29}\]

with

\[k^{2}=\epsilon\mu\frac{\omega^{2}}{c^{2}}. \tag{18.30}\]

In a stationary configuration the field is expected to be periodic in \(\varphi\) while the \(z\)-dependence is the same as for rectangular waveguides. Using the derivatives

Figure 18.2: Cylindrical resonant cavity (pill box cavity)

\(-{\rm i}m\), where \(m\) is an integer eigenvalue, and \(\frac{\partial}{\partial z}=-{\rm i}k_{z}\), we get from (18.29) for the \(z\)-component of the electric field

\[\frac{\partial^{2}E_{z}}{\partial r^{2}}+\frac{1}{r}\frac{\partial E_{z}}{ \partial r}+\left(k_{\rm c}^{2}-\frac{m^{2}}{r^{2}}\right)E_{z}=0, \tag{18.31}\]

where \(k_{\rm c}^{2}=k^{2}-k_{z}^{2}\) consistent with its previous definition. This differential equation can be solved with Bessel's functions in the form [5]

\[E_{z}=E_{0}{\rm J}_{m}(k_{\rm c}r){\rm e}^{{\rm i}(\omega t-m\varphi-k_{z}z)}, \tag{18.32}\]

which must meet the boundary condition \(E_{z}=0\) for \(r=a\), where \(a\) is the radius of the cylindrical waveguide. The location of the cylindrical boundaries are determined by the roots of Bessel's functions of order \(m\). For the lowest order \(m=0\) the first root \(a_{1}\) is (see Fig. 18.3)

\[k_{\rm c}a_{1}=2.405\qquad\mbox{or at a radius}\qquad a_{1}=\frac{2.405}{k_{ \rm c}}. \tag{18.33}\]

Figure 18.3: Electromagnetic field pattern for a TM\({}_{010}\)-mode in a circular waveguide. Three dimensional field configuration (**a**) and radial dependence of fields (**b**)

To define a cylindrical cavity two counter propagating waves are created by adding end caps at \(z=\pm\frac{1}{2}d\) and with \(\mathrm{i}k_{z}=\frac{\pi p}{d}\) and (18.12)

\[k_{\mathrm{c}}^{2}=\epsilon\mu\frac{\omega^{2}}{c^{2}}-\frac{\pi^{2}p^{2}}{d^{2}}. \tag{18.34}\]

Solving for the resonance frequency \(\omega\) of the lowest order or the TM\({}_{010}\)-mode, we get with (18.33), \(m=0\) and \(p=0\)

\[\omega_{010}=\frac{c}{\sqrt{\epsilon\mu}}\frac{2.405}{a_{1}} \tag{18.35}\]

and the \(z\)-component of the electrical field is

\[E_{z}=E_{z,010}J_{0}\left(2.405\frac{r}{a_{1}}\right)\mathrm{e}^{\mathrm{i} \omega_{010}t}. \tag{18.36}\]

The waveguide wavenumber

\[k_{z}^{2}=k^{2}-k_{\mathrm{c}}^{2} \tag{18.37}\]

must be positive in order to obtain a travelling wave rather than a wave decaying exponentially along the waveguide \(\left(k_{z}^{2}<0\right)\). Solving for \(k_{z}\) we get with \(\omega_{\mathrm{c}}=ck_{\mathrm{c}}\)

\[k_{z}^{2}=k^{2}\left(1-\frac{\omega_{\mathrm{c}}^{2}}{\omega^{2}}\right). \tag{18.38}\]

The cutoff frequency is determined by the diameter of the waveguide and limits the propagation of electromagnetic waves in circular waveguides to wavelengths which are less than the diameter of the pipe. To determine the phase velocity of the wave we set \(\psi=\omega t-k_{z}z=\mathrm{const}\) and get from the derivative \(\dot{\psi}=\omega-k_{z}\dot{z}=0\) the phase velocity

\[v_{\mathrm{ph}}=\dot{z}=\frac{\omega}{k_{z}}. \tag{18.39}\]

Inserting (18.38) into (18.39) we get again a phase velocity which exceeds the velocity of light and therefore any velocity a material particle can reach. We were able to modify plane electromagnetic waves in such a way as to produce the desired longitudinal electric field component but note that these fields are not yet suitable for particle acceleration because the phase rolls over the particles and the net acceleration is zero. To make such electromagnetic waves useful for particle acceleration further modifications of the waveguide are necessary to slow down the phase velocity.

To complete our discussion we determine also the group velocity which is the velocity of electromagnetic energy transport along the waveguide. The 


### Rf-Cavities

#### Square Cavities

The waveguide modes are not yet ready to be used for particle acceleration because of excessive phase velocities. This problem can be solved by considering two waves travelling in opposite directions on the same axis of the waveguide. Both fields have the form of (18.18) and the superposition of both waves gives

\[E_{z}=2E_{0}\cos\frac{m\pi x}{a}\cos\frac{n\pi y}{b}\cos\frac{p\pi z}{d}\mathrm{ e}^{\mathrm{i}\,\omega t}, \tag{18.45}\]

where \(d\) is defined by

\[d=\frac{\pi p}{k_{z}} \tag{18.46}\]

and \(p\) is an integer.

The superposition of two equal but opposite waves form a standing wave with nodes half a waveguide length apart. Closing off the waveguide at such nodes points with a metallic surface fulfills automatically all boundary conditions. The resulting rectangular box forms a resonant cavity enclosing a standing electro-magnetic wave which can be used for particle acceleration.In analogy to the waveguide mode terminology we extend the nomenclature to cavities by adding a third index for the eigenvalue \(p\). The lowest cavity mode is the TM\({}_{110}\)-mode. The indices \(m\) and \(n\) cannot be zero because of the boundary conditions for \(E_{z}\). For \(p=0\) we find \(E_{z}\) to be constant along the axis of the cavity varying only with \(x\) and \(y\).The boundary conditions are met automatically at the end caps since with \(p=0\) also \(k=0\) and the transverse field components vanish everywhere. The electrical field configuration for the TM\({}_{110}\)-mode consists therefore of a finite \(E_{z}\)-component being constant only along \(z\) and falling off transversely from a maximum value to zero at the walls. In practical applications rectangular boxes are rarely used as accelerating cavities. There are, however, special applications like beam position monitors where rectangular cavities are preferred.

#### Cylindrical Cavity

Similarly, we may form a cylindrical cavity by two counter propagating waves. By adding endcaps at \(z=\pm\frac{1}{2}\,d\) standing waves are established and with \(k_{z}=\frac{p\pi}{d}\) we get from (18.37)

\[k_{\mathrm{c}}^{2}=\epsilon\mu\frac{\omega^{2}}{c^{2}}-\frac{p^{2}\pi^{2}}{d^{ 2}}\,. \tag{18.47}\]Solving for the resonance frequency \(\omega\) of the lowest order or the TM\({}_{010}\)-mode, we get with (18.33), \(m=0\) and \(p=0\)

\[\omega_{010}=\frac{c}{\sqrt{\epsilon\mu}}\frac{2.405}{a_{1}} \tag{18.48}\]

and the \(z\)-component of the electrical field is

\[E_{z}=2E_{z,010}\mathrm{J}_{0}\left(2.405\frac{r}{a_{1}}\right)\cos\left( \omega_{010}t\right)\,. \tag{18.49}\]

The resonance frequency is inversely proportional to the radius of the cavity and to keep the size of accelerating cavities manageable, short wave radio frequencies are chosen. For electron linear accelerators a wavelength of \(\lambda=10\,\mathrm{cm}\) is often used corresponding to a frequency of \(2997.93\,\mathrm{MHz}\) and a cavity radius of \(a_{1}=3.83\,\mathrm{cm}\). For storage rings a common frequency is \(499.65\,\mathrm{MHz}\) or \(\lambda=60\,\mathrm{cm}\) and the radius of the resonance cavity is \(a_{1}=22.97\,\mathrm{cm}\). The size of the cavities is in both cases quite reasonable. For much lower rf-frequencies the size of a resonant cavity becomes large. Where such low frequencies are desired the diameter of a cavity can be reduced at the expense of efficiency by loading it with magnetic material like ferrite with a permeability \(\mu>1\) as indicated by (18.48). This technique also allows the change of the resonant frequency during acceleration to synchronize with low energy protons, for example, which have not yet reached relativistic energies. To keep the rf-frequency synchronized with the revolution frequency, the permeability of the magnetic material in the cavity can be changed by an external electrical current. The drawback of using materials like ferrites is that they are lossy in electromagnetic fields, get hot and produce significant outgassing in vacuum environments.

The nomenclature for different modes is similar to that for rectangular waveguides and cavities. The eigenvalues are equal to the number of field maxima in \(\varphi\), \(r\) and \(z\) and are indicated as indices in this order. The TM\({}_{010}\)-mode, therefore exhibits only a radial variation of field strength independent of \(\varphi\) and \(z\). Again, we distinguish TM-modes and TE-modes but continue to consider only TM-modes for particle acceleration. Electrical fields in such a cavity have all the necessary properties for particle acceleration. Small openings along the \(z\)-axis allow the beam to pass through the cavity and gain energy from the accelerating field. Cylindrical cavities can be excited in many different modes with different frequencies. For particle acceleration the dimensions of the cavity are chosen such that at least one resonant frequency satisfies the synchronicity condition of the circular accelerator. In general this is the frequency of the TM\({}_{010}\)-mode which is also called the fundamental cavity mode or frequency.

From the expressions (18.44) we find that the lowest order TM-mode does not include transverse electrical field components since \(k_{z}=0\) and \(m=0\). The only transverse field is the azimuthal magnetic field which is with (18.49)

\[\frac{c}{\sqrt{\epsilon\mu}}B_{\psi}=-{\rm i}\,E_{z,010}{\rm J}_{1}\left(2.405 \frac{r}{a_{1}}\right){\rm e}^{{\rm i}\omega_{010}t}\,. \tag{18.50}\]

#### Energy Gain

The kinetic energy gained in such a cavity can be obtained by integrating the time dependent field along the particle path. The cavity center be located at \(z=0\) and a particle entering the cavity at time \(\omega_{010}t=-\pi/2\) or at \(z=-d/2\) may encounter the phase \(\delta\) of the microwave field. The electric field along the \(z\)-axis as seen by the particle travelling with velocity \(v\) has the form \(E_{z}=E_{z0}\sin\left(\omega\frac{z}{v}+\delta\right)\) and we get for the kinetic energy gain of a particle passing through the cavity with velocity \(v\)

\[\Delta E_{\rm kin}=eE_{z0}\int_{-\frac{1}{2}d}^{\frac{1}{2}d}\cos\left(\omega \frac{z}{v}+\delta\right){\rm d}z\,. \tag{18.51}\]

In general, the change in the particle velocity is small during passage of one rf-cavity and the integral is a maximum for \(\delta=\pi/2\) when the field reaches a maximum at the moment the particle is half way through the cavity. Defining an accelerating cavity voltage

\[V_{\rm rf}=E_{z0}d=E_{010}d \tag{18.52}\]

the kinetic energy gain is after integration

\[\Delta E_{\rm kin}=eV_{\rm rf}\frac{\sin\frac{\omega d}{2v}}{\frac{\omega d}{ 2v}}=eV_{\rm cy}\,, \tag{18.53}\]

where we have defined an effective cavity voltage and the transit-time factor is

\[T=\frac{\sin\frac{\omega d}{2v}}{\frac{\omega d}{2v}}\,. \tag{18.54}\]

The transit-time factor provides the correction on the particle acceleration due to the time variation of the field while the particles traverse the cavity. In a resonant pill box cavity (Fig. 18.4a) we have \(d=\lambda/2\) and the transit-time factor for a particle traveling approximately at the speed of light is

\[T_{\rm pillbox}=\frac{2}{\pi}<1\,. \tag{18.55}\]As the cavity length or the active accelerating gap in the cavity is reduced, the transient time factor can be increased. The simple pill box cavity may be modified by adding nose cones (Fig. 18.4b) or by adding drift tubes at the entrance and exit of the cavity as shown in Fig. 18.4c. In this case the parameter \(d\) in (18.54) is the active accelerating gap.

For small velocities (\(v\ll c\)) the transit time factor and thereby the energy gain is small or maybe even negative. Maximum energy gain is obtained for particles travelling at or close to the speed of light. Externally driven accelerating cavity

#### Rf-Cavity as an Oscillator

Accelerator cavities can be described as damped oscillators with external excitation. Damping occurs due to energy losses in the walls of the cavity and transfer of energy to the particle beam while an external rf-power source is connected to the cavity to sustain the rf-fields. Many features of an accelerating cavity can be expressed in well-known terms of a damped, externally excited harmonic oscillator which is described in the form

\[\ddot{x}+2\alpha\dot{x}+\omega_{0}^{2}x=D\mathrm{e}^{\mathrm{i}\omega t}, \tag{18.56}\]

where \(\alpha\) is the damping decrement, \(\omega_{0}\) the unperturbed oscillator frequency and \(D\) the amplitude of the external driving force with frequency \(\omega\). The equilibrium solution can be expressed in the form \(x=A\mathrm{e}^{\mathrm{i}\omega t}\), where the complex amplitude \(A\) is determined after insertion of this ansatz into (18.56)

\[A=\frac{D}{\omega_{0}^{2}-\omega^{2}+\mathrm{i}2\alpha\omega}=a\mathrm{e}^{ \mathrm{i}\psi} \tag{18.57}\]

Figure 18.4: Resonant cavities with drift tubes (schematic). (**a**) Pill box cavity. (**b**) Cavity with nose cones. (**c**) Cavity with drift tubes

The angle \(\Psi\) is the phase shift between the external excitation and the oscillator and the amplitude \(a=\)Re(\(A\)) is from (18.57)

\[a=\frac{D}{\sqrt{\left(\omega_{0}^{2}-\omega^{2}\right)^{2}+4\alpha^{2}\omega^{2 }}}\,, \tag{18.58}\]

Plotting the oscillation amplitude \(a\) as a function of the excitation frequency \(\omega\), we get the resonance curve for the oscillator as shown in Fig. 18.5. The resonance frequency at which the oscillator reaches the maximum amplitude depends on the damping and is

\[\omega_{\rm r}=\sqrt{\omega_{0}^{2}-2\alpha^{2}}. \tag{18.59}\]

For an undamped oscillator the resonance amplitude becomes infinite but is finite whenever there is damping. The oscillator can be excited within a finite distance from the resonance frequency and the width of the resonance curve at half maximum amplitude is

\[\Delta\omega_{\frac{1}{2}}\approx\pm 2\sqrt{3}\alpha\qquad\mbox{for}\qquad \alpha\ll\omega_{\rm r}\,. \tag{18.60}\]

If there were no external excitation to sustain the oscillation, the amplitude would decay like \(a\propto\)e\({}^{-\alpha t}\). The energy of the oscillator scales like \(W\propto A^{2}\) and the energy loss per unit time \(P=-{\rm d}W/{\rm d}t=2\alpha W\), which can be used to determine the quality factor of this oscillator as defined in (18.80)

\[Q=\frac{\omega_{\rm r}}{2\alpha}. \tag{18.61}\]

Figure 18.5: Resonance curve for a damped oscillator

The quality factor is reduced as damping increases. For the case of an accelerating cavity, we expect therefore a higher \(Q\)-value called the unloaded \(Q_{0}\) when there is no beam, and a reduced quality factor called loaded \(Q\) when there is a beam extracting energy from the cavity. The time constant for the decay of oscillation amplitudes or the cavity damping time is

\[t_{\mathrm{d}}=\frac{1}{\alpha}=\frac{2Q}{\omega_{\mathrm{r}}} \tag{18.62}\]

and the field amplitude decays to 1/e during \(Q/\pi\) oscillations.

Coming back to the equation of motion (18.56) for this oscillator, we have the solution

\[x(t)=a\,\mathrm{e}^{\mathrm{i}(\omega t+\psi)} \tag{18.63}\]

noting that the oscillator assumes the same frequency as the external excitation but is out of synchronism by the phase \(\Psi\). The magnitude and sign of this phase shift depends on the excitation frequency and can be derived from (18.57) in the form

\[\omega_{\mathrm{r}}^{2}-\omega^{2}+\mathrm{i}2\alpha\omega=\frac{D}{a}\, \mathrm{e}^{-\mathrm{i}\psi}=\frac{D}{a}\left(\cos\Psi-\mathrm{i}\sin\Psi \right)\,.\]

Both the real and imaginary parts must separately be equal and we get for the phase shift between excitation and oscillator

\[\cot\Psi=\frac{\omega^{2}-\omega_{\mathrm{r}}^{2}}{2\alpha\omega}\approx 2Q \frac{\omega-\omega_{\mathrm{r}}}{\omega_{\mathrm{r}}}\,, \tag{18.64}\]

where we have made use of (18.61) and the approximation \(\omega\approx\omega_{\mathrm{r}}\). For excitation at the resonance frequency we find the oscillator to lag behind the driving force by \(\frac{1}{2}\pi\) and is almost in phase or totally out of phase for very low or very high frequencies, respectively. In rf-jargon this phase shift is called the tuning angle.

#### Cavity Losses and Shunt Impedance

Radio frequency fields can be enclosed within conducting surfaces only because electrical surface currents are induced by these fields which provide the shielding effect. For a perfect conductor with infinite surface conductivity these currents would be lossless and the excitation of such a cavity would persist indefinitely. This situation is achieved to a considerable degree, albeit not perfect, in superconducting cavities. In warm cavities constructed of copper or aluminum the finite resistance of the material causes surface currents to produce heating losses leading to a depletion of field energy. To sustain a steady field in the cavity, radio frequency power must be supplied continuously. The surface currents in the conducting cavity boundaries can be derived from Maxwell's curl equation or Ampere's law (18.2). In cylindrical coordinates this vector equation becomes for the lowest order TM-mode in component form

\[-\frac{\partial B_{\varphi}}{\partial z} = \mu_{0}\mu j_{r}\,,\] \[0 = j_{\varphi}\,, \tag{18.65}\] \[\frac{\partial rB_{\varphi}}{r\,\partial r} = \frac{B_{\varphi}}{r}\,+\,\frac{\partial B_{\varphi}}{\partial r }\,=\,\mu_{0}\mu j_{z}+\mathrm{i}\,\frac{\epsilon\mu}{c^{2}}\omega E_{z}\,.\]

Because we do not consider perfectly but only well conducting boundaries, we expect fields and surface currents to penetrate somewhat into the conducting material. The depth of penetration of fields and surface currents into the conductor is well-known as the skin depth [3]

\[\delta_{\mathrm{s}} = \sqrt{\frac{2}{\mu_{0}\mu_{\mathrm{w}}\omega\,\sigma_{\mathrm{w} }}}\,, \tag{18.66}\]

where \(\sigma_{\mathrm{w}}\) is the conductivity of the cavity wall and \(\mu_{\mathrm{w}}\) the permeability of the wall material. The azimuthal magnetic field component induces surface currents in the cylindrical walls as well as in the end caps. In both cases the magnetic field decays within a skin depth from the surface inside the conductor. The first Eq. (18.65) applies to the end caps and the integral through the skin depth is

\[\int_{S}^{S+\delta_{\mathrm{s}}}\frac{\partial B_{\varphi}(r)}{\partial z} \mathrm{d}z\approx B_{\varphi}(r)|_{S}^{S+\delta_{\mathrm{s}}}\approx-B_{ \varphi}(r,S)\,, \tag{18.67}\]

since \(B_{\varphi}(r,S+\delta_{\mathrm{s}})\approx 0\) just under the surface \(S\) of the wall. We integrate also the third Eq. (18.65) at the cylindrical walls and get for the first term \(\int B_{\varphi}/r\,\mathrm{d}r\approx B_{\varphi}\delta_{\mathrm{s}}/a_{1}\), which is negligible small, while the second term has a form similar to (18.67). The electrical term \(E_{z}\) vanishes because of the boundary condition and the surface current densities for the cylindrical wall and end caps, respectively, are therefore related to the magnetic fields by

\[\begin{split}\mu_{0}\mu j_{z}\delta_{\mathrm{s}}&= B_{\varphi}(a_{1},z)\,,\\ \mu_{0}\mu j_{r}\delta_{\mathrm{s}}&=B_{\varphi}(r, \,\pm\,\tfrac{1}{2}d)\,.\end{split} \tag{18.68}\]

The cavity losses per unit wall surface area are given by

\[\frac{\mathrm{d}P_{\mathrm{cy}}}{\mathrm{d}S}=\tilde{r}_{\mathrm{s}}j_{ \mathrm{s}}^{2}\,, \tag{18.69}\]where \(j_{\rm s}\) is the surface current density and \(\tilde{r}_{\rm s}\) is the surface resistance given by

\[\tilde{r}_{\rm s}=\sqrt{\frac{\mu_{0}\mu_{\rm w}\omega}{2\,\sigma_{\rm w}}}. \tag{18.70}\]

With \(j_{\rm s}=j_{r,z}\,\delta_{\rm s}\), (18.50), (18.66) and the integration of (18.69) is performed over all inside surfaces of the cavity to give

\[P_{\rm cy}=\tfrac{1}{4}\epsilon_{0}\omega\delta_{\rm s}\epsilon\frac{\mu_{\rm w }}{\mu}E_{010}^{2}\int_{S}\mathrm{J}_{1}^{2}\left(2.405\frac{r}{a_{1}}\right) \mathrm{d}S\,, \tag{18.71}\]

where \(\epsilon\) and \(\mu\) is the dielectric constant and permeability of the material inside the cavity, respectively and \(\mu_{\rm w}\) the wall permeability. Evaluating the integral over all surfaces, we get for the cylindrical wall the integral value \(2\pi\,a_{1}dJ_{1}^{2}(2.405)\). For each of the two end caps the integral \(2\pi\,\int_{0}^{a_{1}}\mathrm{J}_{1}^{2}(2.405\frac{r}{a_{1}})\,r\,\mathrm{d}r\) must be evaluated and is from integration tables [6]

\[2\pi\int_{0}^{a_{1}}\mathrm{J}_{1}^{2}\left(2.405\frac{r}{a_{1}}\right)r \mathrm{d}r=\pi\,a_{1}^{2}\mathrm{J}_{1}^{2}(2.405)\;. \tag{18.72}\]

The total cavity wall losses become finally with \(V_{\rm rf}=E_{010}d\) from (18.52)

\[P_{\rm cy}= \tfrac{1}{2}\pi\epsilon_{0}\omega\delta_{\rm s}\epsilon\frac{\mu _{\rm w}}{\mu}V_{\rm rf}^{2}\,J_{1}^{2}(2.405)\frac{a_{1}(a_{1}+d)}{d^{2}}\;. \tag{18.73}\]

It is convenient to separate fixed cavity parameters from adjustable parameters. Once the cavity is constructed, the only adjustable parameter is the strength of the electrical field \(E_{010}\) or the effective cavity voltage \(V_{\rm cy}\). Expressing the cavity losses in terms of an impedance, we get from (18.73) and (18.53)

\[P_{\rm cy}=\frac{V_{\rm cy}^{2}}{2R_{\rm s}}\,, \tag{18.74}\]

where the cavity shunt impedance including transient time factor is defined by1

Footnote 1: The shunt impedance is defined in the literature sometimes by \(P_{\rm cy}=V_{\rm cy}^{2}/R_{\rm s}\) in which case the numerical value of the shunt impedance is larger by a factor of two.

\[R_{\rm s}= \frac{1}{\pi\epsilon_{0}}\frac{1}{\omega\delta_{\rm s}\epsilon} \frac{\mu}{\mu_{\rm w}}\,\frac{d^{2}}{a_{1}(a_{1}+d)}\frac{1}{\mathrm{J}_{1}^ {2}(2.405)}\left(\frac{\sin\frac{\omega d}{2v}}{\frac{\omega d}{2v}}\right)^{ 2}. \tag{18.75}\]

The factor of 2 in (18.74) results from the fact that on average the rf-voltage is \(\left\langle V_{\rm cy}^{2}=\hat{V}_{\rm cy}^{2}\sin^{2}\omega t\right\rangle= \frac{1}{2}\hat{V}_{\rm cy}^{2}\). In accelerator design, we prefer sometimes to use the shunt impedance per unit length or the specific shunt impedance. The required length depends on the accelerating voltage needed and the rf-power available. With the cavity shunt impedance per unit length

\[r_{\rm s}=\frac{R_{\rm s}}{d} \tag{18.76}\]

the cavity losses are instead of (18.74)

\[P_{\rm cy}=\frac{\hat{V}_{\rm cy}^{2}}{2r_{\rm s}L_{\rm cy}}\,, \tag{18.77}\]

where \(L_{\rm cy}\) is the total length of all cavities producing the voltage \(\hat{V}_{\rm cy}\). Since the cavity shunt impedance scales like \(R_{\rm s}\propto 1/\sqrt{\omega}\) and the length for a resonant cavity like \(d\propto 1/\omega\), the specific shunt impedance is proportional to the square root of the rf-frequency \(r_{\rm s}\propto\sqrt{\omega}\) favoring high frequencies. A practical limit is reached when the cavity apertures become too small for the particle beam to pass through or when the size of the cavities prevents an efficient cooling of wall losses.

As an example, we calculate from (18.75) the shunt impedance for a pill box cavity designed for a resonance frequency of 358 MHz. The wavelength is \(\lambda=85\,\)cm, the cavity length \(d=42.5\,\)cm and the cavity radius \(a_{1}=32.535\,\)cm. This cavity was constructed with nose cones for the storage ring PEP [7] from aluminum. With a skin depth of \(\delta_{\rm s}=4.44\,\mu\)m the specific shunt impedance becomes \(r_{\rm s}=15.2\) M\(\Omega\)/m while the measured value for this cavity is 18.0 M\(\Omega\)/m.

The difference is due to two competing effects. The open aperture along the axis for the beam has the tendency to reduce the shunt impedance while the nose cones being a part of the actual cavity increase the transient time factor and thereby the effective shunt impedance (18.75). The simple example of a pill box cavity produces rather accurate results, however, for more precise estimates computer programs have been developed to calculate the mode frequencies and shunt impedances for all modes in arbitrary rotational symmetric cavities (for example, SUPERFISH [8] or URMEL [9]). More sophisticated three-dimensional programs are available (for example, MAFIA [9]) to simulate rf-properties of arbitrary forms of cavities.

The specific shunt impedance for a pill box cavity can be expressed in a simple form as a function of the rf-frequency only and is for realistic cavities approximately

\[\begin{array}{ll}r_{\rm s}({\rm M}\Omega/{\rm m})&\approx\,1.28\sqrt{f_{\rm rf }\,({\rm MHz})}&\qquad\mbox{for copper and}\\ r_{\rm s}({\rm M}\Omega/{\rm m})&\approx\,1.06\sqrt{f_{\rm rf}\,({\rm MHz})} &\qquad\mbox{for aluminum}\,.\end{array} \tag{18.78}\]

The shunt impedance should be maximum in order to minimize cavity losses for a given acceleration. Since the interior of the cavity must be evacuated \(\mu=\epsilon=1\) and \(\mu_{\rm w}=1\) because we do not consider magnetic materials to construct a cavity. The only adjustable design parameters left are the skin depth and the transient time factor. The skin depth can be minimized by using well conducting materials like copper or aluminum.

To derive the quality factor of the cavity the energy \(W\) stored in the electromagnetic field within the cavity must be calculated. The field energy is the volume integral of the square of the electrical or magnetic field and we have in case of a TM\({}_{010}\)-mode with \(W=\frac{1}{2}\epsilon_{0}\,\epsilon\int_{V}E_{z}^{2}\,\mathrm{d}V\) and (18.49) for the stored cavity energy

\[W=\tfrac{1}{2}\epsilon_{0}\epsilon E_{010}^{2}da_{1}^{2}J_{1}^{2}(2.405)\,. \tag{18.79}\]

The quality factor \(Q\) of a resonator is defined as the ratio of the stored energy to the energy loss per radian

\[Q=2\pi\,\frac{\text{stored energy}}{\text{energy loss/cycle}}=\omega\frac{W}{P_{ \text{cy}}}\, \tag{18.80}\]

or with (18.73), (18.79)

\[Q=\frac{d}{\delta_{\text{s}}}\frac{\mu_{\text{w}}}{\mu}\frac{a_{1}}{a_{1}+d}. \tag{18.81}\]

The quality factor determines the cavity time constant since the fields decay exponentially like \(\mathrm{e}^{-t/\tau_{\text{cy}}}\) due to wall losses, where \(\tau_{\text{cy}}\) is the cavity time constant and the decay rate of the stored energy in the cavity is

\[\frac{\mathrm{d}W}{\mathrm{d}t}=-\frac{2}{\tau_{\text{cy}}}W\,. \tag{18.82}\]

The change in the stored energy is equal to the cavity losses \(P_{\text{cy}}\) and the cavity time constant is with (18.80)

\[\tau_{\text{cy}}=\frac{2W}{P_{\text{cy}}}=\frac{2Q}{\omega}\, \tag{18.83}\]

which is equal to (18.62) and also called the cavity filling time because it describes the build up time of fields in a cavity following a sudden application of rf-power.

### 18.3 Rf-Parameters

A variety of rf-parameters has to be chosen for a circular accelerator. Some parameters relate directly to beam stability criteria and are therefore easy to determine. Other parameters have less of an impact on beam stability and are often determined by nonphysical criteria like availability and economics. Before rf-parameters can be determined a few accelerator and lattice parameters must be known. Specifically, we need to know the desired minimum and maximum beam energy, the beam current, the circumference of the ring, the momentum compaction factor, and the bending radius of the magnets. Further, we make a choice of the maximum desired rate of particle acceleration per turn or determine the energy loss per turn to synchrotron radiation which needs to be compensated. During the following discussion we assume that these parameters are known.

One of the most prominent parameters for rf-accelerating systems is the rf-frequency of the electromagnetic fields. For highly relativistic beams there is no fundamental reason for a particular choice of the rf-frequency and it can therefore be selected on technical and economic grounds. The rf-frequency must, however, be an integer multiple, the harmonic number, of the particle revolution frequency. The harmonic number can be any integer from a beam stability point of view. In specific cases, the harmonic number need to be a multiple of a smaller number. Considering, for example, a colliding beam facility with \(N_{\rm IP}\) collision points an optimum harmonic number is divisible by \(N_{\rm IP}/2\). In this case \(N_{\rm IP}/2\) bunches could be filled in each of the two counter rotating beams leading to a maximum collision rate. Other such considerations may require the harmonic number to contain additional factors. In general, most flexibility is obtained if the harmonic number is divisible by small prime numbers.

Within these considerations the harmonic number can be chosen from a large range of rf-frequencies without generally affecting beam stability. Given complete freedom of choice, however, a low frequency is preferable to a high frequency. For low rf-frequencies the bunch length is longer and electromagnetic interaction with the beam environment is reduced since high frequency modes are not excited significantly. A longer bunch length also reduces the particle density in the bunch and thereby potentially troublesome intra-beam scattering [10, 11]. In proton and heavy ion beams a longer bunch length leads to a reduced space charge tune shift and therefore allows to accelerate a higher beam intensity. For these reasons lower frequency systems are used mostly in low energy circular accelerators. The downside of low rf-frequencies is the fact that the accelerating cavities become very large or less efficient and rf-sources are limited in power capability.

The size of circular accelerators imposes a lower limit on the rf-frequency since the synchronicity condition requires that the rf-frequency be at least equal to the revolution frequency in which case the harmonic number is equal to unity. A higher harmonic number to accommodate more than a single particle bunch further increases the required rf-frequency. Most electron and very high energy proton accelerators operate at rf-frequencies of a few hundred MHz, while lower frequencies are preferred for ion or medium energy proton accelerators.

For some applications it is critical to obtain short particle bunches which is much easier to achieve with a high rf-frequency. The appropriate choice of the rf-frequency therefore dependents much on the desired parameters for the particular application and is mostly chosen as a compromise between competing requirements including economic considerations like cost and availability.

#### Synchronous Phase and Rf-voltage

The most common use of an rf-system is for acceleration while particles pass through a resonant cavity at the moment when the voltage reaches the crest of the rf-wave and particles gain a kinetic energy equivalent to the full cavity voltage. This is the general accelerating mode in linear accelerators. In circular accelerators, however, the principle of phase focusing requires that particles be accelerated off the crest at a synchronous phase \(\psi_{\mathrm{s}}\), where the effective accelerating voltage is \(V_{\mathrm{a}}=\hat{V}_{\mathrm{cy}}\sin\psi_{\mathrm{s}}\). The peak rf-voltage \(\hat{V}_{\mathrm{cy}}\) and the synchronous phase are determined by the desired energy acceptance and acceleration per turn.

The energy acceptance of a circular accelerator has been derived in Chap. 9, is proportional to the square root of the cavity voltage and must be adjusted for the larger of several energy acceptance requirements. To successfully inject a beam into a circular accelerator the voltage must be sufficiently large to accept the finite energy spread in the injected beam. In addition, any phase spread or timing error of the incoming beam translates into energy errors due to synchrotron oscillations. For acceleration of a high intensity beam an additional allowance to the rf-voltage must be made to compensate beam loading, which will be discussed later in more detail.

After injection into a circular accelerator an electron beam may change considerably its energy spread due to quantum excitation as a result of emitting synchrotron radiation. This energy spread has a Gaussian distribution and to assure long beam lifetime the energy acceptance must be large enough to contain at least seven standard deviations. In proton and heavy ion accelerators some phase space manipulation may be required during the injection process which contributes another lower limit for the required rf-voltage. In general, there are a number of requirements that determine the ultimate energy acceptance of an accelerator and the most stringent requirement may very well be different for different accelerator designs and applications. Generally, circular accelerators are designed for an energy acceptance of a few percent.

### 18.4 Linear Accelerator

The phase velocity \(v_{\mathrm{ph}}\) must be equal to the particle velocity \(v_{\mathrm{p}}\) for efficient acceleration and we need therefore to modify or "load" the wave guide structure to reduce the phase velocity to become equal to the particle velocity. This can be done by inserting metallic structures into the aperture of the circular wave guide. Many different ways are possible, but we will consider only the disk loaded waveguide which is the most common accelerating structure for electron linear accelerators.

In a disk loaded waveguide metallic plates are inserted normal to the waveguide axis at periodic intervals with iris apertures to allow for the passage of the particle beam as shown in Fig. 18.6.

The boundary conditions and therefore the electromagnetic fields in such a structure are significantly more complicated than those in a simple circular tube. It would exceed the goal of this text to derive the theory of disk loaded waveguides and the interested reader is referred to the review article by Slater [12].

Insertion of disks in periodic intervals into a uniform waveguide essentially creates a sequence of cavities with electromagnetic coupling through either the central hole, holes at some radius on the disks or external coupling cavities. The whole arrangement of cells acts like a band pass filter allowing electromagnetic fields of certain frequencies to propagate. By proper choice of the geometric dimensions the pass band can be adjusted to the desired frequency and the phase velocity can be designed to be equal to the velocity of the particles. For electron linear accelerators the phase velocity is commonly adjusted to the velocity of light since electrons quickly reach such a velocity.

##### Basic Waveguide Parameters

Without going into structure design and detailed determination of geometric parameters we can derive parameters relating to the acceleration capability of such structures. Conservation of energy requires that

\[\frac{\partial W}{\partial t}+\frac{\partial P}{\partial z}+P_{\rm w}+nevE_{z} =0, \tag{18.84}\]

where \(W\) is the stored energy per unit length, \(P\) the energy flux along \(z\), \(P_{\rm w}\) wall losses per unit length and \(nev\,E_{z}\) the energy transferred to \(n\) particles with charge \(e\) each moving with the velocity \(v\) in the electric field \(E_{z}\). The wall losses are related to the quality factor \(Q\) of the structure defined by

\[Q=\frac{\omega W}{P_{\rm w}}, \tag{18.85}\]

Figure 18.6: Disk loaded accelerating structure for an electron linear accelerator (schematic)

where \(P_{\rm w}/\omega\) are wall losses per unit length and per radian of field oscillation. The energy flux \(P\) is with the group velocity \(v_{\rm g}\)

\[P=v_{\rm g}W. \tag{18.86}\]

In case of equilibrium, the stored energy in the accelerating structure does not change with time, \(\partial W/\partial t=0\), and

\[\frac{\partial P}{\partial z}=-P_{\rm w}-i_{\rm b}E_{z}=-\frac{\omega P}{v_{ \rm g}Q}-i_{\rm b}E_{z}\,, \tag{18.87}\]

where \(i_{\rm b}=nev\) is the beam current. Considering the case of small beam loading \(i_{\rm b}\,E_{z}\ll\omega P/(v_{\rm g}Q)\) we may integrate (18.87) to get

\[P=P_{0}\exp\left(-\frac{\omega}{v_{\rm g}Q}z\right)=P_{0}\,{\rm e}^{-2\alpha z}\,, \tag{18.88}\]

where we have defined the attenuation coefficient

\[2\alpha=\frac{\omega}{v_{\rm g}Q}. \tag{18.89}\]

Equation (18.88) shows an exponential decay of the energy flux along the accelerating structure with the attenuation coefficient \(2\alpha\). The wall losses are often expressed in terms of the total voltage or the electrical field defined by

\[P_{\rm w}=\frac{\hat{V}_{0}^{2}}{Z_{\rm s}L}=\frac{\hat{E}^{2}}{r_{\rm s}}, \tag{18.90}\]

where \(Z_{\rm s}=r_{\rm s}L\) is the shunt impedance for the whole section, \(\hat{E}\) the maximum value of the accelerating field, \(E_{z}=\hat{E}\cos\psi_{\rm s}\), \(\psi_{\rm s}\) the synchronous phase at which the particle interacts with the wave, \(r_{\rm s}\) the shunt impedance per unit length, and \(L\) the length of the cavity. From (18.90) we get with (18.87) and (18.89) for negligible beam current the accelerating field

\[\hat{E}^{2}=\frac{\omega}{v_{\rm g}}\frac{r_{\rm s}}{Q}P=2\alpha r_{\rm s}P. \tag{18.91}\]

The total accelerating voltage along a structure of length \(L\) is

\[V_{0}=\int_{0}^{L}E_{z}{\rm d}z=\hat{E}\cos\psi_{\rm s}\int_{0}^{L}{\rm e}^{- \alpha z}\,{\rm d}z \tag{18.92}\]or after integration

\[V_{0}=\frac{1-\mathrm{e}^{-\alpha L}}{\alpha}\hat{E}\cos\psi_{\mathrm{s}}\,. \tag{18.93}\]

Defining an attenuation factor \(\tau\) by

\[\tau=\alpha L \tag{18.94}\]

we get with (18.91) for the total accelerating voltage per section of length \(L\)

\[V_{0}=\sqrt{r_{\mathrm{s}}LP_{0}}\sqrt{2\tau}\,\frac{1-\mathrm{e}^{-\tau}}{ \tau}\cos\psi_{\mathrm{s}}\,. \tag{18.95}\]

The maximum energy is obtained if the particles are accelerated at the crest of the wave, where \(\psi_{\mathrm{s}}=0\).

Tacitly it has been assumed that the shunt impedance \(r_{\mathrm{s}}\) is constant resulting in a variation of the electrical field strength along the accelerating section. Such a structure is called a constant impedance structure and is characterized physically by equal geometric dimensions for all cells.

In a constant impedance structure the electric field is maximum at the beginning of the section and drops off toward the end of the section. A more efficient use of accelerating sections would keep the electric field at the maximum possible value just below field break down throughout the whole section. A structure with such characteristics is called a constant gradient structure because the field is now constant along the structure.

As an example for an electron linear accelerator, the SLAC constant gradient linac structure has the following parameters [13]

\[\begin{array}{ll}f_{\mathrm{rf}}=2856\;\mathrm{MHz}&L=10\,\mathrm{ft}=3.048 \,\mathrm{m}\\ r_{\mathrm{s}}=53\;\mathrm{M}\Omega/\mathrm{m}&a_{\mathrm{i}}=0.040\;\mathrm{m }\\ Q\approx 12000&\tau=0.57\end{array} \tag{18.96}\]

A constant gradient structure can be realized by varying the iris holes in the disks to smaller and smaller apertures along the section. This kind of structure is actually used in the SLAC accelerator as well as in most modern linear electron accelerators. The field \(\hat{E}=\mathrm{const}\) and therefore from (18.88) with (18.94)

\[\frac{\partial P}{\partial z}=\frac{P(L)-P_{0}}{L}=-(1-\mathrm{e}^{-2\tau}) \frac{P_{0}}{L} \tag{18.97}\]

On the other hand, we have from (18.87)

\[\frac{\partial P}{\partial z}=-\frac{\omega P_{0}}{Qv_{\mathrm{g}}}=\mathrm{ const} \tag{18.98}\]and to make \(\partial P/\partial z\) constant the group velocity must vary linearly with the local rf-power like

\[v_{\rm g}\sim P(z)=P_{0}+\frac{\partial P}{\partial z}z\,. \tag{18.99}\]

Furthermore, since \(\partial P/\partial z<0\) the group velocity is made to decrease along the section by reducing gradually the iris radii. From (18.98)

\[v_{\rm g}(z)=-\frac{\omega}{Q}\frac{P(z)}{\partial P/\partial z} \tag{18.100}\]

or with (18.97)

\[v_{\rm g}(z)=-\frac{\omega}{Q}\frac{P_{0}+\frac{\partial P}{\partial z}z}{ \partial P/\partial z}=+\frac{\omega}{Q}\frac{L-(1-{\rm e}^{-2\tau})\,z}{1-{ \rm e}^{-2\tau}} \tag{18.101}\]

and the filling time is after integration of (18.101)

\[t_{\rm F}=\int_{0}^{L}\frac{{\rm d}z}{v_{\rm g}}=2\tau\frac{Q}{\omega}. \tag{18.102}\]

The electric field in the accelerating section is from (18.90) with (18.87)

\[\hat{E}=\sqrt{r_{\rm s}}\,\left|\frac{\partial P}{\partial z}\right| \tag{18.103}\]

and the total accelerating voltage \(V_{0}\) or gain in kinetic energy per section is

\[\Delta E_{\rm kin}=eV_{0}=e\int_{0}^{L}E_{z}{\rm d}z=e\sqrt{r_{\rm s}LP_{0}} \sqrt{1-{\rm e}^{-2\tau}}\cos\psi_{\rm s}\,, \tag{18.104}\]

where \(\psi_{\rm s}\) is the synchronous phase at which the particles travel with the electromagnetic wave. The energy gain scales with the square root of the accelerating section length and rf-power delivered.

As a numerical example, we find for the SLAC structure from (18.104) the gain of kinetic energy per 10 ft section as

\[\Delta E_{\rm kin}\,({\rm MeV})=10.48\sqrt{P_{0}\,({\rm MW})}\,, \tag{18.105}\]

where \(P_{0}\) is the rf-power delivered to the section. The energy gain (18.105) is the maximum value possible ignoring beam loading or energy extraction from the fields by the beam. The total accelerating voltage is reduced when we include beam loading due to a pulse current \(i_{\rm b}\). Referring the interested reader to reference [13]we only quote the result for the energy gain in a linear accelerator with constant gradient sections including beam loading

\[V_{i}=\sqrt{r_{\mathrm{s}}LP_{0}}\sqrt{1-\mathrm{e}^{-2\tau}}-\tfrac{1}{2}i_{ \mathrm{b}}r_{\mathrm{s}}L\left(1-\frac{2\tau\mathrm{e}^{-2\tau}}{1-\mathrm{e} ^{-2\tau}}\right)\,. \tag{18.106}\]

For the SLAC linac structure this equation becomes with \(\tau=0.57\)

\[E_{\mathrm{kin}}\left(\mathrm{MeV}\right)=10.48\sqrt{P_{0}\left(\mathrm{MW} \right)}-37.47i_{\mathrm{b}}\left(\mathrm{A}\right)\,. \tag{18.107}\]

The beam loading depends greatly on the choice of the attenuation factor \(\tau\) as is shown in Figs. 18.7 and 18.8 where the coefficients \(f_{\mathrm{v}}=\sqrt{1-\mathrm{e}^{2\tau}}\) and \(f_{\mathrm{i}}=\frac{1}{2}\left(1-\frac{2\tau\,\mathrm{e}^{-2\tau}}{1- \mathrm{e}^{-2\tau}}\right)\) are plotted as functions of \(\tau\). Both coefficients increase as the attenuation factor is increased and reach asymptotic limits. The ratio \(f_{\mathrm{v}}/f_{\mathrm{i}}\), however, decreases from infinity to a factor two which means that beam loading occurs much stronger for large values of the attenuation factor compared to low values. During the

Figure 18.8: Beam loading coefficient \(f_{\mathrm{i}}\) as a function of \(\tau\)

Figure 18.7: Energy coefficient \(f_{\mathrm{v}}\) as a function of \(\tau\)

design of the linac structure it is therefore useful to know the intended use requiring different optimization for high-energy or high-current acceleration.

We may also ask for the efficiency of transferring rf-power into beam power which is defined by

\[\eta=\frac{i_{\mathrm{b}}V_{\mathrm{i}}}{P_{0}}=i_{\mathrm{b}}\sqrt{\frac{r_{ \mathrm{s}}L}{P_{0}}}\sqrt{1-\mathrm{e}^{-2\tau}}-\frac{1}{2}i_{\mathrm{b}}^{2} r_{\mathrm{s}}\frac{L}{P_{0}}\left(1-\frac{2\tau\,\mathrm{e}^{-2\tau}}{1- \mathrm{e}^{-2\tau}}\right)\,. \tag{18.108}\]

The linac efficiency has clearly a maximum and the optimum beam current is

\[i_{\mathrm{b,opt}}=\sqrt{\frac{P_{0}}{r_{\mathrm{s}}L}}\frac{(1-\mathrm{e}^{- 2\tau})^{3/2}}{1-(1+2\tau)\mathrm{e}^{-2\tau}}\,. \tag{18.109}\]

The optimum beam current is plotted in Fig. 18.9 as a function of the attenuation coefficient \(\tau\) and the linac efficiency is shown in Fig. 18.10 as a function of beam current in units of the optimum current with the attenuation factor as a parameter.

Figure 18.9: Optimum beam current as a function of \(\tau\)

Figure 18.10: Linac efficiency as a function of beam current

The optimum beam current increases as the attenuation factor is reduced while the linac efficiency reaches a maximum for the optimum beam current.

#### Particle Capture in a Linear Accelerator Field*

The capture of particles and the resulting particle energy at the end of the accelerating section depends greatly on the relative synchronism of the particle and wave motion. If particles with velocity \(v_{\rm p}\) are injected at low energy \(\left(v_{\rm p}\ll c\right)\) into an accelerator section designed for a phase velocity \(v_{\rm ph}\geq v_{\rm p}\) the electromagnetic wave would roll over the particles with reduced acceleration. The particle velocity and phase velocity must be equal or at least close to each other. Because small mismatches are quite common, we will discuss particle dynamics under those circumstances and note that there is no fundamental difference between electron and proton linear accelerators. The following discussion is therefore applicable to any particle type being accelerated by traveling electromagnetic fields in a linear accelerator.

We observe the relative motion of both the particle and the wave from the laboratory system. During the time \(\Delta t\) particles move a distance \(\Delta z_{\rm p}=v_{\rm p}\,\Delta t\) and the wave a distance \(\Delta z_{\rm ph}=v_{\rm ph}\,\Delta t\). The difference in the distance traveled can be expressed in terms of a phase shift

\[\Delta\psi=-k(\Delta z_{\rm ph}-\Delta z_{\rm p})=-k(v_{\rm ph}-v_{\rm p})\, \frac{\Delta z_{\rm p}}{v_{\rm p}}\,. \tag{18.110}\]

The wave number \(k\) is

\[k=\frac{\omega}{v_{\rm ph}}=\frac{2\pi c}{\lambda_{\rm rf}v_{\rm ph}} \tag{18.111}\]

and inserted into (18.110) the relative phase shift over a distance \(\Delta z_{\rm p}\) becomes

\[\Delta\psi=-\frac{2\pi c}{\lambda_{\rm rf}}\frac{v_{\rm ph}-v_{\rm p}}{v_{\rm ph }v_{\rm p}}\,\Delta z_{\rm p}\,. \tag{18.112}\]

To complete the equation of motion we consider the energy gain of the particles along the same distance \(\Delta z_{\rm p}\) which is

\[\Delta E_{\rm kin}=-eE_{z}(\psi)\Delta z_{\rm p}. \tag{18.113}\]

Equations (18.112) and (18.113) form the equations of motion for particles in phase space. Both equations are written as difference equations for numerical integration since no analytic solution exists. For the most trivial case \(v_{\rm ph}=v_{\rm p}\) and \(\psi={\rm const}\) allowing easy integration of (18.113). This trivial case becomes the overwhelming common case for electrons which reach a velocity very close to the speed of light. Consistent with this, most accelerating sections are dimensioned for a phase velocity equal to the speed of light.

As an illustrative example, we integrate (18.112) and (18.113) numerically to determine the beam parameters at the end of a single 3 m long accelerating section \(\left(v_{\rm ph}=c\right)\) for an initial particle distribution in phase and momentum at the entrance to the accelerating section. This situation is demonstrated in Fig. 18.11 for a constant field gradient of \(\hat{E}=12.0\) MeV/m. The momentum and phase at the end of the accelerating section are shown as functions of the initial momentum and phase. We note from Fig. 18.11 that particles can be captured in the accelerating field only in the vicinity of \(\psi_{0}\approx 0\) to \(+90^{\circ}\) at almost any initial phase and momentum. At phases from \(\psi_{0}\approx-45\) to \(-160^{\circ}\) slow particles at sub-relativistic energies loose whatever little energy they had to move randomly in the rf-wave rolling over them. On the other hand, particles which enter the accelerating section ahead of the crest (\(\psi_{0}\gtrsim 0^{\circ}\)) gain maximum momentum while the wave's crest moves over them.

Such diagrams calculated for particular parameters under consideration provide valuable information needed to prepare the beam for optimum acceleration. The most forgiving operating parameters are, where the contour lines are far apart. In those areas a spread in initial phase or energy has little effect on the final phase or energy. If a beam with a small energy spread at the end of acceleration is desired, the initial phase should be chosen to be at small positive values or just ahead of the wave crest as shown in Fig. 18.11. Even for a long bunch the final energy spread is small while reaching the highest total energy.

On the other hand, if a short bunch length at the end of acceleration is of biggest importance, an initial phase of around \(\psi_{0}\approx 100^{\circ}\) seems to be more appropriate. In

Figure 18.11: Capture of electrons in a 3 m linac section for initial phase \(\psi_{0}\) and initial kinetic energy \(E_{\rm kin,0}\). Contour lines are lines of constant particle energy in MeV at the end of the section. The phase \(\psi_{0}=0\) corresponds to the crest of the accelerating wave

this case, however, the final energy is lower than the maximum possible energy and the energy spread is large.

Once the particular particle distribution delivered to the linear accelerator and the desired beam quality at the end is known one can use Fig. 18.11 for optimization. Conversely such diagrams can be used to judge the feasibility of a particular design to reach the desired beam characteristics.

### Preinjector and Beam Preparation*

Although the proper choice of the initial rf-phase with respect to the particle beam greatly determines the final beam quality, the flexibility of such adjustments is limited. Special attention must be given to the preparation of the beam before acceleration. In most cases, particles are generated in a continuous stream or from a microwave source of different frequency. Depending on the particle source, special devices are used for initial acceleration and bunching of the beam. We will discuss basic principles of beam preparation.

#### 18.5.1 Prebuncher

Many particle sources, be it for electrons, protons or ions, produce a continuous stream of particles at modest energies limited by electrostatic acceleration between two electrodes. Not all particles of such a beam will be accelerated because of the oscillatory nature of the accelerating field. For this reason and also in case short bunches or a small energy spread at the end of the linac is desired, the particles should be concentrated at a particular phase. This concentration of particles in the vicinity of an optimum phase maximizes the particle intensity in the bunch in contrast to a mechanical chopping of a continuous beam. To bunch particles requires specific beam manipulation which we will discuss here in more detail.

A bunched beam can be obtained from a continuous stream of nonrelativistic particles by the use of a prebuncher. The basic components of a prebuncher is an rf-cavity followed by a drift space. As a continuous stream of particles passes through the prebuncher, some particles get accelerated and some are decelerated. The manipulation of the continuous beam into a bunched beam is best illustrated in the phase space diagrams of Fig. 18.12.

Figure 18.12a shows the continuous particle distribution in energy and phase at the entrance of the prebuncher. Depending on the phase of the electric field in the prebuncher at the time of passage, a particle becomes accelerated or decelerated and the particle distribution at the exit of the prebuncher is shown in Fig. 18.12b. The particle distribution has been distorted into a sinusoidal energy variation. Since the particles are nonrelativistic the energy modulation reflects also a velocity modulation. We concentrate on the origin of the distribution at \(\varphi=0\) and

\(\Delta E_{\rm kin}=0\) as the reference phase and note that particles ahead of this reference phase have been decelerated and particles behind the reference phase have been accelerated. Following this modulated beam through the drift space we observe due to the velocity modulation a bunching of the particle distribution which reaches a maximum at some distance as shown in Fig. 18.12c. A significant beam intensity has been concentrated close to the reference phase of the prebuncher.

The frequency used in the prebuncher depends on the desired bunch distribution. For straight acceleration in a linear accelerator one would choose the same frequency for both systems. Often, however, the linear accelerator is only an injector into a bigger circular accelerator which generally operates at a lower frequency. For optimum matching to the final circular accelerator the appropriate prebuncher frequency would be the same as the cavity frequency in the circular accelerator cavity.

Figure 18.12: Phase space diagrams for a continuous beam passing through a prebuncher. Before acceleration (**a**) and right after (**b**). A distance \(L\) downstream of the buncher cavity the phase space distribution shows strong bunching (**c**). [Note: the beam moves from left to right]

The effect of the prebuncher can be formulated analytically in the vicinity of the reference phase. At the exit of the prebuncher, operating at a voltage \(V=V_{0}\,\sin\varphi\), the energy spread is

\[\Delta E_{\rm kin}=eV_{0}\sin\varphi\,=mc^{2}\beta\gamma^{3}\Delta\beta\, \tag{18.114}\]

which is related to a velocity spread \(\Delta\beta\). Perfect bunching occurs a time \(\Delta t\) later when for \(\sin\varphi\approx\varphi\)

\[c\Delta\beta\Delta t=\frac{\varphi}{2\pi}\lambda_{\rm rf}\, \tag{18.115}\]

where \(\lambda_{\rm rf}\) is the rf-wavelength in the prebuncher cavity. Solving for \(\Delta t\) we get for nonrelativistic particles with \(\gamma=1\) and \(\beta\ll 1\)

\[\Delta t=\frac{\lambda_{\rm rf}}{2\pi}\frac{mv}{eV_{0}} \tag{18.116}\]

and optimum bunching occurs a distance \(L\) downstream from the cavity

\[L=v_{0}\Delta t=\frac{2E_{\rm kin}}{k_{\rm rf}eV_{0}}\, \tag{18.117}\]

where \(v_{0}\) is the velocity of the reference particle and \(k_{\rm rf}=2\pi/\lambda_{\rm rf}\). The minimum bunch length in this case is then

\[\delta L=\frac{\delta E_{\rm kin}}{k_{\rm rf}eV_{0}}\, \tag{18.118}\]

where \(\delta E_{\rm kin}\) is the total energy spread in the beam before the prebuncher.

In this derivation, we have greatly idealized the field variation being linear instead of sinusoidal. The real bunching is therefore less efficient than the above result and shows some wings as is obvious from Fig. 18.12c. In a compromise between beam intensity and bunch length one might let the bunching go somewhat beyond the optimum and thereby pull in more of the particle intensity in the wings.

There are still particles between the bunches which could either be eliminated by an rf-chopper or let go to be lost in the linear accelerator because they are mainly distributed over the decelerating field period in the linac.

##### Beam Chopper

A conceptually simple way to produce a bunched beam is to pass a continuous beam from the source through a chopper system, where the beam is deflected across a narrow slit resulting in a pulsed beam behind the slit. The principle components of such a chopper system are shown in Fig. 18.13.

As was mentioned in the previous section this mode of bunching is rather wasteful and therefore an rf-prebuncher which concentrates particles from a large range of phases towards a particular phase is more efficient for particle bunching. However, we still might want to add a beam chopper.

One such reason could be to eliminate most of the remaining particles between the bunches from the prebuncher. Although these particles most likely get lost during the acceleration process a significant fraction will still reach the end of the linac with an energy spread between zero and maximum energy. Because of their large energy deviation from the energy of the main bunches, such particles will be lost in a subsequent beam transport system and therefore create unnecessary high radiation levels. It is therefore prudent to eliminate such particles at low energies. A suitable device for that is a chopper which consists of an rf-cavity excited similar to the prebuncher cavity but with the beam port offset by a distance \(r\) from the cavity axis. In this case the same rf-source as for the prebuncher or main accelerator can be used and the deflection of particles is effected by the azimuthal magnetic field in the cavity.

The prebuncher produces a string of bunches at the prebuncher frequency. For many applications, however, a different bunch structure is desired. Specifically it often occurs that only one single bunch is desired. To produce a single pulse, the chopper system may consist of a permanent magnet and a fast pulsed magnet. The permanent magnet deflects the beam into an absorber while the pulsed magnet deflects the beam away from the absorber across a small slit. The distance between the center of the pulsed magnet and the slit be \(D\) (Fig. 18.13), the slit aperture \(\Delta\) and the rate of change of the magnetic field \(\dot{B}\). For an infinitely thin beam the pulse length behind the slit is then

\[\tau_{b}=\frac{\Delta}{D\dot{\varphi}}=\frac{\Delta}{D}\frac{cp}{e\dot{B}\ell }\, \tag{18.119}\]

where \(\varphi\) is the deflection angle, \(\ell\) the effective magnetic length of the pulsed magnet and \(cp\) the momentum of the particles. In order to clean the beam between bunches

Figure 18.13: Principal functioning of a chopper system

or to select a single bunch from a train of bunches the chopper parameters must be chosen such that only the desired part of the beam passes through.

#### Buncher Section

A buncher section is similar to an ordinary electron linac section but dimensioned for sub relativistic particle. Particles arriving from the source, prebuncher or chopper may not be at relativistic energies and therefore cannot follow the rf-wave in an ordinary linac section which have a phase velocity equal to the velocity of light. To optimize the whole acceleration system a buncher section is inserted as the first linac section. The length of each cell is shorter than in a normal linac structure and changes from a length appropriate for the velocity of the incoming particles to longer and longer cell lengths until the particle has reached the constant velocity of light for which the normal linac structure is dimensioned. In this case the particle would be accelerated from beginning at the desired phase. In case of an rf-gun the electrons emerge relativistic and no buncher is necessary. In many cases, especially at smaller facilities, the buncher section is omitted at some degradation of beam performance.

### Problems

#### 18.1 (S)

Determine within a factor of two or less the longest TE or TM-mode wavelength that can propagate through a round tube of diameter \(2R\).

#### 18.2 (S)

Consider a pill box cavity made of copper and calculate the frequency shift per \(1\,\mathrm{{}^{\circ}C}\) temperature change. The linear expansion coefficient for copper is \(\eta_{\mathrm{T}}=16.6\cdot 10^{-6}\) m/(m\({}^{\circ}\)C). What is the temperature tolerance if the rf frequency should not change by more than \(\pm 10^{-6}\).

#### 18.3 (S)

Determine the frequency scaling of cavity dimensions, transit time factor, quality factor, shunt impedance, specific shunt impedance and cavity filling time.

#### 18.4 (S)

In electron linear accelerators operating at \(3\,\mathrm{GHz}\) accelerating fields of more than \(50\,\mathrm{MeV/m}\) can be reached. Why can such high fields not be used in a storage ring? Discuss quantitatively, while scaling linac parameters to the frequency of your choice in the storage ring.

#### 18.5 (S)

Discuss the graph in Fig. 18.11. Specifically explain in words the particle dynamics within random features. How come particles get accelerated even though they enter the linac while the accelerating field is negative? (note: interpretation of the graph for initial energies \(\ll\,1\mathrm{MeV}\) does not have enough resolution to be reliable.)

**18.6 (S).**: Design a \(500\,\mathrm{MHz}\) prebuncher system for a \(3\,\mathrm{GHz}\) linear accelerator. Particles in a continuous beam from the source have a kinetic energy of \(E_{0}=100\,\mathrm{keV}\) with an energy spread of \(\pm 0.02\,\%\). Specify the optimum prebuncher voltage and drift length to compress the maximum intensity into a bunch with a phase distribution of less than \(\pm 12^{\circ}\) at \(3\,\mathrm{GHz}\).
**18.7 (S).**: Calculate for a SLAC type linac section the no-load energy gain and the energy gain for a pulse current of \(i_{\mathrm{b}}=20\,\mathrm{mA}\). The rf-power is \(P_{0}=15\,\mathrm{MW}\) per section at a pulse length of \(2.5\,\mathrm{\SIUnitSymbolMicro s}\). Compare the efficiency to the situation when only one bunch of \(n^{\mathrm{b}}=10^{10}\) electrons is accelerated. What is the linac efficiency for this current and what is the energy gain in this case?
**18.8**: Consider a rectangular box cavity with copper walls and dimensioned for an rf-wavelength of \(\lambda=10.5\,\mathrm{cm}\). Calculate the wall losses due to the fundamental field only and determine the shunt impedance per unit length \(r_{\mathrm{s}}\) and the quality factor \(Q\) for this cavity. These losses are due to surface currents within a skin depth generated by the rf-fields on the cavity surface. Compare these parameters with those of (18.96). Is the shape of the cavity very important? Determine the resonance width and temperature tolerance for the cavity.
**18.9**: Plot the electrical and magnetic field distribution for the three lowest order modes in a rectangular and cylindrical cavity. Calculate the shunt impedance and compare the results. Which type of cavity is more efficient?
**18.10**: Derive a general expression of the shunt impedance for general TM-modes in a cylindrical cavity.
**18.11**: Derive expressions for the maximum electric field strength and the waveguide losses per unit length for the \(\mathrm{TE}_{10}\) mode in a rectangular waveguide. Use this result to design a waveguide for 3 GHz. Calculate the cut-off frequency, the phase and group velocities and the waveguide wavelength. What criteria did you use to choose the dimensions \(a\) and \(b\)? Sketch the electrical and magnetic fields.
**18.12**: Consider a \(8\,\mathrm{GeV}\) electron storage ring with a FODO lattice and a beam current of \(200\,\mathrm{mA}\). Determine the equilibrium energy spread and specify rf-parameters which will be sufficient to compensate for synchrotron radiation losses and provide an energy acceptance for all particles in a Gaussian energy distribution up to \(7\sigma_{\epsilon}/E\). What is the synchrotron tune and the bunch length in your storage ring?
**18.13**: Consider a pill box cavity with copper walls for a storage ring and choose a rf-frequency of \(750\,\mathrm{MHz}\). Derive an expression for the wall losses due to the fundamental field only and derive an expression for the shunt impedance of the cavity defined by \(R_{\mathrm{cy}}=V_{\mathrm{rf}}^{2}/P_{\mathrm{rf}}\), where \(V_{\mathrm{rf}}\) is the maximum rf-voltage and \(P_{\mathrm{rf}}\) the cavity wall losses. What are the rf-losses if this cavity is used in the ring of Problem 18.12? Assume that you can cool only about \(150\,\mathrm{kW/m}\) of cavity length. How many cavities would you need for your ring example?

**18.14.** The electromagnetic field for a cylindrical waveguide have been derived in Sect. 18.1.3. Derive in a similar way expressions for resonant field modes in a rectangular waveguide.

## References

* [1] F.T. Cole, Nonlinear transformations in action-angle variables. Technical Report TM-179, FERMI Lab, Batavia (1969)
* longitudinal motion, in _Physics of Particle Accelerators_, vol. 184 (American Institute of Physics, New York, 1989), p. 4243
* [3] J.D. Jackson, _Classical Electrodynamics_, 2nd edn. (Wiley, New York, 1975)
* [4] S. Ramo, J.R. Whinnery, T. van Duzer, _Fields and Waves in Communication Electronic_ (Wiley, New York, 1984)
* [5] M. Abramovitz, I. Stegun, _Handbook of Mathematical Functions_ (Dover, New York, 1972)
* [6] I.S. Gradshteyn, I.M. Ryzhik, _Table of Integrals, Series, and Products_, 4th edn. (Academic, New York, 1965)
* [7] M.A. Allen, L.G. Karvonen, J.L. Pelegrin, P.B. Wilson, IEEE Trans. Nucl. Sci. **24**, 1780 (1977)
* [8] K. Halbach, F. Holsinger, Part. Accel. **7**, 213 (1976)
* [9] T. Weiland, Nucl. Instrum. Methods **212**, 13 (1983)
* [10] A. Piwinski, _CERN Accelerator School, CAS_, CERN 85-19 (CERN, Geneva, 1986), p. 29
* [11] J.D. Bjorken, S.K. Mtingwa, Part. Accel. **13**, 115 (1983)
* [12] J.C. Slater, Rev. Mod. Phys. **20**, 473 (1948)
* [13] R. Neal (ed.), _The 2 Mile Linear Accelerator_ (Benjamin, New York, 1968)

