## Part IV

## Chapter 15 Perturbations in Beam Dynamics

The study of beam dynamics under ideal conditions is the first basic step toward the design of a beam transport system. In the previous sections we have followed this path and have allowed only the particle energy to deviate from its ideal value. In a real particle beam line or accelerator we may, however, not assume ideal and linear conditions. More sophisticated beam transport systems require the incorporation of nonlinear sextupole fields to correct for chromatic aberrations. Deviations from the desired field configurations can be caused by transverse or longitudinal misplacements of magnets with respect to the ideal beam path. Of similar concern are errors in the magnetic field strength, undesirable field effects caused in the field configurations at magnet ends, or higher order multipole fields resulting from design, construction, and assembly tolerances. Still other sources of errors may be beam-beam perturbations, magnetic detectors for high energy physics experiments, insertion devices in beam transport systems or accelerating sections, which are not part of the magnetic lattice configurations. The impact of such errors is magnified in strong focusing beam transport systems as has been recognized soon after the invention of the strong focusing principle. Early overviews and references can be found for example in [1, 2].

A horizontal bending magnet has been characterized as a magnet with only a vertical field component. This is true as long as this magnet is perfectly aligned, in most cases perfectly level. Small rotations about the magnet axis result in the appearance of horizontal field components which must be taken into account for beam stability calculations.

We also assumed that the magnetic field in a quadrupole or higher multipole vanishes at the center of magnet axis. Misalignments of any multipole generates all lower order perturbations which is known as "spill-down".

In addition, any multipole can be rotated by a small angle with respect to the reference coordinate system. As a result we observe the appearance of a small component of a rotated or skew quadrupole causing coupling of horizontal and vertical betatron oscillations.

Although such misalignments and field errors are unintentional and undesired, we have to deal with their existence since there is no way to avoid such errors in a real environment. The particular effects of different types of errors on beam stability will be discussed. Tolerance limits on these errors as well as corrective measures must be established to avoid destruction of the particle beam. Common to all these perturbations from ideal conditions is that they can be considered small compared to forces of linear elements. We will therefore discuss mathematical perturbation methods that allow us to determine the effects of perturbations and to apply corrective measures for beam stability. The equations of motion with perturbations are

\[u^{\prime\prime}+\left(k+\kappa_{u}^{2}\right)\,u=p_{r\!n}\left(z\right)\,x^{ \prime}y^{n-s}\qquad\text{where $u=x$ or $y$} \tag{15.1}\]

and \(r,s=0,1,2\dots\) with \(r+s+1=n\) and \(n\) the order of perturbation. In the remainder of this text whenever \(r=0\) or \(s=0\) we use \(p_{n}\) instead of \(p_{r\!n}\).1

Footnote 1: Note that for this definition \(p_{0}=0\) and therefore there is no conflict with the momentum \(p_{0}\).

### Magnet Field and Alignment Errors

First we consider field errors created by magnet misalignments like displacements or rotations from the ideal positions. Such magnet alignment errors, however, are not the only cause for field errors. External sources like the earth magnetic field, the fields of nearby electrical current carrying conductors, magnets connected to vacuum pumps or ferromagnetic material in the vicinity of beam transport magnets can cause similar field errors. For example electrical power cables connected to other magnets along the beam transport line must be connected such that the currents in all cables are compensated. This occurs automatically for cases, where the power cables to and from a magnet run close together. In circular accelerators one might, however, be tempted to run the cables around the ring only once to save the high material and installation costs. This, however, causes an uncompensated magnetic field in the vicinity of cables which may reach as far as the particle beam pipe. The economic solution is to seek electrical current compensation among all magnet currents by running electrical currents in different directions around the ring. Careful design of the beam transport system can in most cases minimize the impact of such field perturbations while at the same time meeting economic goals.

Multipole errors in magnets are not the only cause for perturbations. For beams with large divergence or a large cross section, kinematic perturbation terms may have to be included. Such terms are neglected in paraxial beam optics discussed here, but will be derived in detail later.

#### Self Compensation of Perturbations

The linear superposition of individual dipole contributions to the dispersion function can be used in a constructive way. Any contribution to the dispersion function by a short magnet can be eliminated again by a similar magnet located 180\({}^{\circ}\) in betatron phase downstream from the first magnet. If the betatron function at the location of both magnets is the same, the magnet strengths are the same too. For quantitative evaluation we assume two dipole errors introducing a beam deflection by the angles \(\theta_{1}\) and \(\theta_{2}\) at locations with betatron functions of \(\beta_{1}\) and \(\beta_{2}\) and betatron phases \(\psi_{1}\) and \(\psi_{2}\), respectively. Since the dispersion function or fractions thereof evolve like a sine like function, we find for the variation of the dispersion function at a phase \(\psi(z)\geq\psi_{2}\)

\[\Delta D(z)=\theta_{1}\sqrt{\beta\beta_{1}}\sin\left[\psi(z)-\psi_{1}\right]+ \theta_{2}\sqrt{\beta\beta_{2}}\sin\left[\psi(z)-\psi_{2}\right]. \tag{15.2}\]

For the particular case where \(\theta_{1}=\theta_{2}\) and \(\beta_{1}=\beta_{2}\) we find

\[\Delta D(z)=0\quad\mbox{for}\quad\psi_{2}-\psi_{1}=(2n+1)\pi. \tag{15.3}\]

If \(\theta_{1}=-\theta_{2}\)

\[\Delta\,D(z)=0\quad\mbox{for}\quad\psi_{2}-\psi_{1}=2n\pi, \tag{15.4}\]

where \(n\) is an integer. This property of the dispersion function can be used in periodic lattices if, for example, a vertical displacement of the beam line is desired. In this case we would like to deflect the beam vertically and as soon as the beam reaches the desired elevation a second dipole magnet deflects the beam by the same angle but opposite sign to level the beam line parallel to the horizontal plane again. In an arbitrary lattice such a beam displacement can be accomplished without creating a residual dispersion outside the beam deflecting section if we place two vertical or rotated bending magnets of opposite sign at locations separated by a betatron phase of \(2\pi\).

Similarly, a deflection in the same direction by two dipole magnets does not create a finite dispersion outside the deflecting section if both dipoles are separated by a betatron phase of \((2n+1)\pi\). This feature is important to simplify beam-transport lattices since no additional quadrupoles are needed to match the dispersion function.

Sometimes it is necessary to deflect the beam in both the horizontal and vertical direction. This can be done in a straightforward way by a sequence of horizontal and vertical bending sections leading, however, to long beam lines. In a more compact version, we combine the beam deflection in both planes within one or a set of magnets. To obtain some vertical deflection in an otherwise horizontally deflecting beam line, we may rotate a whole arc section about the beam axis at the start of this section to get the desired vertical deflection. Some of the horizontal deflectionis thereby transformed into the vertical plane. At the start of such a section we introduce by the rotation of the coordinate system a sudden change in all lattice functions. Specifically, a purely horizontal dispersion function is coupled partly into a vertical dispersion. If we rotate the beam line and coordinate system back at a betatron phase of \(2n\pi\) downstream from the start of rotation, the coupling of the dispersion function as well as that of other lattice functions is completely restored. For that to work without further matching, however, we require that the rotated part of the beam line has a phase advance of \(2n\pi\) in both planes as, for example, a symmetric FODO lattice would have. This principle has been used extensively for the terrain following beam transport lines of the SLAC Linear Collider to the collision point.

### Dipole Field Perturbations

Dipole fields are the lowest order magnetic fields and therefore also the lowest order field perturbations. The equation of motion (15.1) is in this case

\[u^{\prime\prime}+\left(k+\kappa_{u}^{2}\right)u=p_{1}\left(z\right)\,, \tag{15.5}\]

where \(p_{1}\left(z\right)\) represents any dipole field error, whether it be chromatic or not. In trying to establish expressions for dipole errors due to field or alignment errors, we note that the bending fields do not appear explicitly anymore in the equations of motions because of the specific selection of the curvilinear coordinate system and it is therefore not obvious in which form dipole field errors would appear in the equation of motion (15.5). In (6.95) or (6.96) we note, however, a dipole field perturbation due to a particle with a momentum error \(\delta\). This chromatic term \(\kappa_{x0}\,\delta\) is similar to a dipole field error as seen by a particle with the momentum \(\beta E_{0}(1+\delta)\). For particles with the ideal energy we may therefore replace the chromatic term \(\kappa\delta\) by a field error \(-\Delta\kappa\). Perturbations from other sources may be obtained by variations of magnet positions (\(\Delta x\), \(\Delta y\)) or magnet strengths. Up to second order, the horizontal dipole perturbation terms due to magnet field (\(\Delta\kappa\)) and alignment errors (\(\Delta x\), \(\Delta y\)) are from (6.95)

\[p_{1,x}(z)=-\Delta\kappa_{x0}+\left(\kappa_{x0}^{2}+k_{0} \right)\Delta x+\left(2\kappa_{x0}\,\Delta\kappa_{x0}+\Delta k\right)\Delta x \tag{15.6}\] \[-\tfrac{1}{2}m\left(\Delta x^{2}-2x_{\rm c}\Delta x-\Delta y^{2} +2y_{\rm c}\Delta y\right)+\mathcal{O}(3)\,,\]

where we used \(x=x_{\beta}+x_{\rm c}-\Delta x\) and \(y=y_{\beta}+y_{\rm c}-\Delta y\) with \((x_{\beta},y_{\beta})\) the betatron oscillations and \((x_{\rm c},y_{\rm c})\) the closed orbit deviation in the magnet. In the presence of multipole magnets the perturbation depends on the displacement of the beam with respect to the center of multipole magnets.

There is a similar expression for vertical dipole perturbation terms and we get from (6.96) ignoring vertical bending magnets \(\left(\kappa_{y0}=0\right)\) but not vertical dipole errors, \(\Delta k_{y0}\neq 0\),

\[p_{1,y}(z)=-\Delta\kappa_{y0}-k_{0}\Delta y-m(x_{\rm c}\Delta y+y_{\rm c}\, \Delta x)+{\cal O}(3)\;. \tag{15.7}\]

Such dipole field errors deflect the beam from its ideal path and we are interested to quantify this perturbation and to develop compensating methods to minimize the distortions of the beam path. In an open beam transport line the effect of dipole field errors on the beam path can be calculated within the matrix formalism.

A dipole field error at point \(z_{\rm k}\) deflects the beam by an angle \(\theta\). If \({\bf M}(z_{\rm m}|z_{\rm k})\) is the transformation matrix of the beam line between the point \(z_{\rm k}\), where the kick occurs, and the point \(z_{\rm m}\), where we observe the beam position, we find a displacement of the beam center line, for example, in the \(x\)-plane by

\[\Delta x=M_{12}\,\theta\;, \tag{15.8}\]

where \(M_{12}\) is the element of the transformation matrix in the first row and the second column. Due to the linearity of the equation of motion, effects of many kicks caused by dipole errors can be calculated by summation of individual beam center displacements at the observation point \(z_{\rm m}\) for each kick. The displacement of a beam at the location \(z_{\rm m}\) due to many dipole field errors is then given by

\[\Delta x(z_{\rm m})=\sum_{\rm k}M_{12}(z_{\rm m}|z_{\rm k})\;\theta_{\rm k}\;, \tag{15.9}\]

where \(\theta_{\rm k}\) are kicks due to dipole errors at locations \(z_{\rm k}<z_{\rm m}\) and \(M_{12}(z_{\rm m}|z_{\rm k})\) the \(M_{12}\)-matrix element of the transformation matrix from the perturbation at \(z_{\rm k}\) to the monitor at \(z_{\rm m}\).

Generally, we do not know the strength and location of errors. Statistical methods are applied therefore to estimate the expectation value for beam perturbation and displacement. With \(M_{12}(z_{\rm m}|z_{\rm k})=\sqrt{\beta_{\rm m}\beta_{\rm k}}\sin(\psi_{\rm m}- \psi_{\rm k})\) we calculate the root-mean-square of (15.9) noting that the phases \(\psi_{\rm k}\) are random and cross terms involving different phases cancel. With \(\langle\theta_{\rm k}^{2}\rangle=\sigma_{\theta}^{2}\) and \(\langle\Delta u^{2}\rangle=\sigma_{u}^{2}\) we get finally from (15.9) the expectation value of the path distortion \(\sigma_{u}\) at \(z_{\rm m}\) due to statistical errors with a standard value \(\sigma_{\theta}\)

\[\sigma_{u}=\sqrt{\beta_{\rm m}\langle\beta_{\rm k}\rangle}\sqrt{N_{\theta}} \sigma_{\theta}\;, \tag{15.10}\]

where \(\langle\beta_{\rm k}\rangle\) is the average betatron function at the location of errors and \(N_{\theta}\) the number of dipole field errors. Random angles are not obvious, but if we identify the potential sources, we may be in a better position to estimate \(\sigma_{\theta}\). For example, alignment errors \(\sigma_{\Delta u}\) of quadrupoles are related to \(\sigma_{\theta}\) by \(\sigma_{\theta}=k\ell_{\rm q}\sigma_{\Delta u}\), where \(\frac{1}{f}=k\ell_{\rm q}\) are the inverse focal lengths of the quadrupoles.

#### Dipole Field Errors and Dispersion Function

The dispersion function of a beam line is determined by the strength and placement of dipole magnets. As a consequence, dipole field errors also contribute to the dispersion function and we determine such contributions to the dispersion function due to dipole field errors. First, we note from the general expression for the linear dispersion function that the effect of dipole errors adds linearly to the dispersion function by virtue of the linearity of the equation of motion. We may therefore calculate separately the effect of dipole errors and add the results to the ideal solution for the dispersion function.

##### Perturbations in Open Transport Lines

While these properties are useful for specific applications, general beam dynamics requires that we discuss the effects of errors on the dispersion function in a more general way. To this purpose we use the general equation of motion up to linear terms in \(\delta\) and add constant perturbation terms. In the following discussion we use only the horizontal equation of motion, but the results can be immediately applied to the vertical plane as well. The equation of motion with only linear chromatic terms and a quadratic sextupole term is then

\[x^{\prime\prime}+(k+\kappa_{x}^{2})x=kx\delta-{\frac{1}{2}}mx^{2}(1- \delta)-\Delta\kappa_{x}(1-\delta)+{\cal O}(2)\,. \tag{15.11}\]

We observe two classes of perturbation terms, the ordinary chromatic terms and those due to field errors. Taking advantage of the linearity of the solution we decompose the particle position into four components

\[x=x_{\beta}+x_{\rm c}+\eta_{x}\delta+v_{x}\delta\,, \tag{15.12}\]

where \(x_{\beta}\) is the betatron motion, \(x_{\rm c}\) the distorted beam path or orbit, \(\eta_{x}\) the ideal dispersion function and \(v_{x}\) the perturbation of the dispersion that derives from field errors. The individual parts of the solution then are determined by the following set of differential equations:

\[x_{\beta}^{\prime\prime}+(k+\kappa_{x}^{2})\,x_{\beta} = -{\frac{1}{2}}mx_{\beta}^{2}+mx_{\beta}x_{\rm c}\,, \tag{15.13a}\] \[x_{\rm c}^{\prime\prime}+(k+\kappa_{x}^{2})\,x_{\rm c} = -\Delta\kappa_{x}-{\frac{1}{2}}mx_{\rm c}^{2}\,,\] (15.13b) \[\eta_{x}^{\prime\prime}+(k+\kappa_{x}^{2})\,\eta_{x} = \kappa_{x}\,,\] (15.13c) \[v_{x}^{\prime\prime}+(k+\kappa_{x}^{2})\,v_{x} = +\Delta\kappa_{x}+{\frac{1}{2}}mx_{\rm c}^{2}+kx_{\rm c}-mx_{\rm c }\eta_{x}\,. \tag{15.13d}\]In the ansatz (15.12) we have ignored the energy dependence of the betatron function since it will be treated separately as an aberration and has no impact on the dispersion. We have solved (15.13a)-(15.13c) before and concentrate therefore on the solution of (15.13d). Obviously, the field errors cause distortions of the beam path \(x_{\mathrm{c}}\) which in turn cause additional variations of the dispersion function. The principal solutions are

\[C(z) = \sqrt{\beta(z)/\beta_{0}}\cos\left[\psi(z)-\psi_{0}\right]\,, \tag{15.14}\] \[S(z) = \sqrt{\beta(z)\beta_{0}}\sin\left[\psi(z)-\psi_{0}\right], \tag{15.15}\]

and the Greens function becomes

\[G(z,\sigma)=S(z)C(\sigma)-S(\sigma)C(z)=\sqrt{\beta(z)\beta(\sigma)}\sin\left[ \psi(z)-\psi(\sigma)\right]. \tag{15.16}\]

With this the solution of (15.13d) is

\[v_{x}(z) = -x_{\mathrm{c}}(z)\] \[\quad+\sqrt{\beta_{x}(z)}\int_{0}^{z}(k-m\eta_{x})\sqrt{\beta_{x }(\xi)}x_{\mathrm{c}}(\xi)\sin[\psi_{x}(z)-\psi_{x}(\xi)]\mathrm{d}\xi.\]

Here, we have split off the solution for the two last perturbation terms in (15.13d) which, apart from the sign, is exactly the orbit distortion (15.13b). In a closed lattice we look for a periodic solution of (15.17), which can be written in the form

\[v_{x}\left(z\right) = -x_{\mathrm{c}}(z)+\frac{\sqrt{\beta_{x}(z)}}{2\sin\!\pi\,v_{x}}\] \[\quad\times\int_{z}^{z+L_{\mathrm{p}}}(k-m\eta_{x})\sqrt{\beta_{ x}(\xi)}x_{\mathrm{c}}(\xi)\cos\left\{v_{x}\left[\varphi_{x}(z)-\varphi_{x}( \xi)+\pi\right]\right\}\mathrm{d}\xi,\]

where \(x_{\mathrm{c}}(z)\) is the periodic solution for the distorted orbit and \(L_{\mathrm{p}}\) the length of the orbit. In the vertical plane we have exactly the same solution except for a change in sign for some terms

\[v_{y}(z) = -y_{\mathrm{c}}(z)-\frac{\sqrt{\beta_{y}(z)}}{2\sin\pi\,v_{y}}\] \[\quad\times\int_{z}^{z+L_{\mathrm{p}}}(k-m\eta_{x})\sqrt{\beta_{ y}(\xi)}y_{\mathrm{c}}(\xi)\cos\left[v_{y}\left(\varphi_{y}(z)-\varphi_{y}( \xi)+\pi\right)\right]\mathrm{d}\xi\,.\]

For reasons of generality we have included here sextupoles to permit chromatic corrections in long curved beam lines with bending magnets. The slight asymmetry due to the term \(m\eta_{x}\) in the vertical plane derives from the fact that in real accelerators only one orientation of the sextupoles is used. Due to this orientation the perturbation in the horizontal plane is \(-\frac{1}{2}mx^{2}(1-\delta)\) and in the vertical plane \(mxy(1-\delta)\). In both cases we get the term \(m\eta_{x}\) in the solution integrals.

Again we may ask how this result varies as we add acceleration to such a transport line. Earlier in this section we found that the path distortion is independent of acceleration under certain periodic conditions. By the same arguments we can show that the distortion of the dispersions (15.18) and (15.19) are also independent of acceleration and the result of this discussion can therefore be applied to any periodic focusing channel.

#### Existence of Equilibrium Orbits

Particles orbiting around a circular accelerator perform in general betatron oscillations about the equilibrium orbit and we will discuss properties of this equilibrium orbit. Of fundamental interest is of course that such equilibrium orbits exist at all. We will not try to find conditions for the existence of equilibrium orbits in arbitrary electric and magnetic fields but restrict this discussion to fields with midplane symmetry as they are used in particle beam systems. The existence of equilibrium orbits can easily be verified for particles like electrons and positrons because these particles radiate energy in form of synchrotron radiation as they orbit around the ring.

We use the damping process to find the eventual equilibrium orbit in the presence of arbitrary dipole perturbations. To do this, we follow an orbiting particle starting with the parameters \(x=0\) and \(x^{\prime}=0\). This choice of initial parameters will not affect the generality of the argument since any other value of initial parameters is damped independently because of the linear superposition of betatron oscillations.

As an electron orbits in a circular accelerator it will encounter a number of kicks from dipole field errors or field errors due to a deviation of the particle energy from the ideal energy. After one turn the particle position is the result of the superposition of all kicks the particle has encountered in that turn. Since each kick leads to a particle oscillation given by

\[x(z)=\sqrt{\beta(z)\beta_{\theta}}\theta\sin[v\varphi(z)-v\varphi_{\theta}]\]

we find for the superposition of all kicks in one turn

\[x(z)=\sqrt{\beta(z)}\sum_{i}\sqrt{\beta_{i}}\theta_{i}\sin[v\varphi(z)-v \varphi_{i}], \tag{15.20}\]

where the index \(i\) indicates the location of the kicks. We ask ourselves now what is the oscillation amplitude after many turns. For that we add up the kicks from all past turns and include damping effects expressed by the factor \(\mathrm{e}^{-kT_{0}/\tau}\) on the particle oscillation amplitude, where \(T_{0}\) is the revolution time, \(kT_{0}\) is the time passed since the kick occurred \(k\) turns ago, and \(\tau\) the damping time. The contribution to the betatron oscillation due to kicks \(k\) turns ago, is then given by

\[\Delta x_{k}\left(z\right)=\sqrt{\beta(z)}\mathrm{e}^{-kT_{0}/\tau}\sum_{i}\sqrt {\beta_{i}}\theta_{i}\sin[2\pi vk+v\varphi(z)-v\varphi_{i}]\,. \tag{15.21}\]

Adding the contributions from all past turns results in the position \(x(z)\) of the particle

\[x(z)=\sum_{k=0}^{\infty}\sqrt{\beta(z)}\mathrm{e}^{-kT_{0}/\tau}\sum_{i}\sqrt {\beta_{i}}\theta_{i}\sin[2\pi vk+v\varphi(z)-v\varphi_{i}]\,. \tag{15.22}\]

After some rearranging (15.22) becomes

\[x(z)=C_{\theta}\sum_{k=0}^{\infty}\mathrm{e}^{-kT_{0}/\tau}\sin(2\pi vk)+S_{ \theta}\sum_{k=0}^{\infty}\mathrm{e}^{-kT_{0}/\tau}\cos(2\pi vk)\,, \tag{15.23}\]

where

\[\begin{array}{l}C_{\theta}=\sum_{i}\sqrt{\beta(z)\beta_{i}}\theta_{i}\cos [\varphi(z)-\varphi_{i}],\\ S_{\theta}=\sum_{i}\sqrt{\beta(z)\beta_{i}}\theta_{i}\sin[\varphi(z)-\varphi_ {i}].\end{array} \tag{15.24}\]

With the definition \(q=\mathrm{e}^{-T_{0}/\tau}\) we use the mathematical identities

\[\sum_{k=0}^{\infty}\mathrm{e}^{-kT_{0}/\tau}\sin(2\pi vk)=\frac{q\sin 2\pi v}{ 1-2q\cos 2\pi v+q^{2}} \tag{15.25}\]

and

\[\sum_{k=0}^{\infty}\mathrm{e}^{-kT_{0}/\tau}\cos(2\pi vk)=\frac{1-q\cos 2\pi v }{1-2q\cos 2\pi v+q^{2}} \tag{15.26}\]

and get finally instead of (15.23)

\[x(z)=\frac{C_{\theta}q\sin 2\pi v+S_{\theta}(1-q\cos 2\pi v)}{1-2q\cos 2\pi v +q^{2}}\,. \tag{15.27}\]

The revolution time is generally much shorter than the damping time \(T_{0}\ll\tau\) and therefore \(q\approx 1\). In this approximation we get after some manipulation and using (15.24)

\[x(z)=\frac{\sqrt{\beta(z)}}{2\sin\pi v}\sum_{i}\sqrt{\beta_{i}}\theta_{i}\cos [v\varphi(z)-v\varphi_{i}+v\pi]\,. \tag{15.28}\]Equation (15.28) describes the particle orbit reached by particles after some damping times. The solution does not include anymore any reference to earlier turns and kicks except those in one turn and the solution therefore is a steady state solution defined as the equilibrium orbit.

The cause and nature of the kicks \(\theta_{i}\) is undefined and can be any perturbation, random or systematic. A particular set of such errors are systematic errors in the deflection angle for particles with a momentum error \(\delta\) for which \(\theta_{i}=\kappa_{i}\ell_{i}\delta\) is the deflection angle of the bending magnet \(i\). These errors are equivalent to those that led to the dispersion or \(\eta\)-function. Indeed, setting \(\eta(z)=x(z)/\delta\) in (15.28) we get the solution (10.91) for the \(\eta\) function. The trajectories \(x(z)=\eta(z)\delta\) therefore are the equilibrium orbits for particles with a relative momentum deviation \(\delta=\Delta p/p_{0}\) from the ideal momentum \(p_{0}\).

In the next subsection we will discuss the effect of random dipole field errors \(\theta_{i}\) on the beam orbit. These kicks, since constant in time, are still periodic with the periodicity of the circumference and lead to a distorted orbit which turns out to be again equal to the equilibrium orbit found here.

To derive the existence of equilibrium orbits we have made use of the damping of particle oscillations. Since this damping derives from the energy loss of particles due to synchrotron radiation we have proof only for equilibrium orbits for radiating particles like electrons and positrons. The result obtained applies also to any other charged particle. The damping time may be extremely long, but is not infinite and a particle will eventually reach the equilibrium orbit. The concept of equilibrium orbits is therefore valid even though a proton or ion will never reach that orbit in a finite time but will oscillate about it.

##### Closed Orbit Distortion

The solution (15.28) for the equilibrium orbit can be derived also directly by solving the equation of motion. Under the influence of dipole errors the equation of motion is

\[u^{\prime\prime}+K(z)u=p_{1}(z)\,, \tag{15.29}\]

where the dipole perturbation \(p_{0}(z)\) is independent of coordinates \((x,y)\) and energy error \(\delta\). This differential equation has been solved earlier in Sect. 5.5.4, where a dipole field perturbation was introduced as an energy error of the particle. Therefore, we can immediately write down the solution for an arbitrary beam line for which the principal solutions \(C(z)\) and \(S(z)\) are known

\[u(z)=C(z)\,u_{0}+S(z)\,u_{0}^{\prime}+P(z)\,\delta \tag{15.30}\]with

\[P(z)=\int_{0}^{z}p_{1}(\zeta)\left[S(z)C(\zeta)-S(\zeta)C(z)\right]\,\mathrm{d} \zeta\,. \tag{15.31}\]

The result (15.30) can be interpreted as a composition of betatron oscillations with initial values \((u_{0},u_{0}^{\prime})\) and a superimposed perturbation \(P(z)\) which defines the equilibrium trajectory for the betatron oscillations. In (15.31) we have assumed that there is no distortion at the beginning of the beam line, \(P(0)=0\). If there were already a perturbation of the reference trajectory from a previous beam line, we define a new reference path by linear superposition of new perturbations to the continuation of the perturbed path from the previous beam line section. The particle position \((u_{0},u_{0}^{\prime})\) is composed of the betatron oscillation \((u_{0\beta},u_{0\beta}^{\prime})\) and the perturbation of the reference path \((u_{0\mathrm{c}},u_{0\mathrm{c}}^{\prime})\). With \(u_{0}=u_{0\beta}+u_{0\mathrm{c}}\) and \(u_{0}^{\prime}=u_{0\beta}^{\prime}+u_{0\mathrm{c}}^{\prime}\) we get

\[u(z)=\left[u_{0\beta}C(z)+u_{0\beta}^{\prime}S(z)\right]+\left[u_{0\mathrm{c}} C(z)+u_{0\mathrm{c}}^{\prime}S(z)\right]+P(z). \tag{15.32}\]

In a circular accelerator we look for a self-consistent periodic solution. Because the differential equation (15.29) is identical to that for the dispersion function, the solution must be similar to (10.91) and is called the closed orbit, reference orbit or equilibrium orbit given by

\[u_{\mathrm{c}}(z)=\frac{\sqrt{\beta(z)}}{2\sin\pi\,\nu}\oint_{z}^{z+C}p_{1}( \zeta)\sqrt{\beta(\zeta)}\cos\left[\nu\varphi(z)-\nu\varphi(\zeta)+\nu\pi \right]\,\mathrm{d}\zeta\,, \tag{15.33}\]

where \(C\) is the circumference of the accelerator. We cannot anymore rely on a super-periodicity of length \(L_{\mathrm{p}}\) since the perturbations \(p_{\mathit{r}\mathit{s}n}(\zeta)\) due to misalignment or field errors are statistically distributed over the whole ring. Again the integer resonance character discussed earlier for the dispersion function is obvious, indicating there is no stable orbit if the tune of the circular accelerator is an integer. The influence of the integer resonance is noticeable even when the tune is not quite an integer. From (15.33) we find a perturbation \(p_{1}(z)\) to have an increasing effect the closer the tune is to an integer value. The similarity of the closed orbit and the dispersion function in a circular accelerator is deeper than merely mathematical. The dispersion function defines closed orbits for energy deviating particles approaching the real orbit (15.33) as \(\delta\to 0\).

Up to second order the horizontal and vertical dipole perturbation terms due to magnet field and alignment errors are given by (15.6) and (15.7). In the presence of multipole magnets the perturbation depends on the displacement of the beam with respect to the center of multipole magnets.

A vertical closed orbit distortion is shown in Fig. 15.1 for the PEP storage ring. Here, a Gaussian distribution of horizontal and vertical alignment errors with an rms error of 0.05 mm in all quadrupoles has been simulated. In spite of the statistical distribution of errors a strong oscillatory character of the orbit is apparent and counting oscillations we find 18 oscillations being equal to the vertical tune of PEP as we would expect from the denominator of (15.33).

We also note large values of the orbit distortion adjacent to the interaction points (dashed lines), where the betatron function becomes large, again in agreement with expectations from (15.33) since \(u_{\rm c}\propto\sqrt{\beta}\). A more regular representation of the same orbit distortion can be obtained if we plot the normalized closed orbit \(u_{\rm c}(z)/\sqrt{\beta(z)}\) as a function of the betatron phase \(\psi(z)\) shown in Fig. 15.2. In this representation the strong harmonic close to the tune becomes evident while the statistical distribution of perturbations appears mostly in the amplitude of the normalized orbit distortion.

For the sake of simplicity terms of third or higher order as well as terms associated with nonlinear magnets have been neglected in both Eqs. (15.6) and (15.7). All terms in (15.6) and (15.7) are considered small perturbations and can therefore be treated individually and independent of other perturbations terms. Sextupole and

Figure 15.2: Closed orbit distortion of Fig. 15.2 in normalized coordinates as a function of the betatron phase \(\varphi\)

Figure 15.1: Simulation of the closed orbit distortion in the sixfold symmetric PEP lattice due to statistical misalignments of quadrupoles by an amount \(\langle\Delta x\rangle_{\rm ms}=\langle\Delta y\rangle_{\rm ms}=0.05\,{\rm mm}\)

higher multipole perturbations depend on the orbit itself and to get a self-consistent periodic solution of the distorted orbit, iteration methods must be employed.

Solutions for equilibrium orbits can be obtained by inserting the perturbation (15.6) or (15.7) into (15.33). First, we will concentrate on a situation, where only one perturbing kick exists in the whole lattice, assuming the perturbation to occur at \(z=z_{\rm k}\) and to produce a kick \(\theta_{\rm k}=\int p_{1}(\xi)\,{\rm d}\hat{\zeta}\) in the particle trajectory. The orbit distortion at a location \(z<z_{\rm k}\) in the lattice is from (15.33)

\[u_{0}(z)=\tfrac{1}{2}\sqrt{\beta(z)\beta(z_{\rm k})}\,\theta_{\rm k}\frac{\cos \left[v\,\pi-v\varphi(z_{\rm k})\,+\,v\varphi(z)\right]}{\sin\pi\,v}\,. \tag{15.34}\]

If on the other hand we look for the orbit distortion downstream from the perturbation \(z>z_{\rm k}\) the integration must start at \(z\), follow the ring to \(z=C\) and then further to \(z=z+C\). The kick, therefore, occurs at the place \(C+z_{\rm k}\) with the phase \(\varphi(C)+\varphi(z_{\rm k})=2\pi\,+\,\varphi(z_{u})\) and the orbit is given by

\[u_{0}(z)=\tfrac{1}{2}\sqrt{\beta(z)\beta(z_{\rm k})}\,\theta_{\rm k}\frac{\cos [v\pi-v\varphi(z)+v\varphi(z_{\rm k})]}{\sin\pi\,v}. \tag{15.35}\]

This mathematical distinction of cases \(z<z_{\rm k}\) and \(z>z_{\rm k}\) is a consequence of the integration starting at \(z\) and ending at \(z+C\) and is necessary to account for the discontinuity of the slope of the equilibrium orbit at the location of the kick. At the point \(z=z_{\rm k}\) obviously both equations are the same. In Fig. 15.3 the normalized distortion of the ideal orbit due to a single dipole kick is shown. In a linear lattice this distortion is independent of the orbit and adds in linear superposition. If, however, sextupoles or other coupling or nonlinear magnets are included in the lattice, the distortion due to a single or multiple kick depends on the orbit itself and self-consistent solutions can be obtained only by iterations.

In cases where a single kick occurs at a symmetry point of a circular accelerator we expect the distorted orbit to also be symmetric about the kick. This is expressed in the asymmetric phase terms of both equations. Indeed, since \(\varphi(z_{\rm k})-\varphi(z)=\Delta\varphi\) for \(z_{\rm k}>z\) and \(\varphi(z)-\varphi(z_{\rm k})=\Delta\varphi\) for \(z>z_{\rm k}\) the orbit distortion extends symmetrically in either direction from the location of the kick.

Figure 15.3: Distorted orbit due to a single dipole kick for a tune just above an integer (_left_) and for a tune below an integer (_right_)

The solution for the perturbed equilibrium orbit is specially simple at the place where the kick occurs. With \(\varphi(z)=\varphi(z_{k})\) the orbit distortion is

\[u_{\rm k}=\tfrac{1}{2}\beta_{\rm k}\theta_{\rm k}\cot\pi\,v. \tag{15.36}\]

In situations where a short bending magnet like an orbit correction magnet and a beam position monitor are at the same place or at least close together we may use these devices to measure the betatron function at that place \(z_{\rm k}\) by measuring the tune \(v\) of the ring and the change in orbit \(u_{\rm k}\) due to a kick \(\theta_{\rm k}\). Equation (15.36) can then be solved for the betatron function \(\beta_{\rm k}\) at the location \(z_{\rm k}\). This procedure can obviously be applied in both planes to experimentally determine \(\beta_{x}\) as well as \(\beta_{y}\).

#### Statistical Distribution of Dipole Field

and Alignment Errors

In a real circular accelerator a large number of field and misalignment errors of unknown location and magnitude must be expected. If the accelerator is functional we may measure the distorted orbit with the help of beam position monitors and apply an orbit correction as discussed later in this section. During the design stage, however, we need to know the sensitivity of the ring design to such errors in order to determine alignment tolerances and the degree of correction required. In the absence of detailed knowledge about errors we use statistical methods to determine the most probable equilibrium orbit. All magnets are designed, fabricated, and aligned within statistical tolerances, which are determined such that the distorted orbit allows the beam to stay within the vacuum pipe without loss. An expectation value for the orbit distortion can be derived by calculating the root-mean-square of (15.33)

\[u_{0}^{2}(z) = \frac{\beta(z)}{4\sin^{2}\pi\,v}\oint_{z}^{z+C}\oint_{z}^{z+C}p_ {1}(\sigma)\,p_{1}(\tau)\sqrt{\beta(\sigma)}\sqrt{\beta(\tau)}\] \[\times\cos\left[v\left(\varphi_{z}-\varphi_{\sigma}+\pi\right) \right]\cos\left[v\left(\varphi_{z}-\varphi_{\tau}+\pi\right)\right]\,{\rm d} \sigma\,{\rm d}\tau\,,\]

where for simplicity \(\varphi_{z}=\varphi(z)\) etc. This double integral can be evaluated by expanding the cosine functions to separate the phases \(\varphi_{\sigma}\) and \(\varphi_{\tau}\). We get terms like \(\cos v\varphi_{\sigma}\cos v\varphi_{\tau}\) and \(\sin v\varphi_{\sigma}\sin v\varphi_{\tau}\) or mixed terms. All these terms tend to cancel except when \(\sigma=\tau\) since both the perturbations and their locations are statistically distributed in phase. Only for \(\sigma=\tau\) will we get quadratic terms that contribute to a finite expectation value for the orbit distortion

\[\langle p_{1}^{2}(\tau)\left[\cos^{2}v(\varphi_{z}+\pi)\cos^{2} v\varphi_{\tau}+\sin^{2}v(\varphi_{z}+\pi)\sin^{2}v\varphi_{\tau}\right]\rangle\] \[= \langle p_{1}^{2}(\tau)\rangle[\cos^{2}v(\varphi_{z}+\pi)\langle \cos^{2}v\varphi_{\tau}\rangle+\sin^{2}v(\varphi_{z}+\pi)\langle\sin^{2}v \varphi_{\tau}\rangle]\] \[= \langle p_{1}^{2}(\tau)\rangle\tfrac{1}{2},\]and with this (15.37) becomes

\[\langle u_{0}^{2}(z)\rangle = \frac{\beta(z)}{8\sin^{2}\pi\,v}\sum_{i}\langle p_{1}^{2}(\sigma_{i })\beta(\sigma_{i})\ell_{i}^{2}\rangle\,, \tag{15.38}\]

where the integrals have been replaced by a single sum over all perturbing fields of length \(\ell_{i}\). This can be done since we assume that the betatron phase does not change much over the length of individual perturbations. Equation (15.38) gives the expectation value for the orbit distortion at the point \(z\) and since the errors are statistically distributed we get from the central limit theorem a Gaussian distribution of the orbit distortions with the standard deviation \(\sigma_{u}^{2}(z)=\langle u_{0}^{2}(z)\rangle\) from (15.38). In other words if an accelerator is constructed with tolerances \(\langle p_{1}^{2}(\sigma_{i})\rangle\) there is a 68 % probability that the orbit distortions are of the order \(\sqrt{\langle u_{0}^{2}(z)\rangle}\) as calculated from (15.38) and a 98 % probability that they are not more than twice that large.

As an example, we consider a uniform beam transport line, where all quadrupoles have the same strength and the betatron functions are periodic like in a FODO channel. This example seems to be very special since hardly any practical beam line has these properties, but it is still a useful example and may be used to simulate more general beam lines for a quick estimate of alignment tolerances. Assuming a Gaussian distribution of quadrupole misalignments with a standard deviation \(\sigma_{\Delta u}\) and quadrupole strength \(k\), the perturbations are \(p_{1}\left(z\right)=k\sigma_{\Delta u}\) and the expected orbit distortion is

\[\sqrt{\langle u_{0}^{2}(z)\rangle} = \sqrt{\beta(z)}A\sigma_{\Delta u}\,, \tag{15.39}\]

where \(A\) is called the error amplification factor defined by

\[A^{2} = \frac{N}{8\sin^{2}\pi\,v}\langle(k\ell_{\rm q})^{2}\beta\rangle \approx \frac{N}{8\sin^{2}\pi\,v}\frac{\overline{\beta}}{f^{2}}\,, \tag{15.40}\]

\(\langle(k\ell_{\rm q})^{2}\beta\rangle\) is taken as the average value for the expression in all \(N\) misaligned quadrupoles, \(f\) is the focal length of the quadrupoles, and \(\overline{\beta}\) the average betatron function.

The expectation value for the maximum value of the orbit distortion \(\langle\hat{u}_{0}^{2}(z)\rangle\) is larger. In (15.38) we have averaged the trigonometric functions

\[\langle\cos^{2}\nu\varphi(\tau)\rangle = \langle\sin^{2}\nu\varphi(\tau)\rangle=\tfrac{1}{2}\]

and therefore

\[\langle\hat{u}_{0}^{2}\rangle = 2\,\langle u_{0}^{2}(z)\rangle\,. \tag{15.41}\]

These methods obviously require a large number of misalignments to become statistically accurate. While this is not always the case for shorter beam lines it is still useful to perform such calculations. In cases where the statistical significance is really poor, one may use 100 or more sets of random perturbations and apply them to the beam line or ring lattice. This way a better evaluation of the distribution of possible perturbations is possible.

Clearly, the tolerance requirements increase as the average value of betatron functions, the quadrupole focusing, or the size of the accelerator or number of magnets \(N\) is increased. No finite orbit can be achieved if the tune is chosen to be an integer value. Most accelerators work at tunes which are about one quarter away from the next integer to maximize the trigonometric denominator \(|\sin\pi\,v|\,\approx\,1\). From a practical standpoint we may wonder what compromise to aim for between a large aperture and tight tolerances. It is good practice to avoid perturbations as reasonable as possible and then, if necessary, enlarge the magnet aperture to accommodate distortions which are too difficult to avoid. As a practical measure it is possible to restrict the uncorrected orbit distortion in most cases to 5-10 mm and provide magnet apertures that will accommodate this.

What happens if the expected orbit distortions are larger than the vacuum aperture which is virtually sure to happen at least during initial commissioning of more sensitive accelerators? In this case one relies on fluorescent screens or electronic monitoring devices located along the beam line, which are sensitive enough to detect even small beam intensities passing by only once. By empirically employing corrector magnets the beam can be guided from monitor to monitor thus establishing a path and eventually a closed orbit. Once all monitors receive a signal, more sophisticated and computerized orbit control mechanism may be employed.

#### Dipole Field Errors in Insertion Devices

Periodic magnet arrays like wiggler and undulator magnets are used often in synchrotron radiation sources to produce specific radiation characteristics. The requirement for such insertion devices is that the total deflection angle be zero as to not affect the closed orbit in the storage ring

\[\int_{-\infty}^{+\infty}B_{\perp}\mathrm{d}z=0.\]

In reality that is not possible because of manufacturing tolerances. A real trajectory through an undulator may look like shown in Fig. 15.4[3].

From Fig. 15.4 it is obvious that a particle entering the undulator on axis will exit the magnet with a large distance from the axis and with a significant angle. Both will contribute to the orbit distortions. The problem here is that this orbit distortion is gap dependent and as the experimental user changes the gap the orbit changes all around the storage ring moving at the same time the source position for all other users. It is therefore imperative to correct this distortion before it can affect the orbit.

There are two quantities which must be corrected, both being called the first and second integral

\[I_{1} =\int B\,\mathrm{d}z=0\quad\text{ and} \tag{15.42}\] \[I_{2} =\int\mathrm{d}z^{\prime}\int B\,\mathrm{d}z=0.\]

Both integrals should be zero because \(I_{1}\) is proportional to the exit angle and \(I_{2}\) proportional to the position at the undulator exit. Both errors should and can be corrected by a steering magnet before the entrance and right after the undulator exit. By adjusting the entrance steering magnet the exit angle can be varied and the exit beam displacement can be made to be zero. After adjusting the exit beam position to zero the angle still may be wrong which can be adjusted to zero with the exit steering magnet. With this correction the undulator effect on the orbit is eliminated. Unfortunately, the first and second integral can be in a permanent magnet device gap-dependent. Therefore, before using the undulator the steering corrections must be determined experimentally as a function of gap size. This information is stored in the control computer and as the gap size is changed by the user the computer will also change the steering field such that the orbit stays constant during change of the gap. This procedure is know as feed-forward. With this correction the undulkator has become a true insertion device from an accelerator physics point of view.

We notice, however, in Fig. 15.4 that the oscillating trajectory within the undulator is not along a straight line. In the particular case of Fig. 15.4 the trajectory resembles an arc which can reduce the radiation characteristics especially for higher harmonics. This can be corrected by two long coils one each around the full array of poles. This coil can deflect the beam on a dipole trajectory such that it compensates the average curvature within the undulator. Of course if this is done then the

Figure 15.4: Trajectory through an undulator without any special corrections [3]

correction of the first and second integral must be repeated again. The long coil current is also gap dependent and for successful feed-forward three tables must be prepared for the computer control of undulator gap changes.

#### Closed Orbit Correction

Due to magnetic field and alignment errors a distorted equilibrium orbit is generated as discussed in the previous section. Specifically for distinct localized dipole field errors at position \(z_{k}\)

\[u_{0}(z)=\frac{\sqrt{\beta(z)}}{2\sin\pi\,v}\sum_{k}\sqrt{\beta_{k}}\theta_{k} \cos[v\varphi(z)-v\varphi_{k}+v\pi]\,. \tag{15.43}\]

Since orbit distortions reduce the available aperture for betatron oscillations and can change other beam parameters it is customary in accelerator design to include a special set of magnets for the correction of distorted orbits. These orbit correction magnets produce orbit kicks and have, therefore, the same effect on the orbit as dipole errors. However, now the location and the strength of the kicks are known. Before we try to correct an orbit it must have been detected with the help of beam position monitors. The position of the beam in these monitors is the only direct information we have about the distorted orbit. From the set of measured orbit distortions \(u_{i}\) at the \(m\) monitors \(i\) we form a vector

\[\mathbf{u}_{m}=(u_{1},u_{2},u_{3},\ldots,u_{m}) \tag{15.44}\]

and use the correctors to produce additional "orbit distortions" at the monitors through carefully selected kicks \(\theta_{k}\) in orbit correction magnets which are also called trim magnets. For \(n\) corrector magnets the change in the orbit at the monitor \(i\) is

\[\Delta u_{i}=\frac{\sqrt{\beta_{i}}}{2\sin\pi\,v}\sum_{k=1}^{n}\sqrt{\beta_{k }}\theta_{k}\cos\left[v(\varphi_{i}-\varphi_{k}+\pi)\right], \tag{15.45}\]

where the index \(k\) refers to the corrector at \(z=z_{k}\). The orbit changes at the beam position monitors due to the corrector kicks can be expressed in a matrix equation

\[\mathbf{\Delta}\mathbf{u}_{=}=\mathcal{M}\mathbf{\theta}_{\,n}\,, \tag{15.46}\]

where \(\mathbf{\Delta}\mathbf{u}_{=}\) is the vector formed from the orbit changes at all \(m\) monitors, \(\mathbf{\theta}_{\,n}\) the vector formed by all kicks in the \(n\) correction magnets, and \(\mathcal{M}\) the response matrix \(\mathcal{M}=(\mathcal{M}_{ik})\) with

\[\mathcal{M}_{ik}=\frac{\sqrt{\beta_{i}\beta_{k}}}{2\sin\pi\,v}\cos\left[v( \varphi_{i}-\varphi_{k}+\pi)\right]\,. \tag{15.47}\]The distorted orbit can be corrected at least at the position monitors with corrector kicks \(\theta_{k}\) chosen such that \(\mathbf{\Delta}\mathbf{u}_{{}_{m}}=-\mathbf{u}_{{}_{m}}\) or

\[\boldsymbol{\theta}_{{}_{n}}=-\mathcal{M}^{-1}\mathbf{u}_{{}_{m}}\,. \tag{15.48}\]

Obviously, this equation can be solved exactly if \(n=m\) and also for \(n>m\) if not all correctors are used. Additional conditions could be imposed in the latter case like minimizing the corrector strength.

While an orbit correction according to (15.48) is possible it is not always the optimum way to do it. A perfectly corrected orbit at the monitors still leaves finite distortions between the monitors. To avoid large orbit distortions between monitors sufficiently many monitors and correctors must be distributed along the beam line. A more sophisticated orbit correction scheme would only try to minimize the sum of the squares of the orbit distortions at the monitors

\[\left(\mathbf{u}_{{}_{m}}-\mathbf{\Delta}\mathbf{u}_{m}\right)_{\min}^{2}= \left(\mathbf{u}_{m}-\mathcal{M}\boldsymbol{\theta}_{n}\right)_{\min}^{2}, \tag{15.49}\]

thus avoiding extreme corrector settings due to an unnecessary requirement for perfect correction at monitor locations.

This can be achieved for any number of monitors \(m\) and correctors \(n\) although the quality of the resulting orbit depends greatly on the actual number of correctors and monitors. To estimate the number of correctors and monitors needed we remember the similarity of dispersion function and orbit distortion. Both are derived from similar differential equations. The solution for the distorted orbit, therefore, can also be expressed by Fourier harmonics similar to (10.99). With \(F_{n}\) being the Fourier harmonics of \(-\beta^{3/2}(z)\;\Delta\kappa(z)\), the distorted orbit is

\[u_{0}(z)=\sqrt{\beta(z)}\sum_{n=-\infty}^{+\infty}\,\frac{v^{2}F_{n}\,\mathrm{ e}^{inp}}{v^{2}-n^{2}}, \tag{15.50}\]

which exhibits a resonance for \(v=n\). The harmonic spectrum of the uncorrected orbit \(u_{0}(z)\) has therefore also a strong harmonic content for \(n\approx v\). To obtain an efficient orbit correction both the beam position monitor and corrector distribution around the accelerator must have a strong harmonic close to the tune \(v\) of the accelerator. It is, therefore, most efficient to distribute monitors and correctors uniformly with respect to the betatron phase \(\psi(z)\) rather than uniform with \(z\) and use at least about four units of each per betatron wave length.

With sufficiently many correctors and monitors the orbit can be corrected in different ways. One could excite all correctors in such a way as to compensate individual harmonics in the distorted orbit as derived from beam position measurement. Another simple and efficient way is to look for the first corrector that most efficiently reduces the orbit errors then for the second most efficient and so on. This latter method is most efficient since the practicality of other methods can be greatly influenced by errors of the position measurements as well as field errors in the correctors. The single most effective corrector method can be employed repeatedly to obtain an acceptable orbit. Of similar practical effectiveness is the method of beam bumps. Here, a set of three to four correctors are chosen and powered in such a way as to produce a beam bump compensating an orbit distortion in that area. This method is a local orbit correction scheme while the others are global schemes.

As a practical example, we show the vertical orbit in the storage ring PEP before and after correction (Fig. 15.5) in normalized units. The orbit distortions are significantly reduced and the strong harmonic close to the betatron frequency has all but disappeared. Even in normalized units the orbit distortions have now a statistical appearance and further correction would require many more correctors. The peaks at the six interaction points of the lattice, indicated by dashed lines, are actually small orbit distortions and appear large only due to the normalization to a very small value of the betatron function \(\beta_{y}\) at the interaction point.

#### Response Matrix

In the last section we found the relation of the beam position response at each position monitor (BPM) due to a change in any of the steering magnets in the circular accelerator. The matrix made up of these relations (15.47) is called the response matrix. The elements of this response matrix gives us an inside look at perturbations, calibration errors and alignment and field tolerances. Each element of the response matrix is defined by the movement of the beam at a particular beam position monitor due to a known change in a particular steering magnet. That response is made up of all fields encountered by the beam from the steerer to the BPM, all ideal bending, quadrupole and sextupole fields, but also all undesired fields originating from manufacturing tolerances, alignment errors, field errors, stray fields etc. Also included are calibration and alignment errors (offset) of BPMs and steering magnets. Therefore these response matrix elements contain all the information of perturbations which we would like to know. Unfortunately such errors are not spelled out in clear text but must be assumed. To find such errors one uses a computer program like LOCO [4; 5] which allows the user to choose a specific source of suspected errors. The program then uses such errors and tries

Figure 15.5: Orbit of Fig. 15.2 before and after correction

a fit to measured response matrix data. Usually most if not all errors are found to be corrected in a series of different approximations. Of course, the measurement of some 5,000 to 10,000 response matrix elements takes time. It is a very repetitive measurement which is best left to a computer. Such programs are part of the Accelerator Tool box (AT) [6] which includes many more routines to analyse and optimize electron storage rings.

##### Orbit Correction with Single Value Decomposition (SVD)

Space age developments have had their impact on accelerator physics too. Transmission of pictures from space craft and communication in general push for methods to get the information with a minimum of data transfers. Mathematicians found a way to invert a big matrix and determine the most significant eigenvalues. In accelerator physics, we know from the last section that a series of approximations based on the most significant corrector leads to a greatly improved closed orbit. Now, with the new approach, which is called Singe Value Decomposition (SVD), we get the desired result in one application. The method of SVD provides us with the matrix inversion (15.48) such that all steering corrections are listed in order of magnitude.

##### Single Value Decomposition (SVD)

Assume a \(n\times m\) matrix \(\mathcal{A}\) like the one in (15.46). This matrix can be decomposed into three matrices such that

\[\mathcal{A}=\mathcal{U}\mathcal{W}\mathcal{V}^{\mathrm{T}}. \tag{15.51}\]

Here, the columns of \(\mathcal{U}\) are the eigenvectors of \(\mathcal{A}\mathcal{A}^{\mathrm{T}}\) and \(\mathcal{V}\) is made up of rows which are the eigenvectors of \(\mathcal{A}^{\mathrm{T}}\mathcal{A}\). Finally, \(\mathcal{W}\) is a diagonal matrix with elements being the "singular values" equal to the square root of the eigenvalues of both \(\mathcal{A}\mathcal{A}^{\mathrm{T}}\) and \(\mathcal{A}^{\mathrm{T}}\mathcal{A}\). The inverse \(\mathcal{A}^{-1}\) is

\[\mathcal{A}^{-1}=\mathcal{V}\,\mathcal{W}^{-1}\mathcal{U}^{\mathrm{T}} \tag{15.52}\]

and the desired corrector strength are from (15.48)

\[\boldsymbol{\theta}_{{}_{n}}=-\mathcal{A}^{-1}\mathbf{u}_{{}_{m}}. \tag{15.53}\]

where \(\theta_{1}\) is the strongest and \(\boldsymbol{\theta}_{{}_{n}}\) the weakest corrector.

More visually, we show the process in some graphs from the synchrotron light source PLS-II in Pohang, Korea.2 The response matrix before correction is shown in Fig. 15.6 in a 3-D graph (lower left) where the two axis are the number of steering magnets and number of BPMs.

Footnote 2: I thank Dr. Sheungwan Shin, PLS-II, Pohang, Korea for providing the data and graphs.

There are somewhat regular oscillation in the values of the matrix elements visible, which are due to the repetition of the betatron oscillation in each super-periods. Applying SVD the inverse response matrix is of diagonal form and shown in Fig. 15.6 (lower right). The strength of the steering magnets to correct the closed orbit is finally shown in Fig. 15.7 where it is quite obvious that only a few correctors are very effective. Of the 96 corrector magnets installed in PLS-II only about 30 to 50 are effective. All others do not contribute to orbit correction but rather fight each other and therefore should not be used.

While it is not known which corrector magnets and BPMs are the most effective a sufficient number of both, about 6 per betatron oscillation, should be installed. Eventually, correction of the orbit defines the most effective correctors and BPMs, about 4 each per betatron oscillation. In the vertical plane the process is exactly the same with some variation of numbers due to the different tune.

Figure 15.6: 3-D graph of the horizontal response matrix elements before correction (_left_) and after applying SVD (_right_). The values are shown versus the number of steering magnets and the number of BPMs each close to 100

### Quadrupole Field Perturbations

The dipole perturbation terms cause a shift in the beam path or closed orbit without affecting the focusing properties of the beam line. The next higher perturbation terms which depend linearly on the transverse particle offset from the ideal orbit will affect focusing because these perturbations act just like quadrupoles. Linear perturbation terms are of the form

\[u^{\prime\prime}+\left(k_{u}+\kappa_{u}^{2}\right)u=p_{2}\left(z\right)u, \tag{15.54}\]

where \(u\) stand for \(x\) and \(y\), respectively. More quantitatively, these linear perturbations are from (15.6) and (15.7)

\[\begin{array}{l}p_{2,x}\left(z\right)=-\Delta\left(k_{x}+\kappa_{x}^{2} \right)\,x-m\Delta x\,x+\ldots\\ p_{2,y}\left(z\right)=+\Delta k_{y}\,y+m\Delta xy\,+\ldots\end{array} \tag{15.55}\]

As a general feature, we recognize the "feed down" from misalignments of higher order multipoles. A misaligned sextupole, for example, generates dipole as well as gradient fields. Any misaligned multipole produces fields in all lower orders.

Quadrupole fields determine the betatron function as well as the phase advance or tune in a circular accelerator. We expect therefore that linear field errors will modify these parameters and we will derive the effect of gradient errors on lattice functions and tune.

Figure 15.7: Strength of the corrector magnets ordered from most to least effective

#### 15.3.1 Betatron Tune Shift

Gradient field errors have a first order effect on the betatron phase and tune. Specifically in circular accelerators we have to be concerned about the tune not to deviate too much from stable values to avoid beam loss. The effect of a linear perturbation on the tune can be easily derived in matrix formulation for one single perturbation. For simplicity we choose a symmetry point in the lattice of a circular accelerator and insert on either side of this point a thin half-lens perturbation with the transformation matrix

\[{\cal M}_{\rm p}=\left(\matrix{1&0\cr-1/f&1}\right), \tag{15.56}\]

where \(f^{-1}=-\frac{1}{2}\int p_{2}(z)\,{\rm d}z\) and \(p_{2}(z)\) is the total perturbation. Combining this with the transformation of an ideal ring (8.74) with \(\beta=\beta_{0},\alpha=\alpha_{0}=0\) and \(\psi_{0}=2\pi\,v_{0}\)

\[{\cal M}_{0}=\left(\matrix{C(z)&S(z)\cr C^{\prime}(z)&S^{\prime}(z)}\right)= \left(\matrix{\cos\psi_{0}&\beta_{0}\sin\psi_{0}\cr-\frac{1}{\beta_{0}}\sin \psi_{0}&\cos\psi_{0}}\right)\]

we get for the trace of the total transformation matrix \({\cal M}={\cal M}_{\rm p}\,{\cal M}_{0}\,{\cal M}_{\rm p}\)

\[{\rm Tr}{\cal M}=2\cos\psi_{0}-2\frac{\beta_{0}}{f}\sin\psi_{0}\,, \tag{15.57}\]

where \(\beta_{0}\) is the unperturbed betatron function at the location of the perturbation and \(\psi_{0}=2\pi\,v_{0}\) the unperturbed phase advance per turn. The trace of the perturbed ring is \({\rm Tr}{\cal M}=2\cos\psi\) and we have therefore

\[\cos\psi=\cos\psi_{0}-\frac{\beta_{0}}{f}\sin\psi_{0}\,. \tag{15.58}\]

With \(\psi=2\pi\,v=2\pi\,v_{0}+2\pi\,\delta v\) and \(\cos\psi=\cos\psi_{0}\,\cos 2\pi\,\delta v-\sin\psi_{0}\,\sin 2\pi\,\delta v\) we get for small perturbations the tune shift

\[\delta v=\frac{1}{2\pi}\frac{\beta_{0}}{f}=-\frac{\beta_{0}}{4\pi}\int p_{2}( z)\,{\rm d}z\,. \tag{15.59}\]

For more than a single gradient error one would simply add the individual contribution from each error to find the total tune shift. The same result can be obtained from the perturbed equation of motion (15.54). To show this, we introduce normalized coordinates \(w=u/\sqrt{\beta}\) and \(\varphi=\int\frac{{\rm d}z}{v\beta}\) and (15.54) becomes

\[\ddot{w}+v_{0}^{2}w=v_{0}^{2}\beta^{2}(z)\,p_{2}(z)\,w\,. \tag{15.60}\]For simplicity, we drop the index \({}_{u}\) and recognize that all equations must be evaluated separately for \(x\) and \(y\). Since both the betatron function \(\beta(z)\) and perturbations \(p_{2}(z)\) are periodic, we may Fourier expand the coefficient of \(v_{0}w\) on the r.h.s. and get for the lowest, non-oscillating harmonic

\[F_{0}=\frac{1}{2\pi}\int_{0}^{2\pi}v_{0}\beta^{2}p_{2}\left(z\right)\,\mathrm{d }\varphi=\frac{1}{2\pi}\oint\beta(z)p_{2}(z)\,\mathrm{d}z\,. \tag{15.61}\]

Inserting this into (15.60) and collecting terms linear in \(w\) we get

\[\ddot{w}+(v_{0}^{2}-v_{0}F_{0})w=0 \tag{15.62}\]

and the new tune \(v=v_{0}+\delta v\) is determined by

\[v^{2}=v_{0}^{2}-v_{0}\,F_{0}\approx v_{0}^{2}+2v_{0}\delta v\,. \tag{15.63}\]

Solving for \(\delta v\) gives the linear tune perturbation

\[\delta v=-\tfrac{1}{2}F_{0}=-\frac{1}{4\pi}\oint\beta(z)p_{2}(z)\,\mathrm{d}z \tag{15.64}\]

in complete agreement with the result obtained in (15.59). The tune shift produced by a linear perturbation has great diagnostic importance. By varying the strength of an individual quadrupole and measuring the tune shift it is possible to derive the value of the betatron function in this quadrupole.

The effect of linear perturbations contributes in first approximation only to a static tune shift. In higher approximation, however, we note specific effects which can affect beam stability and therefore must be addressed in more detail. To derive these effects we solve (15.60) with the help of a Green's function as discussed in Sect. 5.5.4 and obtain the perturbation

\[P(\varphi)=\int_{0}^{\varphi}v_{0}\beta^{2}(\chi)p_{2}(\chi)w(\chi)\sin\left[ v_{0}(\varphi-\chi)\right]\,\mathrm{d}\chi\,, \tag{15.65}\]

where we have made use of the principal solutions. We select a particular, unperturbed trajectory, \(w(\chi)=w_{0}\,\cos\left(v\chi\right)\) with \(\dot{w}_{0}=0\) and get the perturbed particle trajectory

\[w(v\varphi)=w_{0}\cos\left(v_{0}\varphi\right)+w_{0}v_{0}\int_{0}^{\varphi} \beta^{2}p_{2}\cos\left(v_{0}\chi\right)\sin\left[v_{0}(\varphi-\chi)\right] \,\mathrm{d}\chi\,, \tag{15.66}\]

where \(\beta=\beta(\chi)\) and \(p_{2}=p_{2}(\chi)\). If, on the other hand, we consider the perturbations to be a part of the lattice, we would describe the same trajectory by

\[w(\varphi)=w_{0}\cos v\varphi\,. \tag{15.67}\]Both solutions must be equal. Specifically the phase advance per turn must be the same and we get from (15.66), (15.67) after one turn \(\varphi=2\pi\) for the perturbed tune \(v=v_{0}+\delta v\)

\[\cos 2\pi(v_{0}+\delta v)=\cos 2\pi\,v_{0}+v_{0}\int_{0}^{2\pi}\beta^{2}( \varphi)p_{2}(\varphi)\cos(v_{0}\varphi)\sin\left[v_{0}(2\pi-\varphi)\right] \,\mathrm{d}\varphi\,, \tag{15.68}\]

which can be solved for the tune shift \(\mathrm{d}v\). Obviously the approximation breaks down for large values of the perturbation as soon as the r.h.s. becomes larger than unity. For small perturbations, however, we expand the trigonometric functions and get

\[\delta v =-\frac{1}{4\,\pi}\oint\beta(z)p_{2}(z)\,\mathrm{d}z \tag{15.69}\] \[-\frac{1}{4\pi\,\sin 2\pi v_{0}}\oint\beta(z)p_{2}(z)\sin\left\{2v _{0}\left[\pi-\varphi(z)\right]\right\}\,\mathrm{d}z\,.\]

The first term is the average tune shift which has been derived before, while the second term is of oscillatory nature averaging to zero over many turns if the tune of the circular accelerator is not equal to a half integer or multiples thereof. We have found hereby a second resonance condition to be avoided which occurs for half integer values of the tunes

\[v_{0}\neq\tfrac{1}{2}n\,. \tag{15.70}\]

This resonance is called a half integer resonance and causes divergent solutions for the lattice functions.

#### Optics Perturbation Due to Insertion Devices

The use of insertion devices in synchrotron light sources can introduce significant focusing perturbations. Undulators and wiggler magnets are a series of dipole magnets with end effects causing vertical focusing. This focusing will perturb the periodic betatron function in the storage ring and with it all correction that have been made. Because the perturbation scales like the square of the magnet field it is for most undulators too small to be significant. However, wiggler magnet may cause some problems. Like for orbit correction we do not like to spread the correction of the betatron functions all around the ring. To localize the correction to the vicinity of the insertion device we consider only the closest lattice quadrupoles on either side of the wiggler magnet which are not yet beyond the next insertion device. To minimize the number of quadrupoles needed for correction we start away from the wiggler where we expect the betatron functions to stay unperturbed, say in the middle of the next long straight section. Starting from there we adjust quadrupoles such that in the middle of the wiggler magnet \(\alpha_{x,y}=0\) and \(\eta^{\prime}=0\). That requires three quadrupoles for matching. If we match from the middle of the wiggler magnet to the middle of the next straight section we would need six quadrupoles to match \(\beta_{x,y}\), \(\alpha_{x,y}\), \(\eta\) and \(\eta^{\prime}\). For optimum localization of the perturbation one should use the three quadrupole closest to the wiggler magnet. This correction does not take care of the perturbations in the betatron phase. If two more quadrupoles are available one could try to use them for tune correction. This, however, is not always possible and one might therefore use two quadrupole families for the whole ring to readjust the tunes to the original value. Because there usually are many superperiods, the effect of a small tune correction is distributed around the ring and causes little variation in the beatron functions. The corrections in the quadrupoles are again for a permanent magnet wiggler gap-dependent and must be determined before general use to establish feed-forward of the computer control of the wiggler.

#### Resonances and Stop Band Width

Calculating the tune shift from (15.68) we noticed that there is no solution if the perturbation is too large such that the absolute value of the r.h.s. becomes larger than unity. In this case the tune becomes imaginary leading to ever increasing betatron oscillation amplitudes and beam loss. This resonance condition occurs not only at a half integer resonance but also in a finite vicinity. The region of instability is called the stop band and the width of unstable tune values is called the stop band width which can be calculated by using a higher approximation for the perturbed solution. Following the arguments of Courant and Snyder [1] we note that the perturbation (15.65) depends on the betatron oscillation \(w(\varphi)\) itself and we now use in the perturbation integral the first order approximation (15.66) rather than the unperturbed solution to calculate the perturbation (15.65). Then instead of (15.68) we get with

\[v_{0}\beta^{2}(\varphi)\,p_{2}(\varphi)=g\left(\varphi\right) \tag{15.71}\]

\[\cos 2\pi(v_{0}+\delta v)-\cos 2\pi\,v_{0}= +\int_{0}^{2\pi}g(\varphi)\cos(v_{0}\varphi)\sin v_{0}(2\pi- \varphi)\,\mathrm{d}\varphi\] \[+\ v_{0}\int_{0}^{2\pi}g(\chi)\sin v_{0}(2\pi-\chi) \tag{15.72}\] \[\times\int_{0}^{\chi}g(\zeta)\cos\left(v_{0}\zeta\right)\sin v_{0 }(\chi-\zeta)\,\mathrm{d}\zeta\,\mathrm{d}\chi\.\]

This expression can be used to calculate the stop band width due to gradient field errors which we will do for the integer resonance \(v_{0}=n+\delta v\) and for the half integer resonance \(v_{0}=n+1/2+\delta v\) where \(n\) is an integer and \(\delta v\) the deviation of the tune from these resonances. To evaluate the first integral \(I_{1}\), on the r.h.s. of (15.72) we make use of the relation

\[\cos\left(v_{0}\varphi\right)\sin\left[v_{0}(2\pi-\varphi)\right]=\tfrac{1}{2} \sin\left(2\pi v_{0}\right)+\tfrac{1}{2}\sin[2v_{0}(\pi-\varphi)] \tag{15.73}\]

and get with \(\mathrm{d}z=v_{0}\beta\mathrm{d}\varphi\) and

\[\oint\beta(z)\,p_{2}(z)\,\mathrm{d}z=\int_{0}^{2\pi}g(\varphi)\,\mathrm{d} \varphi=2\pi F_{0} \tag{15.74}\]

from (15.62) for the first integral

\[I_{1}=\pi F_{0}\sin\left(2\pi v_{0}\right)+\tfrac{1}{2}\int_{0}^{2\pi}g( \varphi)\sin\left[2v_{0}(\pi-\varphi)\right]\mathrm{d}\varphi\,. \tag{15.75}\]

The second term of the integral \(I_{1}\) has oscillatory character averaging to zero over many turns and with \(\delta\ll 1\)

\[I_{1}=\pi F_{0}\sin 2\pi v_{0}\approx\left\{\begin{aligned} & 2\pi^{2}F_{0}\,\delta v &\text{for }v_{0}=n+\delta v\\ &-2\pi^{2}F_{0}\,\delta v&\text{for }v_{0}=n+\tfrac{1}{2}+\delta v \end{aligned}\right.\,. \tag{15.76}\]

The second integral \(I_{2}\) in (15.72) can best be evaluated while expressing the trigonometric functions in their exponential form. Terms like \(\mathrm{e}^{\pm\mathrm{i}v(2\pi-2\zeta)}\) or \(\mathrm{e}^{\pm\mathrm{i}v(2\pi-2\zeta)}\) vanish on average over many turns. With

\[\int_{0}^{2\pi}f\left(\chi\right)\mathrm{d}\chi\int_{0}^{\chi}f\left(\zeta \right)\mathrm{d}\zeta=\tfrac{1}{2}\int_{0}^{2\pi}f\left(\chi\right)\mathrm{d }\chi\int_{0}^{2\pi}f\left(\zeta\right)\mathrm{d}\zeta\]

we get for the second integral

\[I_{2}=-\frac{v_{0}}{16}\int_{0}^{2\pi}g(\chi)\int_{0}^{2\pi}g( \zeta)\\ \times\left\{(\mathrm{e}^{\mathrm{i}2\pi v_{0}}+\mathrm{e}^{- \mathrm{i}2\pi v_{0}})-[\mathrm{e}^{\mathrm{i}2v_{0}(\pi-\chi+\zeta)}+ \mathrm{e}^{-\mathrm{i}2v_{0}(\pi-\chi+\zeta)}]\right\}\,\mathrm{d}\zeta\, \mathrm{d}\chi.\]

Close to the integer resonance \(v_{0}=n+\delta v\) and

\[I_{2,n}=-\frac{v_{0}}{16}\int_{0}^{2\pi}g(\chi)\int_{0}^{2\pi}g( \zeta)\\ \times\left\{(\mathrm{e}^{\mathrm{i}2\pi\delta v}+\mathrm{e}^{- \mathrm{i}2\pi\delta v})-[\mathrm{e}^{\mathrm{i}2\pi(\zeta-\chi)}+\mathrm{e}^ {-2n(\zeta-\chi)}]\right\}\,\mathrm{d}\zeta\,\mathrm{d}\chi\]while in the vicinity of the half integer resonance \(v_{0}=n+\frac{1}{2}+\delta v\)

\[I_{2,n+\frac{1}{2}}=-\frac{v_{0}}{16}\int_{0}^{2\pi}g(\chi)\int_{0 }^{2\pi}g(\zeta)\left\{-(\mathrm{e}^{\mathrm{i}2\pi\delta v}+\mathrm{e}^{- \mathrm{i}2\pi\delta v})\right.\\ \left.+\left[\mathrm{e}^{\mathrm{i}2\left(n+\frac{1}{2}\right)( \zeta-\chi)}+\mathrm{e}^{-\mathrm{i}2\left(n+\frac{1}{2}\right)(\zeta-\chi)} \right]\right\}\,\mathrm{d}\zeta\,\mathrm{d}\chi\,.\]

The integralscan now be expressed in terms of Fourier harmonics of \(v_{0}\beta^{2}\left(\varphi\right)p_{1}(\varphi)\), where the amplitudes of the harmonics \(F_{q}\) with integer \(q>0\) are given by

\[\left|F_{q}\right|^{2}=F_{q}F_{q}^{\star}=\frac{v_{0}}{\pi^{2}}\int_{0}^{2\pi} g(\chi)\,\mathrm{e}^{-\mathrm{i}q\chi}\,\mathrm{d}\chi\int_{0}^{2\pi}g(\zeta)\, \mathrm{e}^{\mathrm{i}q\zeta}\,\mathrm{d}\zeta\,. \tag{15.79}\]

For \(F_{0}\) we have from the Fourier transform the result

\[F_{0}=\left\langle g(\varphi)\right\rangle=v_{0}\left\langle\beta^{2}(\varphi )p_{2}(\varphi)\right\rangle \tag{15.80}\]

and we get for (15.78) with this and ignoring terms quadratic in \(\delta v\)

\[I_{2,n}\approx\tfrac{1}{8}\pi^{2}\left(F_{2n}^{2}-4\,F_{0}^{2}\cos 2\pi\delta v\right) \tag{15.81}\]

and for (9.57)

\[I_{2,n+\frac{1}{2}}\approx-\tfrac{1}{8}\pi^{2}\left(F_{2n+1}^{2}-4\,F_{0}^{2} \cos 2\pi\delta v\right)\,. \tag{15.82}\]

At this point we may collect the results and get on the l.h.s. of (15.72) for \(v_{0}=n+\delta v\)

\[\cos 2\pi(v_{0}+\delta v)-\cos 2\pi v_{0}=\cos 2\pi(v_{0}+\delta v)-1+2\pi^{2} \,\delta v^{2}\,.\]

This must be equated with the r.h.s. which is the sum of integrals \(I_{1}\) and \(I_{2}\) and with \(F_{0}^{2}\,\cos 2\pi\delta v=\mathcal{O}(\delta^{4}v)\)

\[\cos 2\pi(v_{0}+\delta v)-1=-2\pi^{2}\delta v^{2}+2\pi^{2}F_{0}\,\delta v+ \tfrac{1}{8}\pi^{2}(F_{2n}^{2}-4F_{0}^{2})\,. \tag{15.83}\]

The boundaries of the stop band on either side of the integer resonance \(v_{0}\approx n\) can be calculated from the condition that \(\cos 2\pi(v_{0}+\delta v)\leq 1\) which has two solutions \(\delta v_{1,2}\). From (15.83) we get therefore

\[\delta v^{2}-F_{0}\,\delta v=\tfrac{1}{16}(\left|F_{2n}\right|^{2}-4F_{0}^{2})\]

and solving for \(\delta v\)

\[\delta v_{1,2}=\tfrac{1}{2}F_{0}\pm\tfrac{1}{4}|F_{2n}\,| \tag{15.84}\]the stop band width is finally

\[\Delta v=\delta v_{1}-\delta v_{2}=\tfrac{1}{2}|F_{2n}|=\frac{1}{2\pi}\oint\beta(z )\,p_{2}(z)\,\mathrm{e}^{-\mathrm{i}2n\oint(z)}\,\mathrm{d}z\,. \tag{15.85}\]

The stop band width close to the integer tune \(v\approx n\) is determined by the second harmonic of the Fourier spectrum for the perturbation. The vicinity of the resonance for which no stable betatron oscillations exist increases with the value of the gradient field error and with the value of the betatron function at the location of the field error. For the half integer resonance \(v_{0}\approx n+\tfrac{1}{2}\), the stop band width has a similar form

\[\Delta v_{\tfrac{1}{2}}=\tfrac{1}{2}|F_{2n+1}|=\frac{1}{2\pi}\int_{0}^{2\pi} \beta(z)\,p_{2}(z)\,\mathrm{e}^{-\mathrm{i}(2n+1)\oint(z)}\,\mathrm{d}z\,. \tag{15.86}\]

The lowest order Fourier harmonic \(n=0\) determines the static tune shift while the resonance width depends on higher harmonics. The existence of finite stop bands is not restricted to linear perturbation terms only. Nonlinear, higher order perturbation terms lead to higher order resonances and associated stop bands. In such cases one would replace in (15.60) the linear perturbation \(\beta^{\frac{1}{2}}p_{2}(z)\,w\) by the \(n\)th order nonlinear perturbation \(\beta^{n/2}p_{n}(z)\,w^{n-1}\) and basically go through the same derivation. Later in this chapter, we will use a different way to describe resonance characteristics caused by higher order perturbations. At this point we note only that perturbations of order \(n\) are weighted by the \(n/2\) power of the betatron function at the location of the perturbation and increased care must be exercised, where large values of the betatron functions cannot be avoided. Undesired fields at such locations must be minimized.

#### Perturbation of Betatron Function

The existence of linear perturbation terms causes not only the tunes but also betatron functions to vary around the ring or along a beam line. This variation, also called beta-beat can be derived by observing the perturbation of a particular trajectory like for example the sine-like solution given by

\[S_{0}(z_{0}|z)=\sqrt{\beta(z)}\sqrt{\beta_{0}}\sin v_{0}[\varphi(z)-\varphi_{ 0}]\,. \tag{15.87}\]

The sine-like function or trajectory in the presence of linear perturbation terms is by the principle of linear superposition the combination of the unperturbed solution (8.74) and perturbation (5.75)

\[S(z_{0}|z)=\sqrt{\beta(z)}\sqrt{\beta_{0}}\sin v_{0}\varphi(z)\\ +\sqrt{\beta(z)}\int_{z_{0}}^{z}p_{2}(\xi)\sqrt{\beta(\xi)}S_{0} (z_{0}|\xi)\sin v_{0}[\varphi(z)-\varphi(\xi)]\,\mathrm{d}\xi\,.\]Following the sinusoidal trajectory for the whole ring circumference or length of a superperiod \(L_{\rm p}\), we have with \(z=z_{0}+L_{\rm p}\), \(\beta(z_{0}+L_{\rm p})=\beta(z_{0})=\beta_{0}\) and \(\varphi(z_{0}+L_{\rm p})=2\pi+\varphi_{0}\)

\[S(z_{0}|z_{0}+L_{\rm p}) = \beta_{0}\sin 2\pi\,v_{0}+\beta_{0}\oint_{z_{0}}^{z_{0}+L_{\rm p}} \beta(\xi)\,p_{1}(\xi)\] \[\times \sin v_{0}[\varphi(\xi)-\varphi_{0}]\sin\left[v_{0}\left(2\pi\,+ \varphi_{0}-\varphi(\xi)\right)\right]\,d\xi\,.\]

The difference due to the perturbation from the unperturbed trajectory (15.87) at \(z=z_{0}+L_{\rm p}\) is

\[\Delta S = S(z_{0}|z_{0}+L_{\rm p})-S_{0}(z_{0}|z_{0}+L_{\rm p})\] \[= \beta_{0}\int_{z_{0}}^{z_{0}+L_{\rm p}}\beta(\xi)\,p_{2}(\xi)\sin [v_{0}(\varphi_{\xi}-\varphi_{0})]\sin[v_{0}(2\pi\,+\varphi_{0}-\varphi_{\xi} )]\,{\rm d}\xi\,,\]

where we abbreviated \(\varphi(z_{0})=\varphi_{0}\) etc.

The variation of the sine like function can be derived also from the variation of the \(M_{12}\) element of the transformation matrix for the whole ring

\[\Delta S=\Delta(\beta\sin 2\pi\,v)=\Delta\beta\,\sin 2\pi\,v_{0}+\beta_{0}\,2 \pi\,\Delta v\cos 2\pi\,v_{0}\,. \tag{15.90}\]

We use (15.64) for the tune shift \(\delta v=-\frac{1}{2}\,F_{0}\), equate (15.90) with (15.89) and solve for \(\Delta\beta/\beta\). After some manipulations, where we replace temporarily the trigonometric functions by their exponential expressions, the variation of the betatron function becomes at \(\varphi(z)\)

\[\frac{\Delta\beta(z)}{\beta(z)}=\frac{1}{2\sin 2\pi\,v_{0}}\oint\beta(\xi)\,p_{2 }(\xi)\cos\left[2v_{0}\left(\varphi(z)-\varphi(\xi)+\pi\right)\right]\,{\rm d }\xi\,. \tag{15.91}\]

The perturbation of the betatron function shows clearly resonance character and a half integer tune must be avoided. We observe a close similarity with the solution (10.91) of the dispersion function or the closed orbit (15.28). Setting \({\rm d}\xi=v_{0}\beta(\xi)\,{\rm d}\varphi\), we find by comparison that the solution for the perturbed betatron function can be derived from a differential equation similar to a modified equation (10.88)

\[\frac{{\rm d}^{2}}{{\rm d}\varphi^{2}}\left(\frac{\Delta\beta}{\beta}\right)+ (2v_{0})^{2}\frac{\Delta\beta}{\beta}=(2v_{0})^{2}\frac{1}{2}\beta^{2}(z)\,p_{ 2}(z)\,. \tag{15.92}\]

Expanding the periodic function \(v_{0}\beta^{2}p_{2}=\sum_{q}F_{q}\,{\rm e}^{{\rm i}q\varphi}\) we try the periodic ansatz

\[\frac{\Delta\beta}{\beta}=\sum_{q}B_{q}F_{q}\,{\rm e}^{{\rm i}q\varphi} \tag{15.93}\]and get from (15.92)

\[\sum_{q}\left[-q^{2}+(2v_{0})^{2}\right]B_{q}F_{q}\,\mathrm{e}^{\mathrm{i}q \varphi}=2v_{0}\sum_{q}F_{q}\,\mathrm{e}^{\mathrm{i}q\varphi}\;.\]

This can be true for all values of the phase \(\varphi\) only if the coefficients of the exponential functions vanish separately for each value of \(q\) or if

\[B_{q}=\frac{2v_{0}}{(2v_{0})^{2}-q^{2}}\;. \tag{15.94}\]

Inserting into the periodic ansatz (15.93) the perturbation of the betatron function in another form is

\[\frac{\Delta\beta}{\beta}=\frac{v_{0}}{2}\sum_{q}\frac{F_{q}\,\mathrm{e}^{ \mathrm{i}q\varphi}}{v_{0}^{2}-(q/2)^{2}}\;. \tag{15.95}\]

Again we recognize the half inter resonance leading to an infinitely large perturbation of the betatron function. In the vicinity of the half integer resonance \(v_{0}\approx n+\frac{1}{2}=q/2\) the betatron function can be expressed by the resonant term only

\[\frac{\Delta\beta}{\beta}\approx\tfrac{1}{2}|F_{2n+1}|\frac{\cos(2n+1)\varphi} {v_{0}-\left(n+\frac{1}{2}\right)}\]

and with \(|F_{2n+1}|=2\Delta v_{\frac{1}{2}}\) from (15.86) we get again the perturbation of the betatron function (15.91). The beat factor for the variation of the betatron function is define by

\[BF=1+\left(\frac{\Delta\beta}{\beta_{0}}\right)_{\text{max}}=1+\frac{\Delta v _{2n+1}}{2v_{0}-(2n+1)}\;, \tag{15.96}\]

where \(\Delta v_{2n+1}\) is the half integer stop band width. The beating of the betatron function is proportional to the stop band width and therefore depends greatly on the value of the betatron function at the location of the perturbation. Even if the tune is chosen safely away from the next resonance, a linear perturbation at a large betatron function may still cause an unacceptable beat factor. It is generally prudent to design lattices in such a way as to avoid large values of the betatron functions. As a practical note, any value of the betatron function which is significantly larger than the quadrupole distances should be considered large. For many beam transport problems this is easier said than done. Therefore, where large betatron functions cannot be avoided or must be included to meet our design goals, results of perturbation theory warn us to apply special care for beam line component design, alignment and to minimize undesirable stray fields.

### 15.4 Chromatic Effects in a Circular Accelerator

Energy independent perturbations as discussed in previous sections can have a profound impact on the stability of particle beams in the form of perturbations of the betatron function or through resonances. Any beam transport line must be designed and optimized with these effects in mind since it is impossible to fabricate ideal magnets and align them perfectly. Although such field and alignment errors can have a destructive effect on a beam, it is the detailed understanding of these effects that allow us to minimize or even avoid such problems by a careful design within proven technology.

To complete the study of perturbations, we note that a realistic particle beam is never quite mono-energetic and includes a finite distribution of particle energies. Bending as well as focusing is altered if the particle momentum is not the ideal momentum. We already derived the momentum dependent reference path in transport lines involving bending magnets. Beyond this basic momentum dependent effect we observe other chromatic aberrations which contribute in a significant way to the perturbations of lattice functions. The effect of chromatic aberrations due to a momentum error is the same as that of a corresponding magnet field error and for beam stability we must include chromatic aberrations.

#### Chromaticity

Perturbations of beam dynamics can occur in beam transport systems even in the absence of magnet field and alignment errors. Deviations of particle energies from the ideal design energy cause perturbations in the solutions of the equations of motion. We have already derived the variation of the equilibrium orbit for different energies. Energy related or chromatic effects can be derived also for other lattice functions. Focusing errors due to an energy error cause such particles to be imaged at different focal points causing a blur of the beam spot. In a beam transport system, where the final beam spot size is of great importance as, for example, at the collision point of linear colliders, such a blur causes a severe degradation of the attainable luminosity. In circular accelerators we have no such direct imaging task but note that the tune of the accelerator is determined by the overall focusing and tune errors occur when the focusing system is in error.

In this chapter we will specifically discuss effects of energy errors on tunes of a circular accelerator and means to compensate for such chromatic aberrations. The basic means of correction are applicable to either circular or open beam transport systems if, for the latter case, we only replace the tune by the phase advance of the transport line in units of \(2\pi\). The control of these chromatic effects in circular accelerators is important for two reasons, to avoid loss of particles due to tune shifts into resonances and to prevent beam loss due to an instability, which we call the head tail instability to be discussed in more detail in Sect. 22.5.

The lowest order chromatic perturbation is caused by the variation of the focal length of the quadrupoles with energy (Fig. 15.8). This kind of error is well known from light optics, where a correction of this chromatic aberration can at least partially be obtained by the use of different kinds of glasses for the lenses in a composite focusing system.

In particle beam optics no equivalent approach is possible. To still correct for the chromatic perturbations we remember that particles with different energies can be separated by the introduction of a dispersion function. Once the particles are separated by energy we apply different focusing corrections depending on the energy of the particles. Higher energy particles are focused less than ideal energy particles and lower energy particles are over focused. For a correction of these focusing errors we need a magnet which is focusing for higher energy particles and defocusing for lower energy particles (Fig. 15.9). A sextupole has just that property.

The variation of tunes with energy is called the chromaticity and is defined by

\[\xi=\frac{\Delta v}{\Delta p/p_{0}}. \tag{15.97}\]

The chromaticity derives from second and higher order perturbations in \((x,y,\delta)\) and the relevant equations of motion are from

\[\begin{array}{l}x^{\prime\prime}+k\,x=kx\delta-\frac{1}{2}m(x^{2}-y^{2})\;, \\ y^{\prime\prime}-k\,x=-ky\delta+mxy\;.\end{array} \tag{15.98}\]

Figure 15.8: Chromatic focusing errors

Figure 15.9: Chromaticity correction with sextupoles

Setting \(x=x_{\beta}+\eta_{x}\,\delta\) and \(y=y_{\beta}\), assuming that \(\eta_{y}\equiv 0\), we retain only betatron oscillation terms involving \(x_{\beta}\) or \(y_{\beta}\) to derive chromatic tune shifts. In doing so we note three types of chromatic perturbation terms, those depending on the betatron motion only, those depending on the momentum error only, and terms depending on both. With these expansions (15.98) becomes

\[\begin{array}{l}x_{\beta}^{\prime\prime}+kx_{\beta}=\quad\,\,\,\,kx_{\beta} \delta-m\eta_{x}x_{\beta}\delta-\tfrac{1}{2}m(x_{\beta}^{2}-y_{\beta}^{2})+ \mathcal{O}(3)\,,\\ y_{\beta}^{\prime\prime}-ky_{\beta}=-ky_{\beta}\delta+m\eta_{x}y_{\beta}\delta+ mx_{\beta}y_{\beta}+\mathcal{O}(3)\,.\end{array} \tag{15.99}\]

We ignore for the time being non chromatic terms of second order which will be discussed later as geometric aberrations and get

\[\begin{array}{l}x_{\beta}^{\prime\prime}+k\,x_{\beta}=\quad\left(k-m\,\eta_ {x}\right)x_{\beta}\delta\,,\\ y_{\beta}^{\prime\prime}-k\,y_{\beta}=-\left(k-m\,\eta_{x}\right)y_{\beta} \delta\,.\end{array} \tag{15.100}\]

The perturbation terms now are linear in the betatron amplitude and therefore have the character of a gradient error. From Sect. 15.3 we know that these types of errors lead to a tune shift which by comparison with (15.64) becomes in terms of a phase shift

\[\begin{array}{l}\Delta\psi_{x}=-\tfrac{1}{2}\delta\,\dot{\phi}\,\beta_{x}(k- m\eta_{x})\,\mathrm{d}z,\\ \Delta\psi_{y}=+\tfrac{1}{2}\delta\,\dot{\phi}\,\beta_{y}(k-m\,\eta_{x})\, \mathrm{d}z\,.\end{array} \tag{15.101}\]

Equations (15.101) are applicable for both circular and open beam lines. Using the definition of the chromaticity for circular accelerators we have finally

\[\begin{array}{l}\dot{\xi}_{x}=-\tfrac{1}{4\pi}\,\dot{\phi}\,\beta_{x}(k-m \eta_{x})\,\mathrm{d}z\,,\\ \dot{\xi}_{y}=+\tfrac{1}{4\pi}\,\dot{\phi}\,\beta_{y}(k-m\eta_{x})\,\mathrm{d}z \,.\end{array} \tag{15.102}\]

Similar to the definition of tunes the chromaticities are also an integral property of the circular accelerator lattice. Setting the sextupole strength \(m\) to zero one gets the natural chromaticities determined by focusing terms only

\[\begin{array}{l}\dot{\xi}_{x0}=-\tfrac{1}{4\pi}\,\dot{\phi}\,\beta_{x}k\, \mathrm{d}z\,,\\ \dot{\xi}_{y0}=+\tfrac{1}{4\pi}\,\dot{\phi}\,\beta_{y}k\,\mathrm{d}z\,.\end{array} \tag{15.103}\]

The natural chromaticities are always negative which is to be expected since focusing is less effective for higher energy particles (\(\delta>0\)) and therefore the number of betatron oscillations is reduced.

For a thin lens symmetric FODO lattice the calculation of the chromaticity becomes very simple. With the betatron function \(\beta^{+}\) at the center of a focusing quadrupole of strength \(k^{+}=k\) and \(\beta^{-}\) at the defocusing quadrupole of strength \(k^{-}=k\), the chromaticity of one FODO half cell is

\[\xi_{x0}=-\frac{1}{4\pi}\left(\beta^{+}\int k^{+}\mathrm{d}z+\beta^{-}\int k^{-} \mathrm{d}z\right)=-\frac{\beta^{+}-\beta^{-}}{4\pi}\int k\,\mathrm{d}z\,. \tag{15.104}\]

With \(\beta^{+}\) (10.3) and \(\beta^{-}\) (10.5) and \(\int k\mathrm{d}z=1/f=1/(\kappa L)\), where \(\kappa\) is the FODO strength parameter and \(L\) the length of a FODO half cell, we get the chromaticity per FODO half-cell in a more practical formulation

\[\xi_{x0}=-\frac{1}{2\pi}\,\frac{1}{\sqrt{\kappa^{2}-1}}=-\frac{1}{\pi}\tan \left(\frac{1}{2}\psi_{x}\right)\,, \tag{15.105}\]

where \(\psi_{x}\) is the horizontal betatron phase for the full FODO cell. The same result can be obtained for the vertical plane.

The natural chromaticity for each \(90^{\circ}\) FODO cell is therefore equal to \(1/\pi\). Although this value is rather small, the total chromaticity for the complete lattice of a storage ring or synchrotron, made up of many FODO cells, can become quite large. For the stability of a particle beam and the integrity of the imaging process by quadrupole focusing it is important that the natural chromaticity be corrected.

It is interesting at this point to discuss for a moment the chromatic effect if, for example, all bending magnets have a systematic field error with respect to other magnets. In an open beam transport line the beam would follow an off momentum path as determined by the difference of the beam energy and the bending magnet "energy". Any chromatic aberration from quadrupoles as well as sextupoles would occur just as discussed.

In a circular accelerator the effect of systematic field errors might be different. We consider, for example, the case where we systematically change the strength of all bending magnets. In an electron storage ring, the particle beam would automatically stay at the ideal design orbit with the particle energy being defined by the strength of the bending magnets. The strength of the quadrupoles and sextupole magnets, however, would now be systematically too high or too low with respect to the bending magnet field and particle energy. Quadrupoles introduce therefore a chromatic tune shift proportional to the natural chromaticity while the sextupoles are ineffective because the beam orbit leads through magnet centers. Changing the strength of the bending magnets by a fraction \(\Delta\) in an electron circular accelerator and measuring the tune change \(\Delta v\) one can determine experimentally the natural chromaticity (\(\xi_{0}=-\Delta v/\Delta\)) of the ring. In Fig. 15.10 the measurement of the tunes as a function of the bending magnet current is shown for the storage ring SPEAR. From the slope of the graphs we derive the natural chromaticities of the SPEAR storage ring as \(\xi_{x}=-11.4\) and \(\xi_{y}=-11.7\).

In a proton accelerator the beam energy must be changed through acceleration or deceleration together with a change of the bending magnet strength to keep the beam on the reference orbit before this measurement can be performed.

#### Chromaticity Correction

Equations (15.102) clearly suggest the usefulness of sextupole magnets for chromatic correction. Sextupoles must be placed along the orbit of a circular accelerator or along a beam transport line at locations, where the dispersion function does not vanish, \(\eta_{x}\neq 0\). A single sextupole is sufficient, in principle, to correct the chromaticity for the whole ring or transport line but its strength may exceed technical limits or cause problems of geometric aberrations to the beam stability. This is due to the nonlinear nature of sextupole fields which causes dynamic instability for large amplitudes for which the sextupole field is no more a perturbation. The largest betatron oscillation amplitude which is still stable in the presence of nonlinear fields is called the dynamic aperture. To maximize the dynamic aperture it is prudent to distribute many chromaticity correcting sextupoles along the beam line or circular accelerator.

To correct both the horizontal and the vertical chromaticity two different groups of sextupoles are required. For a moment we assume that there be only two sextupoles. To calculate the required strength of these sextupoles for chromaticity correction we use thin lens approximation and replacing integrals in (15.102) by a sum the corrected chromaticities are

\[\begin{split}\xi_{x}&=\xi_{x0}+\tfrac{1}{4\pi}(m_ {1}\eta_{x1}\beta_{x1}+m_{2}\eta_{x2}\beta_{x2})\ell_{\mathrm{s}}=0\,,\\ \xi_{y}&=\xi_{y0}+\tfrac{1}{4\pi}(m_{1}\eta_{x1} \beta_{y1}+m_{2}\eta_{x2}\beta_{y2})\ell_{\mathrm{s}}=0\,.\end{split} \tag{15.106}\]

Here we assume that two different sextupoles, each of length \(\ell_{s}\), are available at locations \(z_{1}\) and \(z_{2}\). Solving for the sextupole strengths we get from (15.106)

\[m_{1}\ell_{\mathrm{s}} =-\frac{4\pi}{\eta_{x1}}\frac{\xi_{x0}\,\beta_{y2}-\xi_{y0}\, \beta_{x2}}{\beta_{x1}\beta_{y2}-\beta_{x2}\beta_{y1}}\,, \tag{15.107a}\] \[m_{2}\ell_{\mathrm{s}} =-\frac{4\pi}{\eta_{x2}}\frac{\xi_{x0}\,\beta_{y1}-\xi_{y0}\, \beta_{x1}}{\beta_{x1}\beta_{y2}-\beta_{x2}\beta_{y1}}\,. \tag{15.107b}\]

Figure 15.10: Experimental determination of the natural chromaticity in a storage ring by measuring the tunes as a function of the excitation current \(I=I_{0}+\Delta I\) in the bending magnetsIt is obvious that the dispersion function at sextupoles should be large to minimize sextupoles strength. It is also clear that the betatron functions must be different preferably with \(\beta_{x}\gg\beta_{y}\) at the \(m_{1}\) sextupole and \(\beta_{x}\ll\beta_{y}\) at the \(m_{2}\) sextupole to avoid "fighting" between sextupoles leading to excessive strength requirements.

In general this approach based on only two sextupoles in a ring to correct chromaticities leads to very strong sextupoles causing both magnetic design problems and strong higher order aberrations. A more gentle correction uses two groups or families of sextupoles with individual magnets distributed more evenly around the circular accelerator and the total required sextupole strength is spread over all sextupoles. In cases of severe aberrations, as discussed later, we will need to subdivide all sextupoles into more than two families for a more sophisticated correction of chromaticities. Instead of (15.106) we write for the general case of chromaticity correction

\[\begin{array}{l}\xi_{x}=\xi_{x0}+\frac{1}{4\pi}\sum_{i}m_{i}\eta_{xi}\beta_ {xi}\ell_{si}\\ \xi_{y}=\xi_{y0}+\frac{1}{4\pi}\sum_{i}m_{i}\eta_{xi}\beta_{yi}\ell_{si}\,, \end{array} \tag{15.108}\]

where the sum is taken over all sextupoles. In the case of a two family correction scheme we still can solve for \(m_{1}\) and \(m_{2}\) by grouping the terms into two sums.

The chromaticity of a circular accelerator as defined in this section obviously does not take care of all chromatic perturbations. Since the function \((k-m\eta_{x})\) in (15.100) is periodic, we can Fourier analyze it and note that the chromaticity only describes the effect of the non-oscillating lowest order Fourier component (15.103). All higher order components are treated as chromatic aberrations. In Sect. 17.2 we will discuss in more detail such higher order chromatic and geometric aberrations.

#### Chromaticity in Higher Approximation

So far we have used only quadrupole and sextupole fields to define and calculate the chromaticity. From the general equations of motion we know, however, that many more perturbation terms act just like sextupoles and therefore cannot be omitted without further discussion. To derive the relevant equations of motion from (6.95), (6.96) we set \(x=x_{\beta}+\eta_{x}\delta\) and \(y=y_{\beta}+\eta_{y}\delta\) where we keep for generality the symmetry between vertical and horizontal plane. Neglecting, however, coupling terms we get with perturbations quadratic in \((x,y,\delta)\) but at most linear in \(\delta\) and after separating the dispersion function a differential equation of the form (\(u=x_{\beta}\) or \(y_{\beta}\))

\[u^{\prime\prime}_{\beta}+K\,u_{\beta}=-\Delta K\,u_{\beta}\delta-\Delta L\,u^ {\prime}_{\beta}\,\delta\,, \tag{15.109}\]where

\[K_{x} = \kappa_{x}^{2}+k\,, \tag{15.110}\] \[K_{y} = \kappa_{y}^{2}-k\,,\] (15.111) \[-\Delta K_{x} = 2\kappa_{x}^{2}+k-(m+2\kappa_{x}^{3}+4\kappa_{x}k)\eta_{x}\] \[\quad-(\underline{m}+2\kappa_{x}\underline{k}+2\kappa_{y}k)\eta_ {y}+\kappa_{x}^{\prime}\eta_{x}^{\prime}-\kappa_{y}^{\prime}\eta_{y}^{\prime}\,,\] \[-\Delta K_{y} = 2\kappa_{y}^{2}-k+(m-2\kappa_{y}\underline{k}+2\kappa_{x}k)\eta _{x}\] \[\quad+(\underline{m}-2\kappa_{y}^{3}+4\kappa_{y}k)\eta_{y}- \kappa_{x}^{\prime}\eta_{x}^{\prime}+\kappa_{y}^{\prime}\eta_{y}^{\prime}\,,\] \[-\Delta L_{x} = -\Delta L_{y}=+\kappa_{x}^{\prime}\eta_{x}+\kappa_{y}^{\prime} \eta_{y}+\kappa_{x}\eta_{x}^{\prime}+\kappa_{y}\eta_{y}^{\prime}\] \[= +\frac{\mathrm{d}}{\mathrm{d}z}(\kappa_{x}\eta_{x}+\kappa_{y}\eta _{y})\,.\]

The perturbation terms (15.109) depend on the betatron oscillation amplitude as well as on the slope of the betatron motion. If by some manipulation we succeed in transforming (15.109) into an equation with terms proportional only to \(u\) we obtain immediately the chromaticity. We try a transformation of the form \(u=v\,f(z)\) where \(f(z)\) is a still to be determined function of \(z\). With \(u^{\prime}=v^{\prime}\!f+v\,f^{\,\prime}\) and \(u^{\prime\prime}=v^{\prime\prime}\!f+2v^{\prime}\!f^{\,\prime}+v\,f^{\,\prime\prime}\) (15.109) becomes

\[v^{\prime\prime}\!f+2v^{\prime}\!f^{\,\prime}+v\,f^{\,\prime\prime}+Kvf+\, \Delta Kvf\,\delta+\,\Delta Lv^{\prime}\!f\,\delta+\,\Delta Lvf^{\,\prime}\! \delta=0\,. \tag{15.115}\]

Now we introduce a condition defining the function \(f\) such that in (15.115) the coefficients of \(v^{\prime}\) vanish. This occurs if

\[2\!f^{\,\prime}=-\Delta Lf\delta\,. \tag{15.116}\]

To first order in \(\delta\) this equation can be solved by

\[f=1+\tfrac{1}{2}\delta(\kappa_{x}\eta_{x}+\kappa_{y}\eta_{y}) \tag{15.117}\]

and (15.115) becomes

\[v^{\prime\prime}+\big{[}K+(f^{\,\prime\prime}+\delta\Delta K)\big{]}\,v=0\,. \tag{15.118}\]

The chromaticity in this case is \(\xi=\frac{1}{4\pi}\oint(\frac{f^{\,\prime\prime}}{\delta}+\Delta K)\beta \mathrm{d}z\), which becomes with \(\frac{2\!f^{\,\prime\prime}}{\delta}=\frac{\mathrm{d}^{2}}{\mathrm{d}z^{2}}\,( \kappa_{x}\eta_{x}+\kappa_{y}\eta_{y})\) and (15.112)

\[\xi_{x} = \frac{1}{4\pi}\oint\left(\frac{f^{\,\prime\prime}}{\delta}+ \,\Delta K_{x}\right)\beta_{x}\,\mathrm{d}z\] \[= \frac{1}{4\pi}\oint\!\tfrac{1}{2}\frac{\mathrm{d}^{2}}{\mathrm{d}z ^{2}}(\kappa_{x}\eta_{x}+\kappa_{y}\eta_{y})\beta_{x}\,\mathrm{d}z\]\[-\frac{1}{4\pi}\oint\beta_{x}\left[(2\kappa_{x}^{2}+k)+\kappa_{x}^{ \prime}\eta_{x}^{\prime}-\kappa_{y}^{\prime}\eta_{y}^{\prime}\right.\] \[\left.-(m+2\kappa_{x}^{3}+\left.4\kappa_{x}k)\,\eta_{x}-(\underline {m}+2\kappa_{x}\underline{k}+2\kappa_{y}k)\,\eta_{y}\right]\,\mathrm{d}z\,.\]

The first integral can be integrated twice by parts to give \(\oint\frac{1}{2}(\kappa_{x}\eta_{x}+\kappa_{y}\eta_{y})\,\beta^{\prime\prime} \mathrm{d}z\). Using \(\frac{1}{2}\beta^{\prime\prime}=\gamma_{x}-K\beta\), and (15.119) the horizontal chromaticity is finally

\[\xi_{x} = \frac{1}{4\pi}\oint\left[-(k+2\,\kappa_{x}^{2})-\kappa_{x}^{ \prime}\eta_{x}^{\prime}-\kappa_{y}^{\prime}\eta_{y}^{\prime}\right.\] \[\left.+(m+\kappa_{x}^{3}+3\kappa_{x}k)\eta_{x}+(\underline{m}+2 \kappa_{x}\underline{k}+\kappa_{y}k)\eta_{y}\right]\beta_{x}\,\mathrm{d}z\] \[+\frac{1}{4\pi}\oint(\kappa_{x}\eta_{x}+\kappa_{y}\eta_{y})\gamma _{x}\,\mathrm{d}z\,.\]

A similar expression can be derived for the vertical chromaticity

\[\xi_{y} = \frac{1}{4\pi}\oint\left[(-2\kappa_{y}^{2}+k)+\kappa_{x}^{ \prime}\eta_{x}^{\prime}-\kappa_{y}^{\prime}\eta_{y}^{\prime}\right.\] \[\left.-(m+2\kappa_{x}\underline{k}+\kappa_{y}k)\eta_{x}-( \underline{m}-\kappa_{y}^{3}+3\kappa_{y}k)\eta_{y}\right]\beta_{y}\,\mathrm{d}z\] \[+\frac{1}{4\pi}\oint(\kappa_{x}\eta_{x}+\kappa_{y}\eta_{y}) \gamma_{y}\,\mathrm{d}z\,.\]

In deriving the chromaticity we used the usual curvilinear coordinate system for which the sector magnet is the natural bending magnet. For rectangular or wedge magnets the chromaticity must be determined from (15.121) by taking the edge focusing into account. Generally, this is done by applying a delta function focusing at the edges of dipole magnets with a focal length of

\[\frac{1}{f_{x}}=\frac{1}{\rho}\tan\epsilon\int\delta(z_{\mathrm{edge}})\, \mathrm{d}z\,. \tag{15.122}\]

Similarly, we proceed with all other terms which include focusing.

The chromaticity can be determined experimentally simply by measuring the tunes for a beam circulating with a momentum slightly different from the lattice reference momentum. In an electron ring, this is generally not possible since any momentum deviation of the beam is automatically corrected by radiation damping within a short time. To sustain an electron beam at a momentum different from the reference energy, we must change the frequency of the accelerating cavity. Due to the mechanics of phase focusing, a particle beam follows such an orbit that the particle revolution time in the ring is an integer multiple of the rf-oscillation period in the accelerating cavity. By proper adjustment of the rf-frequency the beam orbit is centered along the ideal orbit and the beam momentum is equal to the ideal momentum as determined by the actual magnetic fields.

If the rf-frequency is raised, for example, the oscillation period becomes shorter and the revolution time for the beam must become shorter too. This is accomplished only if the beam momentum is changed in such a way that the particles now follow a new orbit that is shorter than the ideal reference orbit. Such orbits exist for particles with momenta less than the reference momentum. The relation between revolution time and momentum deviation is a lattice property expressed by the momentum compaction which we write now in the form

\[\frac{\Delta f_{\rm rf}}{f_{\rm rf}}=-\eta_{\rm c}\,\frac{\Delta cp}{cp_{0}}\,. \tag{15.123}\]

Through the knowledge of the lattice and momentum compaction we can relate a relative change in the rf-frequency to a change in the beam momentum. Measurement of the tune change due to the momentum change determines immediately the chromaticity.

##### Non-linear Chromaticity*

The chromaticity of a circular accelerator is defined as the linear rate of change of the tunes with the relative energy deviation \(\delta\). With the increased amount of focusing that is being applied in modern circular accelerators, especially in storage rings, to obtain specific particle beam properties like very high energies in large rings or a small emittance, the linear chromaticity term is no longer sufficient to describe the chromatic dynamics of particle motion. Quadratic and cubic terms in \(\delta\) must be considered to avoid severe stability problems for particles with energy error. Correcting the chromaticity with only two families of sextupoles we could indeed correct the linear chromaticity but the nonlinear chromaticity may be too severe to permit stable beam operation.

We derive the nonlinear chromaticity from the equation of motion expressed in normalized coordinates and including up to third-order chromatic focusing terms

\[\ddot{w}+\nu_{00}^{2}w=\nu_{00}^{2}\beta^{2}p_{2}(\varphi)\,w=(a\delta+b \delta^{2}+c\delta^{3})w\,, \tag{15.124}\]

where the coefficients \(a,b,c\) are perturbation functions up to third order in \(\delta\) and linear in the amplitude \(w\), and where \(\nu_{00}\) is the unperturbed tune. From (6.95) and (6.96) these perturbations are

\[a = \nu_{00}^{2}\beta^{2}\left[\left(k+2\kappa_{x}^{2}\right)+m\eta_{ x}+\ldots\right], \tag{15.125}\] \[b = \nu_{00}^{2}\beta^{2}\left[-\left(k+2\kappa_{x}^{2}\right)-m\eta _{x}+\ldots\right],\] (15.126) \[c = \nu_{00}^{2}\beta^{2}\left[\left(k+2\kappa_{x}^{2}\right)+m\eta_ {x}+\ldots\right]. \tag{15.127}\]This equation defines nonlinear terms for the chromaticity which have been solved for the quadratic term [7] and for the cubic term [8, 9]. While second and third-order terms become significant in modern circular accelerators, higher-order terms can be recognized by numerical particle tracking but are generally insignificant.

Since the perturbations on the r.h.s. of (15.124) are periodic for a circular accelerator we may Fourier expand the coefficients

\[\begin{split} a(\varphi)&=a_{0}+\sum_{n\neq 0}a_{n} \operatorname{e}^{\operatorname{i}n\varphi},\\ b(\varphi)&=b_{0}+\sum_{n\neq 0}b_{n}\operatorname{e}^{ \operatorname{i}n\varphi},\\ c(\varphi)&=c_{0}+\sum_{n\neq 0}c_{n}\operatorname{e}^{ \operatorname{i}n\varphi}.\end{split} \tag{15.128}\]

From the lowest-order harmonics of the perturbations we get immediately the first approximation of nonlinear chromaticities

\[\nu_{0}^{2}=\nu_{00}^{2}-\delta\left(a_{0}+b_{0}\delta+c_{0}\delta^{2}\right) \tag{15.129}\]

or

\[\nu_{0}^{2}=\nu_{00}^{2}\left(1-\beta^{2}\int_{0}^{2\pi}p_{1}(\varphi)\, \mathrm{d}\varphi\right)\,. \tag{15.130}\]

With this definition we reduce the equation of motion (15.124) to

\[\ddot{w}+\nu_{00}^{2}\,w=\nu_{0}^{2}\beta^{2}\delta\left(\sum_{n>0}2a_{n}\cos n \varphi+\delta\sum_{n>0}2b_{n}\cos n\varphi+\delta^{2}\sum_{n>0}2c_{n}\cos n \varphi\right)w \tag{15.131}\]

The remaining perturbation terms on the r.h.s. look oscillatory and therefore seem not to contribute to an energy dependent tune shift. In higher-order approximation, however, we find indeed resonant terms which do not vanish but contribute to a systematic tune shift. Such higher-order tune shifts cannot be ignored in all cases and therefore an analytical expression for this chromatic tune shift will be derived. To solve the differential equation (15.131), we consider the r.h.s. as a small perturbation with \(\delta\) serving as the smallness parameter. Mathematical methods for a solution have been developed and are based on a power series in \(\delta\). We apply this method to both the cosine and sine like principal solution and try the ansatz

\[C(\varphi)=\sum_{n\geq 0}C_{n}(\varphi)\,\delta^{n}\qquad\text{and}\qquad S( \varphi)=\sum_{n\geq 0}S_{n}(\varphi)\,\delta^{n} \tag{15.132}\]

Concentrating first on the cosine like solution we insert (15.132) into (15.131) and sort for same powers in \(\delta\) noting that each term must vanish separately to make the ansatz valid for all values of \(\delta\). The result is a set of differential equations for the individual solution terms

\[\begin{split}\tilde{\mathcal{C}}_{0}+\nu_{0}^{2}C_{0}& =0\,,\\ \tilde{\mathcal{C}}_{1}+\nu_{0}^{2}C_{1}&=v_{0}^{2} \beta^{2}(\varphi)p_{2}(\varphi)\,C_{0},\\ &\cdots\cdots\\ \tilde{\mathcal{C}}_{n}+\nu_{0}^{2}C_{n}&=v_{0}^{2} \beta^{2}(\varphi)p_{2}(\varphi)\,C_{n-1},\end{split} \tag{15.133}\]

where derivatives \(\tilde{\mathcal{C}}_{i}\) are taken with respect to the phase \(\varphi\), e.g. \(\tilde{\mathcal{C}}_{i}=\partial^{2}C_{i}/\partial\varphi^{2}\). These are defining equations for the functions \(C_{0},C_{1},\cdots C_{n}\) with \(C_{i}=C_{i}(\varphi)\) and each function depending on a lower-order solution. The lowest-order solutions are the principal solutions of the unperturbed motion

\[C_{0}(\varphi)=\cos v_{0}\varphi\qquad\text{and}\qquad S_{0}(\varphi)=\frac{1 }{v_{0}}\sin v_{0}\varphi\,. \tag{15.134}\]

The differential equations (15.133) can be solved with the Green's Function method which we have applied earlier to deal with perturbation terms. All successive solutions can now be derived from the unperturbed solutions through

\[\begin{split} C_{n+1}(\varphi)&=\tfrac{1}{v_{0}} \int_{0}^{\varphi}\beta(\xi)p_{2}(\xi)\sin\left[v_{0}\left(\zeta-\varphi \right)\right]C_{n}(\xi)\,\mathrm{d}\zeta\,,\\ S_{n+1}(\varphi)&=\tfrac{1}{v_{0}}\int_{0}^{\varphi }\beta(\xi)p_{2}(\xi)\sin\left[v_{0}\left(\zeta-\varphi\right)\right]S_{n}(\xi )\,\mathrm{d}\zeta\,.\end{split} \tag{15.135}\]

With the unperturbed solution \(C_{0}\) we get for \(C_{1}\)

\[C_{1}(\varphi)=\frac{1}{v_{0}}\int_{0}^{\varphi}\beta(\xi)p_{2}(\xi)\sin\left[ v_{0}\left(\zeta-\varphi\right)\right]\cos\left(v_{0}\xi\right)\,\mathrm{d} \zeta\,, \tag{15.136}\]

and utilizing this solution \(C_{2}\) becomes

\[C_{2}(\varphi)=\frac{1}{v_{0}^{2}}\int_{0}^{\varphi}\beta(\xi)p _{2}(\xi)\sin\left[v_{0}\left(\zeta-\varphi\right)\right]\cos\left(v_{0}\xi\right) \tag{15.137}\] \[\times\int_{0}^{\xi}\beta(\xi)p_{2}(\xi)\sin\left[v_{0}\left(\xi -\xi\right)\right]\cos\left(v_{0}\xi\right)\,\mathrm{d}\xi\,\mathrm{d}\zeta.\]

Further solutions are derived following this procedure although the formulas get quickly rather elaborate. With the cosine and sine like solutions we can formulate the transformation matrix for the whole ring

\[\mathcal{M}=\left(\begin{array}{cc}C(2\pi)&S(2\pi)\\ \dot{C}(2\pi)&\dot{S}(2\pi)\end{array}\right) \tag{15.138}\]and applying Floquet's theorem, the tune of the circular accelerator can be determined from the trace of the transformation matrix

\[2\cos 2\pi\,v = C(2\pi)+\hat{S}(2\pi)\,, \tag{15.139}\]

where \(\hat{S}=\)d\(S\)/d\(\varphi\). With the ansatz (15.132) this becomes

\[2\cos 2\pi\,v = \sum_{n\geq 0}C_{n}(2\pi)\,\delta^{n}+\sum_{n\geq 0}\hat{S}_{n}(2 \pi)\,\delta^{n}\,, \tag{15.140}\]

Retaining only up to third-order terms in \(\delta\), we finally get after some manipulations with (15.135)

\[\cos 2\pi\,v = \cos 2\pi\,v_{0}-\frac{1}{2v_{0}}\sin 2\pi\,v_{0}\int_{0}^{2\pi}p _{2}(\zeta)\,\mathrm{d}\zeta\] \[\quad+\,\frac{1}{2v_{0}^{2}}\int_{0}^{2\pi}\int_{0}^{\zeta}p_{1} (\zeta)\,p_{2}(\beta)\sin\left[v_{0}\left(\zeta-\beta-2\pi\right)\right]\] \[\qquad\qquad\qquad\qquad\qquad\times\sin\left[v_{0}\left(\beta- \alpha\right)\right]\mathrm{d}\beta\,\mathrm{d}\alpha\] \[\quad+\,\frac{1}{2v_{0}^{3}}\int_{0}^{2\pi}\int_{0}^{\zeta}\int_ {0}^{\xi}p_{2}(\zeta)\,p_{2}(\xi)\,p(\nu)\sin\left[v_{0}\left(\zeta-\gamma-2 \pi\right)\right]\] \[\qquad\qquad\qquad\qquad\times\sin\left[v_{0}\left(\xi-\zeta \right)\right]\sin\left[v_{0}\left(\gamma-\xi\right)\right]\mathrm{d}\gamma \,\mathrm{d}\xi\,\mathrm{d}\zeta\,.\]

These integrals can be evaluated analytically and (15.141) becomes after some fairly lengthy but straightforward manipulations

\[\cos 2\pi\,v = \cos 2\pi\,v_{0}-\delta^{2}\left(\frac{\pi\,\sin 2\pi\,v_{0}}{2v _{0}}\sum_{n>0}\frac{a_{n}^{2}}{n^{2}-4v_{0}^{2}}\right)\] \[\quad-\delta^{3}\left(\frac{\pi\,\sin 2\pi\,v_{0}}{v_{0}}\sum_{n >0}\frac{a_{n}^{2}\,b_{n}^{2}}{n^{2}-4v_{0}^{2}}\right)\] \[\quad-\delta^{3}\left\{\left(\frac{\pi\,\sin 2\pi\,v_{0}}{4v_{0}} \sum_{s>0}\sum_{t>0}\frac{a_{s}\,b_{t}}{t^{2}-4v_{0}^{2}}\right)\right.\] \[\qquad\qquad\times\left.\left[a_{s+t}\frac{1+\frac{4v_{0}^{2}}{t(s +t)}}{(s+t)^{2}-4v_{0}^{2}}+a_{|s-t|}\frac{1-\frac{4v_{0}^{2}}{t(s-t)}}{(s+t)^ {2}-4v_{0}^{2}}\right]\right\}+\mathcal{O}(\delta^{4})\,.\]

This expression defines the chromatic tune shift up to third order. Note that the tune \(v_{0}\) is not the unperturbed tune but already includes the lowest-order approximation of the chromaticity (15.129). The relevant perturbations here are linear in the betatron amplitude and drive therefore half-integer resonances as is obvious from (15.142). The main contribution to the perturbation observed here are from the quadrupole and sextupole terms

\[p_{2}(\varphi)=v_{0}^{2}\beta^{2}\left(k-m\eta_{x}\right)\left(\delta-\delta^{2} +\delta^{3}\dots\right)\,. \tag{15.143}\]

In large storage rings the nonlinear chromaticity becomes quite significant as demonstrated in Fig. 15.11. Here the tune variation with energy in the storage ring PEP is shown both for the case where only two families of sextupoles are used to compensate for the natural chromaticities [8]. Since in this ring an energy acceptance of at least \(\pm 1\,\%\) is required, we conclude from Fig. 15.11 that insufficient stability is available because of the nonlinear chromaticity terms shifting the tunes for off-momentum particles to an integer resonance within the desired energy acceptance.

For circular accelerators or rings with a large natural chromaticity it is important to include in the calculation of the nonlinear chromaticity higher-order terms of the dispersion function \(\eta_{x}\). Following the discussion in Sect. 9.4.1 we set in (15.143)

\[\eta_{x}(\varphi)=\eta_{x0}+\eta_{1}\delta+\eta_{2}\delta^{2}+\dots \tag{15.144}\]

and find the Fourier components \(a_{n}\) and \(b_{n}\) in (15.142) defined by

\[v_{0}^{2}\beta^{2}(k-m\eta_{x0})=\sum_{n\geq 0}2a_{n}\cos n \varphi\,, \tag{15.145}\] \[-v_{0}^{2}\beta^{2}(k-m\eta_{x0}+m\eta_{1}) = \sum_{n\geq 0}2b_{n}\cos n\varphi\,. \tag{15.146}\]

Nonlinear energy terms in the \(\eta\)-function can sometimes become quite significant and must be included to improve the accuracy of analytical expressions for the nonlinear chromaticity. In such cases more sophisticated methods of chromaticity correction are required to control nonlinear chromaticities as well. One procedure is to distribute sextupoles in more than two families while keeping their total strength to retain the desired chromaticity. Using more than two families of sextupoles

Figure 15.11: Variation of the vertical tune with energy in the storage ring PEP if the chromaticities are corrected by only two families of sextupoles

allows us to manipulate the strength of specific harmonics \(a_{n}\) such as to minimize the nonlinear chromaticities. Specifically, we note in (15.142) that the quadratic chromaticity term originates mainly from the resonant term \(\frac{a_{n}^{2}}{n^{2}-4v_{0}^{2}}\). This term can be minimized by a proper distribution of sextupoles suppressing the \(n^{\text{th}}\)-harmonic of the chromaticity function \(v^{2}\beta^{2}(k-m\eta)\). Special computer programs like PATRICIA [8] calculate the contribution of each sextupole to the Fourier coefficients \(a_{n}\) and provide thereby the information required to select optimum sextupole locations and field strength to minimize quadratic and cubic chromaticities.

### Kinematic Perturbation Terms*

The rules of linear beam dynamics allow the design of beam transport systems with virtually any desired beam characteristics. Whether such characteristics actually can be achieved depends greatly on our ability or lack thereof to control the source and magnitude of perturbations. Only the lowest-order perturbation terms were discussed in the realm of linear, paraxial beam dynamics. With the continued sophistication of accelerator design and increased demand on beam quality it becomes more and more important to also consider higher-order magnetic field perturbations as well as kinematic perturbation terms.

The effects of such terms in beam-transport lines, for example, may compromise the integrity of a carefully prepared very low emittance beam for linear colliders or may contribute to nonlinear distortion of the chromaticity in circular accelerators and associated reduced beam stability. Studying nonlinear effects we will not only consider nonlinear fields but also the effects of linear fields errors in higher order, whether it be higher-order perturbation terms or higher-order approximations for the equations of motion. The sources and physical nature of perturbative effects must be understood to determine limits to beam parameters and to design correcting measures.

Perturbations of beam dynamics not only occur when there are magnetic field and alignment errors present. During the derivation of the general equation of motion in Chap. 5 we encountered in addition to general multipole fields a large number of kinematic perturbation terms or higher-order field perturbations which appear even for ideal magnets and alignment. Generally, such terms become significant for small circular accelerators or wherever beams are deflected in high fields generating bending radii of order unity or less. If, in addition, the beam sizes are large the importance of such perturbations is further aggravated. In many cases well-known aberration phenomena from light optics can be recognized.

Of the general equations of motion, we consider terms up to third order for ideal linear upright magnets and get the equation of motion in the horizontal and deflecting plane

\[x^{\prime\prime}+\left(\kappa_{x}^{2}+k\right)x = -\kappa_{x}^{3}x^{2}+2\kappa_{x}k\,x^{2}+\left(\tfrac{1}{2}\kappa_{x }k+\tfrac{1}{2}\kappa_{x}^{\prime\prime}\right)y^{2}\] \[+\tfrac{1}{2}\kappa_{x}\left({x^{\prime}}^{2}-{y^{\prime}}^{2} \right)+\kappa_{x}^{\prime}\left(xx^{\prime}+yy^{\prime}\right)\] \[+\tfrac{1}{12}(-10\kappa_{x}^{2}k+k^{\prime\prime}+\kappa_{x} \kappa_{x}^{\prime\prime}+{x^{\prime}}^{2})\,x^{3}-\left(2\kappa_{x}^{2}+ \tfrac{3}{2}k\right)\,{xxx^{\prime}}^{2}\] \[+\tfrac{1}{4}(+6\kappa_{x}^{2}k+k^{\prime\prime}+5\kappa_{x} \kappa_{x}^{\prime\prime}+{x^{\prime}}^{2})\,xy^{2}\] \[-\kappa_{x}\kappa_{x}^{\prime}\,x^{2}x^{\prime}+k^{\prime}xyy^{ \prime}-\tfrac{1}{2}kxy^{\prime 2}+\tfrac{1}{2}\kappa_{x}^{2}xy^{\prime 2}\] \[+\kappa_{x}\delta-\kappa_{x}\delta^{2}+\kappa_{x}\delta^{3}+ \left(2\kappa_{x}^{2}+k\right)x\,\delta-\kappa_{x}^{\prime}yy^{\prime}\delta\] \[+\tfrac{1}{2}\kappa_{x}({x^{\prime}}^{2}+{y^{\prime}}^{2})\delta +\left(-\tfrac{1}{2}\kappa_{x}k-\tfrac{1}{2}\kappa_{x}^{\prime\prime}\right)y ^{2}\delta\] \[+\left(2\kappa_{x}k+\kappa_{x}^{3}\right)x^{2}\delta-\left(k+2 \kappa_{x}^{2}\right)x\delta^{2}+\mathcal{O}(4)\,.\]

In the nondeflecting or vertical plane the equation of motion is

\[y^{\prime\prime}-ky = +2\kappa_{x}k\,xy-\kappa_{x}^{\prime}\left(x^{\prime}y-xy^{ \prime}\right)+\kappa_{x}x^{\prime}y^{\prime}\] \[-\tfrac{1}{12}(+2\kappa_{x}^{2}k+k^{\prime\prime}+\kappa_{x} \kappa_{x}^{\prime\prime}+\kappa_{x}^{\prime 2})\,y^{3}\] \[-\tfrac{1}{4}(k^{\prime\prime}+\kappa_{x}\kappa_{x}^{\prime\prime }-2\kappa_{x}^{2}k+\kappa_{x}^{\prime 2})\,x^{2}y\] \[+\tfrac{1}{2}k\,yy^{\prime 2}-\kappa_{x}\kappa_{x}^{\prime} \,x^{2}y^{\prime}\] \[-k^{\prime}\,xx^{\prime}y+\tfrac{1}{2}kx^{\prime 2}y-\left(2 \kappa_{x}^{2}+k\right)xx^{\prime}y^{\prime}\] \[-k\,y\delta+\kappa_{x}^{\prime}\,x^{\prime}y\delta-2\kappa_{x}k \,xy\delta+k\,y\delta^{2}+\mathcal{O}(4)\,.\]

It is quite clear from these equations that most perturbations become significant only for large amplitudes and oblique particle trajectories or for very strong magnets. The lowest-order quadrupole perturbations are of third order in the oscillation amplitudes and therefore become significant only for large beam sizes. Second-order perturbations occur only in combined-function magnets and for particle trajectories traversing a quadrupole at large amplitudes or offsets from the axis. Associated with the main fields and perturbation terms are also chromatic variations thereof and wherever beams with a large energy spread must be transported such perturbation terms become important. Specifically, the quadrupole terms \(kx\delta\) and \(ky\delta\) determine the chromatic aberration of the focusing system and play a significant role in the transport of beams with large momentum spread. In most cases of beam dynamics, all except the linear chromatic terms can be neglected.

Evaluating the effect of perturbations on a particle beam, we must carefully select the proper boundary conditions for bending magnets. Only for sector magnets is the field boundary normal to the reference path and occurs therefore at the same location \(z\) independent of the amplitude. Generally, this is not true and we must adjust the integration limits according to the particle oscillation amplitudes \(x\) and \(y\) and actual magnet boundary just as we did in the derivation of linear transformation matrices for rectangular or wedge magnets.

### Perturbation Methods in Beam Dynamics

In this chapter, mathematical procedures have been developed to evaluate the effect of specific perturbations on beam dynamics parameters. It is the nature of perturbations that they are unknown and certain assumptions as to their magnitude and distribution have to be made. Perturbations can be systematic, statistical but periodic or just statistical and all can have a systematic or statistical time dependence.

Systematic perturbations in most cases become known through careful magnetic measurements and evaluation of the environment of the beam line. By construction magnet parameters may be all within statistical tolerances but systematically off the design values. This is commonly the case for the actual magnet length. Such deviations generally are of no consequences since the assumed magnet length in the design of a beam-transport line is arbitrary within limits. After the effective length of any magnet type to be included in a beam line is determined by magnetic measurements, beam optics calculations need to be repeated to reflect the variation in length. Similarly, deviations of the field due to systematic errors in the magnet gap or bore radius can be cancelled by experimental calibration of the fields with respect to the excitation current. Left are then only small statistical errors in the strength and length within each magnet type.

One of the most prominent systematic perturbation is an energy error a particle may have with respect to the ideal energy. We have treated this perturbation in much detail leading to dispersion or \(\eta\)-functions and chromaticities.

Other sources of systematic field errors come from the magnetic field of ion pumps or rf-klystrons, from earth magnetic field, and current carrying cables along the beam line. The latter source can be substantial and requires some care in the choice of the direction the electrical current flows such that the sum of currents in all cables is mostly if not completely compensated. Further sources of systematic field perturbations originate from the vacuum chamber if the permeability of the chamber or welding material is too high, if eddy currents exist in cycling accelerators or due to persistent currents in superconducting materials which are generated just like eddy currents during the turn on procedure. All these effects are basically accessible to measurements and compensatory measures in contrast to statistical perturbations as a result of fabrication tolerances.

#### Periodic Distribution of Statistical Perturbations

Whatever statistical perturbations exist in circular accelerators, we know that these perturbations are periodic, with the ring circumference being the period length. The perturbation can therefore always be expressed by a Fourier series. The equation of motion in the presence of, for example, dipole field errors is in normalized coordinates

\[\ddot{w}+v_{0}^{2}w=-v_{0}^{2}\beta^{3/2}\Delta\kappa\,. \tag{15.149}\]

The dipole perturbation \(\beta^{3/2}\Delta\kappa\) is periodic and can be expressed by the Fourier series

\[\beta^{3/2}\,\Delta\kappa=\sum_{n}F_{n}\,\mathrm{e}^{\mathrm{i}n\varphi}\,, \tag{15.150}\]

where \(v_{0}\varphi\) is the betatron phase and the Fourier harmonics \(F_{n}\) are given by

\[F_{n}=\frac{1}{2\pi}\oint\left[\sqrt{\beta^{1/2}(\xi)}\Delta\kappa(\xi)\right] \mathrm{e}^{-\mathrm{i}n\varphi(\xi)}\,\mathrm{d}\xi\,. \tag{15.151}\]

The location of the errors is not known and we may therefore only calculate the expectation value for the perturbation by multiplying (15.151) with its complex conjugate. In doing so, we note that each localized dipole perturbation deflects the beam by an angle \(\theta\) and replace therefore the integral in (15.151) by a sum over all perturbations. With \(\int\Delta\kappa\mathrm{d}\xi\approx\theta\) we get for \(F_{n}F_{n}^{\star}=|F_{n}|^{2}\)

\[|F_{n}|^{2}=\frac{1}{4\pi^{2}}\left[\sum_{k}\beta_{k}\theta_{k}^{2}+\sum_{k \neq j}\sqrt{\beta_{k}\beta_{j}}\theta_{k}\theta_{j}\,\mathrm{e}^{-\mathrm{i} n(\varphi_{k}-\varphi_{j})}\right]\,, \tag{15.152}\]

where \(\beta_{k}\) is the betatron function at the location of the dipole perturbation. The second sum in (15.152) vanishes in general, since the phases for the perturbations are randomly distributed.

For large circular accelerators composed of a regular lattice unit like FODO cells we may proceed further in the derivation of the effects of perturbations letting us determine the field and alignment tolerances of magnets. For simplicity, we assume that the lattice magnets are the source of dipole perturbations and that the betatron functions are the same at all magnets. Equation (15.152) then becomes

\[|F_{n}|^{2}=\frac{1}{4\pi^{2}}N_{m}\beta_{m}\sigma_{\theta}^{2}\,, \tag{15.153}\]

where \(\sigma_{\theta}\) is the expectation value for the statistical deflection angle due to dipole perturbations. In a little more sophisticated approach, we would separate all magnetsinto groups with the same strength and betatron function and (15.153) becomes

\[|F_{n}|^{2} = \frac{1}{4\pi^{2}}\sum_{m}N_{m}\beta_{m}\sigma_{\theta,m}^{2}\,, \tag{15.154}\]

where the sum is taken over all groups of similar perturbations and \(N_{m}\) is the number of perturbations within the group \(m\). In a pure FODO lattice, for example, obvious groups would be all QF's, all QD's and all bending magnets. From now on we will, however, not distinguish between such groups anymore to simplify the discussion.

Periodic dipole perturbations cause a periodic orbit distortion which is from (15.149)

\[w(\varphi)=-\sum_{n}\frac{\nu_{0}^{2}\,F_{n}}{(\nu_{0}^{2}-n^{2})}\,\mathrm{e }^{\mathrm{i}n\varphi}\,. \tag{15.155}\]

The expectation value for the orbit distortion is obtained by multiplying (15.155) with it's complex conjugate and we get with \(w(\varphi)=u(z)/\sqrt{\beta\,\left(z\right)}\)

\[u\,u^{\ast} = \beta(z)\nu^{4}|F_{n}|^{2}\sum_{n=-\infty}^{+\infty}\frac{\mathrm{ e}^{\mathrm{i}n\varphi}}{(\nu^{2}-n^{2})}\sum_{m=-\infty}^{+\infty}\frac{ \mathrm{e}^{-im\varphi}}{(\nu^{2}-m^{2})}\,. \tag{15.156}\]

The sums can be replaced by \(-\frac{\pi\cos v\,(\pi-\varphi)}{v\,\sin v\pi}\) and we get finally for the expectation value of the orbit distortion \(\sigma_{u}\) at locations with a betatron function \(\beta\)

\[\sigma_{u}^{2} = \beta\frac{N\bar{\beta}\sigma_{\theta}^{2}}{8\sin^{2}\pi\,v}\,, \tag{15.157}\]

where \(\bar{\beta}\) is the average betatron function at the locations of perturbations. This result is in full agreement with the result (15.39) for misaligned quadrupoles, where \(\sigma_{\theta}=\sigma_{u}/f\), \(\sigma_{u}\) the statistical quadrupole misalignment and \(f\) the focal length of the quadrupole.

This procedure is not restricted to dipole errors only but can be applied to most any perturbation occurring in a circular accelerator. For this we determine which quantity we would like to investigate, be it the tunes, the chromaticity, perturbation of the dispersion functions, or any other beam parameter. Variation of expressions for such quantities due to variations of magnet parameters and squaring such variation we get the perturbation of the quantity under investigation. Generally, perturbation terms of order \(n\) in normalized coordinates are expressed by

\[P_{n}(z) = \nu_{0}^{2}\beta^{3/2}\beta^{n/2}p_{n}\left(z\right)\,w^{n-1}. \tag{15.158}\]

Because the perturbations are assumed to be small, we may replace the oscillation amplitudes \(w^{n}\) in the perturbation term by their principle unperturbed solutions.

Considering that the beam position \(w\) is a composite of, for example, betatron oscillation \(w_{\beta}\), orbit distortion \(w_{\rm c}\), and energy error \(w_{\eta}\) we set

\[w=w_{\beta}+w_{\rm c}+w_{\eta} \tag{15.159}\]

and note that any higher-order perturbation contributes to the orbit, the eta-function, the tunes, betatron functions, and other beam parameters. Orbit distortions in sextupoles of strength \(m\), for example, produce the perturbations

\[P_{2}(z)=\tfrac{1}{2}v_{0}^{2}\theta^{5/2}mw^{2} \tag{15.160}\]

which for \(w_{\eta}=0\) can be decomposed into three components

\[\begin{array}{rcl}P_{20}(z)&=&\tfrac{1}{2}v_{0}^{2}\beta^{5/2}mw_{\rm c}^{2} \,,\\ P_{21}(z)&=&v_{0}^{2}\theta^{5/2}mw_{\rm c}w_{\beta}\,,\\ P_{22}(z)&=&\tfrac{1}{2}v_{0}^{2}\beta^{5/2}mw_{\beta}^{2}\,.\end{array} \tag{15.161}\]

The perturbation \(P_{20}\) causes an orbit distortion and since the perturbations are randomly distributed the contribution to the orbit according to (15.157) is

\[\sigma_{u}^{2}=\beta_{u}\frac{N_{\rm s}\,\beta_{us}\,\sigma_{\theta}^{2}}{8 \sin^{2}\pi\,v_{u}}\,, \tag{15.162}\]

where \(N_{\rm s}\) is the number of sextupoles, \(\beta_{us}\) the value of the betatron function and \(\sigma_{\rm c}\) the rms orbit distortion at the sextupoles, \(\sigma_{\theta}=\tfrac{1}{2}m\sigma_{\rm c}^{2}\ell_{\rm s}\) and \(\ell_{\rm s}\) is the effective sextupole deflection and length, respectively. In cases of very strong sextupoles iteration methods must be applied since the orbit perturbation depends on the orbit. Similarly, we could have set \(w_{\rm c}=0\) to calculate the perturbation of the \(\eta\)-function due to sextupole magnets.

The linear perturbation \(P_{21}\) in (15.161) causes a statistical tune shift and a perturbation of the betatron function. Following the derivation of tune shifts in Sect. 15.3, we find the expectation value for the tune shift to be

\[\langle\delta^{2}v\rangle=\frac{1}{16\pi^{2}}\sum_{k}\beta_{k}\,m_{k}\ell_{k} \,\langle u_{0}^{2}\rangle_{k}\,, \tag{15.163}\]

where \(\langle u_{0}^{2}\rangle\) is the random misalignment of the sextupole magnets or random orbit distortions in the sextupoles.

We find the interesting result, that sextupoles contribute to a tune error only if there is a finite orbit distortion or misalignment \(u_{0}\), while a finite betatron oscillation amplitude of a particle in the same sextupoles does not contribute to a tune shift. Similarly, we may use the effects of systematic errors to get expressions for the probable variation of the betatron function (15.91) due to gradient errors from misaligned sextupoles.

In the approximation of small perturbations, we are able to determine the expectation value for the effect of statistical errors on a particular beam parameter or lattice function. This formalism is used particularly when we are interested to define tolerances for magnetic field quality and magnet alignment by calculating backwards from the allowable perturbation of beam parameters to the magnitude of the errors. Some specific statistical effects will be discussed in subsequent sections.

#### Periodic Perturbations in Circular Accelerators

Alignment and field errors in circular accelerators not only cause a distortion of the orbit but also a perturbation of the \(\eta\)-functions. Although these perturbations occur in both the horizontal and vertical plane, we will discuss only the effect in the vertical plane. While the derivations are the same for both planes the errors contribute only to a small perturbation of the already existing horizontal \(\eta\)-function while the ideal vertical \(\eta\)-function vanishes, and therefore the perturbation can contribute a large variation of beam parameters. This is specifically true for electron storage ring where the vertical beam emittance is very small and a finite vertical \(\eta\)-function may increase this emittance considerably.

Similar to (15.11) we use the equation of motion

\[y^{\prime\prime}-ky=+\Delta\kappa_{y}-\Delta\kappa_{y}\delta-ky\delta+mxy \tag{15.164}\]

with the decomposition

\[y=y_{\rm c}+v_{y}\,\delta \tag{15.165}\]

and get in normalized coordinates \(\tilde{y}=y/\sqrt{\beta_{y}}\), while ignoring the betatron motion, the differential equations for the orbit distortion \(\tilde{y}_{\rm c}\)

\[\tilde{\tilde{y}}_{\rm c}+v_{y}^{2}\tilde{y}_{\rm c}=+v_{y}^{2}\beta_{y}^{3/2 }(\Delta\kappa_{y}+m\kappa_{\rm c}y_{\rm c}) \tag{15.166}\]

and for the perturbation of the \(\eta\)-function \(\tilde{v}_{y}=v_{y}/\sqrt{\beta_{y}}\)

\[\tilde{\tilde{v}}_{y}+v_{y}^{2}\tilde{v}_{y}=-v_{y}^{2}\beta_{y}^{3/2}\Delta \kappa_{y}+v_{y}^{2}\beta_{y}^{2}(k-m\eta_{x})\tilde{y}_{\rm c}\,. \tag{15.167}\]

First, we note in a linear lattice where \(m=0\) that the differential equations for both the closed orbit distortion and the \(\eta\)-function perturbation are the same except for a sign in the perturbation. Therefore, in analogy to (15.157)

\[\langle v_{y}^{2}(z)\rangle=\frac{\beta(z)\bar{\beta}_{\theta}}{8\sin^{2}\pi v }\sum_{i}\sigma_{i\theta}^{2}\,. \tag{15.168}\]The perturbation of the \(\eta\)-function becomes more complicated in strong focusing lattices, where the chromaticity is corrected by sextupole fields. In this case, we note that all perturbation terms on the r.h.s. are periodic and we express them in Fourier series

\[v^{2}\beta_{y}^{3/2}\,\Delta\kappa_{y}=\sum_{n=-\infty}^{n=+\infty}F_{n}\,{\rm e }^{{\rm i}n\varphi} \tag{15.169}\]

with

\[F_{n}=\frac{v^{2}}{2\pi}\int_{0}^{2\pi}\beta^{3/2}\Delta\kappa_{y}\,{\rm e}^{- {\rm i}n\tau}\,{\rm d}\tau \tag{15.170}\]

and

\[v^{2}\beta_{y}^{2}(k-m\eta_{x})=\sum_{n=-\infty}^{n=+\infty}A_{n}\,{\rm e}^{{ \rm i}n\varphi} \tag{15.171}\]

with

\[A_{n}=\frac{v^{2}}{2\pi}\int_{0}^{2\pi}\beta^{2}(k-m\eta_{x})\,{\rm e}^{-{\rm i }n\tau}\,{\rm d}\tau. \tag{15.172}\]

We also make use of the periodicity of the perturbation of the \(\eta\)-function and set

\[\tilde{v}_{y}=\sum_{n=-\infty}^{n=+\infty}E_{n}\,{\rm e}^{{\rm i}n\varphi}. \tag{15.173}\]

Inserting (15.169)-(15.173) into (15.167) we get with the periodic solution of the closed orbit

\[\tilde{y}_{\rm c}(\varphi)=\sum_{n}\frac{F_{n}}{v^{2}-n^{2}}\,\,{\rm e}^{{\rm i }n\varphi} \tag{15.174}\]

\[\sum_{n}[(v^{2}-n^{2})E_{n}+F_{n}]\,{\rm e}^{{\rm i}n\varphi}-\sum_{m,r}\frac {A_{m}F_{r}}{v^{2}-n^{2}}\,{\rm e}^{{\rm i}(m+r)\varphi}=0. \tag{15.175}\]

Noting that this equation must be true for all phases \(\varphi\) all terms with the same exponential factor must vanish separately and we may solve for the harmonics of the \(\eta\)-function

\[E_{n}=-\,\frac{F_{n}}{v^{2}-n^{2}}\,+\sum_{r}\frac{A_{n-r}\,F_{r}}{(v^{2}-n^{2 })\,(v^{2}-r^{2})}. \tag{15.176}\]The perturbation of the \(\eta\)-function is therefore

\[\tilde{v}_{y}(\varphi)=-\tilde{y}_{c}(\varphi)+\sum_{n,r}\frac{A_{n-r}\,F_{r}}{( \nu^{2}-n^{2})\,(\nu^{2}-r^{2})}\,\e^{\mathrm{i}n\varphi}\,, \tag{15.177}\]

We extract from the double sum on the r.h.s. of (15.177) all terms with \(n=r\) and get from those terms the expression \(A_{0}\sum_{n}\frac{F_{n}}{(\nu^{2}-n^{2})^{2}}\e^{\mathrm{i}n\varphi}\). The coefficient \(A_{0}\), however, is just the natural chromaticity \(A_{0}=2\xi_{0y}/\nu_{y}\) and the perturbation of the \(\eta\)-function is from (15.177)

\[\tilde{v}_{y}(\varphi)=-\tilde{y}_{c}(\varphi)+\frac{2\xi_{y}^{\,\,\,\,\,}}{ \nu_{y}}\sum_{n}\frac{F_{n}\e^{\mathrm{i}n\varphi}}{(\nu^{2}-n^{2})^{2}}+\sum _{n\neq r}\frac{A_{n-r}\,F_{r}\e^{\mathrm{i}n\varphi}}{(\nu^{2}-n^{2})(\nu^{2 }-r^{2})}\,. \tag{15.178}\]

By correcting the orbit distortion and compensating the chromaticity, we are able to greatly reduce the perturbation of the vertical \(\eta\)-function. All terms with \(r=0\) vanish for a truly random distribution of misalignment errors since \(F_{0}=0\). Taking the quadrupole lattice as fixed we find the remaining terms to depend mainly on the distribution of the orbit correction \(F_{r}\) and sextupole positions \(A_{i}\). For any given sextupole distribution the orbit correction must be done such as to eliminate as much as possible all harmonics of the orbit in the vicinity of the tunes \(r\not\propto\nu_{y}\) and to center the corrected orbit such that \(F_{0}=0\).

Furthermore, we note that some care in the distribution of the sextupoles must be exercised. While this distribution is irrelevant for the mere correction of the natural chromaticities, higher harmonics of the chromaticity function must be held under control as well. The remaining double sum is generally rather small since the resonance terms have been eliminated and either \(v-n\) or \(v-r\) is large. However, in very large rings or very strong focusing rings this contribution to the perturbation of the \(\eta\)-function may still be significant.

#### Statistical Methods to Evaluate Perturbations

In an open beam-transport line the perturbation effect at a particular point depends only on the upstream perturbations. Since perturbations cannot change the position but only the slope of particle trajectories, we merely transform the random kick angle \(\theta_{k}\) from the location of the perturbation to the observation point. Adding all perturbations upstream of the observation point we get with \(\psi=\psi(z)\)

\[\begin{array}{l}u(z){=}\,\,\sqrt{\beta(z)}\sum\limits_{\begin{subarray}{c}k \\ \psi_{k}\sim\psi(z)\end{subarray}}\sqrt{\beta_{k}}\sin(\psi-\psi_{k})\,\theta_{ k}\,,\\ u^{\prime}(z){=}\,\,\frac{1}{\sqrt{\beta(z)}}\sum\limits_{\begin{subarray}{c}k\\ \psi_{k}\sim\psi(z)\end{subarray}}\sqrt{\beta_{k}}\cos(\psi-\psi_{k})\,\theta_{ k}\,.\end{array} \tag{15.179}\]The expectation value for the position of the beam center at the observation point becomes from the first equation (15.179) noting the complete independence of the perturbations

\[\sigma_{u}(z)=\sqrt{\beta(z)}\tfrac{1}{2}N\sqrt{\langle\beta\rangle}\sigma_{ \theta}\,. \tag{15.180}\]

Random variations of the beam position are customarily corrected by special steering magnets if such correction is required at all. In long beam-transport systems like those required in linear colliders a mere correction of the beam position at the collision point, for example, may not be acceptable. Specifically, nonlinear perturbations lead to an incoherent increase of the beam size which can greatly reduce the usefulness of the colliding-beam system. In the next subsection we will therefore discuss general perturbations in beam-transport lines and their effect on the beam cross section.

### 15.7 Control of Beam Size in Transport Lines

For the transport of beams with a very small beam size or beam emittance like in a linear collider facilities we are specially concerned about the impact of any kind of perturbation on the integrity of a small beam emittance. Errors can disturb the beam size in many ways. We have discussed already the effect of dipole errors on the dispersion. The distortion of the dispersion causes an increase in the beam size due to the energy spread in the beam. Quadrupole field errors affect the value of betatron functions and therefore the beam size. Vertical orbit distortions in sextupoles give rise to vertical--horizontal coupling. In this section we will try to evaluate these effects on the beam size.

We use the equations of motion (6.95), (6.96) up to second order in \(x,y\) and \(\delta\), and assume the curvature to be small of the order or less than \((x,y,\delta)\). This is a proper assumption for high-energy beam transport lines like in linear colliders. For lower-energy beam lines very often this assumption is still correct and where a better approximation is needed more perturbation terms must be considered. For the horizontal plane we get

\[x^{\prime\prime}+(\kappa_{x}^{2}+k)\,x =\kappa_{x}\delta-\kappa_{x}\delta^{2}-\underline{k}y-\tfrac{1}{ 2}m(x^{2}-y^{2})(1-\delta) \tag{15.181}\] \[\quad-\Delta\kappa_{x}(1-\delta)+kx\delta+\Delta kx(1-\delta)+O(3)\]

and for the vertical plane

\[y^{\prime\prime}-k\,y =\kappa_{y}\delta-\kappa_{y}\delta^{2}-\underline{k}x+mxy(1- \delta)-ky\delta \tag{15.182}\] \[\quad-\Delta\kappa_{y}(1-\delta)-\Delta k\,y(1-\delta)+\mathcal{O }(3)\]In these equations rotated magnets \((\kappa_{y},\underline{k},\underline{m})\) are included as small quantities because rotational alignment errors of upright magnets cause rotated error fields although no rotated magnets per se are used in the beam line. For the solution of (15.181) and (15.182) we try the ansatz

\[\begin{split} x=& x_{\beta}\,+\,x_{\rm c}\,+\,\eta_{x} \delta\,+\,v_{x}\delta\,+\,w_{x}\delta^{2}\,,\\ y=&\ y_{\beta}\,+\,y_{\rm c}\,+\,\eta_{y}\delta\,+\,v_{ y}\delta\,+\,w_{y}\delta^{2}\,.\end{split} \tag{15.183}\]

Here we define \((x_{\beta},y_{\beta})\) as the betatron oscillations, \((x_{\rm c},y_{\rm c})\) the orbit distortions, \((\eta_{x},\eta_{y})\) the dispersion function, \((v_{x},\,v_{y})\) the perturbations of the dispersion functions due to magnetic field errors, and \((w_{x},w_{y})\) the first-order chromatic perturbation of the dispersion functions (\(\eta_{\rm tot}\,=\,\eta+v+w\delta+\ldots\)). This ansatz leads to the following differential equations in the horizontal plane where we assume the bending radii to be large and \(\kappa_{x}\), \(\kappa_{y}\) are therefore treated as small quantities

\[\begin{split} x_{\beta}^{\prime\prime}+kx_{\beta}& =-\underline{k}y_{\beta}-\tfrac{1}{2}m(x_{\beta}^{2}-y_{\beta}^{2})-m(x_{ \beta}x_{\rm c}-y_{\beta}y_{\rm c})\,+\,\Delta kx_{\beta}\,,\qquad\qquad\qquad \qquad\qquad\text{a)}\\ x_{\rm c}^{\prime\prime}+kx_{\rm c}&=-\Delta\kappa_ {x}+\Delta kx_{\rm c}-\underline{k}y_{\rm c}-\tfrac{1}{2}m(x_{\rm c}^{2}-y_{ \rm c}^{2})\,,\qquad\qquad\qquad\qquad\qquad\text{b)}\\ \eta_{x}^{\prime\prime}+k\eta_{x}&=+\,\kappa_{x}\,, \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\text{c)}\\ v_{x}^{\prime\prime}+kv_{x}&=-\,\underline{k}v_{y}-m \,(x_{\beta}+x_{\rm c})(\eta_{x}+v_{x})+m\,(y_{\beta}+y_{\rm c})(\eta_{y}+v_{ y})\qquad\qquad\qquad\text{d)}\\ &+\Delta k(x_{\rm c}+x_{\beta})+\Delta kx_{\rm c}+\Delta k\,( \eta_{x}+v_{x})\\ &+kx_{\beta}\,+\,kx_{\rm c}+\,\tfrac{1}{2}m(x_{\rm c}^{2}-y_{ \rm c}^{2})+\underline{k}y_{\rm c}\,,\\ w_{x}^{\prime\prime}+k\,w_{x}&=-\kappa_{x}-\tfrac{1}{2}m( \eta_{x}^{2}+2\eta_{x}v_{x}-2\eta_{y}v_{y}-v_{y}^{2})+k(\eta_{x}+v_{x})\\ &=+m\,(x_{\rm c}\eta_{x}+x_{\rm c}v_{x}-y_{\rm c}\eta_{y}-y_{\rm c }v_{y})+(\eta_{x}+v_{x})x_{\beta}-v_{y}y_{\beta}\,.\end{split} \tag{15.184}\]

Similarly, we get for the vertical plane

\[\begin{split} y_{\beta}^{\prime\prime}-ky_{\beta}& =-\underline{k}x_{\beta}\,+\,m\,x_{\beta}y_{\beta}-\Delta ky_{ \beta}\,+\,m(x_{\rm c}y_{\beta}+x_{\beta}y_{\rm c})\,,\qquad\qquad\qquad\qquad \text{a)}\\ y_{\rm c}^{\prime\prime}-ky_{\rm c}&=-\Delta k_{y}- \Delta ky_{\rm c}-\underline{k}x_{\rm c}+mx_{\rm c}y_{\rm c}\,,\qquad\qquad \qquad\qquad\qquad\qquad\text{b)}\\ \eta_{y}^{\prime\prime}-k\eta_{y}&=+\kappa_{y}\,, \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\text{c)}\\ v_{y}^{\prime\prime}-kv_{y}&=+\,\Delta\kappa_{y}- \underline{k}(\eta_{y}+v_{y})+m(x_{\beta}+x_{\rm c})(\eta_{y}+v_{y})\qquad \qquad\qquad\qquad\text{d)}\\ &+m\,(\eta_{x}+v_{x})(y_{\beta}+y_{\rm c})+\,\Delta k(y_{\beta}+y_{ \rm c})+\,\underline{k}\,x_{\rm c}-mx_{\rm c}y_{\rm c}\\ &-k(y_{\beta}+y_{\rm c})-\Delta k(\eta_{y}+v_{y})\,,\qquad\qquad \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\text{e)}\\ w_{y}^{\prime\prime}-kw_{y}&=-\kappa_{y}+k(\eta_{y}+v_{y}) \\ &=+m\,(\eta_{x}\eta_{y}+\eta_{x}v_{y}+v_{x}\eta_{y}+v_{x}v_{y})\,. \end{split} \tag{15.185}\]

The solution of all these differential equations is, if not already known, straightforward. We consider every perturbation to be localized as a thin element causing just a kick which propagates along the beam line. If \(\beta_{j}\) is the betatron function at the observation point and \(\beta_{i}\) that at the point of the perturbation \(p_{ni}\) the solutionsof (15.184), (15.185) have the form

\[u_{j}=\sqrt{\beta_{j}}\sum_{i}\sqrt{\beta_{i}}\sin\psi_{ji}\int p_{ni}\,\mathrm{d }z\,. \tag{15.186}\]

The kick due to the perturbation is \(\theta_{i}=\int p_{ni}\,\mathrm{d}z\), where the integral is taken along the perturbation assumed to be short. To simplify the equations to follow we define the length \(\ell_{i}=\theta_{i}/\langle p_{ni}\rangle\). Since most errors derive from real magnets, this length is identical with that of the magnet causing the perturbation and \(\psi_{ji}=\psi_{j}-\psi_{i}\) is the betatron phase between perturbation and observation point. A closer look at (15.184) and (15.185) shows that many perturbations depend on the solution itself requiring an iterative solution process. Here we will, however, concentrate only an the first iteration.

Ignoring coupling terms we have in (15.184) two types of perturbations, statistically distributed focusing errors \(\Delta k\) and geometric aberration effects due to sextupoles. We assume here that the beam line is chromatically corrected by the use of self-correcting achromats in which case the term \(\frac{1}{2}m(x_{\beta}^{2}-y_{\beta}^{2})\) is self-canceling. The expectation value for the betatron oscillation amplitude due to errors is then

\[x_{\beta}^{2}(z)=\beta_{x}(z)\sum_{i}\beta_{xi}\left\langle(p_{ni}\ell_{i})^{ 2}\right\rangle\sin^{2}\psi_{ji} \tag{15.187}\]

or

\[\langle x_{\beta}^{2}(z)\rangle=\beta_{x}(z)\overline{\beta}_{x}(\underline{ k}^{2}y_{\beta}^{2}+\Delta k^{2}x_{\beta}^{2}+m^{2}(x_{\beta}^{2}x_{\rm c}^{2}+y_{ \beta}^{2}y_{\rm c}^{2}))\frac{1}{2}N_{\rm M}\ell^{2}\,, \tag{15.188}\]

where \(\overline{\beta}_{x}\) is the average value of the betatron functions at the errors, \(N_{\rm M}\) the number of perturbed magnets and \(\ell\) the magnet length. With \(\underline{k}=k\alpha\), where \(\alpha\) is the rotational error, we get

\[\langle x_{\beta}^{2}(z)\rangle = \frac{1}{2}\beta_{x}(z)\overline{\beta}_{x}N_{\rm M}k^{2}\ell^{2}\] \[\times\left[\sigma_{\alpha}^{2}\sigma_{y}^{2}+\sigma_{k}^{2} \sigma_{x}^{2}+\frac{m^{2}}{k^{2}}(\sigma_{x}^{2}\sigma_{y{\rm c}}^{2}-\sigma _{y}^{2}\sigma_{y{\rm c}}^{2})\right].\]

We have assumed the errors to have a Gaussian distribution with standard width \(\sigma\). Therefore, \(\sigma_{\alpha}^{2}=\langle\alpha^{2}\rangle\), \(\sigma_{k}^{2}=\langle(\Delta k/k)^{2}\rangle,\sigma_{xc}=\langle x_{\rm c}^{2}\rangle\), etc., and \(\sigma_{y},\sigma_{x}\) the standard beam size for the Gaussian particle distribution. Since \(\langle x_{\beta}^{2}(z)\rangle/\beta(z)=\Delta\epsilon_{x}\) is the increase in beam emittance and \(\sigma_{x}^{2}=\epsilon_{x}\overline{\beta}_{x},\sigma_{y}^{2}=\epsilon_{y} \overline{\beta}_{y}\) we get for a round beam for which \(\epsilon_{x}=\epsilon_{y}\) and the average values for the betatron functions are the same (\(\overline{\beta}_{x}=\overline{\beta}_{y}\))

\[\frac{\Delta\epsilon_{x}}{\epsilon_{x}}=\frac{1}{2}\overline{\beta}^{2}N_{\rm M }k^{2}\ell^{2}\left[\sigma_{\alpha}^{2}+\sigma_{k}^{2}+\frac{m^{2}}{k^{2}}( \sigma_{xc}^{2}+\sigma_{y{\rm c}}^{2})\right]\,. \tag{15.190}\]To keep the perturbation of the beam small the alignment \(\sigma_{\alpha}\) and magnet field quality \(\sigma_{k}\) must be good and the focusing weak which, however, for other reasons is not desirable. For a chromatically corrected beam line we have \(k/m=\bar{\eta}_{x}\), which can be used in (15.190). The perturbation of the vertical beam emittance follows exactly the same results because we used a round beam.

The expectation value for the shift of the beam path is derived from (15.184a), (15.185b) with (15.186) in a similar way as for the betatron oscillations

\[\langle x_{\rm c}^{2}(z)\rangle\,=\,{{1\over 2}}\beta_{x}(z)\overline{\beta}_{x}N_{\rm M} \ell^{2}\left[\langle\Delta\kappa_{x}^{2}\rangle\,+\,k^{2}\sigma_{k}^{2} \langle x_{\rm c}^{2}\rangle\,+\,k^{2}\sigma_{\alpha}^{2}\langle y_{\rm c}^{2} \rangle\right]\,. \tag{15.191}\]

This expression for the path distortion, however, is not to be used to calculate the perturbation of the dispersion. In any properly operated beam line one expects this path distortion to be corrected leading to a smaller residual value depending on the correction scheme applied and the resolution of the monitors. With some effort the path can be corrected to better than 1mm rms which should be used to evaluate path dependent perturbation terms in (15.184), (15.185). In the vertical plane we get

\[\langle y_{\rm c}^{2}(z)\rangle\,=\,{{1\over 2}}\beta_{y}(z) \overline{\beta}_{y}N_{\rm M}\ell^{2}\left[\langle\Delta\kappa_{y}^{2}\rangle \,+\,k^{2}\sigma_{k}^{2}\left\langle y_{\rm c}^{2}\right\rangle\,+\,k^{2} \sigma_{\alpha}^{2}\langle x_{\rm c}^{2}\rangle\right]\,. \tag{15.192}\]

The perturbation of the dispersion is with (15.184d) and (15.186)

\[v_{x}(z)\,=\,-x_{\rm c}(z)\,+\,\sqrt{\beta_{x}(z)}\sum_{i}\sqrt{\beta_{xi}}p_ {xi}\ell_{i}\sin\psi_{xi}\,. \tag{15.193}\]

In (15.184d) we note the appearance of the same perturbation terms as for the path distortion apart from the sign and we therefore separate that solution in (15.193). The perturbations left are then

\[p_{xi}=(k-m\eta_{x})(x_{\beta}+x_{\rm c})\,+\,m(y_{\beta}+y_{\rm c})\eta_{y} \,+\,\Delta k\eta_{x}\,+\cdots \tag{15.194}\]

In this derivation the betatron phase \(\psi_{ji}\) does not depend on the energy since the chromaticity is corrected. Without this assumption, we would get another contribution to \(v_{x}\) from the beam-path distortion. We also note that the chromaticity factor \((k-m\eta_{x})\) can to first order be set to zero for chromatically corrected beam lines. The expectation value for the distortion of the dispersion is finally given by

\[\langle v_{x}^{2}(z)\rangle\,=\,x_{\rm c}^{2}(z)\,+\,{{1\over 2}} \beta_{x}(z)\overline{\beta}_{x}N_{\rm M}\ell^{2}\left[\langle\Delta k^{2} \rangle\overline{\eta}_{x}^{2}\,+\,m^{2}\overline{\eta}_{y}^{2}\langle y_{ \beta}^{2}\rangle\,+\,m^{2}\overline{\eta}_{y}^{2}\langle y_{\rm c}^{2} \rangle\right] \tag{15.195}\]

or with some manipulation

\[\langle v_{x}^{2}(z)\rangle\,=\,\langle x_{\rm c}^{2}(z)\rangle\,+\,{{1 \over 2}}\beta_{x}(z)\overline{\beta}_{x}N_{M}k^{2}\ell^{2}\left[\sigma_{k}^{2}\, \,\overline{\eta}_{x}^{2}\,+\left(\overline{\overline{\eta}_{y}}\over \overline{\overline{\eta}_{x}}\right)^{2}\,(\overline{\beta}_{y}\epsilon_{y} \,+\,\sigma_{y\rm c}^{2})\right] \tag{15.196}\]The perturbation of the dispersion function is mainly caused by quadrupole field errors while the second term vanishes for a plane beam line where \(\eta_{y}=0\). In principle, the perturbation can be corrected if a specific path distortion is introduced which would compensate the perturbation at the point \(z\) as can be seen from (15.193).

In the vertical plane we proceed just the same and get instead of (15.193)

\[v_{y}(z)=-y_{\rm c}(z)+\sqrt{\beta_{y}(z)}\sum_{i}\sqrt{\beta_{yi}}\left(p_{yi} \ell_{i}\right)\sin\psi_{y\bar{\imath}} \tag{15.197}\]

with

\[p_{yi}=-(k-m\eta_{x})(y_{\beta}+y_{\rm c})+mv_{x}(y_{\beta}+y_{ \rm c}) \tag{15.198}\] \[\qquad\qquad+m(\eta_{y}+v_{y})(x_{\beta}+x_{\rm c})-\Delta k(\eta _{y}+v_{y})-\underline{k}(\eta_{y}+v_{y})\,.\]

Again due to chromaticity correction we have \((k-m\eta_{x})\approx 0\) and get for the expectation value of \(\left\langle v_{y}^{2}\right\rangle\) in first approximation with \(v_{y}\equiv 0\) in (15.198) and the average values \(\bar{\eta}_{x}\) and \(\bar{v}_{x}\)

\[\left\langle v_{y}^{2}(z)\right\rangle=\left\langle v_{\rm c}^{2 }(z)\right\rangle+\tfrac{1}{2}\beta_{y}(z)\bar{\beta}_{y}N_{\rm M}k^{2}\ell^{2} \tag{15.199}\] \[\qquad\qquad\qquad\times\left[\left(\sigma_{k}^{2}+\sigma_{ \alpha}^{2}+\frac{\left\langle x_{\rm c}^{2}\right\rangle}{\bar{\eta}_{x}^{2} }\right)\bar{\eta}_{y}^{2}+\frac{\bar{v}_{x}^{2}}{\bar{\eta}_{x}^{2}}\left( \beta_{y}\epsilon_{y}+\left\langle y_{\rm c}^{2}\right\rangle\right)\right]\,.\]

For a plane beam line where \(\eta_{y}\equiv 0\), we clearly need to go through a further iteration to include the perturbation of the dispersion which is large compared to \(\eta_{y}=0\). In this approximation, we also set \(y_{\rm c}(z)=0\) and

\[\left\langle v_{y}^{2}(z)\right\rangle=\frac{\bar{v}_{x}^{2}}{\bar{\eta}_{x}^{ 2}}\left(\beta_{y}\epsilon_{y}+\left\langle y_{\rm c}^{2}\right\rangle\right)\,. \tag{15.200}\]

Using this in a second iteration gives finally for the variation of the vertical dispersion function due to field and alignment errors

\[\left\langle v_{y}^{2}(z)\right\rangle=\left\langle y_{\rm c}^{2 }(z)\right\rangle+\tfrac{1}{2}\beta_{y}(z)\bar{\beta}_{y}N_{\rm M}k^{2}\ell^{2} \left(\sigma_{k}^{2}+\sigma_{\alpha}^{2}+\frac{\left\langle x_{\rm c}^{2} \right\rangle}{\bar{\eta}_{x}^{2}}\right) \tag{15.201}\] \[\qquad\qquad\qquad\qquad\times\left[(\bar{\eta}_{y}^{2}+\bar{v}_ {y}^{2})+\frac{\bar{v}_{y}^{2}}{\bar{\eta}_{x}^{2}}\beta_{x}\epsilon_{x}+\frac {\bar{v}_{x}^{2}}{\bar{\eta}_{x}^{2}}(\beta_{y}\epsilon_{y}+\left\langle y_{ \rm c}^{2}\right\rangle)\right]\,.\]

This second-order dispersion due to dipole field errors is generally small but becomes significant in linear-collider facilities where extremely small beam emittances must be preserved along beam lines leading up to the collision point.

## Problems

### 15.1 (S)

Use the perturbation terms \(P_{22}\) (\(z\)) in (15.161) and show that pure betatron oscillations in sextupoles do not cause a tune shift in first approximation. Why is there a finite tune shift for the \(P_{21}\) (\(z\))-term?

### 15.2 (S)

Show analytically that the dispersion function for a single bending magnet with a bending angle \(\theta\) seems approximately to emerge from the middle of the magnet with a slope \(D^{\prime}=\theta\).

### 15.3 (S)

Use the lattice of example #3 in Table 10.1 and introduce vertical rms misalignments of all quadrupoles by \(\left\langle\delta x\right\rangle_{\rm rms}=0.1\,\)mm. Calculate the vertical rms dispersion function. Then, add also rotational alignment errors of the bending magnets by \(\left\langle\delta\alpha\right\rangle_{\rm rms}=0.17\,\)mrad and calculate again the vertical rms dispersion.

### 15.4 (S)

Use two bending magnets separated by a drift space of length \(\ell\). Both bending magnets are of equal but opposite strength. Such a deflection arrangement causes a parallel displacement \(d\) of the beam path. Show that in this case the contribution to the dispersion at the end of the second bending magnet is \(D=-d\) and \(D^{\prime}=0\).

### 15.5 (S)

For the rings in Problems 15.6 or 15.7 calculate the rms tolerance on the quadrupole strength to avoid the integer or half integer resonance. What is the corresponding tolerance on the quadrupole length? To avoid gradient fields in bending magnets the pole profiles must be aligned parallel with respect to the horizontal midplane. What is the angular tolerance for parallelism of the poles?

Use parameters of example #4 in Table 10.1 for a FODO lattice and construct a full ring. Adjust the quadrupole strength such that both tunes are an integer plus a quarter. Calculate the rms alignment tolerance on the quadrupoles required to keep the beam within \(\sigma_{x}=0.1\,\)mm and \(\sigma_{x}=0.1\,\)mm of the ideal orbit. What is the amplification factor? Determine the rms deflection tolerance of the bending magnets to keep the beam within \(0.1\,\)mm of the ideal orbit. A rotation of the bending magnets about its axis creates vertical orbit distortions. If the magnets are aligned to a rotational tolerance of \(\sigma_{\alpha}=0.17\,\)mrad (this is about the limit of conventional alignment techniques) what is the expectation value for the vertical orbit distortion?

Repeat the calculation of Problem 15.6 with the lattice example #1 in Table 10.1. The alignment tolerances are much relaxed with respect to the ring in Problem 15.6. What are the main three contributions influencing the tolerance requirements? Make general recommendations to relax tolerances.

Design an electrostatic quadrupole with an aperture radius of \(3\,\)cm which is strong enough to produce a tune split of \(\delta v=0.01\) between a counter rotating particle and antiparticle beam at an energy of \(3\,\)GeV. Assume the quadrupole to be placed at a location with a betatron function of \(\beta=10\,\)m. How long must the quadrupole be if the electric field strength is to be limited to no more than \(15\,\)kV/cm?
Consider a long straight beam-transport line for a beam with an emittance of \(\epsilon=10^{-12}\,\)rad-m from the end of a 500 GeV linear collider linac toward the collision point. Use a FODO channel with \(\beta_{\max}=5\) m and determine statistical tolerances for transverse and rotational alignment and strength tolerances for the FODO cell quadrupoles to prevent the beam emittance from dilution by more than 10 %.

Use parameters of example #4 in Table 10.1 for a FODO lattice and construct a full ring. Adjust the quadrupole strength such that both tunes are an integer plus a quarter. Calculate the rms alignment tolerance on the quadrupoles required to keep the beam within \(\sigma_{x}=0.1\,\)mm and \(\sigma_{x}=0.1\,\)mm of the ideal orbit. What is the amplification factor? Determine the rms deflection tolerance of the bending magnets to keep the beam within 0.1 mm of the ideal orbit. A rotation of the bending magnets about its axis creates vertical orbit distortions. If the magnets are aligned to a rotational tolerance of \(\sigma_{\alpha}=0.17\,\)mrad (this is about the limit of conventional alignment techniques) what is the expectation value for the vertical orbit distortion?

Consider statistical transverse alignment errors of the quadrupoles in the large hadron collider lattice example #4 in Table 10.1 of \(\left\langle\delta x\right\rangle_{\mathrm{rms}}=0.1\,\)mm. What is the rms path distortion at the end of one turn? Determine the allowable rotational alignment error of the bending magnets to produce a vertical path distortion of no more than that due to quadrupole misalignments. How precise must the bending magnet fields be aligned to not contribute more path distortion than the quadrupole misalignments.

Repeat the calculation of Problem 15.10 with the lattice example #1 in Table 10.1. The alignment tolerances are much relaxed with respect to the ring in Problem 15.10. What are the main three contributions influencing the tolerance requirements? Make general recommendations to relax tolerances.

Calculate the expectation value for the integer and half integer stop band width of the ring in Problem 15.5. Gradient errors introduce a perturbation of the betatron functions. What is the probable perturbation of the betatron function for the case in Problem 15.5?

Consider a FODO cell equal to examples #1,#2, and #4 in Table 10.1, adjust the phase advance per cell to equal values and calculate the natural chromaticities. Insert thin sextupoles into the center of the quadrupoles and adjust to zero chromaticities. How strong are the sextupoles? Why must the sextupoles for lattice #2 be so much stronger compared with lattice #4 even though the chromaticity per cell is about the same?

Consider the transformation of phase ellipses through one full FODO cell of the examples in Problem 15.14. Let the emittance for the phase ellipses be \(\epsilon=10\) mm mrad. First transform the equation for the phase ellipse into a circle by setting \(u=x\) and \(v=\alpha_{x}+\beta x^{\prime}\). Transform the phase circle from the center of the QF through one full FODO cell to the center of the next QF ignoring any sextupole terms. Repeat this transformation but include now the sextupole in the first QF only as calculated in Problem 15.14. Discuss the distortions of the phase circle for the three different FODO lattices.

## Bibliography

* (1) E.D. Courant, H.S. Snyder, Appl. Phys. **3**, 1 (1959)
* (2) G. Luders, Nuovo Cimento Suppl. **2**, 1075 (1955)
* (3) S. Chunjarean, Supercond. Sci. Technol. **24**, 055013 (2011)
* (4) J. Safranek, Experimental determination of storage ring optics using orbit response measurements. NIMA **388**, 27-36 (1997)
* (5) J. Safranek, G. Portmann, A. Terebilo, C. Steier, Matlab-based loco, in _European Particle Accelerator Conference, EPAC 2002_, Paris, France, p. 1184 (2002)
* (6) A. Terebilo, Accelerator toolbox for matlab. Technical Report SLAC-PUB-8732, Stanford University, SLAC (2001)
* (7) N. Vogt-Nielsen, Expansion of the characteristic exponents and the floquet solutions for the linear homogeneous second order differential equation with periodic coefficients. Technical Report MURA/NVN/3, MURA, Chicago, IL (1956)
* (8) H. Wiedemann, Chromaticity correction in large storage rings. Technical Report PEP-Note 220, Stanford Linear Accelerator Center, Stanford, CA (1976)
* (9) P.L. Morton, Derivation of nonlinear chromaticity by higher-order smooth approximation. Technical Report PEP Note-221, SLAC, Stanford, CA (1976)

