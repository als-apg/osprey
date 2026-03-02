## Chapter 12 Vlasov and Fokker-Planck Equations1
Footnote 1: This chapter has been made Open Access under a CC BY 4.0 license. For details on rights and licenses please read the Correction [https://doi.org/10.1007/978-3-319-18317-6_28](https://doi.org/10.1007/978-3-319-18317-6_28)

###### Abstract

Mathematical tools have been derived in previous chapters to describe the dynamics of singly charged particles in electromagnetic fields. While the knowledge of single-particle dynamics is essential for the development of particle beam transport systems, we are still missing a formal treatment of the behavior of multiparticle beams. In principle a multiparticle beam can be described simply by calculating the trajectories of every single particle within this beam, a procedure that is obviously too inefficient to be useful for the description of any real beam involving a very large number of particles.

In this paragraph we will derive concepts to describe the collective dynamics of a beam composed of a large number of particles and its evolution along a transport line utilizing statistical methods that lead to well defined descriptions of the total beam parameters. Mathematical problems arise only when we have a particle beam with neither few particles nor very many particles. Numerical methods must be employed if the number of particles are of importance and where statistical methods would lead to incorrect results.

The evolution of a particle beam has been derived based on Liouville's theorem assuring the constancy of the particle density in phase space. However, this concept has not allowed us to determine modifications of particle distributions due to external forces. Particle distributions are greatly determined by particle source parameters, quantum effects due to synchrotron radiation, nonlinear magnetic fields, collisions with other particles in the same beam, with particles in another beam or with atoms of the residual gases in the beam environment to name only a few phenomena that could influence that distribution. In this chapter, we will derive mathematical methods that allow the determination of particle distributions under the influence of various external electromagnetic forces.

### The Vlasov Equation

To study the development of a particle beam along a transport line, we will concentrate on the evolution of a particle density distribution function \(\Psi(\mathbf{r,p},t)\) in six-dimensional phase space where every particle is represented by a single point. We consider a volume element of phase space that is small enough that we may assume the particle density to be constant throughout that element and determine its evolution in time. In doing so, we will further assume a large, statistically significant number of particles in each volume element and only a slow variation of the particle density from one volume element to any adjacent volume element. To simplify the equations, we restrict the following discussion to two-dimensional phase space \((w,p_{w})\) and use exclusively normalized coordinates. The derivation is exactly the same for other coordinates.

The dynamics of a collection of particles can be studied by observing the evolution of their phase space. Specifically, we may choose a particular phase space element and follow it along its path taking into account the forces acting on it. To do this, we select a phase space element in form of a rectangular box defined by the four corner points \(P_{i}\) in Fig. 12.1.

At the time \(t\) these corners have the coordinates

\[\begin{split}& P_{1}(w,p_{w})\,,\\ & P_{2}(w+\Delta w,p_{w})\,,\\ & P_{3}(\,w+\Delta w,p_{w}+\Delta p_{w})\,,\\ & P_{4}(w,p_{w}+\Delta p_{w})\,.\end{split} \tag{12.1}\]

A short time \(\Delta t\) later, this rectangular box will have moved and may be deformed into a new form of a quadrilateral \((\mathrm{Q}_{1},\mathrm{Q}_{2},\mathrm{Q}_{3},\mathrm{Q}_{4})\) as shown in Fig. 12.1. In determining the volume of the new box at time \(t+\Delta t\) we will assume the conservation of particles allowing no particles to be generated or getting lost. To keep the derivation general the rate of change in the conjugate variables is defined by

\[\begin{split}&\dot{w}=f_{w}(w,p_{w},t)\,,\\ &\dot{p}_{w}=g_{w}(w,p_{w},t)\,,\end{split} \tag{12.2}\]

Figure 12.1: Two-dimensional motion of a rectangle in phase space


speed thus distorting the rectangle \(P_{i}\) into the shape \(Q_{i}\) of Fig. 12.1. To calculate the new vectors defining the distorted area we expand the functions \(f_{w}\) and \(g_{w}\) in a Taylor's series at the point \((w,p_{w})\). While, for example, the \(w\)-component of the movement of point \(P_{1}\) along the \(w\) coordinate is given by \(f_{w}\,\Delta t\) the same component for \(P_{2}\) changes by \(f_{w}\,\Delta t+\frac{\partial f_{w}}{\partial w}\,\Delta w\,\Delta t\). The \(w\)-component of the vector \(\boldsymbol{q}_{w}=Q_{1}-Q_{2}\) therefore becomes \(\Delta w+\frac{\partial f_{w}}{\partial w}\,\Delta w\,\Delta t\). Similarly, we can calculate the \(p\)-component of this vector as well as both components for the vector \(\boldsymbol{q}_{p}=Q_{1}-Q_{4}\). The phase space area of the distorted rectangle \((Q_{1},Q_{2},Q_{3},Q_{4})\) at time \(t+\Delta t\) with these vector components is then given by

\[|\boldsymbol{q}_{w},\boldsymbol{q}_{p}|=\left|\begin{array}{cc}\Delta w+ \frac{\partial f_{w}}{\partial w}\Delta w\,\Delta t&\frac{\partial f_{w}}{ \partial p_{w}}\Delta p_{w}\,\Delta t\\ \frac{\partial g_{w}}{\partial w}\Delta w\,\Delta t&\Delta p_{w}+\frac{ \partial g_{w}}{\partial p_{w}}\Delta p_{w}\,\Delta t\end{array}\right|=\, \Delta A_{Q}. \tag{12.8}\]

Dropping second-order terms in \(\Delta t\) we get indeed the expression (12.6). Obviously, the phase space volume does not change if

\[\frac{\partial f_{w}}{\partial w}+\frac{\partial g_{w}}{\partial p_{w}}=0 \tag{12.9}\]

in agreement with the result obtained in Chap. 8, where we have assumed that the Lorentz force is the only force acting on the particle. In this paragraph, however, we have made no such restrictions and it is this generality that allows us to derive, at least in principle, the particle distribution under the influence of any forces. Equation (12.9) tells us that there is no damping if the velocity \(\dot{w}=f_{w}\) is independent of the position and the forces \(\dot{p}=g_{w}\) are independent of the momentum.

The factor

\[\left[1+\left(\frac{\partial f_{w}}{\partial w}+\frac{\partial g_{w}}{ \partial p_{w}}\right)\Delta t\right] \tag{12.10}\]

in (12.6) is the general Wronskian of the transformation and is not necessarily equal to unity. We have such an example in the form of adiabatic damping. Indeed we have damping or anti-damping whenever the Wronskian is different from unity.

To illustrate this, we use the example of a damped harmonic oscillator, which is described by the second-order differential equation \(\ddot{w}+2\alpha_{w}\dot{w}+\omega_{0}^{2}w=0\), or in form of a set of two linear differential equations

\[\begin{split}\dot{w}=\omega_{0}p_{w}=f_{w}(w,p_{w},t),\\ \dot{p}_{w}=-\omega_{0}w-2\alpha_{w}p_{w}=g_{w}(w,p_{w},t).\end{split} \tag{12.11}\]

From this we find indeed the relation

\[\frac{\partial f_{w}}{\partial w}+\frac{\partial g_{w}}{\partial p_{w}}=-2 \alpha_{w}, \tag{12.12}\]where \(\alpha_{w}\) is the damping decrement1 of the oscillator. We have obtained on a general basis that the phase space density for harmonic oscillators will vary only if damping forces are present. Here we use the term damping in a very general way including excitation depending on the sign of the damping decrement \(\alpha_{w}\). The designation \(\alpha_{w}\) for the damping decrement may potentially lead to some confusion with the same use for the betatron function \(\alpha=-\frac{1}{2}\beta^{\prime}\). However, we choose here to rather require some care than introduce against common use new designations for the damping decrement or the betatron functions. We also note that for all cases where the damping time is long compared to the oscillation time, and we consider here only such cases, the damping occurs for both conjugate trajectories.

Footnote 1: The letter \(\alpha_{u}\) is used here for the damping decrement. Since in beam dynamics \(\alpha_{u}\) is also used to identify a lattice function, a mixup of the quantities could occur. We have chosen not to use a different nomenclature, however, since this choice is too deeply entrenched in the community. With some care, confusion can be avoided.

The derivation in two-dimensional phase space can easily be generalized to six-dimensional phase space with the generalized volume element

\[\Delta V_{P}=\Delta\mathbf{r}\Delta\mathbf{p} \tag{12.13}\]

at time \(t\) and a time interval \(\Delta t\) later

\[\Delta V_{Q}=\Delta\mathbf{r}\Delta\mathbf{p}[1+\mathbf{\nabla}_{\mathbf{s}}\mathbf{f}\ \Delta t+\mathbf{\nabla}_{p}\mathbf{g}\ \Delta t]\,. \tag{12.14}\]

The Nabla operators are defined by

\[\mathbf{\nabla}_{r}=\left(\frac{\partial}{\partial w},\ \frac{\partial}{ \partial v},\ \frac{\partial}{\partial u}\right)\quad\text{and}\quad\mathbf{\nabla}_{p}= \left(\frac{\partial}{\partial p_{w}},\ \frac{\partial}{\partial p_{v}},\ \frac{ \partial}{\partial p_{u}}\right)\,, \tag{12.15}\]

where \((w,v,u)\) are normalized variables and the vector functions \(\mathbf{f}\) and \(\mathbf{g}\) are defined by the components \(\mathbf{f}=(f_{w},f_{v},f_{u})\) and \(\mathbf{g}=(g_{w},g_{v},g_{u})\).

Equation (12.4) can now be reduced further after applying a Taylor's expansion to the density function \(\Psi\). With (12.5), (12.6) and keeping only linear terms

\[\frac{\partial\Psi}{\partial t}+f_{w}\frac{\partial\Psi}{\partial w}+g_{w} \frac{\partial\Psi}{\partial p_{w}}=-\left(\frac{\partial f_{w}}{\partial w} +\frac{\partial g_{w}}{\partial p_{w}}\right)\Psi\,. \tag{12.16}\]

It is straightforward to generalize this result again to six-dimensional phase space

\[\frac{\partial\Psi}{\partial t}+\mathbf{f}\,\mathbf{\nabla}_{r}\Psi+\mathbf{g}\,\mathbf{ \nabla}_{p}\Psi=-\,\left(\mathbf{\nabla}_{\mathbf{f}}+\mathbf{\nabla}_{p}\mathbf{g}\right)\Psi\,, \tag{12.17}\]

which is called the Vlasov equation. If there is no damping the r.h.s. of the Vlasov equation vanishes and we have

\[\frac{\partial\Psi}{\partial t}+\mathbf{f}\,\mathbf{\nabla}_{r}\Psi+\mathbf{g}\mathbf{\nabla} _{p}\Psi=0\,. \tag{12.18}\]This is simply the total time derivative of the phase space density \(\Psi\) telling us that in the absence of damping it remains a constant of motion. The preservation of the phase space density is Liouville's theorem and we have demonstrated in this paragraph the validity of this theorem for a Hamiltonian system with vanishing dissipating forces \((\nabla_{\boldsymbol{j}}\boldsymbol{f}+\nabla_{\boldsymbol{p}}\boldsymbol{g})=0\).

Equation (12.18) describes the evolution of a multiparticle system in phase space where the physics of the particular particle dynamics is introduced through the functions \(\boldsymbol{f}\left(\boldsymbol{r},\boldsymbol{p},t\right)\) and \(\boldsymbol{g}(\boldsymbol{r},\boldsymbol{p},t)\). The definition of these functions in (12.2) appears similar to that for the Hamiltonian equations of motion. In case \(\boldsymbol{r}\) and \(\boldsymbol{p}\) are canonical variables we may indeed derive these functions from the Hamiltonian

\[\begin{split}\dot{\boldsymbol{r}}=\nabla_{p}\mathcal{H}=& \boldsymbol{f}\,,\\ \dot{\boldsymbol{p}}=-\nabla_{r}\mathcal{H}=& \boldsymbol{g},\end{split} \tag{12.19}\]

where \(\mathcal{H}\) is the Hamiltonian of the system. We are therefore, at least in principle, able to solve the evolution of a multiparticle system in phase space if its Hamiltonian is known. It should be emphasized, however, that the variables \((w,p)\) need not be canonical to be used in the Vlasov equation.

It is interesting to apply the Vlasov equation to simple one-dimensional harmonic oscillators with vanishing perturbation. Introducing the canonical variable \(p\) through \(\dot{w}=vp\), the Hamiltonian becomes \(\mathcal{H}_{0}=\frac{1}{2}vp^{2}+\frac{1}{2}vw^{2}\) and the equations of motion are

\[\begin{split}\dot{w}=&+\frac{\partial\mathcal{H}_{0} }{\partial p}=vp=f,\\ \dot{p}=&-\frac{\partial\mathcal{H}_{0}}{\partial w }=-\nu w=g.\end{split} \tag{12.20}\]

It is customary for harmonic oscillators and similarly for particle beam dynamics to use the oscillation phase as the independent or "time" variable. Since we have not made any specific use of the real time in the derivation of the Vlasov equation, we choose here the phase as the "time" variable. For the simple case of an undamped harmonic oscillator \(\frac{\partial\dot{f}}{\partial w}=0\) and \(\frac{\partial g}{\partial p}=0\) and consequently the Vlasov equation becomes from (12.16) with (12.20)

\[\frac{\partial\Psi}{\partial\varphi}+\nu p\frac{\partial\Psi}{\partial w}-\nu w \frac{\partial\Psi}{\partial p}=0\,. \tag{12.21}\]

In cylindrical phase space coordinates (\(w=r\cos\theta\), \(p=r\sin\theta\), \(\varphi\)) this reduces to the simple equation

\[\frac{\partial\Psi}{\partial\varphi}-\nu\frac{\partial\Psi}{\partial\theta}= 0\,. \tag{12.22}\]Any differentiable function with the argument \((r,\theta\,+\,v\varphi)\) can be a solution of (12.22) describing the evolution of the particle density \(\Psi\) with time

\[\Psi(w,p_{w},\varphi)=F(r,\theta\,+\,v\varphi)\,, \tag{12.23}\]

Any arbitrary particle distribution in \((w,p_{w})\)-phase space merely rotates about the center with the frequency \(v\) and remains otherwise unchanged as shown in Fig. 12.2. This is just another way of saying that an ensemble of many particles behaves like the sum of all individual particles since any interaction between particles as well as damping forces have been ignored. In \((x,x^{\prime})\)-phase space this rotation is deformed into a "rotation" along elliptical trajectories. The equation of motion in \((w,p_{w})\)-phase space is solved by \(r=\) const indicating that the amplitude \(r\) is a constant of motion. In \((x,x^{\prime})\)-phase space we set \(w\,=\,x/\sqrt{\beta}\) and \(p\,=\,\sqrt{\beta}\,x^{\prime}\,+\,\frac{\alpha}{\sqrt{\beta}}\,x\) and get from \(r^{2}\,=\,w^{2}\,+\,p_{w}^{2}\) for this constant of motion

\[\beta\,{x^{\prime}}^{2}\,+\,2\alpha\,xx^{\prime}\,+\,\gamma\,x^{2}=\,{\rm const} \tag{12.24}\]

which is the Courant-Snyder invariant. The Vlasov equation allows us to generalize this result collectively to all particles in a beam. Any particular particle distribution a beam may have at the beginning of the beam transport line or circular accelerator will be preserved as long as damping or other statistical effects are absent.

Figure 12.2: Beam motion in phase space

#### 12.1.1 Betatron Oscillations and Perturbations

The Vlasov equation will prove to be a useful tool to derive particle beam parameters. Specifically, it allows us to study the influence of arbitrary macroscopic fields on particle density in phase space and on the characteristic frequency of particle motion. To demonstrate this, we expand the example of the harmonic oscillator to include also perturbation terms. For such a perturbed system the equation of motion is

\[\ddot{w}+v_{0}^{2}w=v_{0}^{2}\beta^{\frac{3}{2}}\sum_{n>0}p_{n}\,\beta^{\frac{n }{2}}\,w^{n}\,, \tag{12.25}\]

where the coefficients \(p_{n}\) are the strength parameters for the \(n\)th order perturbation term and the amplitude \(w\) is the normalized betatron oscillation amplitude. The Vlasov equation allows us to calculate the impact of these perturbation terms on the betatron frequency. We demonstrate this first with a linear perturbation term (\(n=1\)) caused by a gradient field error \(p_{1}=-\delta k\) in a quadrupole. In this case the equation of motion is from (12.25)

\[\ddot{w}+v_{0}^{2}w=-v_{0}^{2}\beta^{2}\delta k\,w \tag{12.26}\]

or

\[\ddot{w}+v_{0}^{2}(1+\beta^{2}\delta k)w=0\,. \tag{12.27}\]

This second-order differential equation can be replaced by two first-order differential equations which is in general the most straight forward way to obtain the functions (12.2)

\[\begin{split}\dot{w}&=v_{0}\sqrt{1+\beta^{2}\delta k }p\,,\\ \dot{p}&=-v_{0}\sqrt{1+\beta^{2}\delta k}\,w\,.\end{split} \tag{12.28}\]

Here it is assumed that the betatron function \(\beta\) and the quadrupole field error \(\delta k\) are uniformly distributed along the beam line and therefore can be treated as constants. This approach is justified since we are interested only in the average oscillation frequency of the particles and not in fast oscillating terms. The desired result can be derived directly from (12.28) without any further mathematical manipulation by comparison with (12.20). From there the oscillating frequency for the perturbed system is given by

\[v=v_{0}\sqrt{1+\beta^{2}\,\delta k}\approx v_{0}\,(1+\tfrac{1}{2}\beta^{2} \delta k)\,, \tag{12.29}\]for small perturbations. The betatron frequency shift can be expressed by the lowest order harmonic of the Fourier expansion for the periodic perturbation function \(v_{0}\,\beta^{2}\,\delta k\) to give

\[2\pi v_{0}\,\left(\beta^{2}\delta k\right)_{0}=\oint v_{0}\beta^{2}\delta k\, \mathrm{d}\varphi=\oint\beta\delta k\,\mathrm{d}z \tag{12.30}\]

making use of the definition for the betatron phase \(\mathrm{d}\varphi=\)d\(z/v_{0}\beta\). The tune shift \(\delta v\) due to quadrupole field errors is therefore from (12.29)

\[\delta\,v=v-v_{0}=\frac{1}{4\pi}\oint\beta\delta k\mathrm{d}z\,, \tag{12.31}\]

in agreement with (15.64). Again, the Vlasov equation confirms this result for all particles irrespective of the distribution in phase space. This procedure can be expanded to any order of perturbation. From the differential equation (12.25) one gets in analogy to the equations of motion (12.28)

\[\dot{w}=v_{0}\sqrt{1-\beta^{3/2}\sum_{n>0}p_{n}\beta^{n/2}w^{n-1}}\,p\,, \tag{12.32}\]

\[\dot{p}=-v_{0}\sqrt{1-\beta^{3/2}\sum_{n>0}p_{n}\beta^{n/2}w^{n-1}}\,w\,\,.\]

For small perturbations the solution for the unperturbed harmonic oscillator \(w(\varphi)=w_{0}\sin(v_{0}\varphi+\delta)\) may be used where \(\delta\) is an arbitrary phase constant. The tune shift \(\Delta v=v-v_{0}\) is thus while integrating over all perturbations around a circular accelerator

\[\Delta v=-\sum_{n>0}\frac{1}{4\pi}\oint p_{n}\beta^{\frac{n+1}{2}}w_{0}^{n-1} \sin^{n-1}[v_{0}\varphi(z)+\delta]\,\mathrm{d}z, \tag{12.33}\]

where we have changed the independent variable from \(\varphi\) to \(z\) by \(\mathrm{d}z=v_{0}\beta\mathrm{d}\varphi\).

Not all perturbation terms contribute to a tune variation. All even terms \(n=2m\), where \(m\) is an integer, integrate, for example, to zero in this approximation and a sextupole field therefore does not contribute to a tune shift or tune spread. This conclusion must be modified, however, due to higher-order approximations which become necessary when perturbations cannot be considered small anymore. Furthermore, we find from (12.33) that the tune shift is independent of the particle oscillation amplitude only for quadrupole field errors \(n=1\). For higher-order multipoles the tune shift becomes amplitude dependent resulting in a tune spread within the particle beam rather than a coherent tune shift for all particles of the beam.

In a particular example, the tune spread caused by a single octupole (\(n=3\)) in a circular accelerator is given by

\[\Delta v_{3}=-\frac{\epsilon_{w}}{8\pi}\oint p_{3}\beta^{2}\,\mathrm{d}z\,, \tag{12.34}\]

where \(w_{0}^{2}=\epsilon_{w}\) is the emittance of the beam. Similar results can be found for higher-order multipoles.

#### Damping

At the beginning of this section we have decided to ignore damping and have used the undamped Vlasov equation (12.18). Damping or anti-damping effects do, however, occur in real systems and it is interesting to investigate if the Vlasov equation can be used to derive some general insight into damped systems as well. For a damped oscillator we use (12.11), (12.12) to form the Vlasov equation in the form of (12.16). Instead of the phase we now use the real time as the independent variable to allow the intuitive definition of the damping decrement as the relative decay of the oscillation amplitude with time

\[\frac{\partial\Psi}{\partial t}+\omega_{0}p_{w}\frac{\partial\Psi}{\partial w }-(\omega_{0}w+2\alpha_{w}p_{w})\frac{\partial\Psi}{\partial p_{w}}=+2\alpha_ {w}\Psi\,. \tag{12.35}\]

This partial differential equation can be solved analytically in a way similar to the solution of the undamped harmonic oscillator by using cylindrical coordinates. For very weak damping we expect a solution close to (12.23) where the amplitude \(r\) in phase space was a constant of motion. For a damped oscillator we try to form a similar invariant from the solution of a damped harmonic oscillator

\[w=w_{0}\mathrm{e}^{-\alpha_{w}t}\cos\sqrt{\omega_{0}^{2}-\alpha_{w}^{2}}\,t= r\mathrm{e}^{-\alpha_{w}t}\cos\theta\,. \tag{12.36}\]

With the conjugate component \(\omega_{0}\,p_{w}=\dot{w}\), we form the expression

\[\frac{\omega_{0}\,p_{w}+\alpha_{w}w}{\sqrt{\omega_{0}^{2}-\alpha_{w}^{2}}}=-w _{0}\mathrm{e}^{-\alpha_{w}t}\sin\sqrt{\omega_{0}^{2}-\alpha_{w}^{2}}\,t=-r \mathrm{e}^{-\alpha_{w}t}\sin\theta \tag{12.37}\]

and eliminate the phase \(\theta\) from (12.36), (12.37) keeping only terms linear in the damping decrement \(\alpha_{w}\) to obtain the "invariant"

\[r^{2}\mathrm{e}^{-\,2\alpha_{w}\,t}=w^{2}+p_{w}^{2}+2\frac{\alpha_{w}}{\omega _{0}}wp_{w}\,. \tag{12.38}\]Obviously if we set \(\alpha_{w}=0\) we have the invariant of the harmonic oscillator. The time dependent factor due to finite damping modifies this "invariant". However, for cases where the damping time is long compared to the oscillation period we may still consider (12.38) a quasi invariant. The phase coordinate \(\theta\) can be derived from (12.36), (12.37) as a function of \(w\) and \(p_{w}\) as may be verified by insertion into the differential equation (12.35). The solution for the phase space density of a damped oscillator is of the form

\[\Psi(w,p_{w},t)=\mathrm{e}^{2\alpha_{w}t}F(r,\Phi)\,, \tag{12.39}\]

where \(F(r,\Phi)\) is any arbitrary but differentiable function of \(r\) and \(\Phi\) and the phase \(\Phi\) is defined by

\[\Phi=\theta+\sqrt{\omega_{0}^{2}-\alpha_{w}^{2}}\,t=\arctan\left(+\frac{\omega _{0}\,p_{w}+\alpha_{w}w}{\sqrt{\omega_{0}^{2}-\alpha_{w}^{2}w}}\right)+\sqrt{ \omega_{0}^{2}-\alpha_{w}^{2}}\,t\,. \tag{12.40}\]

For very weak damping \(\alpha_{w}\to 0\) and the solution (12.39) approaches (12.23) where \(\alpha_{w}=0\) and \(v\varphi\ =\ \omega_{0}t\) as expected. Therefore even for finite damping a particle distribution rotates in phase space although with a somewhat reduced rotation frequency due to damping. The particle density \(\Psi\), however, changes exponentially with time due to the factor \(\mathrm{e}^{2\alpha_{w}t}\). For damping \(\alpha_{w}>0\), we get an increase in the phase space density at the distance \(R\) from the beam center. At the same time the real particle oscillation amplitudes \((w,p_{w})\) are being reduced proportional to \(\mathrm{e}^{-\alpha_{w}t}\) and the increase in the phase space density at \(R\) reflects the concentration of particles in the beam center from larger amplitudes due to damping.

In conclusion we found that in systems where velocity dependent forces exist, we have damping (\(\alpha_{w}>0\)) or anti-damping (\(\alpha_{w}<0\)) of oscillation amplitudes. As has been discussed such forces do exist in accelerators leading to damping. Mostly, however, the Vlasov equation is applied to situations where particles interact with self or external fields that can lead to instabilities. It is the task of particle beam dynamics to determine the nature of such interactions and to derive the circumstances under which the damping coefficient \(\alpha_{w}\), if not zero, is positive for damping or negative leading to beam instability.

### 12.2 Damping of Oscillations in Electron Accelerators

In electron accelerators we are concerned mainly with damping effects caused by the emission of synchrotron radiation. All six degrees of freedom for particle motion are damped. Damping of energy oscillations occurs simply from the fact that the synchrotron radiation power is energy dependent. Therefore a particle with a higher energy than the reference particle radiates more and a particle with less energy radiates less. The overall effect is that the energy deviation is reduced or damped. Damping of the transverse motion is principally a geometric effect. The photons of synchrotron radiation are emitted into the direction of the particle motion. Therefore part of the energy loss is correlated to a loss in transverse momentum. On the other hand, the lost energy is restored through accelerating fields with longitudinal components only. The overall effect of an energy loss during the course of betatron oscillations is therefore a loss of transverse momentum which leads to a reduction in the transverse oscillation amplitude, an effect we call damping. In the next section, we will discuss the physics leading to damping and derive the appropriate damping decrement for different modes of oscillations.

##### Damping of Synchrotron Oscillations

In a real beam particles are spread over a finite distribution of energies close to the reference energy. The magnitude of this energy spread is an important parameter to be considered for both beam transport systems as well as for experimental applications of particle beams. In general, an energy spread as small as possible is desired to minimize chromatic aberrations and for improved accuracy of experimental observation. We will therefore derive the parametric dependence of damping and discuss methods to reduce the energy spread within a particle beam.

To do this, we consider a beam of electrons being injected with an arbitrary energy distribution into a storage ring ignoring incidental beam losses during the injection process due to a finite energy acceptance. Particles in a storage ring undergo synchrotron oscillations which are oscillations about the ideal momentum and the ideal longitudinal position. Since energy and time or equivalently energy and longitudinal position are conjugate phase space variables, we will investigate both the evolution of the energy spread as well as the longitudinal distribution or bunch length of the particle beam.

The evolution of energy spread or bunch length of the particle beam will depend very much on the nature of particles and their energy. For heavy particles like protons or ions there is no synchrotron radiation damping and therefore the phase space for such beams remains constant. As a consequence, the energy spread or bunch length also stays a constant. A similar situation occurs for electrons or positrons at very low energies since synchrotron radiation is negligible. Highly relativistic electrons, however, produce intense synchrotron radiation leading to a strong damping effect.

The damping decrement \(\alpha_{w}\) is defined in the Vlasov equation by

\[\frac{\partial f}{\partial w}+\frac{\partial g}{\partial p}=-2\alpha_{w} \tag{12.41}\]and can be calculated with the knowledge of the functions \(f\) and \(g\). For the conjugate variables \((w,p_{w})\) we use the time deviation of a particle with respect to the synchronous particle \(w=\tau\) as shown in Fig. 12.3 and the difference of the particle's energy \(E\) from the synchronous or reference energy \(E_{0}\) and set \(p_{w}=\epsilon=E-E_{0}\).

Since \(f=\frac{\mathrm{d}\tau}{\mathrm{d}t}=\dot{\tau}\) and \(g=\frac{\mathrm{d}\epsilon}{\mathrm{d}t}=\dot{\epsilon}\) we have to determine the rate of change for the conjugate variables. The rate of change of \(\tau\) is from (9.17) with \(cp_{0}\approx E_{0}\)

\[\frac{\mathrm{d}\tau}{\mathrm{d}t}=-\eta_{\mathrm{c}}h\frac{\epsilon}{E_{0}}, \tag{12.42}\]

where we have replaced the phase by the time \(\dot{\psi}=c\beta hk_{0}\dot{\tau}\) and the relative momentum error by the relative energy error since we consider here only highly relativistic particles. The latter replacement is a matter of convenience since we will be using the energy gain in accelerating fields.

The energy rate of change \(\dot{\epsilon}\) is the balance of the energy gained in accelerating fields and the energy lost due to synchrotron radiation or other losses

\[\dot{\epsilon}=\frac{1}{T}\left[eV_{\mathrm{rf}}\left(\tau_{\mathrm{s}}+\tau \right)-U(E_{\mathrm{s}}+\epsilon)\right]. \tag{12.43}\]

Here \(T\) is the time it takes the particles to travel the distance \(L\). The energy gain within the distance \(L\) for a particle traveling a time \(\tau\) behind the reference or synchronous particle is \(eV_{\mathrm{rf}}\left(\tau_{\mathrm{s}}+\tau\right)\) and \(U\) is the energy loss to synchrotron radiation along the same distance of travel. here we assume the energy gain or loss to be distributed evenly over the length of \(L\).

Before we go on, we apply these expressions to the simple situation of a linear accelerator of length \(L\) where the momentum compaction factor vanishes \(\alpha_{\mathrm{c}}=0\) and where there is no energy loss due to synchrotron radiation \(U\equiv 0\). Furthermore, we ignore for now other energy losses and have with \(\eta_{\mathrm{c}}=1/\gamma^{2}\)

\[\begin{array}{l}f=\dot{\tau}=\frac{1}{\beta^{2}\gamma^{2}}\frac{\epsilon}{E},\\ g=\dot{\epsilon}=\frac{1}{T}eV_{\mathrm{rf}}\left(\tau_{\mathrm{s}}+\tau \right).\end{array} \tag{12.44}\]

Inserted into (12.41) we find the damping decrement to vanish which is consistent with the constancy of phase space. From the Vlasov equation we learn that in the absence of damping the energy spread \(\epsilon\) stays constant as the particle beam gets accelerated.

Figure 12.3: Longitudinal particle position

The Vlasov equation still can be used to also describe adiabatic damping but we need to use the relative energy spread as one of the variables. Instead of the second equation (12.44) we have then with \(\delta=\frac{\epsilon}{E}\)

\[g=\frac{\mathrm{d}}{\mathrm{d}t}\delta=\frac{\frac{\epsilon}{E_{t}}-\frac{ \epsilon}{E_{0}}}{\Delta t}, \tag{12.45}\]

where \(E_{0}\) and \(E_{t}\) are the energies time \(t_{0}\) and \(t=t_{0}+\mathrm{d}t\), respectively. We choose the time interval \(\mathrm{d}t\) small enough so that the energy increase \(\mathrm{d}E=a\mathrm{d}t\ll E_{0}\) and get

\[g=-\frac{\epsilon}{E_{t}}\frac{a}{E_{0}}. \tag{12.46}\]

The damping decrement becomes from (12.41) with \(\delta=\frac{\epsilon}{E}\) and \(\partial f/\partial\tau=0\)

\[\frac{\partial g}{\partial\delta}=-\frac{a}{E_{0}}=-2\alpha_{w}=\frac{1}{ \delta}\frac{\mathrm{d}\delta}{\mathrm{d}t} \tag{12.47}\]

and after integration

\[\int\frac{\mathrm{d}\delta}{\delta}=\ln\frac{\delta}{\delta_{0}}=-\int\frac{a }{E_{0}}\mathrm{d}t=-\int\frac{\mathrm{d}E}{E_{0}}=+\ln\frac{E_{0}}{E_{t}} \tag{12.48}\]

or

\[\frac{\delta}{\delta_{0}}=\frac{E_{0}}{E_{t}}. \tag{12.49}\]

The relative energy spread in the beam is reduced during acceleration inversely proportional to the energy. The reduction of the relative energy spread is called adiabatic damping. This name is unfortunate in the sense that it does not actually describe a damping effect in phase space as we just found out but rather describes the variation of the relative energy spread with energy which is merely a consequence of the constant phase space density or Liouville's theorem.

Returning to the general case (12.43) we apply a Taylor's expansion to the rf-voltage in (12.44) and get for terms on the r.h.s. keeping only linear terms

\[e\,V_{\mathrm{rf}}(\tau_{\mathrm{s}}+\tau) = e\,V_{\mathrm{rf}}(\tau_{\mathrm{s}})+e\,\left.\frac{\partial V _{\mathrm{rf}}}{\partial\tau}\right|_{\tau_{\mathrm{s}}}\tau\,, \tag{12.50}\] \[-U(E_{\mathrm{s}}+\epsilon) = -U(E_{\mathrm{s}})-\left.\frac{\partial U}{\partial E}\right|_{E_ {\mathrm{s}}}\epsilon\,. \tag{12.51}\]Since the energy gain from the rf-field \(eV_{\rm rf}(\tau_{\rm s})\) for the synchronous particle just compensates its energy loss \(U(E_{\rm s})\), we have instead of (12.43) now

\[\dot{\epsilon}\,=\,\frac{1}{T}\left[\left.e\dot{V}_{\rm rf}(\tau_{\rm s})\ \tau\ -\ \frac{\partial U}{\partial E}\right|_{E_{\rm s}}\epsilon\ \right], \tag{12.52}\]

where we have set \(\dot{V}_{\rm rf}=\frac{\partial V_{\rm rf}}{\partial\tau}\). The synchrotron oscillation damping decrement can now be derived from (12.41) with (12.44), (12.52) to give

\[\alpha_{\rm s}=\left.+\,\frac{1}{2}\frac{1}{T}\ \left.\frac{\partial U}{ \partial E}\right|_{E_{\rm s}}. \tag{12.53}\]

We will now derive the damping decrement for the case that the energy loss is only due to synchrotron radiation. The energy loss along the transport line \(L\) is given by

\[U_{\rm s}=\left.\frac{1}{c}\int_{0}^{L}P_{\gamma}{\rm d}s\right., \tag{12.54}\]

where \(P_{\gamma}\) is the synchrotron radiation power and the integration is taken along the actual particle trajectory \(s\). If \(\rho(z)\) is the bending radius along \(z\), we have \(\frac{{\rm d}s}{{\rm d}z}=1+\frac{x}{\rho}\). With \(x=x_{\beta}+\eta\frac{\epsilon}{E_{\rm s}}\) and averaging over many betatron oscillations, we get \(\langle x_{\beta}\rangle=0\) and

\[\frac{{\rm d}s}{{\rm d}z}=1+\ \frac{\eta}{\rho}\frac{\epsilon}{E}. \tag{12.55}\]

This asymmetric averaging of the betatron oscillation only is permissible if the synchrotron oscillation frequency is much lower than the betatron oscillation frequency as is the case in circular accelerators. With \({\rm d}s=[1+(\eta/\rho)(\epsilon/E_{\rm s})]{\rm d}z\) in (12.54), the energy loss for a particle of energy \(E_{\rm s}+\epsilon\) is

\[U_{\rm s}(E_{\rm s}+\epsilon)=\frac{1}{c}\int_{\rm L}P_{\gamma}\left(1+\frac{ \eta}{\rho}\frac{\epsilon}{E_{\rm s}}\right){\rm d}z \tag{12.56}\]

or after differentiation with respect to the energy

\[\left.\frac{\partial U_{\rm s}}{\partial E}\right|_{E_{\rm s}}=\frac{1}{c} \int_{\rm L}\left[\left.\frac{{\rm d}P_{\gamma}}{{\rm d}E}+P_{\gamma}\ \frac{\eta}{\rho}\frac{1}{E_{\rm s}}\right]_{E_{\rm s}}{\rm d}z\right.. \tag{12.57}\]

The synchrotron radiation power is proportional to the square of the energy and the magnetic field \(P_{\gamma}\sim E_{\rm s}^{2}B_{0}^{2}\) which we use in the expansion

\[\frac{{\rm d}P_{\gamma}}{{\rm d}E}=\frac{\partial P_{\gamma}}{\partial E}+ \frac{\partial P_{\gamma}}{\partial B_{0}}\frac{\partial B}{\partial E}=2 \frac{P_{\gamma}}{E_{\rm s}}+2\frac{P_{\gamma}}{B}\frac{\partial B}{\partial x }\frac{\partial x}{\partial E}. \tag{12.58}\]The variation of the synchrotron radiation power with energy depends directly on the energy but also on the magnetic field if there is a field gradient \(\frac{\partial B}{\partial x}\) and a finite dispersion function \(\eta=E_{\rm s}\frac{\partial x}{\partial E}\). The magnetic field as well as the field gradient is to be taken at the reference orbit. Collecting all these terms and setting \(\frac{1}{B_{0}}\frac{\partial B}{dx}=\rho\,k\) we get for (12.57)

\[\frac{\partial U_{\rm s}}{\partial E}\bigg{|}_{E_{\rm s}} = \frac{1}{c}\int_{L}\,\bigg{(}2\frac{P_{\gamma}}{E_{\rm s}}+2 \frac{P_{\gamma}}{E_{\rm s}}\rho k\eta+\frac{P_{\gamma}}{E_{\rm s}}\frac{\eta }{\rho}\bigg{)}\bigg{|}_{E_{\rm s}}\,{\rm d}z\] \[= \frac{U_{\rm s}}{E_{\rm s}}\left[\,2+\,\frac{1}{cU_{\rm s}}\int_{ L}\,P_{\gamma}\eta\left(\frac{1}{\rho}+2\rho k\right)\right|_{E_{\rm s}}\,{\rm d }z\,\,\right]\,,\]

where we have made use of \(U_{\rm s}=\frac{1}{c}\int_{L}P_{\gamma}(E_{\rm s})\,{\rm d}z\). Recalling the expressions for the synchrotron radiation power and energy loss \(P_{\gamma}=C_{\gamma}\,E_{\rm s}^{4}/\rho^{2}\) and \(U_{\rm s}=C_{\gamma}E_{\rm s}^{4}\int{\rm d}z/\rho^{2}\), we may simplify (12.59) for

\[\frac{\partial U}{\partial E}\bigg{|}_{E_{\rm s}}=\,\frac{U_{\rm s}}{E_{\rm s }}\,(2+\vartheta)\,, \tag{12.60}\]

where the \(\vartheta\)-parameter has been introduced in (11.25). We finally get from (12.53) with (12.60) the damping decrement for synchrotron oscillations

\[\alpha_{\epsilon}\,=\,\frac{U_{\rm s}}{2\mbox{\it TE}_{\rm s}}(2+\vartheta)\, =\,\frac{U_{\rm s}}{2\mbox{\it TE}_{\rm s}}J_{\epsilon}\,=\,\frac{\langle P_ {\gamma}\rangle}{2E_{\rm s}}J_{\epsilon}\,, \tag{12.61}\]

in full agreement with results obtained earlier. Since all parameters except \(\vartheta\) are positive we have shown that the synchrotron oscillations for radiating particles are damped. A potential situation for anti-damping can be created if \(\vartheta\,<-2\).

To calculate the damping decrement, we assume accelerating fields evenly distributed around the ring to restore the lost energy. In practice this is not true since only few rf-cavities in a ring are located at one or more places around the ring. As long as the revolution time around the ring is small compared to the damping time, however, we need not consider the exact location of the accelerating cavities and may assume an even and uniform distribution around the ring.

##### Damping of Vertical Betatron Oscillations

Particles orbiting in a circular accelerator undergo transverse betatron oscillations. These oscillations are damped in electron rings due to the emission of synchrotron radiation. First we will derive the damping decrement for the vertical betatron oscillation. In a plane accelerator with negligible coupling this motion is independent of other oscillations. This is not the case for the horizontal betatron motion which is coupled to the synchrotron oscillation due to the presence of a finite dispersion function. We will therefore derive the vertical damping decrement first and then discuss a very general theorem applicable for the damping in circular accelerators. This theorem together with the damping decrement for the synchrotron and vertical betatron oscillations will enable us to derive the horizontal damping in a much simpler way than would be possible in a more direct way.

In normalized coordinates the functions \(f\) and \(g\) are for the vertical plane

\[\frac{\mathrm{d}w}{\mathrm{d}\varphi}=+\upsilon p=f(w,p,\varphi)\, \tag{12.62}\]

\[\frac{\mathrm{d}p}{\mathrm{d}\varphi}=-\upsilon w=g(w,p,\varphi)\, \tag{12.63}\]

where \(\upsilon=\upsilon_{y},w=y/\sqrt{\beta_{y}}\), \(\frac{1}{\upsilon_{y}}\frac{\mathrm{d}w}{\mathrm{d}\varphi}=\sqrt{\beta_{y}}y^ {\prime}-\frac{1}{2}\frac{\beta_{y}^{\prime}}{\sqrt{\beta_{y}}}y\) and \(\upsilon_{y}\varphi=\psi_{y}\) is the vertical betatron phase.

Due to the emission of a synchrotron radiation photon alone the particle does not change its position \(y\) nor its direction of propagation \(y^{\prime}\). With this we derive now the damping within a path element \(\Delta z\) which includes the emission of photons as well as the appropriate acceleration to compensate for that energy loss. Just after the emission of the photon but before the particle interacts with accelerating fields let the transverse momentum and total energy be \(p_{\perp}\) and \(E_{\mathrm{s}}\),respectively. The slope of the particle trajectory is therefore (Fig. 12.4)

\[y^{\prime}_{0}=\frac{cp_{\perp}}{\beta E_{\mathrm{s}}}. \tag{12.64}\]

Energy is transferred from the accelerating cavity to the particle at the rate of the synchrotron radiation power \(P_{\gamma}\) and the particle energy increases in the cavity of length \(\Delta z\) from \(E_{\mathrm{s}}\) to \(E_{\mathrm{s}}+P_{\gamma}\frac{\Delta z}{\beta\epsilon}\) and the slope of the particle trajectory becomes at the exit of the cavity of length \(\Delta z\) due to this acceleration

\[y^{\prime}_{1}=\frac{cp_{\perp}}{\beta E_{\mathrm{s}}+P_{\gamma}\frac{\Delta z }{c}}\approx\frac{cp_{\perp}}{\beta E_{\mathrm{s}}}\left(1-\frac{P_{\gamma}}{ \beta E_{\mathrm{s}}}\frac{\Delta z}{c}\right). \tag{12.65}\]We are now in a position to express the functions \(f\) and \(g\) in terms of physical parameters. The function \(f\) is expressed by

\[f=\frac{\Delta w}{\Delta\varphi}=\frac{y_{1}-y_{0}}{\sqrt{\beta_{y}}\Delta\varphi }=\frac{y_{0}^{\prime}}{\sqrt{\beta_{y}}}\frac{\Delta z}{\Delta\varphi}=\nu\, \sqrt{\beta_{y}}y_{0}^{\prime}, \tag{12.66}\]

where we made use of \(\Delta\varphi=\Delta z/(\nu\beta)\). The damping decrement will depend on the derivation \(\frac{\mathrm{d}f}{\mathrm{d}w}\) which can be seen from (12.66) to vanish since \(f\) does not depend on \(w\)

\[\frac{\partial f}{\partial w}=0. \tag{12.67}\]

The variation of the conjugate variable \(p\) with phase is from (12.62)

\[\frac{\Delta p}{\Delta\varphi}=\frac{\frac{\mathrm{d}w_{1}}{\mathrm{d}\varphi }-\frac{\mathrm{d}w_{0}}{\mathrm{d}\varphi}}{\nu\,\Delta\varphi}. \tag{12.68}\]

From linear beam dynamics, we find

\[\frac{\mathrm{d}w_{1}}{\mathrm{d}\varphi}-\frac{\mathrm{d}w_{0}}{\mathrm{d} \varphi}=\sqrt{\beta_{y}}(y_{1}^{\prime}-y_{0}^{\prime})-\frac{1}{2}\frac{ \beta_{y}^{\prime}}{\sqrt{\beta_{y}}}(y_{1}-y_{0}) \tag{12.69}\]

and get with (12.65), (12.66)

\[g(w,p,\varphi)=\frac{\Delta p}{\Delta\varphi}=\frac{-\sqrt{\beta_{y}}\frac{p_ {y}}{\beta cE_{\mathrm{s}}}\Delta xy_{0}^{\prime}+F(y)}{\nu\Delta\varphi}. \tag{12.70}\]

The function \(F(y)\) is a collection of \(y\)-dependent terms that become irrelevant for our goal. Damping will be determined by the value of the derivative \(\frac{\partial g}{\partial p}\) which with \(y_{0}^{\prime}=\frac{1}{\sqrt{\beta_{y}}}\frac{\mathrm{d}w}{\mathrm{d}\varphi }+\frac{1}{2}\beta_{y}^{\prime}\frac{1}{\beta_{y}}y_{0}\) becomes

\[\frac{\partial g}{\partial p}=\nu\,\frac{\partial g}{\partial\frac{\mathrm{d} w}{\mathrm{d}\varphi}}=\frac{P_{y}}{\beta cE_{\mathrm{s}}}\frac{\Delta z}{ \Delta\varphi}. \tag{12.71}\]

In the derivation of (12.71) we have used the betatron phase as the "time" and get therefore the damping per unit betatron phase advance. Transforming to the real time with \(\frac{\Delta z}{\beta c\,\Delta\varphi}=\frac{T_{\mathrm{rev}}}{2\pi}\) and (12.41)

\[\frac{\partial g}{\partial p}=\frac{P_{\gamma}}{E_{\mathrm{s}}}\frac{T_{ \mathrm{rev}}}{2\pi}=-2\alpha_{y}\frac{T_{\mathrm{rev}}}{2\pi} \tag{12.72}\]and solving for the vertical damping decrement

\[\alpha_{y}=-\frac{\langle P_{\gamma}\rangle}{2E_{\mathrm{s}}}. \tag{12.73}\]

In this last equation, we have used the average synchrotron radiation power which is the appropriate quantity in case of a non-isomagnetic ring. The damping of the vertical betatron function is proportional to the synchrotron radiation power. This fact can be used to increase damping when so desired by increasing the synchrotron radiation power from special magnets in the lattice structure.

##### Robinson's Damping Criterion

The general motion of charged particles extends over all six degrees of freedom in phase space and therefore the particle motion is described in six-dimensional phase space as indicated in the general Vlasov equation (12.17). It is, however, a fortunate circumstance that it is technically possible to construct accelerator components in such a fashion that there is only little or no coupling between different pairs of conjugate coordinates. As a consequence, we can generally treat horizontal betatron oscillations separate from the vertical betatron oscillations and both of them separate from synchrotron oscillations. Coupling effects that do occur will be treated as perturbations. There is some direct coupling via the dispersion function between synchrotron and particularly the horizontal betatron oscillations but the frequencies are very different with the synchrotron oscillation frequency being in general much smaller than the betatron oscillation frequency. Therefore in most cases the synchrotron oscillation can be ignored while discussing transverse oscillations and we may average over many betatron oscillations when we discuss synchrotron motion.

A special property of particle motion in six-dimensional phase space must be introduced allowing us to make general statements about the overall damping effects in a particle beam. We start from the Vlasov equation (12.17)

\[\frac{\partial\Psi}{\partial t}\,+\mathbf{f}\,\nabla_{r}\Psi\,+\mathbf{g}\nabla_{p} \Psi\,=-\left(\nabla_{r}\mathbf{f}+\nabla_{p}\mathbf{g}\right)\Psi \tag{12.74}\]

and define a total damping decrement \(\alpha_{\mathrm{t}}\) by setting

\[\nabla_{\mathbf{f}}\mathbf{f}+\nabla_{p}\mathbf{g}=-2\alpha_{\mathrm{t}}\,. \tag{12.75}\]

The total damping decrement is related to the individual damping decrements of the transverse and longitudinal oscillations but the precise dependencies are not yet obvious. In the derivation of (12.17), we have expanded the functions \(f\) and \(g\) in a Taylor series neglecting all terms of second or higher order in time and got as a result the simple expression (12.75) for the overall damping. Upon writing (12.75) in component form, we find from the components of the l.h.s. that the overall damping decrement \(\alpha_{\rm t}\) is just the sum of all three individual damping decrements and we may therefore set

\[\nabla_{r}\boldsymbol{f}+\nabla_{p}\boldsymbol{g}=-2\alpha_{\rm t}=-2(\alpha_{x} +\alpha_{y}+\alpha_{\epsilon})\,. \tag{12.76}\]

From this equation and the linearity of the functions \(\boldsymbol{f}\) and \(\boldsymbol{g}\) describing the physics of the dynamical system general characteristics of the damping process can be derived. The damping decrement does not depend on the dynamic variables of the particles and coupling terms do not contribute to damping. The damping rate is therefore the same for all particles within a beam. In the following paragraphs, we will discuss in more detail the general characteristics of synchrotron radiation damping. Specifically, we will determine the functions \(\boldsymbol{f}\) and \(\boldsymbol{g}\) and derive an expression for the total damping.

We consider a small section of a beam transport line or circular accelerator including all basic processes governing the particle dynamics. These processes are focusing, emission of photons and acceleration. All three processes are assumed to occur evenly along the beam line. The six-dimensional phase space to be considered is

\[(x,x^{\prime},y,y^{\prime},\tau,\epsilon)\,. \tag{12.77}\]

During the short time \(\Delta t\) some of the transverse coordinates change and it is those changes that determine eventually the damping rate. Neither the emission of a synchrotron radiation photon nor the absorption of energy in the accelerating cavities causes any change in the particle positions \(x,y\), and \(\tau\). Indicating the initial coordinates by the index \({}_{0}\) and setting \(\beta c\,\Delta t=\Delta z\) we get for the evolution of the particle positions within the length element \(\Delta z\) in the three space dimensions

\[x =x_{0}+x^{\prime}_{0}\,\Delta z\,,\] \[y =y_{0}+y^{\prime}_{0}\,\Delta z\,, \tag{12.78}\] \[\tau =\tau_{0}+\eta_{\rm c}\frac{\epsilon_{0}}{E_{\rm s}}\frac{ \Delta z}{\beta c}\,.\]

The conjugate coordinates vary in a somewhat more complicated way. First we note that the Vlasov equation does not require the conjugate coordinates to be canonical variables. Indeed this derivation will become simplified if we do not use canonical variables but use the slopes of the particle trajectories with the reference path and the energy deviation. The change of the slopes due to focusing is proportional to the oscillation amplitude and vanishes on average. Emission of a synchrotron radiation photon occurs typically within an angle of \(\pm 1/\gamma\) causing a small transverse kick to the particle trajectory. In general, however, this transverse kick will be very small and we may assume for all practical purposes the slope of the transverse trajectory not to be altered by photon emission. Forces parallel to the direction of propagation of the particles can be created, however, through the emission of synchrotron radiation photons. In this case, the energy or energy deviation of the particle will be changed like

\[\epsilon=\epsilon_{0}-P_{\gamma}\frac{\Delta z}{\beta c}+P_{\rm rf}\frac{\Delta z }{\beta c}\,. \tag{12.79}\]

Here we use the power \(P_{\gamma}\) to describe the synchrotron radiation energy loss rate a particle may suffer during the time \(\beta c\Delta t=\Delta z\). No particular assumption has been made about the nature of the energy loss except that during the time \(\Delta t\) it be small compared to the particle energy. To compensate this energy loss the particles become accelerated in rf-cavities. The power \(P_{\rm rf}\) is the energy flow from the cavity to the particle beam, not to be confused with the total power the rf-source delivers to the cavity.

The transverse slopes \(x^{\prime}\) and \(y^{\prime}\) are determined by the ratio of the transverse to the longitudinal momentum \(u^{\prime}=p_{u}/p_{z}\) where \(u\) stands for \(x\) or \(y\), respectively. During acceleration in the rf-cavity the transverse momentum does not change but the total kinetic energy increases from \(E_{\rm s}\) to \(E_{\rm s}+P_{\rm rf}\frac{\Delta z}{\beta c}\) and the transverse slope of the trajectory is reduced after a distance \(\Delta z\) to

\[u^{\prime}=\frac{cp_{u}}{cp_{z}+P_{\rm rf}\beta\frac{\Delta z}{\beta c}}\approx u _{0}^{\prime}-\frac{P_{\rm rf}}{E_{\rm s}}\frac{\Delta z}{\beta c}u_{0}^{ \prime}\,. \tag{12.80}\]

Explicitly, the transverse slopes vary now like

\[\begin{split} x^{\prime}&=x_{0}^{\prime}-\frac{P_{ \rm rf}}{E_{\rm s}}\frac{\Delta z}{\beta c}x_{0}^{\prime}\,,\\ y^{\prime}&=y_{0}^{\prime}-\frac{P_{\rm rf}}{E_{ \rm s}}\frac{\Delta z}{\beta c}y_{0}^{\prime}\,.\end{split} \tag{12.81}\]

All ingredients are available now to formulate expressions for the functions \(\mathbf{f}\) and \(\mathbf{g}\) in component form

\[\begin{split}\mathbf{f}&=\left(x_{0}^{\prime},\,y_{0}^{ \prime},\,\eta_{\rm c}\frac{\epsilon}{E_{\rm s}}\right),\\ \mathbf{g}&=\left(-\frac{P_{\rm rf}}{E_{\rm s}}x_{0}^{ \prime},-\frac{P_{\rm rf}}{E_{\rm s}}y_{0}^{\prime},-P_{\gamma}\,+\,P_{\rm rf} \right).\end{split} \tag{12.82}\]

With these expressions we evaluate (12.76) and find that \(\nabla_{\mathbf{f}}\mathbf{f}=0\). For the determination of \(\nabla_{\mathbf{p}}\mathbf{g}\) we note that the power \(P_{\rm rf}\) from the cavity is just equal to the average radiation power \(\left\langle P_{\gamma}\right\rangle\) and the derivative of the radiation power with respect to the particle energy is

\[-\frac{\partial P_{\gamma}}{\partial\epsilon}=-2\frac{P_{\gamma}}{E_{\rm s}}. \tag{12.83}\]Finally, we note that the rf-power \(P_{\rm rf}\) is equal to the average radiation power \(\left\langle P_{\gamma}\right\rangle\) and get from (12.76)

\[\alpha_{x}+\alpha_{y}+\alpha_{\epsilon}=2\frac{\left\langle P_{\gamma}\right\rangle }{E_{\rm s}}. \tag{12.84}\]

The sum of all damping decrements is a constant, a result which has been derived first by Robinson [1] and is known as the Robinson criterion. The total damping depends only on the synchrotron radiation power and the particle energy and variations of magnetic field distribution in the ring keeping the radiation power constant will not affect the total damping rate but may only shift damping from one degree of freedom to another.

##### Damping of Horizontal Betatron Oscillations

With the help of the Robinson criterion, the damping decrement for the horizontal betatron oscillation can be derived by simple subtraction. Inserting (12.61), (12.75) into (12.84) and solving for the horizontal damping decrement we get

\[\alpha_{x}=\frac{\left\langle P_{\gamma}\right\rangle}{2\,E_{\rm s}}(1-\vartheta). \tag{12.85}\]

The damping decrements derived from the Vlasov equation agree completely with the results obtained in Sect. 11.2 by very different means.

No matter what type of magnet lattice we use, the total damping depends only on the synchrotron radiation power and the particle energy. We may, however, vary the distribution of the damping rates through the \(\vartheta\)-parameter to different oscillation modes by proper design of the focusing and bending lattice in such a way that one damping rate is modified in the desired way limited only by the onset of anti-damping in another mode. Specifically, this is done by introducing gradient bending magnets with a field gradient such as to produce the desired sign of the \(\vartheta\) parameter.

### The Fokker-Planck Equation

From the discussions of the previous section it became clear that the Vlasov equation is a useful tool to determine the evolution of a multiparticle system under the influence of forces depending on the physical parameters of the system through differentiable functions. If, however, the dynamics of a system in phase space depends only on its instantaneous physical parameters where the physics of the particle dynamics cannot be expressed by differentiable functions, the Vlasovequation will not be sufficient to describe the full particle dynamics. A process which depends only on the state of the system at the time \(t\) and not on its history is called a Markoff process.

In particle beam dynamics we have frequently the appearance of such processes where forces are of purely statistical nature like those caused, for example, by the quantized emission of synchrotron radiation photons or by collisions with other particles within the same bunch or residual gas atoms. To describe such a situation we still have variations of the coordinates per unit time similar to those in (12.2) but we must add a term describing the statistical process and we set therefore

\[\dot{w}= f_{w}(w,p_{w},t)+\sum\dot{\xi}_{i}\ \delta(t-t_{i})\, \tag{12.86}\] \[\dot{p}_{w}= g_{w}(w,p_{w},t)+\sum\pi_{i}\ \delta(t-t_{i})\, \tag{12.87}\]

where \(\dot{\xi}_{i}\) and \(\pi_{i}\) are instantaneous statistical changes in the variables \(w\) and \(p_{w}\) with a statistical distribution in time \(t_{i}\) and where \(\delta(t-t_{i})\) is the Dirac delta function. The probabilities \(P_{w}(\dot{\xi})\) and \(P_{p}(\pi)\) for statistical occurrences with amplitudes \(\dot{\xi}\) and \(\pi\) be normalized and centered

\[\begin{array}{ll}\int P_{w}(\dot{\xi})\,\mathrm{d}\dot{\xi}=1\,&\int P_{w}(\dot{\xi})\dot{\xi}\ \mathrm{d}\dot{\xi}=0\,\\ \int P_{p}(\pi)\,\mathrm{d}\pi=1\,&\int P_{p}(\pi)\pi\ \mathrm{d}\pi=0\.\end{array} \tag{12.88}\]

The first equations normalize the probability amplitudes and the second equations are true for symmetric statistical processes. The sudden change in the amplitude by \(\Delta w_{i}\) or in momentum by \(\Delta p_{wi}\) due to one such process is given by

\[\Delta w_{i} =\int\dot{\xi}_{i}\ \delta(t-t_{i})\,\mathrm{d}t=\dot{\xi}_{i}\,, \tag{12.89a}\] \[\Delta p_{wi} =\int\pi_{i}\ \delta(t-t_{i})\,\mathrm{d}t=\pi_{i}\,. \tag{12.89b}\]

Analogous to the discussion of the evolution of phase space in the previous section, we will now formulate a similar evolution including statistical processes. At the time \(t+\Delta t\), the particle density in phase space is taken to be \(\Psi(w,p_{w},t+\Delta t)\) and we intend to relate this to the particle density at time \(t\). During the time interval \(\Delta t\) there are finite probabilities \(P_{w}(\dot{\xi})\), \(P_{p}(\pi)\) that the amplitude (\(w-\dot{\xi}\)) or the momentum (\(p_{w}-\pi\)) be changed by a statistical process to become \(w\) or \(p_{w}\) at time \(t\). This definition of the probability function also covers the cases where particles during the time \(\Delta t\) either jump out of the phase space area \(\Delta A_{P}\) or appear in the phase space area \(\Delta A_{Q}\).

To determine the number of particles ending up within the area \(\Delta A_{Q}\), we look at all area elements \(\Delta A_{P}\) which at time \(t\) are a distance \(\Delta w=\dot{\xi}\) and \(\Delta p_{w}\), \(=\pi\) away from the final area element \(\Delta A_{Q}\) at time \(t+\Delta t\). As a consequence of our assumption that the particle density is only slowly varying in phase space, we may assume that the density \(\Psi\) is uniform within the area elements \(\Delta A_{P}\) eliminating the need for a local integration. We may now write down the expression for the phase space element and the particle density at time \(t+\Delta t\) by integrating over all values of \(\xi\) and \(\pi\)

\[I=\Delta A_{P}\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}\Psi(w-\xi,p_{w}- \pi,t)\,P_{w}(\xi)\,P_{p}(\pi)\,\mathrm{d}\xi\mathrm{d}\pi\, \tag{12.90}\]

where we used the abbreviation \(I=\Psi(w+f_{w}\,\Delta t,\,p_{w}+g_{w}\,\Delta t,\,t+\Delta t)\)\(\Delta A_{Q}\). The volume elements \(\Delta A_{P}\) and \(\Delta A_{Q}\) are given by (12.5), (12.6), respectively. The statistical fluctuations may in general be of any magnitude. In particle beam dynamics, however, we find that the fluctuations with reasonable probabilities are small compared to the values of the variables \(w\) and \(p_{w}\). The phase space density can therefore be expanded into a Taylor series where we retain linear as well as quadratic terms in \(\xi\) and \(\pi\)

\[\Psi(w-\xi,p_{w}-\pi,t) =\Psi_{0}-\xi\frac{\partial\Psi_{0}}{\partial w}-\pi\,\frac{ \partial\Psi_{0}}{\partial p_{w}} \tag{12.91}\] \[+\,\tfrac{1}{2}\xi^{2}\frac{\partial^{2}\Psi_{0}}{\partial w^{2} }+\tfrac{1}{2}\pi^{2}\frac{\partial^{2}\Psi_{0}}{\partial p_{w}^{2}}+\xi\pi\, \frac{\partial^{2}\Psi_{0}}{\partial w\partial p_{w}}\,\]

where \(\Psi_{0}=\Psi(w,p_{w},t)\) and we finally get for the integrals with (12.88)

\[I=\Psi_{0}+\,\tfrac{1}{2}\frac{\partial^{2}\Psi_{0}}{\partial w^{2}}\int\xi^{ 2}P_{w}(\xi)\,\mathrm{d}\xi\,+\,\tfrac{1}{2}\frac{\partial^{2}\Psi_{0}}{ \partial p_{w}^{2}}\int\pi^{2}P_{p}(\pi)\,\mathrm{d}\pi. \tag{12.92}\]

For simplicity, we leave off the integration limits which are still from \(-\infty\) to \(+\infty\). If we now set \(\mathcal{N}\) to be the number of statistical occurrences per unit time we may simplify the quadratic terms on the r.h.s. of (12.92) by setting

\[\tfrac{1}{2}\!\int\,\xi^{2}P_{w}(\xi)\,\mathrm{d}\xi =\,\tfrac{1}{2}\left\langle\mathcal{N}_{\xi}\xi^{2}\right\rangle \Delta t\,, \tag{12.93}\] \[\tfrac{1}{2}\!\int\pi^{2}P_{p}(\pi)\,\mathrm{d}\pi =\,\tfrac{1}{2}\left\langle\mathcal{N}_{\pi}\pi^{2}\right\rangle \Delta t\,, \tag{12.94}\]

and get similarly to the derivation of the Vlasov equation in Sect. 12.1

\[\frac{\partial\Psi_{0}}{\partial t}+f_{w}\frac{\partial\Psi_{0}} {\partial w}+g_{w}\frac{\partial\Psi_{0}}{\partial p_{w}} =-\left(\frac{\partial f_{w}}{\partial w}+\frac{\partial g_{w}} {\partial p_{w}}\right)\Psi_{0} \tag{12.95}\] \[+\,\tfrac{1}{2}\left\langle\mathcal{N}_{\xi}\xi^{2}\right\rangle \frac{\partial^{2}\Psi_{0}}{\partial w^{2}}+\,\tfrac{1}{2}\left\langle \mathcal{N}_{\pi}\pi^{2}\right\rangle\frac{\partial^{2}\Psi_{0}}{\partial p_{w }^{2}}\.\]This partial differential equation is identical to the Vlasov equation except for the statistical excitation terms and is called the Fokker-Planck equation [2]. We define diffusion coefficients describing the flow in \(\xi\) and \(\pi\) space by

\[D_{\xi} = {\frac{1}{2}}\left\{\mathcal{N}_{\sim}\xi^{2}\right\}\,, \tag{12.96}\] \[D_{\pi} = {\frac{1}{2}}\left\{\mathcal{N}_{\approx}\pi^{2}\right\}\,, \tag{12.97}\]

and the Fokker-Planck equation becomes finally

\[\frac{\partial\Psi}{\partial t}\,+f_{w}\frac{\partial\Psi}{\partial w}\,+g_{w }\frac{\partial\Psi}{\partial p_{w}}\,=\,2\alpha_{w}\Psi\,+D_{\xi}\,\frac{ \partial^{2}\Psi}{\partial w^{2}}\,+D_{\pi}\,\frac{\partial^{2}\Psi}{\partial p _{w}^{2}}\,\,. \tag{12.98}\]

For the case of damped oscillators the Fokker-Planck equation can be derived similar to (12.35) and is

\[\frac{\partial\Psi}{\partial t}+\omega_{0}p_{w}\frac{\partial\Psi}{\partial w }-(\omega_{0}w+2\alpha_{w}p_{w})\frac{\partial\Psi}{\partial p_{w}}\,=\,2 \alpha_{w}\Psi+D_{\xi}\,\frac{\partial^{2}\Psi}{\partial w^{2}}\,+D_{\pi}\, \frac{\partial^{2}\Psi}{\partial p_{w}^{2}}\,. \tag{12.99}\]

This form of the Fokker-Planck equation will be very useful to describe a particle beam under the influence of diffusion processes. In the following section, we will derive general solutions which will be applicable to specific situations in accelerator physics.

##### Stationary Solution of the Fokker-Planck Equation

A unique stationary solution exists for the particle density distribution described by the partial differential equation (12.98). To derive this solution we transform (12.98) to cylindrical coordinates \((w,p_{w})\to(r,\theta)\) with \(w=r\cos\theta\) and \(p_{w}=r\sin\theta\) and note terms proportional to derivatives of the phase space density with respect to the angle \(\theta\). One of these terms \(\omega_{0}\Psi_{\theta}\) exists even in the absence of diffusion and damping and describes merely the betatron motion in phase space while the other terms depend on damping and diffusion. The diffusion terms will introduce a statistical mixing of the phases \(\theta\) and after some damping times any initial azimuthal variation of the phase space density will be washed out. We are here only interested in the stationary solution and therefore set all derivatives of the phase space density with respect to the phase \(\theta\) to zero. In addition we find it necessary to average square terms of \(\cos\theta\) and \(\sin\theta\). With these assumptions the Fokker-Planck Equation (12.98) becomes after some manipulations in the new coordinates

\[\frac{\mathrm{d}\Psi}{\mathrm{d}t}\,=\,2\alpha_{w}\Psi\,+\left(\alpha_{w}r+ \frac{D}{r}\right)\frac{\partial\Psi}{\partial r}\,+\,D\frac{\partial^{2} \Psi}{\partial r^{2}}\,\,, \tag{12.100}\]where we have defined a total diffusion coefficient

\[D=\tfrac{1}{2}(D_{\xi}+D_{\pi})\,. \tag{12.101}\]

Equation (12.100) has some similarity with, for example, wave equations in quantum mechanics which are solved by the method of separation of variables and we expect the stationary solution for the phase space density to be of the form \(\Psi(r,t)=\sum_{n}F_{n}(t)\,G_{n}(r)\). The solution \(G_{n}(r)\) must meet some particular boundary conditions. Specifically, at time \(t=0\), we may have any arbitrary distribution of the phase space density \(G_{n0}(r)\). Furthermore, we specify that there be a wall at \(r=R\) beyond which the phase space density drops to zero and consequently, the boundary conditions are

\[\begin{split} G_{n}(r<R)&=\,G_{n0}(r)\,,\\ G_{n}(r>R)&=\,0\,.\end{split} \tag{12.102}\]

By the method of separation of the constants we find for the functions \(F_{n}(t)\)

\[F_{n}(t)=\text{const.}\,\mathrm{e}^{-\alpha_{n}\,t}\,, \tag{12.103}\]

where the quantity \(-\,\alpha_{n}\) is the separation constant. The general form of the solution for (12.100) may now be expressed by a series of orthogonal functions or eigenmodes of the distribution \(G_{n}(r)\) which fulfill the boundary conditions (12.102)

\[\Psi(r,t)=\sum_{n\geq 0}c_{n}G_{n}(r)\,\mathrm{e}^{-\alpha_{n}\,t}\,. \tag{12.104}\]

The amplitudes \(c_{n}\) in (12.104) are determined such as to fit the initial density distribution

\[\Psi_{0}(r,t=0)=\sum_{n\geq 0}c_{n}G_{n0}(r)\,. \tag{12.105}\]

With the ansatz (12.104) we get from (12.100) for each of the eigenmodes the following second-order differential equation:

\[\frac{\partial^{2}G_{n}}{\partial r^{2}}+\left(\frac{1}{r}+\frac{\alpha_{w}}{ D}r\right)\frac{\partial G_{n}}{\partial r}+\frac{\alpha_{w}}{D}\left(2+\frac{ \alpha_{n}}{\alpha_{w}}\right)G_{n}=0\,. \tag{12.106}\]

All terms with a coefficient \(\alpha_{n}>0\) vanish after some time due to damping (12.103). Negative values for the damping decrements \(\alpha_{n}<0\) define instabilities which we will not consider here. Stationary solutions, therefore require the separation constants to be zero \(\alpha_{n}=0\). Furthermore, all solutions \(G_{n}\) must vanish at the boundary \(r=R\) where \(R\) may be any value including infinity if there are no physical boundaries at all to limit the maximum particle oscillation amplitude. In the latter case where there are no walls, the differential equation (12.106) can be solved by the stationary distribution

\[\Psi(r,t)=\sum_{\begin{subarray}{c}n\geq 0\\ a_{n}=0\end{subarray}}c_{n}\,G_{n}(r)\propto\exp\left(-\frac{\alpha_{w}}{2D}r^{ 2}\right)\, \tag{12.107}\]

which can easily be verified by backinsertion into (12.106). The solution for the particle distribution in phase space under the influence of damping \(\alpha_{w}\) and statistical fluctuations \(D\) is a Gaussian distribution with the standard width

\[\sigma_{r}=\sqrt{\frac{D}{\alpha_{w}}}. \tag{12.108}\]

Normalizing the phase space density the stationary solution of the Fokker-Planck equation for a particle beam under the influence of damping and statistical fluctuations is

\[\Psi(r)=\frac{1}{\sqrt{2\pi}\sigma_{r}}\mathrm{e}^{-r^{2}/2\sigma_{r}^{2}}. \tag{12.109}\]

Eigenfunctions for which the eigenvalues \(\alpha_{n}\) are not zero, are needed to describe an arbitrary particle distribution, e.g., a rectangular distribution at time \(t=0\). The Fokker-Planck equation, however, tells us that after some damping times these eigensolutions have vanished and the Gaussian distribution is the only stationary solution left. The Gaussian distribution is not restricted to the \(r\)-space alone. The particle distribution in equilibrium between damping and fluctuations is also Gaussian in the normalized phase space \((w,p_{w})\) as well as in real space. With \(r^{2}=w^{2}+p_{w}^{2}\) we get immediately for the density distribution in \((w,p_{w})\)-space

\[\Psi(w,p_{w})=\frac{1}{2\pi\sigma_{w}\sigma_{p_{w}}}\ \mathrm{e}^{-w^{2}/2 \sigma_{w}^{2}}\ \mathrm{e}^{-p_{w}^{2}/2\sigma_{p_{w}}^{2}}\, \tag{12.110}\]

where we have set \(\sigma_{w}=\sigma_{p_{w}}=\sqrt{\frac{D}{\alpha_{w}}}\). The standard deviation in \(w\) and \(p_{w}\) is the same as for \(r\) which is to be expected since all three quantities have the same dimension and are linearly related.

In real space we have for \(u=x\) or \(y\) by definition \(u=\sqrt{\beta_{u}}w\) and \(p=\frac{\dot{w}}{v}\) where \(\dot{w}=\frac{\mathrm{d}w}{\mathrm{d}\varphi}\). On the other hand, \(p=\sqrt{\beta_{x}}x^{\prime}-\frac{\beta^{\prime}}{2\sqrt{\beta}}x\) and inserted into (12.107) we get the density distribution in real space

\[\Psi(u,\ u^{\prime})\propto\exp\left(-\frac{\gamma_{u}u^{2}-\beta_{u}^{\prime }\ uu^{\prime}+\beta_{u}u^{\prime 2}}{2\,\sigma_{w}^{2}}\right). \tag{12.111}\]This distribution describes the particle distribution in real phase space where particle trajectories follow tilted ellipses. Note that we carefully avoid replacing the derivative of the betatron function with \(\beta^{\prime}=-2\alpha\) because this would lead to a definite confusion between the damping decrement and the betatron function. To further reduce confusion we also use the damping times \(\tau_{i}=\alpha_{i}^{-1}\). Integrating the distribution (12.111) for all values of the angles \(u^{\prime}\), for example, gives the particle distribution in the horizontal or vertical midplane. Using the mathematical relation \(\int\infty\)e\({}^{-p^{2}x^{2}\pm\alpha x}\mathrm{d}x=\frac{\sqrt{\pi}}{p}\) e\({}^{q^{2}/(4\rho^{2})}\)[3], we get

\[\Psi(\ u)=\frac{1}{\sqrt{2\ \pi}\sqrt{\beta_{u}\sigma_{w}}}\mathrm{e}^{-u^{2}/2 \sigma_{u}^{2}}, \tag{12.112}\]

where the standard width of the horizontal Gaussian particle distribution is

\[\sigma_{u}=\sqrt{\beta}\sigma_{w}=\sqrt{\beta}\sqrt{\tau_{u}D_{u}}. \tag{12.113}\]

The index \({}_{u}\) has been added to the diffusion and damping terms to indicate that these quantities are in general different in the horizontal and vertical plane. The damping time depends on all bending magnets, vertical and horizontal, but only on the damping-partition number for the plane under consideration. Similar distinction applies to the diffusion term.

In a similar way, we get the distribution for the angles by integrating (12.111) with respect to \(u\)

\[\Psi(u^{\prime})=\frac{\sqrt{\beta}}{\sqrt{2\pi}\sqrt{1+\frac{1}{4}\ \beta^{\prime\ 2}} \sigma_{w}}\exp\left[-\frac{\beta\ u^{\prime\ 2}}{2(1+\frac{1}{4}\ \beta^{\prime\ 2})\, \sigma_{w}^{2}}\right], \tag{12.114}\]

where the standard width of the angular distribution is

\[\sigma_{u}^{\prime}=\sqrt{\frac{4+\ \beta^{\prime\ 2}}{4\beta}}\sigma_{w}=\sqrt{ \frac{4+\ \beta^{\prime\ 2}}{4\beta}}\sqrt{\tau_{u}D_{u}}. \tag{12.115}\]

We have not made any special assumption as to the horizontal or vertical plane and find in (12.112)-(12.115) the solutions for the particle distribution in both planes.

In the longitudinal phase space the equations of motion are mathematically equal to Eq. (12.11). First we define new variables

\[\dot{w}=-\frac{\Omega_{\mathrm{s}0}}{\eta_{\mathrm{c}}}\dot{t}, \tag{12.116}\]where \(\Omega_{\rm s0}\) is the synchrotron oscillation frequency, \(\eta_{\rm c}\) the momentum compaction and \(\tau\) the longitudinal deviation of a particle from the reference particle. The conjugate variable we define by

\[p=-\frac{\dot{\epsilon}}{E_{0}}, \tag{12.117}\]

where \(\epsilon\) is the energy deviation from the reference energy \(E_{0}\). After differentiation of (12.52) and making use of (12.53) and the definition of the synchrotron oscillation frequency, we use these new variables and obtain the two first-order differential equations

\[\dot{w} = +\Omega_{\rm s}p, \tag{12.118}\] \[\dot{p} = -\Omega_{\rm s}w-2\alpha_{\epsilon}p. \tag{12.119}\]

These two equations are of the same form as (12.11) and the solution of the longitudinal Fokker-Planck equation is therefore similar to (12.112)-(12.115). The energy distribution within a particle beam under the influence of damping and statistical fluctuations becomes with \(p=\delta=\epsilon/E_{0}\)

\[\Psi(\delta)=\frac{1}{\sqrt{2\pi}\sigma_{\delta}}{\rm e}^{-\delta^{2}/2\sigma _{\delta}^{2}}, \tag{12.120}\]

where the standard value for the energy spread in the particle beam is defined by

\[\frac{\sigma_{\epsilon}}{E_{0}}=\sqrt{\tau_{\epsilon}D_{\epsilon}}. \tag{12.121}\]

In a similar way, we get for the conjugate coordinate \(\tau\) with \(w=\frac{\Omega_{\rm s}}{\eta_{\rm c}}\tau\) the distribution

\[\Psi(\tau)=\frac{1}{\sqrt{2\pi}\sigma_{\tau}}{\rm e}^{-\tau^{2}/2\sigma_{ \tau}^{2}}. \tag{12.122}\]

The standard width of the longitudinal particle distribution is finally

\[\sigma_{\tau}=\frac{|\eta_{\rm c}|}{\Omega_{\rm s}}\sqrt{\tau_{\epsilon}D_{ \epsilon}}. \tag{12.123}\]

The deviation in time \(\tau\) of a particle from the synchronous particle is equivalent to the distance of these two particles and we may therefore define the standard value for the bunch length from (12.123) by

\[\sigma_{\ell}=c\beta\frac{|\eta_{\rm c}|}{\Omega_{\rm s}}\sqrt{\tau_{\epsilon }D_{\epsilon}}. \tag{12.124}\]By application of the Fokker-Planck equation to systems of particles under the influence of damping and statistical fluctuations, we were able to derive expressions for the particle distribution within the beam. In fact, we were able to determine that the particle distribution is Gaussian in all six degrees of freedom. Since such a distribution does not exhibit any definite boundary for the beam, it becomes necessary to define the size of the distributions in all six degrees of freedom by the standard value of the Gaussian distribution. Specific knowledge of the nature for the statistical fluctuations are required to determine the numerical values of the beam sizes.

In Chap. 13 we will apply these results to determine the equilibrium beam emittance in an electron positron storage ring where the statistical fluctuations are generated by quantized emission of synchrotron radiation photons.

#### Particle Distribution within a Finite Aperture

The particle distribution in an electron beam circulating in a storage ring is a Gaussian if we ignore the presence of walls containing the beam. All other modes of particle distribution are associated with a finite damping time and vanish therefore after a short time. In a real storage ring we must, however, consider the presence of vacuum chamber walls which cut off the Gaussian tails of the particle distribution. Although the particle intensity is very small in the far tails of a Gaussian distribution, we cannot cut off those tails too tight without reducing significantly the beam lifetime. Due to quantum excitation, we observe a continuous flow of particles from the beam core into the tails and back by damping toward the core. A reduction of the aperture into the Gaussian distribution absorbs therefore not only those particles which populate these tails at a particular moment but also all particles which reach occasionally large oscillation amplitudes due to the emission of a high energy photon. The absorption of particles due to this effect causes a reduction in the beam lifetime which we call the quantum lifetime.

The presence of a wall modifies the particle distribution especially close to the wall. This modification is described by normal mode solutions with a finite damping time which is acceptable now because any aperture less than an infinite aperture absorbs beam particles thus introducing a finite beam lifetime. Cutting off Gaussian tails at large amplitudes will not affect the Gaussian distribution in the core and we look therefore for small variations of the Gaussian distribution which become significant only quite close to the wall. Instead of (12.107) we try the ansatz

\[\Psi(r,t)={\rm e}^{-\frac{\alpha_{W}}{2D}r^{2}}g(r)\,{\rm e}^{-\alpha t}\,, \tag{12.125}\]

where \(1/\alpha\) is the time constant for the distribution, with the boundary condition that the particle density be zero at the aperture or acceptance defining wall \(r=A\) or

\[\Psi(A,t)=0\,. \tag{12.126}\]Equation (12.125) must be a solution of (12.100) and back insertion of (12.125) into (12.100) gives the condition on the function \(g(r)\)

\[g^{\prime\prime}+\left(\frac{1}{r}-\frac{r}{\sigma^{2}}\right)g^{\prime}+\frac{ \alpha}{\alpha_{w}\,\sigma^{2}}g=0. \tag{12.127}\]

Since \(g(r)=1\) in case there is no wall, we expand the correction into a power series

\[g(r)=1+\sum_{k\geq 1}C_{k}\,x^{k}\,,\qquad\mbox{where}\qquad x=\frac{r^{2}}{2 \sigma^{2}}. \tag{12.128}\]

Inserting (12.128) into (12.127) and collecting terms of equal powers in \(r\) we derive the coefficients

\[C_{k}=\frac{1}{(k!)^{2}}\prod_{p=1}^{p=k}(p-1-X)\approx-\frac{(k-1)!}{(k!)^{2} }X\,, \tag{12.129}\]

where \(X=\frac{\alpha}{2\alpha_{w}}\ \ll\ 1\). The approximation \(X\ll 1\) is justified since we expect the vacuum chamber wall to be far away from the beam center such that the expected quantum lifetime \(1/\alpha\) is long compared to the damping time \(1/\alpha_{w}\) of the oscillation under consideration. With these coefficients (12.128) becomes

\[g(r)=1-\frac{\alpha}{2\alpha_{w}}\sum_{k\geq 1}\frac{1}{k\,k!}x^{k}\,. \tag{12.130}\]

For \(x=A^{2}/(2\sigma^{2})\gg 1\) where \(A\) is the amplitude or amplitude limit for the oscillation \(w\), the sum in (12.130) can be replaced by an exponential function

\[\sum_{k\geq 1}\frac{1}{k\,k!}x^{k}\approx\frac{\mbox{e}^{x}}{x}. \tag{12.131}\]

From the condition \(g(A)=0\) we finally get for the quantum lifetime \(\tau_{\rm q}=1/\alpha\)

\[\tau_{\rm q}=\frac{1}{2}\tau_{w}\frac{\mbox{e}^{x}}{x}, \tag{12.132}\]

where

\[x=\frac{A^{2}}{2\sigma^{2}}. \tag{12.133}\]

The quantum lifetime \(\tau_{\rm q}\) is related to the damping time. To make the quantum life time very large of the order of 50 or more hours, the aperture must be at least about \(7\sigma_{w}\) in which case \(x=24.5\) and \(\mbox{e}^{x}/x=1.8\times 10^{9}\).

The aperture \(A\) is equal to the transverse acceptance of a storage ring for a one-dimensional oscillation like the vertical betatron oscillation while longitudinal or energy oscillations are limited through the maximum energy acceptance allowed by the rf-voltage. Upon closer look, however, we note a complication for horizontal betatron oscillations and synchrotron oscillations because of the coupling from energy oscillation into transverse position due to a finite dispersion function. We also have assumed that \(\alpha/(2\alpha_{w})\ \ll\ 1\) which is not true for tight apertures of less than one sigma. Both of these situations have been investigated in detail [4, 5] and the interested reader is referred to those references.

Specifically, if the acceptance \(A\) of a storage ring is defined at a location where there is also a finite dispersion function, Chao [4] derives a combined quantum lifetime of

\[\tau\,=\,\frac{\mathrm{e}^{\pi^{2}/2}}{\sqrt{2\pi}\alpha_{x}n^{3}}\frac{1}{(1+ r)\,\sqrt{r\,(1-r)}}\, \tag{12.134}\]

where \(n=A/\sigma_{\mathrm{ T}}\), \(\sigma_{\mathrm{ T}}^{2}=\sigma_{x}^{2}+\eta^{2}\sigma_{\delta}^{2}\), \(r=\eta^{2}\sigma_{\delta}^{2}/\sigma_{\mathrm{ T}}^{2}\), \(A\) the transverse aperture, \(\eta\) the dispersion function at the same location where the aperture is \(A\), \(\sigma_{x}\) the transverse beam size and \(\sigma_{\delta}=\sigma_{\epsilon}/E\) the standard relative energy width in the beam.

#### Particle Distribution in the Absence of Damping

To obtain a stationary solution for the particle distribution it was essential that there were eigensolutions with vanishing eigenvalues \(\alpha_{n}=0\). As a result, we obtained an equilibrium solution where the statistical fluctuations are compensated by damping. In cases where there is no damping, we would expect a different solution with particles spreading out due to the effect of diffusion alone. This case can become important in very high energy electron positron linear colliders where an extremely small beam emittance must be preserved along a long beam transport line. The differential equation (12.106) becomes in this case

\[\frac{\partial^{2}G_{n}}{\partial r^{2}}\,+\,\frac{1}{r}\frac{\partial G_{n}} {\partial r}\,+\,\frac{\alpha_{n}}{D}G_{n}=0. \tag{12.135}\]

We will assume that a beam with a Gaussian particle distribution is injected into a damping free transport line and we therefore look for solutions of the form

\[\Psi_{n}(r,t)\,=\,c_{n}G_{n}(r)\,\mathrm{e}^{-\alpha_{n}t}\,, \tag{12.136}\]

where

\[G_{n}(r)=\mathrm{e}^{-r^{2}/2\sigma_{0}^{2}} \tag{12.137}\]with \(\sigma_{0}\) being the beam size at \(t=0\). We insert (12.137) into (12.135) and obtain an expression for the eigenvalues \(\alpha_{n}\)

\[\alpha_{n}=\,\frac{2D}{\sigma_{0}^{2}}-\frac{D}{\sigma_{0}^{4}}\,r^{2}\,. \tag{12.138}\]

The time dependent solution for the particle distribution now becomes

\[\Psi(r,t)=A\,\exp\left(-\frac{2D}{\sigma_{0}^{2}}t\right)\exp\left[\left(-\frac {r^{2}}{2\sigma_{0}^{2}}\right)\left(1-\frac{2D}{\sigma_{0}^{2}}t\right) \right]\,. \tag{12.139}\]

Since nowhere a particular mode is used we have omitted the index \(n\). The solution (12.139) exhibits clearly the effect of the diffusion in two respects. The particle density decays exponentially with the decrement \(2D/\sigma_{0}^{2}\). At the same time the distribution remains to be Gaussian although being broadened by diffusion. The time dependent beam size \(\sigma\) is given by

\[\sigma^{2}(t)=\,\frac{\sigma_{0}^{2}}{1-\frac{2D}{\sigma_{0}^{2}}t}\approx \sigma_{0}^{2}\left(1+\frac{2D}{\sigma_{0}^{2}}t\right)\,, \tag{12.140}\]

where we have assumed that the diffusion term is small \((2D/\sigma_{0}^{2})t\ll 1\). Setting \(\sigma^{2}=\sigma_{u}^{2}=\epsilon_{u}\beta_{u}\) for the plane \(u\) where \(\beta_{u}\) is the betatron function at the observation point of the beam size \(\sigma_{u}\). The time dependent beam emittance is

\[\epsilon_{u}=\epsilon_{u0}+\,\frac{2D}{\beta_{u}}t \tag{12.141}\]

or the rate of change

\[\frac{\mathrm{d}\epsilon_{u}}{\mathrm{d}t}=\,\frac{2D}{\beta_{u}}=\,\frac{D_ {\xi}+D_{\pi}}{\beta_{u}}\,\,. \tag{12.142}\]

Due to the diffusion coefficient \(D\) we obtain a continuous increase of the beam emittance in cases where no damping is available.

The Fokker-Planck diffusion equation provides a tool to describe the evolution of a particle beam under the influence of conservative forces as well as statistical processes. Specifically, we found that such a system has a stationary solution in cases where there is damping. The stationary solution for the particle density is a Gaussian distribution with the standard width of the distribution \(\sigma\) given by the diffusion constant and the damping decrement.

In particular, the emission of photons due to synchrotron radiation has the properties of a Markoff process and we find therefore the particle distribution to be Gaussian. Indeed we will see that this is true in all six dimensions of phase space.

Obviously not every particle beam is characterized by the stationary solution of the Fokker-Planck equation. Many modes contribute to the particle distribution and specifically at time \(t=0\) the distribution may have any arbitrary form. However, it has been shown that after a time long compared to the damping times only one nontrivial stationary solution is left, the Gaussian distribution.

## Problems

**12.1 (S).**: Derive from the Vlasov equation an expression for the synchrotron frequency while ignoring damping. A second rf-system with different frequency can be used to change the synchrotron tune. Determine a system that would reduce the synchrotron tune for the reference particle to zero while providing the required rf-voltage at the synchronous phase. What is the relationship between both voltages and phases? Is the tune shift the same for all particles?
**12.2 (S).**: Formulate an expression for the equilibrium bunch length in a storage ring with two rf-systems of different frequencies to control bunch length.
**12.3 (S).**: Energy loss of a particle beam due to synchrotron radiation provides damping. Show that energy loss due to interaction with an external electromagnetic field does not provide beam damping.
**12.4 (S).**: An arbitrary particle distribution of beam injected into a storage ring damps out while a Gaussian distribution evolves with a standard width specific to the ring design. What happens if a beam from another storage ring with a different Gaussian distribution is injected? Explain why this beam changes its distribution to the ring specific Gaussian distribution.
**12.5 (S).**: Consider a 1.5 GeV electron storage ring with a bending field of 1.5 T. The circumference may be covered to 60 % by bending magnets. Let the bremsstrahlung lifetime be 100 h, the Coulomb scattering lifetime 50 h and the Touschek lifetime 60 h. Calculate the total beam lifetime including quantum excitation as a function of aperture. How many "sigma's" (\(A/\sigma\)) must the apertures be in order not to reduce the beam lifetime by more than 10 % due to quantum excitation?
**12.6.**: To reduce coupling instabilities between bunches of a multibunch beam it is desirable to give each bunch a different synchrotron tune. This can be done, for example, by employing two rf-systems operating at harmonic numbers \(h\) and \(h+1\). Determine the ratio or required rf-voltages to split the tunes between successive bunches by \(\Delta v/v_{\rm s}\).
**12.7.**: Attempt to damp out the energy spread of a storage ring beam in the following way. At a location where the dispersion function is finite one could insert a TM\({}_{110}\)-mode cavity. Such a cavity produces accelerating fields which vary linear with the transverse distance of a particle from the reference path. This together with a linear change in particle energy due to the dispersion would allow the correction of the energy spread in the beam. Derive the complete Vlasov equation for such an arrangement and discuss the six-dimensional dynamics. Show that it is impossible to achieve a monochromatic stable beam.

**12.8.**: Derive an expression for the diffusion due to elastic scattering of beam particles on residual gas atoms. How does the equilibrium beam emittance of an electron beam scale with gas pressure and beam energy? Determine an expression for the required gas pressure to limit the emittance growth of a proton or ion beam to no more than 1 % per hour and evaluate numerical for a proton emittance of \(10^{-9}\,\)rad-m at an energy of 300 GeV. Is this a problem if the achievable vacuum pressure is 1 nTorr? Concentrating the allowable scattering to one location of 10 cm length (gas jet as a target) in a ring of 4 km circumference, calculate the tolerable pressure of the gas jet.

**12.9.**: For future linear electron colliders it may be desirable to provide a switching of the beams from one experimental detector to another. Imagine a linear collider system with two experimental stations separated transversely by 50 m. To guide the beams from the linear accelerators to the experimental stations use translating FODO cells and determine the parameters required to keep the emittance growth of a beam to less than 10 % (beam emittance \(10^{-11}\,\)rad-m at 500 GeV).

## Bibliography

* [1] K.W. Robinson, Phys. Rev. **111**, 373 (1958)
* [2] H. Risken, _The Fokker-Planck Equation_ (Springer, Berlin/Heidelberg, 1989)
* [3] I.S. Gradshteyn, I.M. Ryzhik, _Table of Integrals, Series, and Products_, 4th edn. (Academic, New York, 1965)
* [4] A.W. Chao, in _Physics of High Energy Particle Accelerators_, vol. AIP 87, p. 395, ed. by M. Month, M. Dienes (American Institute of Physics, New York, 1982)
* [5] Y.H. Chin, Quantum lifetime. Technical Report DESY Report 87-062, DESY, DESY, Hamburg (1987)

