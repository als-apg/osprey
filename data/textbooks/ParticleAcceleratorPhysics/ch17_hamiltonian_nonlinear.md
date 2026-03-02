## Chapter 17 Hamiltonian Nonlinear Beam Dynamics*

Deviations from linear beam dynamics in the form of perturbations and aberrations play an important role in accelerator physics. Beam parameters, quality and stability are determined by our ability to correct and control such perturbations. Hamiltonian formulation of nonlinear beam dynamics allows us to study, understand and quantify the effects of geometric and chromatic aberrations in higher order than discussed so far. Based on this understanding we may develop correction mechanisms to achieve more and more sophisticated beam performance. We will first discuss higher-order beam dynamics as an extension to the linear matrix formulation followed by specific discussions on aberrations. Finally, we develop the Hamiltonian perturbation theory for particle beam dynamics in accelerator systems.

### 17.1 Higher-Order Beam Dynamics

Chromatic and geometric aberrations appear specifically in strong focusing transport systems designed to preserve carefully prepared beam characteristics. As a consequence of correcting chromatic aberrations by sextupole magnets, nonlinear geometric aberrations are introduced. The effects of both types of aberrations on beam stability must be discussed in some detail. Based on quantitative expressions for aberrations, we will be able to determine criteria for stability of a particle beam.

#### Multipole Errors

The general equations of motion (6.95), (6.96) exhibit an abundance of driving terms which depend on second or higher-order transverse particle coordinates \((x,x^{\prime},y,y^{\prime})\) or linear and higher-order momentum errors \(\delta\). Magnet alignment andfield errors add another multiplicity to these perturbation terms. Although the designers of accelerator lattices and beam guidance magnets take great care to minimize undesired field components and avoid focusing systems that can lead to large transverse particle deviations from the reference orbit, we cannot completely ignore such perturbation terms.

In previous sections we have discussed the effect of some of these terms and have derived basic beam dynamics features as the dispersion function, orbit distortions, chromaticity and tune shifts as a consequence of particle momentum errors or magnet alignment and field errors. More general tools are required to determine the effect of any arbitrary driving term on the particle trajectories. In developing such tools we will assume a careful design of the accelerator under study in layout and components so that the driving terms on the r.h.s. of (6.95), (6.96) can be treated truly as perturbations. This may not be appropriate in all circumstances in which cases numerical methods need to be applied. For the vast majority of accelerator physics applications it is, however, appropriate to treat these higher-order terms as perturbations.

This assumption simplifies greatly the mathematical complexity. Foremost, we can still assume that the general equations of motion are linear differential equations. We may therefore continue to treat every perturbation term separately as we have done so before and use the unperturbed solutions for the amplitude factors in the perturbation terms. The perturbations are reduced to functions of the location \(z\) along the beam line and the relative momentum error \(\delta\) only and such differential equations can be solved analytically as we will see. Summing all solutions for the individual perturbations finally leads to the composite solution of the equation of motion in the approximation of small errors.

The differential equations of motion (6.95), (6.96) can be expressed in a short form by

\[u^{\prime\prime}+K(z)\,u=\sum_{\mu,v,\sigma,\rho,\tau\geq 0}p_{\mu v\sigma \rho\tau}(z)\,x^{u}x^{\prime v}y^{\sigma}y^{\prime\rho}\delta^{\tau}\, \tag{17.1}\]

where \(u=x\) or \(u=y\) and the quantities \(p_{\mu v\sigma\rho\tau}(z)\) represent the coefficients of perturbation terms. The same form of equation can be used for the vertical plane but we will restrict the discussion to only one plane neglecting coupling effects.

Some of the perturbation terms \(p_{\mu v\sigma\rho\tau}\) can be related to aberrations known from geometrical light optics. Linear particle beam dynamics and Gaussian geometric light optics works only for paraxial beams where the light rays or particle trajectories are close to the optical axis or reference path. Large deviations in amplitude, as well as fast variations of amplitudes or large slopes, create aberrations in the imaging process leading to distortions of the image known as spherical aberrations, coma, distortions, curvature and astigmatism. While corrections of such aberrations are desired, the means to achieve corrections in particle beam dynamics are different from those used in light optics. Much of the theory of particle beam dynamics is devoted to diagnose the effects of aberrations on particle beams and to develop and apply such corrections.

The transverse amplitude \(x\) can be separated into its components which under the assumptions made are independent from each other

\[x=x_{\beta}+x_{0}+x_{\delta}+\sum x_{\mu\nu\sigma\rho\tau}\,. \tag{17.2}\]

The first three components of solution (17.2) have been derived earlier and are associated with specific lowest order perturbation terms:

\(x_{\beta}(z)\) is the betatron oscillation amplitude and general solution of the homogeneous differential equation of motion with vanishing perturbations \(p_{\mu\nu\sigma\rho\tau}=0\) for all indices.

\(x_{\rm c}(z)\) is the orbit distortion and is a special solution caused by amplitude and momentum independent perturbation terms like dipole field errors or displacements of quadrupoles or higher multipoles causing a dipole-field error. The relevant perturbations are characterized by \(\mu=v=\sigma=\rho=\tau=0\) but otherwise arbitrary values for the perturbation \(p_{00000}\). Note that in the limit \(p_{00000}\to 0\) we get the ideal reference path or reference orbit \(x_{\rm c}(z)=0\).

\(x_{\delta}(z)\) is the chromatic equilibrium orbit for particles with an energy different from the ideal reference energy, \(\delta\neq 0\), and differs from the reference orbit with or without distortion \(x_{\rm c}(z)\) by the amount \(x_{\delta}(z)\) which is proportional to the dispersion function \(\eta(z)\) and the relative momentum deviation \(\delta,x_{\delta}(z)=\eta\left(z\right)\delta\). In this case \(\mu=v=\sigma=\rho=0\) and \(\tau=1\).

All other solutions \(x_{\mu\nu\sigma\rho\tau}\) are related to remaining higher-order perturbations. The perturbation term \(p_{10000}\), for example, acts just like a quadrupole and may be nothing else but a quadrupole field error causing a tune shift and a variation in the betatron oscillations. Other terms, like \(p_{00100}\) can be correlated with linear coupling or with chromaticity if \(p_{10001}\neq 0\). Sextupole terms \(p_{20000}\) are used to compensate chromaticities, in which case the amplitude factor \(x^{2}\) is expressed by the betatron motion and chromatic displacement

\[x^{2}\approx(x_{\beta}+x_{\delta})^{2}=(x_{\beta}+\eta\delta)^{2}\Longrightarrow 2 \eta x_{\beta}\delta\,. \tag{17.3}\]

The \(x_{\beta}^{2}\)-term, which we neglected while compensating the chromaticity, is the source for geometric aberrations due to sextupolar fields becoming strong for large oscillation amplitudes and the \(\eta^{2}\delta^{2}\)-term contributes to higher-order solution of the \(\eta\)-function. We seem to make arbitrary choices about which perturbations to include in the analysis. Generally therefore only such perturbations are included in the discussion which are most appropriate to the problem to be investigated and solved. If, for example, we are only interested in the orbit distortion \(x_{\rm c}\), we ignore in lowest order of approximation the betatron oscillation \(x_{\beta}\) and all chromatic and higher-order terms. Should, however, chromatic variations of the orbit be of interest one would evaluate the corresponding component separately. On the other hand, if we want to calculate the chromatic variation of betatron oscillations, we need to include the betatron oscillation amplitudes as well as the off momentum orbit \(x_{\delta}\).

In treating higher-order perturbations we make an effort to include all perturbations that contribute to a specific aberration to be studied or to define the order of approximation used if higher-order terms are to be ignored. A careful inspection of all perturbation terms close to the order of approximation desired is prudent to ensure that no significant term is missed. Conversely such an inspection might very well reveal correction possibilities. An example is the effect of chromaticity which is generated by quadrupole field errors for off momentum particles but can be compensated by sextupole fields at locations where the dispersion function is finite. Here the problem is corrected by fields of a different order from those causing the chromaticity.

To become more quantitative we discuss the analytical solution of (17.1). Since in our approximation this solution is the sum of all partial solutions for each individual perturbation term, the problem is solved if we find a general solution for an arbitrary perturbation. The solution of, for example, the horizontal equation of motion

\[x^{\prime\prime}\,+\,K(z)x=p_{\mu\,v\sigma\rho\tau}\,x^{\mu}x^{\prime\nu}y^{ \sigma}y^{\prime\rho}\delta^{\tau} \tag{17.4}\]

can proceed in two steps. First we replace the oscillation amplitudes on the r.h.s. by their most significant components

\[\begin{split} x^{\mu}&\to(x_{\beta}\,+\,x_{0}\,+\,x _{\delta})^{\mu}\,,\,\,\,x^{\prime\nu}&\to(x^{\prime}_{\beta}\,+ \,x^{\prime}_{0}\,+\,x^{\prime}_{\delta})^{\nu}\,,\\ y^{\sigma}&\to(y_{\beta}\,+\,y_{0}\,+\,y_{\delta} )^{\sigma}\,,\,\,\,y^{\prime\rho}&\to(y^{\prime}_{\beta}\,+\,y^ {\prime}_{0}\,+\,y^{\prime}_{\delta})^{\rho}\,.\end{split} \tag{17.5}\]

As discussed before, in a particular situation only those components are eventually retained that are significant to the problem. Since most accelerators are constructed in the horizontal plane we may set the vertical dispersion \(y_{\delta}=0\). The decomposition (17.5) is inserted into the r.h.s of (17.4) and again only terms significant for the particular problem and to the approximation desired are retained. The solution \(x_{\mu\,v\sigma\rho\tau}\) can be further broken down into components each relating to only one individual perturbation term. Whatever number of perturbation terms we decide to keep, the basic differential equation for the perturbation is of the form

\[P^{\prime\prime}\,+\,K(z)P=p(x_{\beta},x^{\prime}_{\beta},x_{\rm c},x^{\prime} _{\rm c},x_{\delta},x^{\prime}_{\delta},y_{\beta},y^{\prime}_{\beta},y_{\rm c },y^{\prime}_{\rm c},y_{\rm c},y^{\prime}_{\delta},\delta,z)\,, \tag{17.6}\]

for which we have discussed the solution in Sect. 5.5.4. Following these steps we may calculate, at least in principle, the perturbations \(P(z)\) for any arbitrary higher-order driving term \(p(z)\). In praxis, however, even principal solutions of particle trajectories in composite beam transport systems can be expressed only in terms of the betatron functions. Since the betatron functions cannot be expressed in a convenient analytical form, we are unable to obtain an analytical solution and must therefore employ numerical methods.

#### Non-linear Matrix Formalism

In linear beam dynamics this difficulty has been circumvented by the introduction of transformation matrices, a principle which can be used also for beam transport systems including higher-order perturbation terms. This non-linear matrix formalism was developed by Karl Brown [1, 2, 3] and we follow his reasoning in the discussion here. The solution to (17.1) can be expressed in terms of initial conditions. Similar to discussions in the context of linear beam dynamics we solve (17.6) for individual lattice elements only where \(K(z)=\mathrm{const}\). In this case (5.75) can be solved for any piecewise constant perturbation along a beam line. Each solution depends on initial conditions at the beginning of the magnetic element and the total solution can be expressed in the form

\[\begin{split}& x(z)=c_{110}\,x_{0}+c_{120}\,x_{0}^{\prime}+c_{130} \,\delta_{0}+c_{111}\,x_{0}^{2}+c_{112}\,x_{0}x_{0}^{\prime}+\dots,\\ & x^{\prime}(z)=c_{210}\,x_{0}+c_{220}\,x_{0}^{\prime}+c_{230}\, \delta_{0}+c_{211}\,x_{0}^{2}+c_{212}\,x_{0}x_{0}^{\prime}+\dots,\end{split} \tag{17.7}\]

where the coefficients \(c_{ijk}\) are functions of \(z\). The nomenclature of the indices becomes obvious if we set \(x_{1}=x,x_{2}=x^{\prime}\), and \(x_{3}=\delta\). The coefficient \(c_{ijk}\) then determines the effect of the perturbation term \(x_{j}x_{k}\) on the variable \(x_{i}\). In operator notation we may write

\[c_{ijk}=\langle x_{i}|x_{0}x_{k0}\rangle. \tag{17.8}\]

The first-order coefficients are the principal solutions

\[\begin{split} c_{110}(z)&=\,C(z)\,,\ \ c_{210}(z)=\,C^{\prime}(z)\,,\\ c_{120}(z)&=S(z)\,,\ \ c_{220}(z)=S^{\prime}(z)\,,\\ c_{130}(z)&=D(z)\,,\ \ c_{230}(z)=D^{\prime}(z)\,. \end{split} \tag{17.9}\]

Before continuing with the solution process, we note that the variation of the oscillation amplitudes \((x^{\prime},y^{\prime})\) are expressed in a curvilinear coordinate system generally used in beam dynamics. This definition, however, is not identical to the intuitive assumption that the slope \(x^{\prime}\) of the particle trajectory is equal to the angle \(\Theta\) between the trajectory and reference orbit. In a curvilinear coordinate system the slope \(x^{\prime}=\mathrm{d}x/\mathrm{d}z\) is a function of the amplitude \(x\). To clarify the transformation, we define angles between the trajectory and the reference orbit by

\[\frac{\mathrm{d}x}{\mathrm{d}s}=\Theta\qquad\mathrm{and}\qquad\frac{\mathrm{d }y}{\mathrm{d}s}=\Phi\,, \tag{17.10}\]

where

\[\mathrm{d}s=(1+\kappa x)\,\mathrm{d}z\,. \tag{17.11}\]In linear beam dynamics there is no numerical difference between \(x^{\prime}\) and \(\Theta\) which is a second-order effect nor is there a difference in straight parts of a beam transport line where \(\kappa=0\). The relation between both definitions is from (17.10), (17.11)

\[\Theta=\frac{x^{\prime}}{1+\kappa x}\qquad\text{and}\qquad\Phi=\frac{y^{\prime} }{1+\kappa x}\,, \tag{17.12}\]

where \(x^{\prime}=\mathrm{d}x/\mathrm{d}z\) and \(y^{\prime}=\mathrm{d}y/\mathrm{d}z\). We will use these definitions and formulate second-order transformation matrices in a Cartesian coordinate system \((x,y,z)\). Following Brown's notation [1], we may express the nonlinear solutions of (17.4) in the general form

\[u_{i}=\sum_{j=1}^{3}c_{ij0}u_{j0}+\sum_{\begin{subarray}{c}j=1\\ k=1\end{subarray}}^{3}T_{ijk}(z)\,u_{j0}\,u_{k0}\,, \tag{17.13}\]

with

\[(u_{1},u_{2},u_{3})=(x,\Theta,\delta)\,, \tag{17.14}\]

where \(z\) is the position along the reference particle trajectory. Nonlinear transformation coefficients \(T_{ijk}\) are defined similar to coefficients \(\mathrm{c}_{ijk}\) in (17.8) by

\[T_{ijk}=\langle u_{i}|u_{j0}u_{k0}\rangle\,, \tag{17.15}\]

where the coordinates are defined by (17.14). In linear approximation both coefficients are numerically the same and we have

\[\left(\begin{array}{cc}c_{110}&c_{120}&c_{130}\\ c_{210}&c_{220}&c_{230}\\ c_{310}&c_{320}&c_{330}\end{array}\right)\,=\left(\begin{array}{cc}C(z)&S(z )&D(z)\\ C^{\prime}(z)&S^{\prime}(z)&D^{\prime}(z)\\ 0&0&1\end{array}\right)\,. \tag{17.16}\]

Earlier in this section we decided to ignore coupling effects which could be included easily in (17.13) if we set for example \(x_{4}=y\) and \(x_{5}=y^{\prime}\) and expand the summation in (17.13) to five indices. For simplicity, however, we will continue to ignore coupling.

The equations of motion (6.95), (6.96) are expressed in curvilinear coordinates and solving (5.75) results in coefficients \(c_{ijk}\) which are different from the coefficients \(T_{ijk}\) if one or more variables are derivatives with respect to \(z\). In the equations of motion all derivatives are transformed like (17.12) generating a \(\Theta\)-term as well as an \(x\,\Theta\)-term. If, for example, we were interested in the perturbations to the particle amplitude \(x\) caused by perturbations proportional to \(x_{0}\,\Theta_{0}\), we are looking for the coefficient \(T_{112}=\langle x\,|x_{0}\,\Theta_{0}\rangle\). Collecting from (6.95) only second-order perturbation terms proportional to \(xx^{\prime}\), we find

\[x=c_{112}\,x_{0}\,x_{0}^{\prime}=c_{112}\,x_{0}\,\Theta_{0}+\mathcal{O}(3)\,. \tag{17.17}\]

An additional second-order contribution appears as a spill over from the linear transformation

\[x=c_{120}\,x_{0}^{\prime}=c_{120}\,(1+\kappa_{x}x_{0})\,\,\Theta_{0}\,. \tag{17.18}\]

Collecting all \(x_{0}\,\Theta_{0}-\)terms, we get finally

\[T_{112}=c_{112}+c_{120}\,\kappa_{x}\,=c_{112}+\kappa_{x}S(z)\,. \tag{17.19}\]

To derive a coefficient like \(T_{212}=\langle\Theta\,|x_{0}\,\Theta_{0}\,\rangle\) we also have to transform the derivative of the particle trajectory at the end of the magnetic element. First, we look for all contributions to \(x^{\prime}\) from \(x_{0}x_{0}^{\prime}\)-terms which originate from \(x^{\prime}=c_{220}\,\,x_{0}^{\prime}+c_{212}x_{0}x_{0}^{\prime}\). Setting in the first term \(x_{0}^{\prime}=\Theta_{0}\,(1+\kappa_{x}x_{0})\) and in the second term \(x_{0}x_{0}^{\prime}\approx x_{0}\,\Theta_{0}\,\), we get with \(c_{220}=S^{\prime}(z)\) and keeping again only second-order terms

\[x^{\prime}=\left[c_{212}+\kappa_{x}S^{\prime}(z)\right]\,x_{0}\Theta_{0}\,. \tag{17.20}\]

On the l.h.s. we replace \(x^{\prime}\) by \(\Theta\,(1+\kappa_{x}x)\) and using the principal solutions we get

\[x\,\Theta\approx\left(C_{x}x_{0}+S_{x}\Theta_{0}\right)\left(C_{x}^{\prime}x_ {0}+S_{x}^{\prime}\Theta_{0}\right)=\left(C_{x}S_{x}^{\prime}+C_{x}^{\prime}S_ {x}\right)x_{0}\Theta_{0} \tag{17.21}\]

keeping only the \(x_{0}\Theta\)-terms. Collecting all results, the second-order coefficient for this perturbation becomes

\[T_{212}=\langle\Theta\,|x_{0}\,\Theta_{0}\,\rangle=c_{212}+\kappa_{x}S^{ \prime}(z)-\kappa_{x}\left(C_{x}S_{x}^{\prime}+C_{x}^{\prime}S_{x}\right)\,. \tag{17.22}\]

In a similar way we can derive all second-order coefficients \(T_{ijk}\). Equations (17.13) define the transformation of particle coordinates in second order through a particular magnetic element. For the transformation of quadratic terms we may ignore the third order difference between the coefficients \(c_{ijk}\) and \(T_{ijk}\) and get

\[\begin{split}& x^{2}=\left(C_{x}x_{0}+S_{x}x_{0}^{\prime}+D_{x} \delta_{0}\right)^{2},\\ & xx^{\prime}=\left(C_{x}x_{0}+S_{x}x_{0}^{\prime}+D_{x}\delta_{ 0}\right)\left(C_{x}^{\prime}x_{0}+S_{x}^{\prime}x_{0}^{\prime}+D_{x}^{\prime }\delta_{0}\right)\\ & x\delta=\left(C_{x}x_{0}+S_{x}x_{0}^{\prime}+D_{x}\delta_{0} \right)\,\delta_{0}\\ &\vdots\quad\quad\text{etc.}\end{split} \tag{17.23}\]All transformation equations can now be expressed in matrix form after correctly ordering equations and coefficients and a general second-order transformation matrix can be formulated in the form

\[\left(\begin{array}{c}x\\ \Theta\\ \delta\\ x^{2}\\ x\,\Theta\\ x\,\delta\\ \Theta^{2}\\ \Theta\,\delta\\ \delta^{2}\end{array}\right)=\mathcal{M}\left(\begin{array}{c}x_{0}\\ \Theta_{0}\\ \delta_{0}\\ x_{0}\,\delta_{0}\\ \Theta_{0}^{2}\\ \Theta_{0}\,\delta_{0}\\ \delta_{0}^{2}\end{array}\right), \tag{17.24}\]

where we have ignored the \(y\)-plane. The second-order transformation matrix is then

\[\mathcal{M}= \tag{17.25}\] \[\left(\begin{array}{cccccccc}C&S&D&T_{111}&T_{112}&T_{116}&T_{12 2}&T_{126}&T_{166}\\ C^{\prime}&S^{\prime}&D^{\prime}&T_{211}&T_{212}&T_{216}&T_{222}&T_{226}&T_{266} \\ 0&0&1&0&0&0&0&0\\ 0&0&0&C^{2}&2CS&2CD&S^{2}&2SD&D^{2}\\ 0&0&0&CC^{\prime}&CS^{\prime}+C^{\prime}S&CD^{\prime}+C^{\prime}D&SS^{\prime}& SD^{\prime}+S^{\prime}D&DD^{\prime}\\ 0&0&0&0&0&C&0&S&D\\ 0&0&0&C^{\prime}&2C^{\prime}S^{\prime}&2C^{\prime}D&S^{\prime}\,{}^{2}&2S^{ \prime}D^{\prime}&D^{\prime}\,{}^{2}\\ 0&0&0&0&0&C^{\prime}&0&S^{\prime}&D^{\prime}\\ 0&0&0&0&0&0&0&1\end{array}\right)\]

with \(C=C_{x},S=S_{x},\ldots\)etc.

A similar equation can be derived for the vertical plane. If coupling effects are to be included the matrix could be further expanded to include also such terms. While the matrix elements must be determined individually for each magnetic element in the beam transport system, we may in analogy to linear beam dynamics multiply a series of such matrices to obtain the transformation matrix through the whole composite beam transport line. As a matter of fact the transformation matrix has the same appearance as (17.24) for a single magnet or a composite beam transport line and the magnitude of the nonlinear matrix elements will be representative of imaging errors like spherical and chromatic aberrations.

To complete the derivation of second-order transformation matrices we derive, as an example, an expression of the matrix element \(T_{111}\) from the equation of motion (6.95). To obtain all \(x_{0}^{2}\)-terms, we look in (6.95) for perturbation terms proportional to \(x^{2},xx^{\prime}\) and \(x^{\prime 2}\), replace these amplitude factors by principal solutionsand collect only terms quadratic in \(x_{0}\) to get the relevant perturbation

\[p(z)=\left[-\left(\tfrac{1}{2}m+2\kappa_{x}k+\kappa_{x}^{3}\right)C_{x}^{2}+ \tfrac{1}{2}\kappa_{x}C_{x}^{\prime 2}+\kappa_{x}^{\prime}C_{x}C_{x}^{\prime} \right]x_{0}^{2}\,. \tag{17.26}\]

First, we recollect that the theory of nonlinear transformation matrices is based on the constancy of magnet strength parameters and we set therefore \(\kappa_{x}^{\prime}=0\). Where this is an undue simplification like in magnet fringe fields one could approximate the smooth variation of \(\kappa_{x}\) by a step function. Inserting (17.26) into (5.75) the second-order matrix element

\[c_{111} =T_{111} \tag{17.27}\] \[=-(\tfrac{1}{2}m+2\kappa_{x}k+\kappa_{x}^{3})\int_{0}^{z}C_{x}^{2 }(\xi)\,G(z,\xi)\,\mathrm{d}\xi-\tfrac{1}{2}\kappa_{x}\int_{0}^{z}C_{x}^{\prime 2 }(\xi)\,G(z,\xi)\,\mathrm{d}\xi\,.\]

The integrands are powers of trigonometric functions and can be evaluated analytically. In a similar way we may now derive any second-order matrix element of interest. A complete list of all second order matrix elements can be found in [1].

This formalism is valuable whenever the effect of second-order perturbations must be evaluated for particular particle trajectories. Specifically, it is suitable for nonlinear beam simulation studies where a large number of particles representing the beam are to be traced through nonlinear focusing systems to determine, for example, the particle distribution and its deviation from linear beam dynamics at a focal point. This formalism is included in the program TRANSPORT [4] allowing the determination of the coefficients \(T_{\mathit{gk}}\) for any beam transport line and providing fitting routines to eliminate such coefficients by proper adjustment and placement of nonlinear elements like sextupoles.

### Aberrations

From light optics we are familiar with the occurrence of aberrations which cause the distortion of optical images. We have repeatedly noticed the similarity of particle beam optics with geometric or paraxial light optics and it is therefore not surprising that there is also a similarity in imaging errors. Aberrations in particle beam optics can cause severe stability problems and must therefore be controlled.

We distinguish two classes of aberrations, geometric aberrations and for off momentum particles chromatic aberrations. The geometric aberrations become significant as the amplitude of betatron oscillations increases while chromatic aberration results from the variation of the optical system parameters for different colors of the light rays or in our case for different particle energies. For the discussion of salient features of aberration in particle beam optics we study the equation of motion in the horizontal plane and include only bending magnets, quadrupoles and sextupole magnets. The equation of motion in this case becomes in normalized coordinates

\[\ddot{w}+v_{0}^{2}w=v_{0}^{2}\beta^{3/2}\kappa\delta+v_{0}^{2}\beta^{2}kw\delta- \tfrac{1}{2}v_{0}^{2}\beta^{5/2}mw^{2}\,, \tag{17.28}\]

where \(\beta=\beta_{x}\). The particle deviation \(w\) from the ideal orbit is composed of two contributions, the betatron oscillation amplitude \(w_{\beta}\) and the shift in the equilibrium orbit for particles with a relative momentum error \(\delta\). This orbit shift \(w_{\delta}\) is determined by the normalized dispersion function at the location of interest \(\left(w_{\delta}=\tilde{\eta}\,\delta=\tfrac{\eta}{\sqrt{\beta}}\delta\right)\) and the particle position can be expressed by the composition

\[w=w_{\beta}+w_{\delta}=w_{\beta}+\tilde{\eta}\delta. \tag{17.29}\]

Inserting (17.29) into (17.28) and employing the principle of linear superposition (17.28) can be separated into two differential equations, one for the betatron motion and one for the dispersion function neglecting quadratic or higher-order terms in \(\delta\). The differential equation for the dispersion function is then

\[\ddot{\tilde{\eta}}+v_{0}^{2}\tilde{\eta}=v_{0}^{2}\beta^{1/2}\kappa+v_{0}^{2 }\beta^{2}k\tilde{\eta}\delta-\tfrac{1}{2}v_{0}^{2}\beta^{5/2}m\tilde{\eta}^{2 }\delta\,, \tag{17.30}\]

which has been solved earlier in Sect. 9.4.1. All other terms include the betatron oscillation \(w_{\beta}\) and contribute therefore to aberrations of betatron oscillations expressed by the differential equation

\[\ddot{w}_{\beta}+v_{0}^{2}w_{\beta}=v_{0}^{2}\beta^{2}kw_{\beta}\,\delta-v_{0 }^{2}\beta^{2}m\eta w_{\beta}\,\delta-\tfrac{1}{2}v_{0}^{2}\beta^{5/2}m\,w_{ \beta}^{2}\,. \tag{17.31}\]

The third term in (17.31) is of geometric nature causing a perturbation of beam dynamics at large betatron oscillation amplitudes and, as will be discussed in Sect. 17.3, also gives rise to an amplitude dependent tune shift. This term appears as an isolated term in second order and no local compensation scheme is possible. Geometric aberrations must therefore be expected whenever sextupole magnets are used to compensate for chromatic aberrations.

The first two terms in (17.31) represent the natural chromaticity from quadrupoles and the compensation by sextupole magnets, respectively. Whenever it is possible to compensate the chromaticity at the location where it occurs both terms would cancel for \(m\eta=k\). Since the strength changes sign for both magnets going from one plane to the other the compensation is correct in both planes. This method of chromaticity correction is quite effective in long beam transport systems with many equal lattice cells. An example of such a correction scheme are the beam transport lines from the SLAC linear accelerator to the collision point of the Stanford Linear Collider, SLC, [5]. This transport line consists of a dense sequence of strong magnets forming a combined function FODO channel (for parameters see example #2 in Table 10.1). In these magnets dipole, quadrupole and sextupole components are combined in the pole profile and the chromaticity compensation occurs locally.

This method of compensation, however, does not work generally in circular accelerators because of special design criteria which often require some parts of the accelerator to be dispersion free and the chromaticity created by the quadrupoles in these sections must then be corrected elsewhere in the lattice. Consequently both chromaticity terms in (17.31) do not cancel anymore locally and can be adjusted to cancel only globally.

The consequence of these less than perfect chromaticity correction schemes is the occurrence of aberrations through higher-order effects. We get a deeper insight for the effects of these aberrations in a circular accelerator by noting that the coefficients of the betatron oscillation amplitude \(w_{\beta}\) for both chromatic perturbations are periodic functions in a circular accelerator and can therefore be expanded into a Fourier series. Only non-oscillatory terms of these expansions cancel if the chromaticity is corrected while all other higher harmonics still appear as chromatic aberrations.

##### Geometric Aberrations

Geometric perturbations from sextupole fields scale proportional to the square of the betatron oscillation amplitude leading to a loss of stability for particles oscillating at large amplitudes. From the third perturbation term in (17.31) we expect this limit to occur at smaller amplitudes in circular accelerators where either the betatron functions are generally large or where the focusing and therefore the chromaticity and required sextupole correction is strong or where the tunes are large. Most generally this occurs in large proton and electron colliding-beam storage rings or in electron storage rings with strong focusing.

##### Compensation of Nonlinear Perturbations

In most older circular accelerators the chromaticity is small and can be corrected by two families of sextupoles. Although in principle only two sextupole magnets for the whole ring are required for chromaticity compensation, this is in most cases impractical since the strength of the sextupoles becomes too large exceeding technical limits or leading to loss of beam stability because of intolerable geometric aberrations. For chromaticity compensation we generally choose a more even distribution of sextupoles around the ring and connect them into two families compensating the horizontal and vertical chromaticity, respectively. This scheme is adequate for most not too strong focusing circular accelerators. Where beam stability suffers from geometric aberrations more sophisticated sextupole correction schemes must be utilized.

To analyze the geometric aberrations due to sextupoles and develop correction schemes we follow a particle along a beam line including sextupoles. Here we understand a beam line to be an open system from a starting point to an image point at the end or one full circumference of a circular accelerator. Following any particle through the beam line and ignoring for the moment nonlinear fields we expect the particle to move along an ellipse in phase space as shown in Fig. 17.1. Travelling through the length of a circular accelerator with phase advance \(\psi\,=\,2\pi\,v_{0}\) a particle moves \(v_{0}\) revolutions around the phase ellipse in Fig. 17.1.

Including nonlinear perturbations due to, for example, sextupole magnets the phase space trajectory becomes distorted from the elliptical form as shown in Fig. 17.2. An arbitrary distribution of sextupoles along a beam line can cause large variations of the betatron oscillation amplitude leading to possible loss of particles on the vacuum chamber wall even if the motion is stable in principle. The PEP storage ring [6] was the first storage ring to require a more sophisticated sextupole correction [7] beyond the mere compensation of the two chromaticities because geometric aberrations were too strong to give sufficient beam stability. Chromaticity correction with only two families of sextupoles in PEP would have produced large amplitude dependent tune shifts leading to reduced beam stability.

Such a situation can be greatly improved with additional sextupole families [7] to minimize the effect of these nonlinear perturbation. Although individual

Figure 17.2: Typical phase space motion in the presence of nonlinear fields

Figure 17.1: Linear particle motion in phase space

perturbations may not be reduced much by this method the sum of all perturbations can be compensated to reduce the overall perturbation to a tolerable level.

In this sextupole correction scheme the location and strength of the individual sextupoles are selected such as to minimize the perturbation of the particle motion in phase space at the end of the beam transport line. Although this correction scheme seems to work in not too extreme cases it is not sufficient to guarantee beam stability. This scheme works only for one amplitude due to the nonlinearity of the problem and in cases where sextupole fields are no longer small perturbations we must expect a degradation of this compensation scheme for larger amplitudes. As the example of PEP shows, however, an improvement of beam stability can be achieved beyond that obtained by a simple two family chromaticity correction. Clearly, a more formal analysis of the perturbation and derivation of appropriate correction schemes are desirable.

##### 17.2.1 Sextupoles Separated by a \(-\mathcal{I}\)-Transformation

A chromaticity correction scheme that seeks to overcomes this amplitude dependent aberration has been proposed by Brown and Servranckx [8]. In this scheme possible sextupole locations are identified in pairs along the beam transport line such that each pair is separated by a negative unity transformation

\[-\mathcal{I}=\left(\begin{array}{cccc}-1&0&0&0\\ 0&-1&0&0\\ 0&0&-1&0\\ 0&0&0&-1\end{array}\right)\,. \tag{17.32}\]

Placing sextupoles of equal strength at these two locations we get an additive contribution to the chromaticity correction. The effect of geometric aberrations, however, is canceled for all particle oscillation amplitudes. This can be seen if we calculate the transformation matrix through the first sextupole, the \(-\mathcal{I}\) section, and then through the second sextupole. The sextupoles are assumed to be thin magnets inflicting kicks on particle trajectories by the amount

\[\Delta x^{\prime}=-\tfrac{1}{2}m_{0}\ell_{\mathrm{s}}\left(x^{2}-y^{2}\right)\,, \tag{17.33}\]

and

\[\Delta y^{\prime}=-m_{0}\ell_{\mathrm{s}}xy\,, \tag{17.34}\]where \(\ell_{\rm s}\) is the sextupole length. We form a \(4\times 4\) transformation matrix through a thin sextupole and get

\[\left(\begin{array}{c}x\\ x^{\prime}\\ y\\ y^{\prime}\end{array}\right) = \mathcal{M}_{\rm s}\left(x_{0},y_{0}\right)\left(\begin{array}{c }x_{0}\\ x_{0}^{\prime}\\ y_{0}\\ y_{0}^{\prime}\end{array}\right) \tag{17.35}\] \[= \left(\begin{array}{cccc}1&0&0&0\\ -\frac{1}{2}m_{0}\ell_{\rm s}x_{0}&1&\frac{1}{2}m_{0}\ell_{\rm s}x_{0}&0\\ 0&0&1&0\\ 0&0&m_{0}\ell_{\rm s}x_{0}&1\end{array}\right)\left(\begin{array}{c}x_{0}\\ x_{0}^{\prime}\\ y_{0}\\ y_{0}^{\prime}\end{array}\right)\]

To evaluate the complete transformation we note that in the first sextupole the particle coordinates are \(\left(x_{0},y_{0}\right)\) and become after the \(-\mathcal{I}\)-transformation in the second sextupole \(\left(-x_{0},-y_{0}\right)\). The transformation matrix through the complete unit is therefore

\[\mathcal{M}_{t}=\mathcal{M}_{\rm s}\left(x_{0},y_{0}\right)\left(-\mathcal{I }\right)\mathcal{M}_{\rm s}\left(-x_{0},-y_{0}\right)\,. \tag{17.36}\]

Independent of the oscillation amplitude we observe a complete cancellation of geometric aberrations in both the horizontal and vertical plane. This correction scheme has been applied successfully to the final focus system of the Stanford Linear Collider [9], where chromatic as well as geometric aberrations must be controlled and compensated to high accuracy to allow the focusing of a beam to a spot size at the collision point of only a few micrometer.

The effectiveness of this correction scheme and its limitations in circular accelerators has been analyzed in more detail by Emery [10] and we will discuss some of his findings. As an example, we use strong focusing FODO cells for an extremely low emittance electron storage ring [10] and investigate the beam parameters along this lattice. Any other lattice could be used as well since the characteristics of aberrations is not lattice dependent although the magnitude may be. The particular FODO lattice under discussion as shown in Fig. 17.3 is a thin lens lattice with \(90^{\rm o}\) cells, a distance between quadrupoles of \(L_{\rm q}=3.6\,\)m and an

Figure 17.3: FODO lattice and chromaticity correction

integrated half quadrupole strength of \((k\ell_{\rm q})^{-1}=\sqrt{2}\,L_{\rm q}\). The horizontal and vertical betatron functions at the symmetry points are 12.29 and 2.1088 m respectively. Three FODO cells are shown in Fig. 17.3 including one pair of sextupoles separated by 180\({}^{\circ}\) in betatron phase space. We choose a phase ellipse for an emittance of \(\epsilon=200\) mm-mrad which is an upright ellipse at the beginning of the FODO lattice, Fig. 17.4a. Due to quadrupole focusing the ellipse becomes tilted at the entrance to the first sextupole, Fig. 17.4b. The thin lens sextupole introduces a significant angular perturbation (Fig. 17.4c) leading to large lateral aberrations in the quadrupole QF (Fig. 17.4d). At the entrance to the second sextupole the distorted phase ellipse is rotated by 180\({}^{\circ}\) and all aberrations are compensated again by this sextupole, Fig. 17.4e. Finally, the phase ellipse at the end of the third FODO cell is again an upright ellipse with no distortions left, Fig. 17.4f. The range of stability therefore extends to infinitely large amplitudes ignoring any other detrimental effects.

The compensation of aberrations works as long as the phase advance between sextupoles is exactly 180\({}^{\circ}\). A shift of the second sextupole by a few degrees or a quadrupole error resulting in a similar phase error between the sextupole pair would greatly reduce the compensation. In Fig. 17.5 the evolution of the phase ellipse from Fig. 17.4 is repeated but now with a phase advance between the sextupole pair of only 175\({}^{\circ}\). A distortion of the phase ellipse due to aberrations can be observed which may build up to instability as the particles pass through many similar cells. Emery has analyzed numerically this degradation of stability and finds empirically

Figure 17.4: Phase ellipses along a FODO channel including nonlinear aberrations due to thin sextupole magnets separated by exactly 180\({}^{\circ}\) in betatron phase (consult text for details)

the maximum stable betatron amplitude to scale with the phase error like \(\Delta\varphi^{-0.52}\)[10]. The sensitivity to phase errors together with unavoidable quadrupole field errors and orbit errors in sextupoles can significantly reduce the effectiveness of this compensation scheme.

The single most detrimental arrangement of sextupoles compared to the perfect compensation of aberrations is to interleave sextupoles which means to place other sextupoles between two pairs of compensating sextupoles [8]. Such interleaved sextupoles introduce amplitude dependent phase shifts leading to phase errors and reduced compensation of aberrations. This limitation to compensate aberrations is present even in a case without apparent interleaved sextupoles as shown in Fig. 17.6 for the following reason.

The assumption of thin magnets is sometimes convenient but, as Emery points out, can lead to erroneous results. For technically realistic solutions, we must allow the sextupoles to assume a finite length and find, as a consequence, a loss of complete compensation for geometric aberrations because sextupoles of finite length are only one particular case of interleaved sextupole arrangements. If we consider the sextupoles made up of thin slices we still find that each slice of the first

Figure 17.6: Phase ellipses along a FODO channel including nonlinear aberrations due to finite length sextupole magnets placed exactly 180 degrees apart. Phase ellipse (**a**) transforms to (**b**) after one FODO triplet cell and to (**c**) after passage through many such cells

Figure 17.5: Thin sextupole magnets separated by 175\({}^{\circ}\) in betatron phase space. The unperturbed phase ellipse (**a**) becomes slightly perturbed (**b**) at the end of the first triple FODO cell (Fig.17.3, and more so after passing through many such triplets (**c**)

#### 17.2.2 Filamentation of Phase Space

Some distortion of the unperturbed trajectory in phase space due to aberrations is inconsequential to beam stability as long as this distortion does not build up and starts growing indefinitely. A finite or infinite growth of the beam emittance enclosed within a particular particle trajectory in phase space may at first seem impossible since we deal with macroscopic, non-dissipating magnetic fields where Liouville's theorem must hold. Indeed numerical simulations indicate that the total phase space occupied by the beam does not seem to increase but an originally elliptical boundary in phase space can grow, for example, tentacles like a spiral galaxy leading to larger beam sizes without actually increasing the phase space density. This phenomenon is called filamentation of the phase space and can evolve like shown in Fig. 17.7.

For particle beams this filamentation is as undesirable as an increase in beam emittance or beam loss. We will therefore try to derive the causes for beam filamentation in the presence of sextupole non-linearities which are the most important non-linearities in beam dynamics. In this discussion we will follow the ideas developed by Autin [11] which observes the particle motion in action-angle phase space under the influence of nonlinear fields.

Figure 17.7: Filamentation of phase space after passage through an increasing number of FODO cellsFor simplicity of expression, we approximate the nonlinear sextupoles by thin magnets. This does not restrict our ability to study the effect of finite length sextupoles since we may always represent such sextupoles by a series of thin magnets. A particle in a linear lattice follows a circle in action-angle phase space with a radius equal to the action \(J_{0}\). The appearance of a nonlinearity along the particle trajectory will introduce an amplitude variation \(\Delta J\) to the action which is from the Courant-Snyder invariant for both the horizontal and vertical plane

\[\begin{array}{l}\Delta J_{x}=v_{x0}w\Delta w+\frac{1}{v_{x0}}\dot{w}\Delta \dot{w}=\frac{1}{v_{x0}}\dot{w}\Delta\dot{w}\,,\\ \Delta J_{y}=v_{y0}v\Delta v+\frac{1}{v_{y0}}\dot{v}\Delta\dot{v}=\frac{1}{v_{y 0}}\dot{v}\Delta\dot{v}\,,\end{array} \tag{17.37}\]

since \(\Delta w=\Delta v=0\) for a thin magnet. Integration of the equations of motion in normalized coordinates over the "length" \(\ell\) of the thin magnet produces a variation of the slopes

\[\begin{array}{l}\Delta\dot{w}=v_{x0}\sqrt{\beta_{x}}\frac{1}{2}m\ell(x^{2}- y^{2})\,,\\ \Delta\dot{v}=-v_{y0}\sqrt{\beta_{y}}m\ell\,xy\,.\end{array} \tag{17.38}\]

We insert (17.38) into (17.37) and get after transformation into action-angle variables and linearization of the trigonometric functions the variation of the action

\[\begin{array}{l}\Delta J_{x}=\frac{m\ell}{4}\sqrt{\frac{2J_{x}\beta_{x}}{v_{ x0}}}\left\{\left(J_{x}\beta_{x}-2J_{y}\beta_{y}\frac{v_{x}}{v_{y}}\right) \sin\psi_{x}+J_{x}\beta_{x}\sin 3\psi_{x}\\ \qquad\qquad-J_{y}\beta_{y}\frac{v_{x}}{v_{y}}\left[\sin(\psi_{x}+2\psi_{y}) +\sin(\psi_{x}-2\psi_{y})\right]\right\}\,,\\ \Delta J_{y}=\frac{m\ell}{2}\sqrt{\frac{2J_{x}\beta_{x}}{v_{x0}}}J_{y}\beta_{y} [\sin(\psi_{x}+2\psi_{y})-\sin(\psi_{x}-2\psi_{y})]\,.\end{array} \tag{17.39}\]

Since the action is proportional to the beam emittance, (17.39) allow us to study the evolution of beam filamentation over time. The increased action from (17.39) is due to the effect of one nonlinear sextupole magnet and we obtain the total growth of the action by summing over all turns and all sextupoles. To sum over all turns we note that the phases in the trigonometric functions increase by \(2\pi\,v_{0,x,y}\) every turn and we have for the case of a single sextupole after an infinite number of turns expressions of the form

\[\sum_{n=0}^{\infty}\sin[(\psi_{xj}+2\pi\,v_{x0}n)+2(\psi_{yj}+2\pi\,v_{y0}n)]\,, \tag{17.40}\]

where \(\psi_{xj}\) and \(\psi_{yj}\) are the phases at the location of the sextupole \(j\). Such sums of trigonometric functions are best solved in the form of exponential functions. In this case the sine function terms are equivalent to the imaginary part of the exponential functions

\[\mathrm{e}^{\mathrm{i}(\psi_{y}+2\psi_{yj})}\,\mathrm{e}^{\mathrm{i}2\pi(v_{x0 }+2v_{y0})n}. \tag{17.41}\]The second factor forms an infinite geometric series and the imaginary part of the sum is therefore

\[\text{Im }\frac{\text{e}^{\text{i}(\psi_{\text{\tiny${\rm{${\rm{${\rm{${\rm{$ \rm{$\rm{$\rm{$\rm{$\rm{$\rm{$}$}$}$}$}$}}}}}}}+2\psi_{\text{\tiny${\rm{${\rm{${ \rm{${\rm{${\rm{$\rm{$\rm{$\rm{${$\rm{$$$$\rm{$$$$$}}$}$}$}}}}}}}}})}}}{1- \text{e}^{\text{i}2\pi(v_{x0}+2v_{y0})}}=\frac{\cos[(\psi_{\text{\tiny${\rm{${ \rm{${\rm{${\rm{${\rm{$\rm{${$\rm{$\rm{${$\rm{${$}}}$}$}$}$}}}}}}}}-\pi v_{x0}) +2(\psi_{\text{\tiny${\rm{${\rm{${\rm{${\rm{${\rm{${$\rm{${$\rm{$$$}}$}$}$}$}}}}}}}} }-\pi v_{y0})}]}}{2\,\sin[\pi(v_{x0}+2v_{y0})]}. \tag{17.42}\]

This solution has clearly resonant character leading to an indefinite increase of the action if \(v_{x0}+2v_{y0}\) is an integer. Similar results occur for the other three terms and Autin's method to observe the evolution of the action coordinate over many turns allows us to identify four resonances driven by sextupolar fields which can lead to particle loss and loss of beam stability if not compensated. Resonant growth of the apparent beam emittance occurs according to (17.39) for

\[\begin{array}{l}v_{x0}=q_{1}\,,\qquad\text{or}\qquad v_{x0}+2v_{y0}=q_{3}\,, \\ 3v_{x0}=q_{2}\,,\qquad\text{or}\qquad v_{x0}-2v_{y0}=q_{4}\,,\end{array} \tag{17.43}\]

where the \(q_{i}\) are integers. In addition to the expected integer and third integer resonance in the horizontal plane, we find also two third order coupling resonances in both planes where the sum resonance leads to beam loss while the difference resonance only initiates an exchange of the horizontal and vertical emittances. The asymmetry is not fundamental and is the result of our choice to use only upright sextupole fields.

So far we have studied the effect of one sextupole on particle motion. Since no particular assumption was made as to the location and strength of this sextupole, we conclude that any other sextupole in the ring would drive the same resonances and we obtain the beam dynamics under the influence of all sextupoles by adding the individual contributions. In the expressions of this section we have tacitly assumed that the beam is observed at the phase \(\psi_{x0,y0}=0\). If this is not the desired location of observation the phases \(\psi_{x\bar{y}}\) need to be replaced by \(\psi_{x\bar{y}}-\psi_{x0}\), etc., where the phases \(\psi_{x\bar{y},y\bar{y}}\) define the location of the sextupole \(j\). Considering all sextupoles in a circular lattice we sum over all such sextupoles and get, as an example, for the sum resonance used in the derivation above from (17.39)

\[\Delta J_{x,v_{x}+2v_{y}}=-\sum_{j}\frac{m_{j}\ell_{j}}{4}\sqrt{\frac{2J_{x} \beta_{x\bar{y}}}{v_{x0}}}J_{y}\beta_{y\bar{y}}\frac{v_{x}}{v_{y}}\;\sin(\psi_{ x\bar{y}}+2\psi_{y\bar{y}})\,. \tag{17.44}\]

Similar expressions exist for other resonant terms. Equation (17.44) indicates a possibility to reduce the severity of driving terms for the four resonances. Sextupoles are primarily inserted into the lattice where the dispersion function is nonzero to compensate for chromaticities. Given sufficient flexibility these sextupoles can be arranged to avoid driving these resonances. Additional sextupoles may be located in dispersion free sections and adjusted to compensate or at least minimize the four resonance driving terms without affecting the chromaticity correction. The perturbation \(\Delta J\) is minimized by distributing the sextupoles such that the resonant driving terms in (17.39) are as small as possible. This is accomplished by harmonic correction which is the process of minimization of expressions

\[\sum\nolimits_{j}m_{j}\ell_{j}\beta_{x}^{3/2}\,\mathrm{e}^{\mathrm{i} \psi_{xj}} \to 0\,, \tag{17.45}\] \[\sum\nolimits_{j}m_{j}\ell_{j}\beta_{x}^{3/2}\,\mathrm{e}^{ \mathrm{i}3\psi_{xj}} \to 0\,,\] (17.46) \[\sum\nolimits_{j}m_{j}\ell_{j}\beta_{x}^{1/2}\beta_{y}\,\mathrm{e }^{\mathrm{i}\psi_{xj}} \to 0\,,\] (17.47) \[\sum\nolimits_{j}m_{j}\ell_{j}\beta_{x}^{1/2}\beta_{y}\,\mathrm{e }^{\mathrm{i}(\psi_{xj}+2\psi_{xj})} \to 0\,,\] (17.48) \[\sum\nolimits_{j}m_{j}\ell_{j}\beta_{x}^{1/2}\beta_{y}\,\mathrm{e }^{\mathrm{i}(\psi_{xj}-2\psi_{xj})} \to 0\,. \tag{17.49}\]

The perturbations of the action variables in (17.39) cancel perfectly if we insert sextupoles in pairs at locations which are separated by a \(-\mathcal{I}\) transformation as discussed previously in this chapter. The distribution of sextupoles in pairs is therefore a particular solution to (17.45) for the elimination of beam filamentation and specially suited for highly periodic lattices while (17.45)-(17.49) provide more flexibility to achieve similar results in general lattices and sextupole magnets of finite length.

Cancellation of resonant terms does not completely eliminate all aberrations caused by sextupole fields. Because of the existence of nonlinear sextupole fields the phases \(\psi_{j}\) depend on the particle amplitude and resonant driving terms are therefore canceled only to first order. For large amplitudes we expect increasing deviation from the perfect cancellation leading eventually to beam filamentation and beam instability. Maximum stable oscillation amplitudes in \((x,y)\)-space due to nonlinear fields form the dynamic aperture which is to be distinguished from the physical aperture of the vacuum chamber. This dynamic aperture is determined by numerical tracking of particles. Given sufficiently large physical apertures in terms of linear beam dynamics the goal of correcting nonlinear aberrations is to extend the dynamic aperture to or beyond the physical aperture. Methods discussed above to increase the dynamic aperture have been applied successfully to a variety of particle storage rings, especially by Autin [11] to the antiproton cooling ring ACOL, where a particularly large dynamic aperture is required.

#### Chromatic Aberrations

Correction of natural chromaticities is not a complete correction of all chromatic aberrations. For sensitive lattices nonlinear chromatic perturbation terms must be included. Both linear as well as nonlinear chromatic perturbations have been discussed in detail in Sect. 9.4.1. Such terms lead primarily to gradient errors and therefore the sextupole distribution must be chosen such that driving terms for half integer resonances are minimized. Together with tune shifts due to gradient field errors we observe also a variation of the betatron function. Chromatic gradient errors in the presence of sextupole fields are

\[p_{1}(z)=(k-m\eta)\,\delta \tag{17.50}\]

and the resulting variation of the betatron function has been derived in Sect. 15.3. For the perturbation (17.50) the linear variation of the betatron function with momentum is from (15.91)

\[\frac{\Delta\beta(z)}{\beta_{0}}=\frac{\delta}{2\sin 2\pi v_{0}}\int_{z}^{z+L} \beta(k-m\eta)\,\cos[2v_{0}(\varphi_{z}-\varphi_{\xi}+2\pi)]\mathrm{d}\xi\,, \tag{17.51}\]

where \(L\) is the length of the superperiod, \(\varphi_{z}=\varphi(z)\) and \(\varphi_{\xi}=\varphi(\xi)\). The same result can be expressed in the form of a Fourier expansion for \(N_{\mathrm{s}}\) superperiods in a ring lattice by

\[\frac{\Delta\beta}{\beta}=\delta\frac{v_{0}}{4\pi}\,\sum_{q}\frac{F_{q} \mathrm{e}^{\mathrm{i}N_{\mathrm{s}}q\varphi}}{v_{0}^{2}-(N_{\mathrm{s}}q/2)^ {2}}\,, \tag{17.52}\]

where

\[F_{q}=\frac{v_{0}}{2\pi}\int_{0}^{2\pi}\beta^{2}(k-m\eta)\mathrm{e}^{\mathrm{ i}N_{\mathrm{s}}q\varphi}\,\mathrm{d}\varphi\,. \tag{17.53}\]

Both expressions exhibit the presence of half integer resonances and we must expect the area of beam stability in phase space to be reduced for off momentum particles because of the increased strength of the resonances. Obviously, this perturbation does not appear in cases where the chromaticity is corrected locally so that \((k-m\eta)\equiv 0\) but few such cases exist. To minimize the perturbation of the betatron function, we look for sextupole distributions such that the Fourier harmonics are as small as possible by eliminating excessive "fighting" between sextupoles and by minimizing the resonant harmonic \(q=2v_{0}\). Overall, however, it is not possible to eliminate this beta-beat completely. With a finite number of sextupoles the beta-beat can be adjusted to zero only at a finite number of points along the beam line.

In colliding-beam storage rings, for example, we have specially sensitive sections just adjacent to the collision points. To maximize the luminosity the lattice is designed to produce small values of the betatron functions at the collision points and consequently large values in the adjacent quadrupoles. In order not to further increase the betatron functions there and make the lattice more sensitive to errors, one might choose to seek sextupole distributions such that the beta-beat vanishes at the collision point and its vicinity.

Having taken care of chromatic gradient errors we are left with the variation of geometric aberrations as a function of particle momentum. Specifically, resonance patterns vary and become distorted as the particle momentum is changed. Generally this should not cause a problem as long as the dynamic aperture can be optimized to exceed the physical aperture. A momentum error will introduce only a small variation to the dynamic aperture as determined by geometric aberrations for on momentum particles only. If, however, the dynamic aperture is limited by some higher-order resonances even a small momentum change can cause a big difference in the stable phase space area.

Analytical methods are useful to minimize detrimental effects of geometric and chromatic aberrations due to nonlinear magnetic fields. We have seen how by careful distribution of the chromaticity correcting sextupoles, resonant beam emittance blow up and excessive beating of the betatron functions for off momentum particles can be avoided or at least minimized within the approximations used. In Sect. 17.3, we will also find that sextupolar fields can produce strong tune shifts for larger amplitudes leading eventually to instability at nearby resonances. Here again a correct distribution of sextupoles will have a significant stabilizing effect. Although there are a number of different destabilizing effects, we note that they are driven by only a few third order resonances. Specifically, in large circular lattices a sufficient number of sextupoles and locations for additional sextupoles are available for an optimized correction scheme. In small rings such flexibility often does not exist and therefore the sophistication of chromaticity correction is limited. Fortunately, in smaller rings the chromaticity is much smaller and some of the higher-order aberrations discussed above are very small and need not be compensated. Specifically, the amplitude dependent tune shift is generally negligible in small rings while it is this effect which limits the dynamic aperture in most cases of large circular accelerators.

The optimization of sextupole distribution requires extensive analysis of the linear lattice and it is best to use a numerical program to do the well known but cumbersome work. At present the program OPA [12] is widely used. This program uses a linear lattice and adjusts the sextupoles such that chromaticities and some harmonics are corrected. With the new sextupole strengths the dynamic aperture can be obtained in the same program.

In trying to solve aberration problems in beam dynamics we are, however, mindful of approximations made and terms neglected for lack of mathematical tools to solve analytically the complete nonlinear dynamics in realistic accelerators. The design goals for circular accelerators become more and more demanding on our ability to control nonlinear aberrations. On one hand the required cross sectional area in the vicinity of the ideal orbit for a stable beam remains generally constant for most designs but the degree of aberrations is increased in an attempt to reach very special beam characteristics. As a consequence, the nonlinear perturbations become stronger and the limits of dynamic aperture occur for smaller amplitudes compared to less demanding lattices and require more and more sophisticated methods of chromaticity correction and control of nonlinear perturbations.

#### Particle Tracking

No mathematical methods are available yet to calculate analytically the limits of the dynamic aperture for any but the most simple lattices. High order approximations are required to treat strong aberrations in modern circular accelerator designs. The most efficient way to determine beam stability characteristics for a particular lattice design is to perform numerical particle tracking studies.

Perturbations of localized nonlinear fields on a particle trajectory are easy to calculate and tracking programs follow single particles along their path incorporating any nonlinear perturbation encountered. Since most nonlinear fields are small, we may use thin lens approximation and passage of a particle through a nonlinear field of any order inflicts therefore only a deflection on the particle trajectory. During the course of tracking the deflections of all non-linearities encountered are accumulated for a large number of turns and beam stability or instability is judged by the particle surviving the tracking or not, respectively. The basic effects of nonlinear fields in numerical tracking programs are therefore reduced to what actually happens to particles travelling through such fields producing results in an efficient way. Of course from an intellectual point of view such programs are not completely satisfactory since they serve only as tools providing little direct insight into actual causes for limitations to the dynamic aperture and instability.

The general approach to accelerator design is to develop first a lattice in linear approximation meeting the desired design goals followed by an analytical approach to include chromaticity correcting sextupoles in an optimized distribution. Further information about beam stability and dynamic aperture can at this point only be obtained from numerical tracking studies. Examples of widely used computer programs to perform such tracking studies are in historical order PATRICIA [7], RACETRACK [13], OPA [12] and more.

Tracking programs generally require as input an optimized linear lattice and allow then particle tracking for single particles as well as for a large number of particles simulating a full beam. Nonlinear fields of any order can be included as thin lenses in the form of isolated multipole magnets like sextupoles or a multipole errors of regular lattice magnets. The multipole errors can be chosen to be systematic or statistical and the particle momentum may have a fixed offset or may be oscillating about the ideal momentum due to synchrotron oscillations.

Results of such computer studies contribute information about particle dynamics which is not available otherwise. The motion of single particles in phase space can be observed together with an analysis of the frequency spectrum of the particle under the influence of all nonlinear fields included and at any desired momentum deviation.

Further information for the dynamics of particle motion can be obtained from the frequency spectrum of the oscillation. An example of this is shown in Fig. 17.8 as a function of oscillation amplitudes. For small amplitudes we notice only the fundamental horizontal betatron frequency \(v_{x}\). As the oscillation amplitude is increased this basic frequency is shifted toward lower values while more frequenciesappear. We note the appearance of higher harmonics of \(\nu_{x}\) due to the nonlinear nature of motion.

The motion of a particle in phase space and its frequency spectrum as a result of particle tracking can give significant insight into the dynamics of a single particle. For the proper operation of an accelerator, however, we also need to know the overall stability of the particle beam. To this purpose we define initial coordinates of a large number of particles distributed evenly over a cross section normal to the direction of particle propagation to be tested for stability. All particles are then tracked for many turns and the surviving particles are displayed over the original cross section at the beginning of the tracking thus defining the area of stability or dynamic aperture.

### Hamiltonian Perturbation Theory

The Hamiltonian formalism has been applied to derive tune shifts and to discuss resonance phenomena. This was possible by a careful application of canonical transformation to eliminate, where possible, cyclic variables from the Hamiltonian and obtain thereby an invariant of the motion. We have also learned that this "elimination" process need not be perfect. During the discussion of resonance theory, we observed that slowly varying terms function almost like cyclic variables giving us important information about the stability of the motion.

During the discussion of the resonance theory, we tried to transform perturbation terms to a higher order in oscillation amplitude than required by the approximation desired and where this was possible we could then ignore such higher-order fast-oscillating terms. This procedure was successful for all terms but resonant terms.

Figure 17.8: Frequency spectrum for betatron oscillations with increasing amplitudes (x) as determined by particle tracking with PATRICIA

In this section we will ignore resonant terms and concentrate on higher-order terms which we have ignored so far [14]. By application of a canonical identity transformation we try to separate from fast oscillating terms those which vary only slowly. To that goal, we start from the nonlinear Hamiltonian (16.30)

\[H=v_{0}\,J+p_{n}(\varphi)\,J^{n/2}\cos^{n}\psi. \tag{17.54}\]

Fast-oscillating terms can be transformed to a higher order by a canonical transformation which can be derived from the generating function

\[G_{1}=\psi\,J_{1}+g(\psi,\varphi)\,J_{1}^{n/2}\,, \tag{17.55}\]

where the function \(g(\psi,\varphi)\) is an arbitrary but periodic function in \(\psi\) and \(\varphi\) which we will determine later. From (17.55) we get for the new angle variable \(\psi_{1}\) and the old action variable \(J\)

\[\begin{split}&\psi_{1}=\tfrac{\mathrm{d}G_{1}}{\mathrm{d}J_{1}}= \psi\,+\,\tfrac{n}{2}g(\psi,\varphi)\,J^{n/2-1},\\ & J=\tfrac{\mathrm{d}G_{1}}{\mathrm{d}\psi}=J_{1}+\tfrac{\partial g }{\partial\psi}\,J_{1}^{n/2-1},\end{split} \tag{17.56}\]

and the new Hamiltonian is

\[H_{1}=H+\,\frac{\mathrm{d}G_{1}}{\mathrm{d}\,\varphi}=H+\,\frac{\partial g( \psi,\varphi)}{\partial\,\varphi}J_{1}^{n/2}\,. \tag{17.57}\]

We replace now the old variables \((\psi,J)\) in the Hamiltonian by the new variables \((J_{1},\psi_{1})\) and expand

\[J^{n/2}=\left(J_{1}+\,\frac{\partial g}{\partial\psi}J_{1}^{n/2}\right)^{n/2}= J_{1}^{n/2}+\,\frac{n}{2}\frac{\partial g}{\partial\,\psi}J_{1}^{n-1}+\cdots\,. \tag{17.58}\]

With (17.56), (17.58) the Hamiltonian (17.57) becomes

\[\begin{split} H_{1}&=v_{0}\,J_{1}+J_{1}^{n/2}\! \left[v_{0}\frac{\partial g}{\partial\psi}\,+p_{n}(\varphi)\cos^{n}\psi\,+\, \frac{\partial g}{\partial\varphi}\right]\\ &+J_{1}^{n-1}\left[\frac{n}{2}p_{n}(\varphi)\cos^{n}\psi\,\, \frac{\partial g}{\partial\psi}\right]+\mathcal{O}\left(J_{1}^{n+1/2}\right) \,.\end{split} \tag{17.59}\]

All terms of order \(n+1/2\) or higher in the amplitude \(J\) as well as quadratic terms in \(g(\psi,\varphi)\) or derivations thereof have been neglected. We still must express all terms of the Hamiltonian in the new variables and define therefore the function

\[Q(\psi,\varphi)=v_{0}\frac{\partial g}{\partial\,\psi}\,+p_{n}(\varphi)\cos^{ n}\psi\,+\,\frac{\partial w}{\partial\,\varphi}\,. \tag{17.60}\]In the approximation of small perturbations we have \(\psi_{1}\approx\psi\) or \(\psi_{1}=\psi+\Delta\psi\) and may expand (17.60) like

\[Q_{1}(\psi_{1},\varphi) = Q(\psi,\varphi)+\frac{\partial Q}{\partial\psi}\Delta\psi \tag{17.61}\] \[= Q(\psi,\varphi)+\frac{n}{2}g(\psi_{1},\varphi)J_{1}^{n/2-1} \frac{\partial Q}{\partial\psi}\,, \tag{17.62}\]

where we used the first equation of (17.56). The Hamiltonian can be greatly simplified if we make full use of the periodic but otherwise arbitrary function \(g(\psi_{1},\varphi)\). With (17.62) we obtain from (17.59)

\[H_{1} = v_{0}\,J_{1}+J_{1}^{n/2}Q_{1}(\psi_{1},\varphi) \tag{17.63}\] \[+ \frac{n}{2}J_{1}^{n-1}\left[p_{n}(\varphi)\cos^{n}\psi_{1}\frac{ \partial w}{\partial\psi}-g(\psi,\varphi)\,\frac{\partial Q}{\partial\psi} \right]+\cdots.\]

and we will derive the condition that

\[Q(\psi,\varphi)=0\,. \tag{17.64}\]

First we set

\[\cos^{n}\psi_{1}=\sum_{m=-n}^{n}a_{nm}\mathrm{e}^{\mathrm{i}m\psi_{1}} \tag{17.65}\]

and try an expansion of \(g(\psi_{1},\varphi)\) in a similar way by setting

\[g(\psi_{1},\varphi)=\sum_{m=-n}^{n}g_{m}(\varphi)\mathrm{e}^{\mathrm{i}m( \psi_{1}-v_{0}\varphi)}, \tag{17.66}\]

where the function \(g\) obviously is still periodic in \(\psi\) and \(\varphi\) as long as \(g_{m}(\varphi)\) is periodic. With

\[\frac{\partial g}{\partial\psi_{1}}=\sum_{m=-n}^{n}g_{m}(\varphi)\,\mathrm{i} m\,\mathrm{e}^{\mathrm{i}m(\psi_{1}-v_{0}\varphi)} \tag{17.67}\]

and

\[\frac{\partial g}{\partial\varphi}=\sum_{m=-n}^{n}\left[\frac{\partial g_{m} }{\partial\varphi}-\mathrm{i}v_{0}mg_{m}(\varphi)\right]\mathrm{e}^{\mathrm{ i}m(\psi_{1}-v_{0}\varphi)} \tag{17.68}\]we get instead of (17.60)

\[Q(\psi_{1},\varphi)\approx Q(\psi,\varphi)=\\ \mathrm{i}v_{0}\sum_{m=-n}^{n}mg_{m}\,\mathrm{e}^{\mathrm{i}m(\psi_ {1}-v_{0}\varphi)}+p_{n}(\varphi)\sum_{m=-n}^{n}a_{nm}\mathrm{e}^{\mathrm{i}m \psi_{1}}\\ +\sum_{m=-n}^{n}\left(\frac{\partial g_{m}}{\partial\varphi}- \mathrm{i}v_{0}mg_{m}\right)\mathrm{e}^{\mathrm{i}m(\psi_{1}-v_{0}\varphi)}=0\]

noting from (17.62) that the difference \(\Delta Q=Q(\psi_{1},\varphi)-Q(\psi,\varphi)\) contributes nothing to the term of order \(J_{1}^{n/2}\) for \(n>2\). The imaginary terms cancel and we get

\[Q(\psi_{1},\varphi)\approx p_{n}(\varphi)\sum_{m=-n}^{n}a_{nm}\mathrm{e}^{ \mathrm{i}m\psi}\,+\,\sum_{m=-n}^{n}\frac{\partial g_{m}}{\partial\varphi} \mathrm{e}^{\mathrm{i}m(\psi-v_{0}\varphi)}=0\,. \tag{17.69}\]

This equation must be true for all values of \(\varphi\) and therefore the individual terms of the sums must vanish independently

\[p_{n}(\varphi)a_{nm}\,+\,\frac{\partial g_{m}}{\partial\varphi}\mathrm{e}^{- \mathrm{i}mv_{0}\varphi}=0 \tag{17.70}\]

for all values of \(m\). After integration we have

\[g_{m}(\varphi)=g_{m0}-a_{nm}\int_{0}^{\varphi}p_{n}(\phi)\mathrm{e}^{\mathrm{ i}mv_{0}\varphi}\mathrm{d}\phi \tag{17.71}\]

and since the coefficients \(g_{m}(\varphi)\) must be periodic \(\left[g_{m}(\varphi)=g_{m}(\varphi+\frac{2\pi}{N})\right]\) where \(N\) is the super-periodicity, we are able to eventually determine the function \(g(\psi_{1},\varphi)\). With

\[g_{m}(\varphi)\mathrm{e}^{\mathrm{i}m(\psi_{1}-v_{0}\varphi)}=g_{m}\left( \varphi\,+\,\frac{2\pi}{N}\right)\mathrm{e}^{\mathrm{i}m\left(\psi_{1}-v_{0} \varphi-\frac{2\pi}{N}v_{0}\right)} \tag{17.72}\]

and (17.71) we have

\[g_{m0}\mathrm{e}^{\mathrm{i}m(\psi-v_{0}\varphi)}-a_{nm}\mathrm{e}^{ \mathrm{i}m(\psi-v_{0}\varphi)}\int_{0}^{\varphi}p_{n}(\bar{\phi})\mathrm{e}^{ \mathrm{i}mv_{0}\bar{\phi}}\mathrm{d}\bar{\phi}\\ =\mathrm{e}^{\mathrm{i}m\left(\psi-v_{0}\varphi-\frac{2\pi}{N}v_{ 0}\right)}\left(g_{m0}-a_{nm}\right)\int_{0}^{\varphi+\,\frac{2\pi}{N}}p_{n}( \bar{\phi})\mathrm{e}^{\mathrm{i}mv_{0}\bar{\phi}}\mathrm{d}\bar{\phi}\.\]Solving for \(g_{m0}\) we get

\[g_{m0}\left(1-\mathrm{e}^{\mathrm{i}m\frac{2\pi}{N}v_{0}}\right)=a_{mn}\int_{0}^{ \frac{2\pi}{N}}p_{n}(\bar{\phi})\mathrm{e}^{\mathrm{i}mv_{0}\bar{\phi}}\mathrm{d} \bar{\phi}. \tag{17.73}\]

A solution for \(g_{m0}\) exists only if there are no perturbations and \(p(\varphi)\equiv 0\) or if \(\left(1-\mathrm{e}^{\mathrm{i}m\frac{2\pi}{N}v_{0}}\right)\neq 0\). In other words we require the condition

\[mv_{0}\neq qN\,, \tag{17.74}\]

where \(q\) is an integer number. The canonical transformation (17.55) leads to the condition (17.64) only if the particle oscillation frequency is off resonance. We have therefore the result that all nonresonant perturbation terms can be transformed to higher-order terms in the oscillation amplitudes while the resonant terms lead to phenomena discussed earlier. From (17.73) we derive \(g_{m0}\), obtain the function \(g_{m}(\varphi)\) from (17.71) and finally the function \(g(\psi_{1},\varphi)\) from (17.66). Since \(Q(\psi_{1},\varphi)=0\), the Hamiltonian is from (17.63)

\[H_{1} = v_{0}\,J_{1}+J_{1}^{n/2}Q_{1}(\psi_{1},\varphi)\] \[\quad+\frac{n}{2}J_{1}^{n-1}\left[p_{n}(\varphi)\,\cos^{n}\psi_{1 }\,\frac{\partial g}{\partial\psi_{1}}-g(\psi_{1},\varphi)\,\frac{\partial Q }{\partial\psi_{1}}\right]+\cdots.\]

Nonresonant terms appear only in order \(J_{1}^{n-1}\). As long as such terms can be considered small we conclude that the particle dynamics is determined by the linear tune \(v_{0}\), a tune shift or tune spread caused by perturbations and resonances. Note that the Hamiltonian (17.75) is not the complete form but addresses only the nonresonant case of particle dynamics while the resonant case of the Hamiltonian has been derived earlier.

We will now continue to evaluate (17.75) and note that the product

\[g(\psi_{1},\varphi)\frac{\partial Q(\psi_{1},\varphi)}{\partial\psi_{1}}=0 \tag{17.76}\]

in this approximation and get

\[T(\psi,\varphi)=\frac{n}{2}p_{n}(\varphi)\cos^{n}\psi\,\frac{\partial g}{ \partial\psi}\, \tag{17.77}\]

where we have dropped the index on \(\psi\) and set from now on \(\psi_{1}=\psi\) which is not to be confused with the variable \(\psi\) used before the transformation (17.55). Using the Fourier spectrum for the perturbations and summing over all but resonant terms\(q\neq q_{\rm r}\) we get from (17.73)

\[g_{m0}\left(1-{\rm e}^{{\rm i}m\frac{2\pi}{N}v_{0}}\right) =a_{nm}\sum_{q\neq q_{\rm r}}\int_{0}^{\frac{2\pi}{N}}p_{nq}{\rm e} ^{{\rm i}(mv_{0}-qN)\varphi}{\rm d}\varphi\] \[=a_{nm}\sum_{q\neq q_{\rm r}}p_{nq}\frac{{\rm e}^{{\rm i}m\frac{2 \pi}{N}v_{0}}-1}{{\rm i}\left(mv_{0}-qN\right)}\;, \tag{17.78}\]

or

\[g_{m0}={\rm i}\,a_{nm}\sum_{q\neq q_{\rm r}}\frac{p_{nq}}{mv_{0}-qN}\;. \tag{17.79}\]

Note that we have excluded in the sums the resonant terms \(q=q_{\rm r}\) where \(m_{r}v_{0}-q_{\rm r}N=0\). These resonant terms include also terms \(q=0\) which do not cause resonances of the ordinary type but lead to tune shifts and tune spreads. After insertion into (17.71) and some manipulations we find

\[g_{m}\left(\varphi\right) ={\rm i}a_{nm}\sum_{q\neq q_{\rm r}}\frac{p_{nq}}{mv_{0}-qN}-a_{nm} \sum_{q\neq q_{\rm r}}\int_{0}^{\varphi}p_{nq}{\rm e}^{{\rm i}(mv_{0}-qN) \varphi}{\rm d}\phi \tag{17.80}\] \[={\rm i}a_{nm}\sum_{q\neq q_{\rm r}}p_{nq}\frac{{\rm e}^{{\rm i}(mv _{0}-qN)\varphi}}{mv_{0}-qN}\;,\]

and with (17.66)

\[g(\psi,\varphi)={\rm i}\sum_{m=-n}^{n}\,\sum_{q\neq q_{\rm r}}\,\frac{a_{nm}p _{nq}}{mv_{0}-qN}\,{\rm e}^{{\rm i}m\psi}\,{\rm e}^{-{\rm i}qN)\varphi}\;. \tag{17.81}\]

From (17.77) we get with (17.65) and (17.81)

\[T(\psi,\varphi)={\rm i}\,\frac{n}{2}\sum_{q\neq q_{\rm r}}p_{nq}\,{\rm e}^{-{ \rm i}qN\varphi}\sum_{m=-n}^{n}a_{nm}\,{\rm e}^{{\rm i}m\psi}\,m\,g(\psi, \varphi)\;. \tag{17.82}\]

This function \(T(\psi,\varphi)\) is periodic in \(\psi\) and \(\varphi\) and we may apply a Fourier expansion like

\[T(\psi,\varphi)=\sum_{r}\sum_{s\neq\frac{r_{0}}{N}}T_{rs}\,{\rm e}^{{\rm i}( r\psi-sN\varphi)}\;, \tag{17.83}\]where the coefficients \(T_{rs}\) are determined by

\[T_{rs}=\,\frac{N}{4\pi^{2}}\int_{0}^{2\pi}\mathrm{e}^{-\mathrm{i}r\psi}\;\mathrm{ d}\psi\int_{0}^{2\pi/N}\mathrm{e}^{\mathrm{i}sN\varphi}T(\psi,\varphi)\, \mathrm{d}\varphi\;. \tag{17.84}\]

To evaluate (17.84) it is most convenient to perform the integration with respect to the betatron phase \(\psi\) before we introduce the expansions with respect to \(\varphi\). Using (17.65), (17.66), (17.77), we get from (17.84) after some reordering

\[T_{rs}=\mathrm{i}\,\frac{nN}{4\pi}\sum_{m=-n}^{n}m\int_{0}^{2\pi} \sum_{j=-n}^{n}\frac{a_{nj}}{2\pi}\mathrm{e}^{\mathrm{i}(j+m-r)\psi}\;\mathrm{ d}\psi\] \[\times\int_{0}^{2\pi/N}p_{n}(\varphi)\,g_{m}(\varphi)\mathrm{e}^{ \mathrm{i}(mv_{0}-sN)\varphi}\;\mathrm{d}\varphi\;.\]

The integral with respect to \(\psi\) is zero for all values \(j+m-r\neq 0\) and therefore equal to \(a_{n,r-m}\)

\[T_{rs}=\mathrm{i}\,\frac{nN}{4\pi}\sum_{m=-n}^{m}m\,a_{m,r-m}\int_{0}^{2\pi/N} p_{n}(\varphi)\,g_{m}(\varphi)\;\mathrm{e}^{-\mathrm{i}(mv_{0}-sN)\varphi}\; \mathrm{d}\varphi\;. \tag{17.85}\]

Expressing the perturbation \(p_{n}(\varphi)\) by its Fourier expansion and replacing \(g_{m}(\varphi)\) by (17.80), (17.85) becomes

\[T_{rs}=-\frac{n}{2}\sum_{m=-n}^{n}ma_{m,r-m}\,a_{n,m}\sum_{q\neq q_{r}}\frac{p _{n,s-q}\,p_{n,q}}{mv_{0}-qN}\;. \tag{17.86}\]

With this expression we have fully defined the function \(T(\psi,\varphi)\) and obtain for the non-resonant Hamiltonian (17.75)

\[H=v_{0}\,J+J^{n-1}\sum_{r}\sum_{s\neq\frac{r}{N}v_{0}}T_{rs}\;\mathrm{e}^{ \mathrm{i}(r\psi-sN\varphi)}\;. \tag{17.87}\]

We note in this result a higher-order amplitude dependent tune spread which has a constant contribution \(T_{00}\) as well as oscillatory contributions.

Successive application of appropriate canonical transformations has lead us to derive detailed insight into the dynamics of particle motion in the presence of perturbations. Of course, every time we applied a canonical transformation of variables it was in the hope of obtaining a cyclic variable. Except for the first transformation to action-angle variables, this was not completely successful. However, we were able to extract from perturbation terms depending on both action-angle variables such elements that do not depend on the angle variable. As a result, we are now able to determine to a high order of approximation shifts in the betatron frequency caused by perturbations as well as the occurrence and nature of resonances.

Careful approximations and simplifications had to be made to keep the mathematical formulation manageable. Specifically we had to restrict the perturbation theory in this section to one order of multipole perturbation and we did not address effects of coupling between horizontal and vertical betatron oscillations.

From a more practical view point one might ask to what extend this higher-order perturbation theory is relevant for the design of particle accelerators. Is the approximation sufficient or is it more detailed than needed? As it turns out so often in physics we find the development of accelerator design to go hand in hand with the theoretical understanding of particle dynamics. Accelerators constructed up to the late sixties were designed with moderate focusing and low chromaticities requiring no or only very weak sextupole magnets. In contrast more modern accelerators require much stronger sextupole fields to correct for the chromaticities and as a consequence, the effects of perturbations, in this case third-order perturbations, become more and more important. The ability to control the effects of such perturbations actually limits the performance of particle accelerators. For example, in colliding-beam storage rings the strongly nonlinear fields introduced by the beam-beam effect limit the attainable luminosity while a lower limit on the attainable beam emittance for synchrotron light sources or damping rings is determined by strong sextupole fields.

##### 17.3.1 Tune Shift in Higher Order

In (16.36) we found the appearance of tune shifts due to even order multipole perturbations only. Third-order sextupole fields, therefore, would not affect the tunes. This was true within the degree of approximation used at that point. In this section, however, we have derived higher-order tune shifts and should therefore discuss again the effect of sextupolar fields on the tune.

Before we evaluate the sextupole terms, however, we like to determine the contribution of a quadrupole perturbation to the higher-order tune shift. In lower order we have derived earlier a coherent tune shift for the whole beam. We use (17.86) and calculate \(T_{00}\) for \(n=2\)

\[T_{00}=\sum_{q\neq q_{e}}p_{2,q}\,p_{2,-q}\sum_{m=-2}^{2}\frac{ma_{2,m}\,a_{2,-m}}{mv_{0}-qN}\,. \tag{17.88}\]With \(4a_{2,2}=a_{2,-2}=2a_{2,0}=1\) and \(a_{2,1}=a_{2,-1}=0\) the term in the bracket becomes

\[\frac{-2}{-2v_{0}-qN}+\frac{2}{2v_{0}-qN}=\frac{2qN}{\left(2v_{0}\right)^{2}- \left(qN\right)^{2}}\]

and (17.88) is simplified to

\[T_{00}=-\sum_{q\neq q_{t}}p_{2,q}\,p_{2,-q}\frac{2qN}{\left(2v_{0}\right)^{2} \,-\,\left(qN\right)^{2}}\,. \tag{17.89}\]

In this summation we note the appearance of the index \(q\) in pairs as a positive and a negative value. Each such pair cancels and therefore

\[T_{00,2}=0\,, \tag{17.90}\]

where the index \({}_{2}\) indicates that this coefficient was evaluated for a second-order quadrupole field. This result is not surprising since all quadrupole fields contribute directly to the tune and formally a quadrupole field perturbation cannot be distinguished from a "real" quadrupole field.

In a similar way we derive the \(T_{00}\) coefficient for a third-order multipole or a sextupolar field. From (17.86) we get for \(n=3\)

\[T_{00,3}=-\frac{3}{2}\sum_{q\neq q_{t}}p_{3,q}\,p_{3,-q}\sum_{m=-3}^{3}\, \frac{ma_{3,m}\,a_{3,-m}}{mv_{0}-qN}\,. \tag{17.91}\]

Since \(\cos^{3}\psi\) is an even function we have \(a_{3,m}=a_{3,-m},a_{3,1}=\frac{3}{8}\) and \(a_{3,3}=\frac{1}{8}\). The second sum in (17.91) becomes now

\[\frac{1}{64}\left(\frac{3}{3v_{0}+qN}+\frac{q}{v_{0}+qN}+\frac{q} {v_{0}-qN}+\frac{3}{3v_{0}-qN}\right)\\ =\frac{1}{64}\left(\frac{18v_{0}}{v_{0}^{2}-\left(qN\right)^{2} }\,+\frac{18v_{0}}{\left(3v_{0}\right)^{3}-\left(qN\right)^{2}}\right)\,,\]

and after separating out the terms for \(q=0\), (17.91) becomes

\[T_{00,3} =-\frac{15}{32v_{0}}p_{3,0}^{2} \tag{17.92}\] \[\qquad-\frac{27v_{0}}{64}\sum_{q\neq q_{t}}p_{3,q}\,p_{3,-q}\, \left[\frac{1}{v_{0}^{2}-\left(qN\right)^{2}}\,+\,\frac{1}{\left(3v_{0} \right)^{3}-\left(qN\right)^{2}}\right].\]This expression in general is nonzero and we found, therefore, that sextupole fields indeed, contribute to a tune shift although in a high order of approximation. This tune shift can actually become very significant for strong sextupoles and for tunes close to an integer or third integer resonances. Although we have excluded resonances (\(q=q_{\mathrm{r}}\)), terms close to resonances become important. Obviously, the tunes should be chosen such as to minimize both terms in the bracket of (17.92). This can be achieved with \(v_{0}\ =qN+\,\frac{1}{2}N\) and \(3v_{0}\ =rN+\,\frac{1}{2}N\) where \(q\) and \(r\) are integers. Eliminating \(v_{0}\) from both equations we get the condition \(3q-r+1=0\) or \(r=3q+1\). With this we finally get from the two tune conditions the relation \(2v_{0}\ =(2q+1)\,N\) or

\[v_{\mathrm{opt}}=\frac{2q+1}{2}N\,. \tag{17.93}\]

Of course, an additional way to minimize the tune shift is to arrange the sextupole distribution in such a way as to reduce strong harmonics in (17.92). In summary, we find for the non-resonant Hamiltonian in the presence of sextupole fields.

\[H_{3}=v_{0}J+T_{00,3}J^{2}+\mathrm{higher\ order\ terms} \tag{17.94}\]

and the betatron oscillation frequency or tune is given by

\[v=v_{0}\ +2T_{00,3}J. \tag{17.95}\]

In this higher-order approximation of beam dynamics we find that sextupole fields cause an amplitude dependent tune shift in contrast to our earlier first-order conclusion

\[\frac{\Delta v}{v_{0}}=\frac{v-v_{0}}{v_{0}}=T_{00,3}\left(\gamma u^{2}+2u\,u ^{\prime}+\beta\ u^{\prime\ 2}\right)=T_{00,3}\,\epsilon\, \tag{17.96}\]

where we have used (5.59) with \(\epsilon\) the emittance of a single particle oscillating with a maximum amplitude \(a^{2}=\beta\epsilon\). We have shown through higher-order perturbation theory that odd order nonlinear fields like sextupole fields, can produce amplitude dependent tune shifts which in the case of sextupole fields are proportional to the square of the betatron oscillation amplitude and therefore similar to the tune shift caused by octupole fields. In a beam where particles have different betatron oscillation amplitudes this tune shift leads to a tune spread for the whole beam.

In practical accelerator designs requiring strong sextupoles for chromaticity correction it is mostly this tune shift which moves large amplitude particles onto a resonance thus limiting the dynamic aperture. Since this tune shift is driven by the integer and third-order resonance, it is imperative in such cases to arrange the sextupoles such as to minimize this driving term for geometric aberration.

## Problems

### 17.1 (S)

Derive the expression for the second-order matrix element \(T_{166}\) and give a physical interpretation for this term.

### 17.2 (S)

Show that the perturbation proportional to \(x_{0}^{2}\) is \(p\left(z\left|x_{0}^{2}\right.\right)=\left[\left(-\frac{1}{2}m-\kappa^{3}-2 \kappa k\right)C+\frac{1}{2}\kappa\ C^{\prime\ 2}\right]x_{0}^{2}\), where \(C=C\left(z\right)=\cos\sqrt{k}z\) and \(C^{\prime}=C^{\prime}\left(z\right)\) and the second-order matrix element

\(T_{111}=\left(-\frac{1}{2}m-\kappa^{3}-2\kappa k\right)\frac{1}{3k}\left[kS^{2 }+\left(1-C\right)\right]+\frac{1}{6}\kappa\left[2\left(1-C\right)-kS^{2}\right]\).

### 17.3 (S)

Consider a large circular accelerator made of many FODO cells with a phase advance of \(90^{\circ}\) per cell. Locate chromaticity correcting sextupoles in the center of each quadrupole and calculate the magnitude for one of the five expressions (17.45)-(17.49). Now place non-interleaved sextupole in pairs \(180^{\circ}\) apart and calculate the same two expressions for the new sextupole distribution.

### 17.4 (S)

Use the lattice of Problem 17.3 and determine the tunes of the ring. Are the tunes the best choices for the super-periodicity of the ring to avoid resonance driven sextupole aberrations? How would you go about improving the situation?

Expand the second-order transformation matrix to include path length terms relevant for the design of an isochronous beam transport system and derive expressions for the matrix elements. Which elements must be adjusted and how would you do this? Which parameters would you observe to control your adjustment?

Sextupoles are used to compensate for chromatic aberrations at the expense of geometric aberrations. Derive a condition for which the geometric aberration has become as large as the original chromatic aberration. What is the average perturbation of geometric aberrations on the betatron motion? Try to formulate a "rule of thumb" stability criteria for the maximum sextupole strength. Is it better to place a chromaticity correcting sextupole at a high beta location (weak sextupole) or at a low beta location (weak aberration)?

Consider both sextupole distributions of Problem 17.3 and form a phasor diagram of one of expressions (17.45)-(17.49) for the first four or more FODO cells. Discuss desirable features of the phasor diagram and explain why the \(-\mathcal{I}\) correction scheme works well. A phasor diagram is constructed by adding vectorially each term of an expression (17.45)-(17.49) going along a beam line.

The higher-order chromaticity of a lattice may include a strong quadratic term. What dependence on energy would one expect in this case for the beta beat? Why? Can your findings be generalized to higher-order terms?

## References

* [1] K.L. Brown, R. Belbeoch, P. Bounin, Rev. Sci. Instrum. **35**, 481 (1964)
* [2] K.L. Brown, The adjustable phase planar helical undulator, in _5th International Conference on High Energy Accelerators_, Frascati, Italy (1965)
* [3] K.L. Brown, Adv. Part. Phys. **1**, 71 (1967)
* [4] K.L. Brown, D.C. Carey, C.H. Iselin, F. Rothacker, Technical Report SLAC-75, CERN 73-16, SLAC-91, CERN-80-4, CERN,FNAL,SLAC (1972)
* [5] Slac linear collider, conceptual design report. Technical Report SLAC-229, SLAC, Stanford, CA (1981)
* [6] Pep technical design report. Technical Report SLAC-189, LBL-4299, SLAC, Stanford, CA (1976)
* [7] H. Wiedemann, Chromaticity correction in large storage rings. Technical Report PEP-Note 220, Stanford Linear Accelerator Center, Stanford, CA (1976)
* [8] K.L. Brown, R.V. Servranckx, in _11th International Conference on High Energy Accelerators_. Stanford linear Accelerator Center, Birkauser, Basel (1980)
* [9] J.J. Murray, K.L. Brown, T. Fieguth, in _1987 IEEE Particle Accelerators Conference_, Washington. IEEE Cat. No. 87CH2387-9 (1987)
* [10] L. Emery, A wiggler-based ultra-low-emittance damping ring lattice and its chromatic correction. Ph.D. thesis, Stanford University, Stanford, CA (1990)
* [11] B. Autin, The cern anti-proton collector. Technical Report CERN 74-2, CERN, CERN, Geneva (1974)
* [12] A. Streun, Opa. Available from PSI (2010)
* [13] A. Wrulich, in _Proceedings of Workshop on Accelerator Orbit and Particle Tracking Programs_. Technical Report BNL-317615, BNL, Brookhaven, NY (1982)
* [14] F.T. Cole, Longitudinal motion in circular accelerators, in _Physics of Particle Accelerators_, vol. AIP 153, ed. by M. Month, M. Dienes (The American Institute of Physics, New York, 1987), p. 44


