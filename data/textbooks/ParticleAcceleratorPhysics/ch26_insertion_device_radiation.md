## Chapter 26 Insertion Device Radiation

Synchrotron radiation from bending magnets is characterized by a wide spectrum from microwaves up to soft or hard x-rays as determined by the critical photon energy. To optimally meet the needs of basic research with synchrotron radiation, it is desirable to provide radiation characteristics that cannot be obtained from ring bending magnets but require special magnets. The field strength of bending magnets and the maximum particle beam energy in circular accelerators like a storage ring is fixed leaving no adjustments to optimize the synchrotron radiation spectrum for particular experiments. To generate specific synchrotron radiation characteristics, radiation is often produced from insertion devices installed along the particle beam path. Such insertion devices introduce no net deflection of the beam and can therefore be incorporated in a beam line without changing its geometry. Motz [1] proposed first the use of wiggler magnets to optimize characteristics of synchrotron radiation. By now, such magnets have become the most common insertion devices consisting of a series of alternating magnet poles deflecting the beam periodically in opposite directions as shown in Fig. 26.1.

In Chap. 24 the properties of wiggler radiation were discussed shortly in an introductory way. Here we concentrate on more detailed and formal derivations of radiation characteristics from relativistic electrons passing through periodic magnets.

There is no fundamental difference between wiggler and undulator radiation. One is the stronger/weaker version of the other. The deflection in an undulator is weak and the transverse particle momentum remains nonrelativistic. The motion is purely sinusoidal in a sinusoidal field, and the emitted radiation is monochromatic at the particle oscillation frequency which is the Lorentz-contracted periodicity of the undulator period. Since the radiation is emitted from a moving source the observer in the laboratory frame of reference then sees a Doppler shifted frequency. We call this monochromatic radiation the fundamental radiation or radiation at the fundamental frequency of the undulator.

As the undulator field is increased, the transverse motion becomes stronger and the transverse momentum starts to become relativistic. As a consequence, the so far purely sinusoidal motion becomes periodically distorted causing the appearance of harmonics of the fundamental monochromatic radiation. These harmonics increase in number and density with further increase of the magnetic field and, at higher frequencies, eventually merge into one broad spectrum characteristic for wiggler or bending magnet radiation. At very low frequencies, the theoretical spectrum is still a line spectrum showing the harmonics of the revolution frequency. Of course, there is a low frequency cut-off at a wavelength comparable or longer than vacuum chamber dimensions which therefore do not show-up as radiation.

An insertion device does not introduce a net deflection of the beam and we may therefore choose any arbitrary field strength which is technically feasible to adjust the radiation spectrum to experimental needs. The radiation intensity from a wiggler magnet also can be made much higher compared to that from a single bending magnet. A wiggler magnet with say ten poles acts like a string of ten bending magnets or radiation sources aligned in a straight line along the photon beam direction. The effective photon source is therefore ten times more intense than the radiation from a single bending magnet with the same field strength.

Wiggler magnets come in a variety of types with the flat wiggler magnet being the most common. In this wiggler type only the component \(B_{y}\) is nonzero deflecting the beam in the horizontal plane. To generate circularly or elliptically polarized radiation, a helical wiggler magnet [2] may be used or a combination of several flat wiggler magnets deflecting the beam in orthogonal planes which will be discussed in more detail in Sect. 26.3.2.

### Particle Dynamics in a Periodic Field Magnet

Insertion devices are characterized by the requirement that

\[\int B_{\perp}\mathrm{d}z=0.\]

As discussed in Chap. 15 this requirement demands that the first and second integral must be made zero with the use of steering magnets before and after the undulator.

Figure 26: Trajectory of a particle beam in a flat wiggler

This correction is sufficient from the beam stability point of view. However, it does not address the effect of field tolerances on the intensity of radiation into harmonics. For example, the curved trajectory in Fig. 15.4 can reduce the radiation intensity because not all periods radiate in the same direction and constructive interference of light emitted by individual periods is not optimum. Therefore the overall trajectory curvature in Fig. 15.4 should be corrected as discussed in Chap. 15. Furthermore, variations in field strength and period length from period to period in the undulator can seriously diminish the radiation intensity especially in the higher harmonics. The effect of such errors on individual harmonic intensities have been studied [3]. A special shimming procedure has been proposed by Elleaume [4] to transform an undulator with construction tolerances to an almost ideal undulator giving close to perfect intensities for about a dozen harmonics. If the shimming is done correctly the long coil mentioned in Chap. 15 is not necessary anymore. In the following discussion we assume that the integrals have been corrected and that the undulator has been shimmed.

Particle dynamics and resulting radiation characteristics for an undulator have been derived first by Motz [1] and later in more detail by other authors [5, 6]. A sinusoidally varying vertical field causes a periodic deflection of particles in the \((x,z)\)-plane shown in Fig. 26.1. To describe the particle trajectory, we use the equation of motion

\[\frac{\mathbf{n}}{\rho}=\frac{ec}{mc^{2}\gamma\beta^{2}}[\boldsymbol{\beta} \times\boldsymbol{B}], \tag{26.1}\]

where \(\beta\) is the particle velocity and get with (6.110) the equations of motion in component form

\[\begin{array}{l}\frac{\mathrm{d}^{2}x}{\mathrm{d}r^{2}}=-\frac{eB_{0}}{ \gamma\beta m}\frac{\mathrm{d}z}{\mathrm{d}t}\cos\left(k_{\mathrm{p}}z\right) \\ \frac{\mathrm{d}^{2}z}{\mathrm{d}r^{2}}=+\frac{eB_{0}}{\gamma\beta m}\frac{ \mathrm{d}x}{\mathrm{d}t}\cos\left(k_{\mathrm{p}}z\right)\end{array}, \tag{26.2}\]

where we have set \(k_{\mathrm{p}}=2\pi/\lambda_{\mathrm{p}}\) and \(\mathrm{d}z=\beta c\mathrm{d}t\) with \(\beta=v/c\).

Equations (26.2) describe the coupled motion of a particle in the sinusoidal field of a flat wiggler magnet. This coupling is common to the particle motion in any magnetic field but generally in beam dynamics we set \(\mathrm{d}z/\mathrm{d}t\approx v\) and \(\mathrm{d}x/\mathrm{d}t\approx 0\) because \(\mathrm{d}x/\mathrm{d}t\ll\mathrm{d}z/\mathrm{d}t\). This approximation is justified in most beam transport applications for relativistic particles, but here we have to be cautious not to neglect effects that might be of relevance on a very short time or small geometric scale comparable to the oscillation period and wavelength of synchrotron radiation.

We will keep the \(\mathrm{d}x/\mathrm{d}t\)-term and get from (26.2) with \(\mathrm{d}z/\mathrm{d}t\approx v\) and after integrating twice that the particle trajectory follows the magnetic field in the sense that the oscillatory motion reaches a maximum where the magnetic field reaches a maximum and crosses the beam axis where the field is zero. We start at the time \(t=0\) in the middle of a magnet pole where the transverse velocity \(\dot{x}_{0}=0\) while the longitudinal velocity \(\dot{z}_{0}=\beta c\) and integrate both equations (26.2) utilizing the integral of the first equation in the second to get

\[\begin{array}{l}\frac{\mathrm{d}x}{\mathrm{d}t}=-\beta c\frac{K}{\bar{\beta} \gamma}\sin\left(k_{\mathrm{p}}z\right),\\ \frac{\mathrm{d}z}{\mathrm{d}t}=\beta c\left[1-\frac{K^{2}}{2\beta^{2}\gamma^ {2}}\sin^{2}\left(k_{\mathrm{p}}z\right)\right].\end{array} \tag{26.3}\]

The transverse motion describes the expected oscillatory motion and the longitudinal velocity \(v\) exhibits a periodic modulation reflecting the varying projection of the velocity vector to the \(z\)-axis. Closer inspection of this velocity modulation shows that its frequency is twice that of the periodic motion. It is convenient to describe the longitudinal particle motion with respect to a Cartesian reference frame moving uniformly along the \(z\)-axis with the average longitudinal particle velocity \(\bar{\beta}c=\left\langle\dot{z}\right\rangle\) which can be derived from (26.3b)

\[\bar{\beta}=\beta\left(1-\frac{K^{2}}{4\beta^{2}\gamma^{2}}\right). \tag{26.4}\]

In this reference frame the particle follows a figure-of-eight trajectory composed of the transverse oscillation and a longitudinal oscillation with twice the frequency. We will come back to this point since both oscillations contribute to the radiation spectrum. A second integration of (26.3b) results finally in the equation of motion in component representation

\[\begin{array}{l}x(t)=\frac{K}{\bar{\beta}\gamma k_{\mathrm{p}}}\cos\left(k_ {\mathrm{p}}\bar{\beta}ct\right),\\ z(t)=\bar{\beta}ct+\frac{K^{2}}{8\beta^{2}\gamma^{2}k_{\mathrm{p}}}\sin\left(2k _{\mathrm{p}}\bar{\beta}ct\right),\end{array} \tag{26.5}\]

where we set \(z=\bar{\beta}ct\). The maximum amplitude \(a\) of the transverse particle oscillation is finally

\[a=\frac{K}{\beta\gamma k_{\mathrm{p}}}=\frac{\lambda_{\mathrm{p}}K}{2\pi\beta \gamma}. \tag{26.6}\]

This last expression gives another simple relationship between the wiggler strength parameter and the transverse displacement of the beam trajectory

\[a\left(\mu\mathrm{m}\right)=0.8133\frac{\lambda_{\mathrm{p}}\left(\mathrm{cm} \right)K}{E\left(\mathrm{GeV}\right)}. \tag{26.7}\]

For most cases, this beam displacement is very small.

### Undulator Radiation

The physical process of undulator radiation is not different from the radiation produced from a single bending magnet. However, the radiation received at great distances from the undulator exhibits special features which we will discus in more detail. Basically, we observe an electron performing \(N_{\mathrm{p}}\) oscillations while passing through an undulator with \(N_{\mathrm{p}}\) undulator periods. The observed radiation spectrum is the Fourier transform of the electron motion and therefore quasi-monochromatic with a finite line width inversely proportional to the number of oscillations performed.

#### Fundamental Wavelength

Undulator radiation can also be viewed as a superposition of radiation fields from \(N_{\mathrm{p}}\) sources yielding quasi-monochromatic radiation as a consequence of interference. To see that, we observe the radiation at an angle \(\vartheta\) with respect to the path of the electron as shown in Fig. 26.2.

The electron travels on its path at an average velocity given by (26.4) and it takes the time

\[\tau\,=\,\frac{\lambda_{\mathrm{p}}}{c\bar{\beta}}\,=\,\frac{\lambda_{\mathrm{ p}}}{c\beta[1-K^{2}/(4\gamma^{2})]} \tag{26.8}\]

to move along one undulator period. During that same time, the radiation front proceeds a distance

\[s_{\mathrm{ph}}\,=\,\tau c\,=\,\frac{\lambda_{\mathrm{p}}}{\beta[1-K^{2}/(4 \gamma^{2})]} \tag{26.9}\]

moving ahead of the particle since \(s_{\mathrm{ph}}>\tau c\bar{\beta}\). For constructive superposition of radiation from all undulator periods, we require that the difference \(s_{\mathrm{ph}}-\lambda_{\mathrm{p}}\,\cos\vartheta\) be equal to an integer multiple of the wavelength \(\lambda_{k}\) or for small observation angles \(\vartheta\,\ll 1\)

\[k\lambda_{k}\,=\,\frac{\lambda_{\mathrm{p}}}{\beta[1-K^{2}/(4\gamma^{2})]}- \lambda_{\mathrm{p}}(1-\tfrac{1}{2}\vartheta^{2}). \tag{26.10}\]

Figure 26.2: Interference of undulator radiationAfter some manipulations, we get with \(K^{2}/\gamma^{2}\ll 1\) and \(\beta\approx 1\) for \(\lambda_{k}\)

\[\lambda_{k}=\frac{\lambda_{\rm p}}{2\gamma^{2}k}\left(1+\tfrac{1}{2}K^{2}+ \gamma^{2}\vartheta^{2}\right). \tag{26.11}\]

The lowest harmonics is defined by \(k=1\) and is called the fundamental undulator wavelength.

From an infinitely long undulator, the radiation spectrum consists of spectral lines at a wavelength determined by (26.11). In particular, we note that the shortest wavelength is emitted into the forward direction while the radiation at a finite angle \(\vartheta\) appears red shifted by the Doppler effect. For an undulator with a finite number of periods, the spectral lines are widened to a width of about \(1/N_{\rm p}\) or less as we will discuss in the next section.

#### Radiation Power

The radiation power is from (25.41)

\[P=\tfrac{2}{3}r_{\rm c}mc|\hat{\beta}^{*}|_{\rm r}^{2}, \tag{26.12}\]

where \({}^{*}\) indicates quantities to be evaluated in the particle reference system. We may use this expression in the particle system to calculate the total radiated energy from an electron passing through an undulator. The transverse particle acceleration is expressed by \(m\dot{\bf v}^{*}={\rm d}{\bf p}_{\perp}/{\rm d}t^{*}=\gamma{\rm d}{\bf p}_{ \perp}/{\rm d}t\) where we used \(t^{*}=t/\gamma\) and inserting into (26.12) we get

\[P=\tfrac{2}{3}\frac{r_{\rm c}}{mc}\frac{\gamma^{2}}{mc}\left(\frac{{\rm d}{\bf p }_{\perp}}{{\rm d}t}\right)^{2}. \tag{26.13}\]

The transverse momentum is determined by the particle deflection in the undulator with a period length \(\lambda_{\rm p}\) and is for a particle of momentum \(cp_{0}\)

\[p_{\perp}=\hat{p}\,\sin\omega_{\rm p}t\,, \tag{26.14}\]

where \(\hat{p}=p_{0}\theta\) and \(\omega_{\rm p}=ck_{\rm p}=2\pi c/\lambda_{\rm p}\). The angle \(\theta=K/\gamma\) is the maximum deflection angle defined in (6.121). With these expressions and averaging over one period, we get from (26.13) for the instantaneous radiation power from a charge \(e\) traveling through an undulator

\[P_{\rm inst}=\tfrac{1}{3}cr_{\rm c}mc^{2}\gamma^{2}K^{2}k_{\rm p}^{2}\,, \tag{26.15}\]

where \(r_{\rm c}\) is the classical electron radius. The duration of the radiation pulse is equal to the travel time through an undulator of length \(L_{\rm u}=\lambda_{\rm p}N_{\rm p}\) and the total radiated energy per electron is therefore

\[\Delta E=\tfrac{1}{3}r_{\mathrm{c}}mc^{2}\gamma^{2}K^{2}k_{\mathrm{p}}^{2}L_{ \mathrm{u}}\,. \tag{26.16}\]

In more practical units

\[\Delta E(\mathrm{eV}) = C_{\mathrm{u}}\frac{E^{2}K^{2}}{\lambda_{\mathrm{p}}^{2}}L_{ \mathrm{u}}=725.69\frac{E^{2}K^{2}}{\lambda_{\mathrm{p}}^{2}(\mathrm{cm})}\,L_ {\mathrm{u}} \tag{26.17}\]

with

\[C_{\mathrm{u}} = \frac{4\pi^{2}r_{\mathrm{c}}}{3mc^{2}}=7.2569\times 10^{-20}\frac{ \mathrm{m}}{\mathrm{eV}}. \tag{26.18}\]

The average total undulator radiation power for an electron beam circulating in a storage ring is then just the radiated energy (26.16) multiplied by the number of particles \(N_{\mathrm{b}}\) in the beam and the revolution frequency or

\[P_{\mathrm{avg}} = \tfrac{1}{3}r_{\mathrm{c}}cmc^{2}\gamma^{2}K^{2}k_{\mathrm{p}}^{ 2}N_{\mathrm{b}}\frac{L_{\mathrm{u}}}{2\pi\,\tilde{R}} \tag{26.19}\]

or

\[P_{\mathrm{avg}}(\mathrm{W}) = 633.6\,E^{2}B_{0}^{2}I\,L_{\mathrm{u}}\,, \tag{26.20}\]

where \(I\) is the circulating electron beam current. The total angle integrated radiation power from an undulator in a storage ring is proportional to the square of the beam energy and maximum undulator field \(B_{0}\) and proportional to the beam current and undulator length.

##### Spatial and Spectral Distribution

For bending magnet radiation, the particle dynamics is relatively simple being determined only by the particle velocity and the bending radius of the magnet. In a wiggler magnet, the magnetic field parameters are different from those in a constant field magnet and we will therefore derive again the synchrotron radiation spectrum for the beam dynamics in a general wiggler magnet. No special assumptions on magnetic field configurations have been made to derive the radiation spectrum (25.71) and we can therefore use this expression together with the appropriate beam dynamics to derive the radiation spectrum from a wiggler magnet

\[\frac{\mathrm{d}^{2}W}{\mathrm{d}o\,\mathrm{d}\Omega} = \frac{r_{\mathrm{c}}\,mc\,\omega^{2}}{4\pi^{2}}\left|\int_{-\infty }^{\infty}\mathbf{n}\times[\mathbf{n}\times\mathbf{\beta}]\mathrm{e}^{-\mathrm{i}\omega \,\left(t_{\mathrm{r}}+\frac{\theta}{c}\right)}\mathrm{d}t_{\mathrm{r}}\right| ^{2}. \tag{26.21}\]The integrand in (26.21) can be evaluated from known particle dynamics in a wiggler magnet noting that all quantities are to be taken at the retarded time. The unit vector from the observer to the radiating particle is from Fig. 26.3

(26.22)

where are coordinate unit vectors. The exponent in (26.21) includes the term. We express again the vector from the observer to the particle by the constant from the origin of the coordinate system to the observer and the vector from the coordinate origin to the particle for as shown in Fig. 26.3.

The -term gives only a constant phase shift and can therefore be ignored. The location vector of the particle with respect to the origin of the coordinate system is

(26.23)

and with the solutions (26.5) we have

(26.24)

where

(26.25)

The velocity vector finally is just the time derivative of (26.24)

(26.26)

Figure 26.3: Particle trajectory and radiation geometry in a wiggler magnet

We use these vector relations to evaluate the integrand in (26.21). First, we express the triple vector product \(\boldsymbol{n}\times[\boldsymbol{n}\times\boldsymbol{\beta}]\) by its components and get with (26.22), (26.25)

\[\begin{split}\boldsymbol{n}\times[\boldsymbol{n}\times \boldsymbol{\beta}]&=\,+\boldsymbol{\hat{x}}\left[-\frac{K}{ \gamma}\,\bar{\beta}\sin^{2}\vartheta\ \cos^{2}\varphi\ \cos\omega_{\rm p}t_{\rm r}\,+\,\frac{K}{ \gamma}\,\bar{\beta}\,\sin\omega_{\rm p}t_{\rm r}\right.\\ &\qquad\left.+\,\bar{\beta}\left(1\,+\,\frac{K^{2}}{4\gamma^{2}} \cos 2\omega_{\rm p}t_{\rm r}\right)\sin\vartheta\ \cos\vartheta\ \cos\varphi\,\right]\\ &\qquad\left.+\,\boldsymbol{\hat{y}}\left[-\frac{K}{\gamma}\,\bar {\beta}\,\sin^{2}\vartheta\ \sin\varphi\cos\varphi\,\sin\omega_{\rm p}t_{\rm r}\right.\\ &\qquad\left.+\,\bar{\beta}\left(1\,+\,\frac{K^{2}}{4\gamma^{2}} \cos 2\omega_{\rm p}t_{\rm r}\right)\sin\vartheta\ \cos\vartheta\ \sin\varphi\,\right]\\ &\qquad\left.+\,\boldsymbol{\hat{z}}\left[-\frac{K}{\gamma}\, \bar{\beta}\,\sin\vartheta\ \cos\vartheta\,\cos\varphi\,\cos\omega_{\rm p}t_{\rm r}\right.\\ &\qquad\left.+\,\bar{\beta}\left(1\,+\,\frac{K^{2}}{4\gamma^{2}} \cos 2\omega_{\rm p}t_{\rm r}\right)\left(\cos^{2}\vartheta\,-1\right)\right]. \end{split} \tag{26.26}\]

This expression can be greatly simplified considering that the radiation is emitted into only a very small angle \(\vartheta\,\ll\,1\). Furthermore, we note that the deflection due to the wiggler field is in most practical cases very small and therefore \(K\ll\gamma\) and \(\bar{\beta}=\beta\,\left(1-\frac{K^{2}}{4\gamma^{2}}\right)\approx\beta\). Finally, we carefully set \(\beta\approx 1\) where this term does not appear as a difference to unity. With this and ignoring second order terms in \(\vartheta\) and \(K/\gamma\) we get from (26.26)

\[\boldsymbol{n}\times[\boldsymbol{n}\times\boldsymbol{\beta}\,]=\left(\bar{ \beta}\vartheta\,\cos\varphi\,+\,\bar{\beta}\frac{K}{\gamma}\sin\left(\omega_{ \rm p}t_{\rm r}\right)\right)\boldsymbol{\hat{x}}+\left(\bar{\beta} \vartheta\,\sin\varphi\right)\boldsymbol{\hat{y}}\,. \tag{26.27}\]

The vector product in the exponent of the exponential function is just the product of (26.22) and (26.23)

\[\frac{1}{c}\boldsymbol{nr}_{\rm p}(t_{\rm r})=-\frac{K\bar{\beta}}{\gamma \omega_{\rm p}}\sin\vartheta\ \cos\varphi\ \cos\left(\omega_{\rm p}t_{\rm r}\right)-\left(\bar{\beta}t_{\rm r}+\frac{K^{ 2-}_{\star}}{8\gamma^{2}\omega_{\rm p}}\sin 2\omega_{\rm p}\,t_{\rm r}\right)\cos \vartheta\,. \tag{26.28}\]

Employing again the approximation \(\vartheta\,\ll\,1\) and keeping only linear terms we get from (26.28)

\[t_{\rm r}+\frac{1}{c}\boldsymbol{nr}_{\rm p}(t_{\rm r})=t_{\rm r}(1-\bar{\beta }\,\cos\vartheta)-\frac{K_{\star}\vartheta}{\gamma\omega_{\rm p}}\cos\varphi\ \cos\left(\omega_{\rm p}t_{\rm r}\right)-\frac{K^{2-}_{\star}}{8\gamma^{2} \omega_{\rm p}}\sin\left(2\omega_{\rm p}\,t_{\rm r}\right)\,. \tag{26.29}\]

With (26.4) and \(\cos\vartheta\,\approx\,1-\frac{1}{2}\,\vartheta^{2}\), the first term becomes \[1-\bar{\beta}\cos\vartheta\,=\,\frac{1}{2\gamma^{2}}\left(1\,+\,\tfrac{1}{2}K^{2} \,+\,\gamma^{2}\vartheta^{2}\right)=\frac{\omega_{\rm p}}{\omega_{1}}\,, \tag{26.30}\]

where we have defined the fundamental wiggler frequency \(\omega_{1}\) by

\[\omega_{1}=\omega_{\rm p}\frac{2\gamma^{2}}{1+\,\tfrac{1}{2}K^{2} +\gamma^{2}\vartheta^{2}} \tag{26.31}\]

or the fundamental wavelength of the radiation

\[\lambda_{1}=\frac{\lambda_{\rm p}}{2\gamma^{2}}\left(1\,+\,\tfrac{1}{2}\,K^{2} \,+\,\gamma^{2}\vartheta^{2}\right) \tag{26.32}\]

in full agreement with (26.11). At this point, it is worth to remember that the term \(\tfrac{1}{2}K^{2}\) becomes \(K^{2}\) for a helical wiggler [2]. With (26.30), the complete exponential term \(-{\rm i}\omega\left[t_{\rm r}\,+\,\tfrac{1}{c}\boldsymbol{n}\boldsymbol{r}_{ \rm p}(t_{\rm r})\right]\) in (26.21) can be evaluated to be equal to

\[-{\rm i}\frac{\omega}{\omega_{1}}\left[\omega_{\rm p}t_{\rm r}- \frac{K\bar{\beta}\vartheta}{\gamma}\frac{\omega_{1}}{\omega_{\rm p}}\cos \varphi\,\cos\left(\omega_{\rm p}t_{\rm r}\right)-\frac{K^{2}\bar{\beta}}{8 \gamma^{2}}\frac{\omega_{1}}{\omega_{\rm p}}\sin\left(2\omega_{\rm p}t_{\rm r }\right)\right], \tag{26.33}\]

and (26.21) can be modified with this expression into a form suitable for integration by inserting (26.27) and (26.30) into (26.21) for

\[\frac{{\rm d}^{2}W}{{\rm d}\omega\,{\rm d}\Omega} = \frac{r_{\rm c}\,mc\,\omega^{2}}{4\pi^{2}}\bar{\beta}^{2}\] \[\times \left|\int_{-\infty}^{\infty}\left[\vartheta\,\cos\varphi\,+\, \frac{K}{\gamma}\sin\left(\omega_{\rm p}t_{\rm r}\right)\right]\boldsymbol{x} \,+\,\left(\vartheta\,\sin\varphi\right)\boldsymbol{y}\,{\rm e}^{X}{\rm d}t_{ \rm r}\right|^{2},\]

where

\[X=\left\{-{\rm i}\,\frac{\omega}{\omega_{1}}\left[\omega_{\rm p}t_{\rm r}- \frac{K\vartheta}{\gamma}\frac{\omega_{1}}{\omega_{\rm p}}\cos\varphi\,\cos \left(\omega_{\rm p}t_{\rm r}\right)-\frac{K^{2}}{8\gamma^{2}}\frac{\omega_{1 }}{\omega_{\rm p}}\sin\left(2\omega_{\rm p}\,t_{\rm r}\right)\right]\right\}\,.\]

We are now ready to perform the integration of (26.34) noticing that the integration over all times can be simplified by separation into an integral along the wiggler magnet alone and an integration over the rest of the time while the particle is traveling in a field free space. We write symbolically

\[\int_{-\infty}^{\infty}=\int_{-\pi N_{\rm p}/\omega_{\rm p}}^{\pi N_{\rm p}/ \omega_{\rm p}}(K\neq 0)\,+\int_{-\infty}^{\infty}(K=0)-\int_{-\pi N_{\rm p}/ \omega_{\rm p}}^{\pi N_{\rm p}/\omega_{\rm p}}(K=0)\,. \tag{26.35}\]

First, we evaluate the second integral for \(K=0\) which is of the form

\[\int_{-\infty}^{\infty}{\rm e}^{{\rm i}x\omega t}\,{\rm d}t=\frac{2\pi}{| \kappa|}\delta(\omega)\,,\]where \(\delta(\omega)\) is the Dirac \(\delta\)-function. The value of the integral is nonzero only for \(\omega=0\) in which case the factor \(\omega^{2}\) in (26.34) causes the whole expression to vanish. The second integral is therefore zero.

The third integral has the same form as the second integral, but since the integration is conducted only over the length of the wiggler magnet we get

\[\int_{-\pi N_{\rm p}/\omega_{\rm p}}^{\pi N_{\rm p}/\omega_{\rm p}}{\rm e}^{-{ \rm i}\,\frac{\omega}{2\gamma^{2}}\,t_{\rm r}}\,{\rm d}t_{\rm r}=\,\frac{2\pi N _{\rm p}}{\omega_{\rm p}}\,\,\frac{\sin\frac{\pi N_{\rm p}}{2\gamma^{2}}\,\, \frac{\omega}{\omega_{\rm p}}}{\frac{\pi N_{\rm p}}{2\gamma^{2}}\,\,\frac{ \omega}{\omega_{\rm p}}}\,. \tag{26.36}\]

The value of this integral reaches a maximum of \(2\pi\frac{N_{\rm p}}{\omega_{\rm p}}\) for \(\omega\to 0\). From (26.34) we note the coefficient of this integral to include the angle \(\vartheta\gtrsim 1/\gamma\) and the whole integral is therefore of the order or less than \(L_{\rm u}/(c\gamma)\), where \(L_{\rm u}=N_{\rm p}\lambda_{\rm p}\) is the total length of the wiggler magnet. This value is in general very small compared to the first integral and can therefore be neglected. Actually, this statement is only partially true since the first integral, as we will see, is a fast varying function of the radiation frequency with a distinct line spectrum. Being, however, primarily interested in the peak intensities of the spectrum we may indeed neglect the third integral. Only between the spectral lines does the radiation intensity from the first integral become so small that the third integral would be a relatively significant although absolutely a small contribution.

To evaluate the first integral in (26.35) with \(K\neq 0\) we follow Alferov [5] and introduce with (26.31) the abbreviations

\[C = \frac{2K\,\,\,\tilde{\beta}\,\,\gamma\vartheta\,\cos\varphi}{1+ \frac{1}{2}K^{2}+\,\gamma^{2}\vartheta^{2}}\,, \tag{26.37a}\] \[S = \frac{K^{2}\,\,\,\tilde{\beta}}{4\left(1+\frac{1}{2}K^{2}+\,\gamma ^{2}\vartheta^{2}\right)} \tag{26.37b}\]

to get from (26.34) the exponential functions in the form

\[{\rm e}^{-{\rm i}\,\frac{\omega_{\rm p}}{\omega_{1}}\omega_{\rm p}t}\,\,{\rm e }^{{\rm i}\,\frac{\omega_{\rm p}}{\omega_{1}}C\cos\omega_{\rm p}t_{\rm r}}\,\,{ \rm e}^{{\rm i}\,\frac{\omega_{\rm p}}{\omega_{1}}S\sin 2\omega_{\rm p}t_{\rm r}}. \tag{26.38}\]

The integral in the radiation power spectrum (26.34) has two distinct forms, one where the integrand is just the exponential function multiplied by a time independent factor while the other includes the sine function \(\sin\omega_{\rm p}t_{\rm r}\) as a factor of the exponential function. To proceed further we replace the exponential functions by an infinite sum of Bessel's functions

\[{\rm e}^{{\rm i}\kappa\,\sin\psi}=\sum_{p=-\infty}^{p=\infty}J_{p}(\kappa)\,{ \rm e}^{{\rm i}p\psi} \tag{26.39}\]and apply this identity to the first integral type in (26.34). Applying the identity (26.39) also to the second and third exponential factors in (26.38), we get with \(\mathrm{e}^{a\cos x}\,=\,\mathrm{e}^{a\sin(x+\pi/2)}\) the product of the exponential functions

\[\mathrm{e}^{-\mathrm{i}\left(\frac{a_{\mathrm{r}}}{a_{1}}\omega_{p}t_{\mathrm{r }}-\frac{\omega}{a_{1}}C\cos\omega_{0}t_{\mathrm{r}}-\frac{\omega}{a_{1}}S \sin 2\omega_{0}t_{\mathrm{r}}\right)}\,=\,\sum_{\begin{subarray}{c}m=\\ -\infty\end{subarray}}^{\infty}\sum_{\begin{subarray}{c}n=\\ -\infty\end{subarray}}^{\infty}J_{m}(u)\,J_{n}(v)\,\mathrm{e}^{\mathrm{i} \frac{1}{2}\pi n}\mathrm{e}^{-\mathrm{i}R_{\omega}\omega_{0}t_{\mathrm{r}}}, \tag{26.40}\]

where

\[R_{\omega}\,=\,\frac{\omega}{\omega_{1}}-n-2m\,,\quad u\,=\,\frac{\omega}{ \omega_{1}}S,\qquad\text{and}\qquad v\,=\,\frac{\omega}{\omega_{1}}C\,. \tag{26.41}\]

The time integration along the length of the wiggler magnet is straight forward for this term since no other time dependent factors are involved and we get

\[\int_{-\pi N_{\mathrm{p}}/\omega_{\mathrm{p}}}^{\pi N_{\mathrm{p}}/\omega_{ \mathrm{p}}}\mathrm{e}^{-\mathrm{i}\left(\frac{\omega}{a_{1}}-n-2m\right) \omega_{\mathrm{p}}t_{\mathrm{r}}}\,\mathrm{d}t_{\mathrm{r}}\,=\,\frac{2\pi N _{\mathrm{p}}}{\omega_{\mathrm{p}}}\,\,\frac{\sin\left(\pi N_{\mathrm{p}}R_{ \omega}\right)}{\pi N_{\mathrm{p}}R_{\omega}}\,. \tag{26.42}\]

In the second form of the integrand, we replace the trigonometric factor, \(\sin\omega_{\mathrm{p}}t_{\mathrm{r}}\), by exponential functions and get with (26.42) integrals of the form

\[\int_{-\pi N_{\mathrm{p}}/\omega_{\mathrm{p}}}^{\pi N_{\mathrm{p} }/\omega_{\mathrm{p}}}\sin\omega_{\mathrm{p}}t_{\mathrm{r}}\,\mathrm{e}^{- \mathrm{i}R_{\omega}\omega_{0}t_{\mathrm{r}}}\,\mathrm{d}t_{\mathrm{r}}\] \[=-\mathrm{i}\frac{1}{2}\int_{-\pi N_{\mathrm{p}}/\omega_{\mathrm{p }}}^{\pi N_{\mathrm{p}}/\omega_{\mathrm{p}}}\left(\mathrm{e}^{\mathrm{i}\, \omega_{0}t_{\mathrm{r}}}-\mathrm{e}^{-\mathrm{i}\,\omega_{0}t_{\mathrm{r}}} \right)\mathrm{e}^{-\mathrm{i}R_{\omega}\omega_{0}t_{\mathrm{r}}}\mathrm{d}t_{ \mathrm{r}} \tag{26.43}\] \[=\mathrm{i}\,\frac{\pi N_{\mathrm{p}}}{\omega_{\mathrm{p}}}\, \frac{\sin\left[\pi N_{\mathrm{p}}(R_{\omega}+1)\right]}{\pi N_{\mathrm{p}}(R _{\omega}+1)}-\mathrm{i}\,\frac{\pi N_{\mathrm{p}}}{\omega_{\mathrm{p}}}\, \frac{\sin\left[\pi N_{\mathrm{p}}(R_{\omega}-1)\right]}{\pi N_{\mathrm{p}}(R _{\omega}-1)}.\]

Both integrals (26.42) and (26.43) exhibit the character of multibeam interference spectra well known from optical interference theory. The physical interpretation here is that the radiation from the \(N_{\mathrm{p}}\) wiggler periods consists of \(N_{\mathrm{p}}\) photon beamlets which have a specific phase relationship such that the intensities are strongly reduced for all frequencies but a few specific frequencies as determined by the \(\frac{\sin x}{x}\)-factors. The resulting line spectrum, characteristic for undulator radiation, is the more pronounced the more periods or beamlets are available for interference. To get a more complete picture of the interference pattern, we collect now all terms derived separately so far and use them in (26.34) which becomes with (26.38)

\[\frac{\mathrm{d}^{2}W}{\mathrm{d}\omega\,\mathrm{d}\Omega} = a\left|\int_{-\pi N_{\mathrm{p}}/\omega_{\mathrm{p}}}^{\pi N_{ \mathrm{p}}/\omega_{\mathrm{p}}}\left[\left(A_{0}+A_{1}\sin\omega_{\mathrm{p }}t_{\mathrm{r}}\right)\boldsymbol{\hat{x}}+B_{0}\boldsymbol{\hat{y}}\right]\right. \tag{26.44}\] \[\times\mathrm{e}^{-\mathrm{i}\,\frac{\omega}{a_{1}}\omega_{0}t_ {\mathrm{r}}}\,\mathrm{e}^{\mathrm{i}\,v\,\cos\omega_{0}t_{\mathrm{r}}}\, \mathrm{e}^{\mathrm{i}\,u\sin 2\omega_{0}t_{\mathrm{r}}}\,\mathrm{d}t_{\mathrm{r}}\right|^{2}\,,\]where \(a=\frac{r_{\rm c}\,mc\tilde{\beta}^{2}}{4\pi^{2}}\omega^{2}\), \(A_{0}=\vartheta\,\cos\varphi,A_{1}=\frac{K}{\gamma}\), and \(B_{0}=\vartheta\,\sin\varphi\). Introducing the identity (26.38), the photon energy spectrum becomes

\[\frac{{\rm d}^{2}W}{{\rm d}\omega\,{\rm d}\Omega} = a\left|\int_{-\pi N_{\rm p}/\omega_{\rm p}}^{\pi N_{\rm p}/ \omega_{\rm p}}\left[\left(A_{0}+A_{1}\,\sin\omega_{\rm p}t_{\rm r}\right) \boldsymbol{\hat{x}}+B_{0}\boldsymbol{\hat{y}}\right]\right. \tag{26.45}\] \[\quad\times\sum_{m=-\infty}^{\infty}\sum_{n=-\infty}^{\infty}J_{ m}(u)\,J_{n}(v)\,{\rm e}^{{\rm i}\frac{1}{2}\pi n}{\rm e}^{-{\rm i}R_{\omega}a_{ \rm p}t_{\rm r}}{\rm d}t_{\rm r}\,\Bigg{|}^{2}\]

and after integration with (26.42) and (26.43)

\[\frac{{\rm d}^{2}W}{{\rm d}\omega\,{\rm d}\Omega} = a\left|\boldsymbol{x}\,A_{0}\sum_{\begin{subarray}{c}m=\\ -\infty\end{subarray}}^{\infty}\sum_{\begin{subarray}{c}n=\\ -\infty\end{subarray}}^{\infty}J_{m}(u)\,J_{n}(v)\,{\rm e}^{{\rm i}\frac{1}{2} \pi n}\frac{2\pi N_{\rm p}}{\omega_{\rm p}}\,\frac{\sin\left(\pi N_{\rm p}R_{ \omega}\right)}{\pi N_{\rm p}R_{\omega}}\right. \tag{26.46}\] \[\quad+\boldsymbol{\hat{x}}\,A_{1}\sum_{\begin{subarray}{c}m=\\ -\infty\end{subarray}}^{\infty}\sum_{\begin{subarray}{c}n=\\ -\infty\end{subarray}}^{\infty}J_{m}(u)\,J_{n}(v)\,{\rm e}^{{\rm i}\frac{1}{2} \pi n}\] \[\quad\times{\rm i}\frac{\pi N_{\rm p}}{2\omega_{\rm p}}\left[ \frac{\sin\left[\pi N_{\rm p}(R_{\omega}+1)\right]}{\pi N_{\rm p}(R_{\omega}+ 1)}-{\rm i}\,\frac{\pi N_{\rm p}}{\omega_{\rm p}}\frac{\sin\left[\pi N_{\rm p }(R_{\omega}-1)\right]}{\pi N_{\rm p}(R_{\omega}-1)}\right]\] \[\quad+\boldsymbol{\hat{y}}\,B_{0}\sum_{\begin{subarray}{c}m=\\ -\infty\end{subarray}}^{\infty}\sum_{\begin{subarray}{c}n=\\ -\infty\end{subarray}}^{\infty}J_{m}(u)\,J_{n}(v)\,{\rm e}^{{\rm i}\frac{1}{2} \pi n}\frac{2\pi N_{\rm p}}{\omega_{\rm p}}\,\frac{\sin\left(\pi N_{\rm p}R_{ \omega}\right)}{\pi N_{\rm p}R_{\omega}}\Bigg{|}^{2}\,.\]

To determine the frequency and radiation intensity of the line maxima, we simplify the double sum of Bessel's functions by selecting only the most dominant terms. The first and third sums in (26.46) show an intensity maximum for \(R_{\omega}=0\) at frequencies

\[\omega=(n+2m)\,\omega_{1}\,, \tag{26.47}\]

and intensity maxima appear therefore at the frequency \(\omega_{1}\) and harmonics thereof. The transformation of a lower frequency to very high values has two physical components. In the system of relativistic particles, the static magnetic field of the wiggler magnet appears Lorentz contracted by the factor \(\gamma\), and particles passing through the wiggler magnet oscillate with the frequency \(\gamma\omega_{\rm p}\) in its own system emitting radiation at that frequency. The observer in the laboratory system receives this radiation from a source moving with relativistic velocity and experiences therefore a Doppler shift by the factor \(2\gamma\). The wavelength of the radiation emitted in the forward direction, \(\vartheta=0\), from a weak wiggler magnet, \(K\ll 1\), with the period length \(\lambda_{\rm p}\) is therefore reduced by the factor \(2\gamma^{2}\). In cases of a stronger wiggler magnet or when observing at a finite angle \(\vartheta\), the wavelength is somewhat longer as one would expect from higher order terms of the Doppler effect.

From (26.46) we determine two more dominant terms originating from the second term for \(R_{\omega}\pm 1=0\) at frequencies

\[\omega = (n+2m-1)\ \omega_{1} \tag{26.48a}\] \[\omega = (n+2m+1)\ \omega_{1}\,, \tag{26.48b}\]

respectively. The summation indices \(n\) and \(m\) are arbitrary integers between \(-\infty\) and \(\infty\). Among all possible resonant terms we collect such terms which contribute to the same harmonic \(k\) of the fundamental frequency \(\omega_{1}\). To collect these dominant terms for the same harmonic we set \(\omega=\omega_{k}=k\,\omega_{1}\) where \(k\) is the harmonic number of the fundamental and express the index \(n\) by \(k\) and \(m\) to get

\[\text{from (\ref{eq:26.47}):}\qquad n=k-2m,\] \[\text{from (\ref{eq:26.48a}):}\qquad n=k-2m+1 \tag{26.49}\] \[\text{and from (\ref{eq:26.48b}):}\qquad n=k-2m-1\,.\]

Introducing these conditions into (26.46) all trigonometric factors assume the form \(\frac{\sin\left(\pi N_{\text{p}}\,\Delta\omega_{\text{f}}/\omega_{1}\right)}{ \pi N_{\text{p}}\,\Delta\omega_{\text{f}}/\omega_{1}}\), where

\[\frac{\Delta\omega_{k}}{\omega_{1}}=\frac{\omega}{\omega_{1}}-k \tag{26.50}\]

and we get the photon energy spectrum of the \(k\)th harmonic for radiation from a single electron passing through an undulator

\[\frac{\text{d}^{2}W_{k}(\omega)}{\text{d}\omega\,\text{d}\Omega} = \frac{r_{\text{c}}\,mc\tilde{\beta}^{2}N_{\text{p}}^{2}}{\gamma^{ 2}}\frac{\omega^{2}}{\omega_{\text{p}}^{2}}\Bigg{[}\,\frac{\sin\left(\pi N_{ \text{p}}\,\Delta\omega_{k}/\omega_{1}\right)}{\pi N_{\text{p}}\Delta\omega_{ k}/\omega_{1}}\Bigg{]}^{2} \tag{26.51}\] \[\times \Bigg{|}+\hat{\pi}A_{0}\sum_{m=-\infty}^{\infty}J_{m}(u)\,J_{k-2 m}(v)\,\text{e}^{\text{i}\,\frac{1}{2}\pi(k-2m)}\] \[+\hat{\gamma}B_{0}\sum_{m=-\infty}^{\infty}J_{m}(u)\,J_{k-2m}(v) \,\text{e}^{\text{i}\,\frac{1}{2}\pi(k-2m)}\] \[+\text{i}\,\hat{\pi}\,\frac{1}{2}A_{1}\sum_{m=-\infty}^{\infty}J_ {m}(u)\,J_{k-2m+1}(v)\,\text{e}^{\text{i}\,\frac{1}{2}\pi(k-2m+1)}\] \[-\text{i}\,\hat{\pi}\,\frac{1}{2}A_{1}\sum_{m=-\infty}^{\infty}J_ {m}(u)\,J_{k-2m-1}(v)\,\text{e}^{\text{i}\,\frac{1}{2}\pi(k-2m-1)}\Bigg{|}^{2}\,.\]All integrals exhibit the resonance character defining the locations of the spectral lines. The \((\sin x/x)\)-terms represents the line spectrum of the radiation. Specifically, the number \(N_{\rm p}\) of beamlets, here source points, determines the spectral purity of the radiation. In Fig. 26.4 the \((\sin x/x)\)-function is shown for \(N_{\rm p}=5\) and \(N_{\rm p}=100\). It is clear that the spectral purity improves greatly as the number of undulator periods is increased. This is one of the key features of undulator magnets to gain spectral purity by maximizing the number of undulator periods.

The spectral purity or line width is determined by the shape of the \((\sin x/x)\)-function. We define the line width by the frequency at which \(\sin x/x=0\) or where \(\pi N_{\rm p}\,\Delta\omega_{k}/\omega_{1}=\pi\) defining the line width for the \(k\)th harmonic

\[\frac{\Delta\omega_{k}}{\omega_{k}}=\pm\frac{1}{kN_{\rm p}}. \tag{26.52}\]

The spectral width of the undulator radiation is reduced proportional to the number of undulator periods, but reduces also proportional to the harmonic number.

The Bessel functions \(J_{m}(u)\) determine mainly the intensity of the line spectrum. For an undulator with \(K\ll 1\), the argument \(u\propto K^{2}\ll 1\) and the contributions of higher order Bessel's functions are very small. The radiation spectrum consists therefore only of the fundamental line. For stronger undulators with \(K>1\), higher order Bessel's functions grow and higher harmonic radiation appears in the line spectrum of the radiation.

Summing over all harmonics of interest, one gets the total power spectrum. In the third and fourth terms of (26.51) we use the identities \(\mathrm{i}\,\mathrm{e}^{\pm\mathrm{i}\pi/2}=\mp 1\), \(J_{m}(u)\,\mathrm{e}^{\mathrm{i}\pi m}=J_{-m}(u)\) and abbreviate the sums of Bessel's functions by the symbols

\[\sum\nolimits_{1} =\sum_{m=-\infty}^{\infty}J_{-m}(u)\,J_{k-2m}(v) \tag{26.53a}\] \[\sum\nolimits_{2} =\sum_{m=-\infty}^{\infty}J_{-m}(u)\,\left[J_{k-2m-1}(v)\,+\,J_{ k-2m+1}(v)\right]\,. \tag{26.53b}\]

The total number of photons \(N_{\mathrm{ph}}\) emitted into a spectral band width \(\Delta\omega/\omega\) by a single electron moving through a wiggler magnet is finally with \(N_{\mathrm{ph}}(\omega)=W(\omega)/(\hbar\omega)\)

\[\frac{\mathrm{d}N_{\mathrm{ph}}(\omega)}{\mathrm{d}\Omega} =\alpha\gamma^{2}\bar{\beta}^{2}N_{\mathrm{p}}^{2}\frac{\Delta \omega}{\omega}\sum_{k=1}^{\infty}k^{2}\Bigg{[}\,\frac{\sin\left(\pi N_{ \mathrm{p}}\,\Delta\omega_{k}/\omega_{1}\right)}{\pi N_{\mathrm{p}}\,\Delta \omega_{k}/\omega_{1}}\,\Bigg{]}^{2} \tag{26.54}\] \[\qquad\times\,\frac{\left(2\gamma\vartheta\sum_{1}\cos\varphi-K \sum_{2}\right)^{2}\hat{\boldsymbol{x}}^{2}+\left(2\gamma\vartheta\sum_{1} \sin\varphi\right)^{2}\hat{\boldsymbol{y}}^{2}}{\left(1\,+\,\frac{1}{2}K^{2} +\,\gamma^{2}\vartheta^{2}\right)^{2}}\,,\]

where \(\alpha\) is the fine structure constant and where we have kept the coordinate unit vectors to keep track of the polarization modes. The vectors \(\mathbf{x}\) and \(\mathbf{y}\) are orthogonal unit vectors indicating the directions of the electric field or the polarization of the radiation. Performing the squares does therefore not produce cross terms and the two terms in (26.54) with the expressions (26.53) represent the amplitude factors for both polarization directions, the \(\sigma\)-mode and \(\pi\)-mode respectively.

We also made use of (26.50) and the resonance condition

\[\frac{\omega}{\omega_{\mathrm{p}}}=\frac{k\omega_{1}\,+\,\Delta\omega_{k}}{ \omega_{\mathrm{p}}}\approx k\frac{\omega_{1}}{\omega_{\mathrm{p}}}=\,\frac{2 \gamma^{2}\,k}{1\,+\,\frac{1}{2}K^{2}\,+\,\gamma^{2}\vartheta^{2}}\,, \tag{26.55}\]

realizing that the photon spectrum is determined by the \((\sin x/x)^{2}\)-function. For not too few periods, this function is very small for frequencies away from the resonance conditions.

Storage rings optimized for very small beam emittance are being used as modern synchrotron radiation sources to reduce the line width of undulator radiation and concentrate all radiation to the frequency desired. The progress in this direction is demonstrated in the spectrum of Fig. 26.5 derived from the first electron storage ring operated at a beam emittance below \(10\,\mathrm{nm}\) at \(7.1\,\mathrm{GeV}\)[7]. In Fig. 26.5 a measured undulator spectrum is shown as a function of the undulator strength \(K\)[8]. For a strength parameter \(K\ll 1\) there is only one line at the fundamental frequency. As the strength parameter increases, additional lines appear in addition to being shifted to lower frequencies. The spectral lines from a real synchrotron radiation source are not infinitely narrow as (26.66) would suggest. Because of the finite size of the pinhole opening, some light at small angles with respect to the axis passes through, and we observe therefore also some signal of the even order harmonic radiation.

Even for an extremely small pin hole, we would observe a similar spectrum as shown in Fig. 26.5 because of the finite beam divergence of the electron beam. The electrons follow oscillatory trajectories due not only to the undulator field but also due to betatron oscillations. We observe therefore always some radiation at a finite angle given by the particle trajectory with respect to the undulator axis. Figure 26.5 also demonstrates the fact that all experimental circumstances must be included to meet theoretical expectations. The amplitudes of the measured low energy spectrum is significantly suppressed compared to theoretical expectations which is due to a Be-window being used to extract the radiation from the ultra high vacuum chamber of the accelerator. This material absorbs radiation significantly below a photon energy of about 3 keV.

While we observe a line spectrum expressed by the \((\sin x/x)^{2}\)-function, we also notice that this line spectrum is red shifted as we increase the observation angle \(\vartheta\). Only, when we observe the radiation though a very small aperture (pin hole)

Figure 26.5: Measured frequency spectrum from an undulator for different strength parameters \(K\)[8]

do we actually see this line spectrum. Viewing the undulator radiation through a large aperture integrates the linespectra over a finite range of angles \(\vartheta\) producing an almost continuous spectrum with small spikes at the locations of the harmonic lines.

The difference between a pin hole undulator spectrum and an angle-integrated spectrum becomes apparent from the experimental spectra shown in Fig. 26.6[7]. While the pin hole spectrum demonstrates well the line character of undulator radiation, much radiation appears between these spectral lines as the pin hole is removed and radiation over a large solid angle is collected by the detector. The pin hole undulator line spectrum shows up as mere spikes on top of a broad continuous spectrum.

The overall spatial intensity distribution includes a complex set of different radiation lobes depending on frequency, emission angle and polarization. In Fig. 26.7 the radiation intensity distributions described by the last factor in (26.54)

\[I_{\sigma,k}=\frac{(2\gamma\vartheta\Sigma_{1}\cos\varphi-K\Sigma_{2})^{2}}{( 1+\frac{1}{2}K^{2}+\gamma^{2}\vartheta^{2})^{2}}\]

for the \(\sigma\)-mode polarization and

\[I_{\pi,k}=\frac{(2\gamma\vartheta\Sigma_{1}\sin\varphi)^{2}}{(1+\frac{1}{2}K^ {2}+\gamma^{2}\vartheta^{2})^{2}}\]

for the \(\pi\)-mode polarization are shown for the lowest order harmonics.

We note clearly the strong forward lobe at the fundamental frequency in \(\sigma\)-mode while there is no emission in \(\pi\)-mode along the path of the particle. The second

Figure 26: Actual radiation spectra from an undulator with a maximum field of 0.2 T and a beam energy of 7.1 GeV through a pin hole and angle-integrated after removal of the pin hole [7]

harmonic radiation vanishes in the forward direction, an observation that is true for all even harmonics. By inspection of (26.54), we note that \(v=0\) for \(\vartheta=0\) and the square bracket in (26.53b) vanishes for all odd indices or for all even harmonics \(k\). There is therefore no forward radiation for even harmonics of the fundamental undulator frequency.

A contour plot of the first harmonic \(\sigma\)- and \(\pi\)-mode radiation is shown in Fig. 26.8. There is a slight asymmetry in the radiation distribution between the deflecting and nondeflecting plane as one might expect. It is obvious that the pin hole radiation is surrounded by many radiation lobes not only from the first harmonics but also from higher harmonics compromising the pure line spectrum for larger apertures.

Figure 26.7: Undulator radiation distribution in \(\sigma\)- and \(\pi\)-mode for the lowest order harmonics

#### Line Spectrum

To exhibit other important and desirable features of the radiation spectrum (26.54), we ignore the actual frequency distribution in the vicinity of the harmonics and set \(\Delta\omega_{k}=0\) because the spectral lines are narrow for large numbers of wiggler periods \(N_{\rm p}\). Further, we are interested for now only in the forward radiation where \(\vartheta=0\) keeping in mind that the radiation is mostly emitted into a small angle \(\langle\vartheta\rangle=1/\gamma\).

There is no radiation for the \(\pi\)-mode in the forward direction and the only contribution to the forward radiation comes from the second term in (26.54) of the \(\sigma\)-mode. From (26.41) we get for this case with \(\omega\,/\omega_{1}=k\)

\[u_{0}=\frac{kK^{2}}{4+2K^{2}}\qquad\quad\mbox{and}\qquad\quad v_{0}=0\,. \tag{26.56}\]

The sums of Bessel's functions simplify in this case greatly because only the lowest order Bessel's function has a nonvanishing value for \(v_{0}=0\). In the expression for \(\Sigma_{2}\) all summation terms vanish except for the two terms for which the index is zero or for which

\[k-2m-1=0,\qquad\mbox{or}\qquad k-2m+1=0 \tag{26.57}\]

and

\[\sum\nolimits_{2} =\sum_{m=-\infty}^{\infty}J_{-m}(u)\left[J_{k-2m-1}(0)+J_{k-2m+1 }(0)\right]\] \[=J_{-\frac{k-1}{2}}(u_{0})+J_{-\frac{k+1}{2}}(u_{0}). \tag{26.58}\]

Figure 26.8: Contour plot of the first harmonic \(\sigma\)-mode (_solid_) and \(\pi\)-mode (_dashed_) undulator radiation distributionThe harmonic condition (26.57) implies that \(k\) is an odd integer. For even integers, the condition cannot be met as we would expect from earlier discussions on harmonic radiation in the forward direction. Using the identity \(J_{-n}=(-1)^{n}J_{n}\) and (26.56), we get finally with \(N_{\rm ph}=W/\hbar o\) the photon flux per unit solid angle from a highly relativistic particle passing through an undulator

\[\left.\frac{{\rm d}N_{\rm ph}(\omega)}{{\rm d}\Omega}\right|_{\theta=0}=\alpha \gamma^{2}N_{\rm p}^{2}\frac{\Delta\omega}{\omega}\frac{K^{2}}{\left(1+\frac{1 }{2}K^{2}\right)^{2}}\sum_{k=1}^{\infty}k^{2}\left(\frac{\sin\pi N_{\rm p}\, \Delta\omega_{k}/\omega_{1}}{\pi N_{\rm p}\Delta\omega_{k}/\omega_{1}}\right)^ {2}JJ^{2}, \tag{26.59}\]

where the \(JJ\)-function is defined by

\[JJ=\left[J_{\frac{1}{2}(k-1)}\left(\frac{kK^{2}}{4+2K^{2}}\right)-J_{\frac{1} {2}(k+1)}\left(\frac{kK^{2}}{4+2K^{2}}\right)\right]. \tag{26.60}\]

The amplitudes of the harmonics are given by

\[A_{k}(K)=\frac{k^{2}K^{2}}{(1+\frac{1}{2}K^{2})^{2}}JJ^{2}\,. \tag{26.61}\]

The strength parameter greatly determines the radiation intensity as shown in Fig. 26.9 for the lowest order harmonics. For the convenience of numerical calculations the values \(A_{k}(K)\) are tabulated for odd harmonics in Table 26.1. For weak magnets (\(K\ll 1\)) the intensity increases with the square of the magnet field or undulator strength parameter. There is an optimum value for the strength parameter for maximum photon flux depending on the harmonic under consideration. In particular, radiation in the forward direction at the fundamental frequency reaches a maximum photon flux for strength parameters \(K\approx 1.3\). The photon flux per unit solid angle increases like the square of the number of wiggler periods \(N_{\rm p}\), which is a result of the interference effect of many beams concentrating the radiation more and more into one frequency and its harmonics as the number of interfering beams is increased.

Figure 26.9: Undulator radiation intensity \(A_{k}(K)\) in the forward direction as a function of the strength parameter \(K\) for the six lowest order odd harmonics

The radiation opening angle is primarily determined by the \((\sin x/x)^{2}\)-term. We define the opening angle for the \(k\)th harmonic radiation by \(\vartheta_{k}\) being the angle for which \(\sin x/x=0\) for the first time. In this case \(x=\pi\) or \(N_{\rm p}\Delta\omega_{k}/\omega_{1}=1\). With \(\omega_{1}=\omega_{\rm p}\frac{2\gamma^{2}}{1+\frac{1}{2}\,K^{2}}\), \(\omega_{k}=k\omega_{\rm p}\frac{2\gamma^{2}}{1+\frac{1}{2}\,K^{2}+\gamma^{2} \vartheta_{k}^{2}}\) and \(\frac{\Delta\omega_{k}}{\omega_{1}}=\left|\frac{\omega_{k}}{\omega_{1}}-k\right|\), we get \(\frac{N_{\rm p}\,k\,\gamma^{2}\,\vartheta_{k}^{2}}{1+\frac{1}{2}\,K^{2}+\gamma ^{2}\vartheta_{k}^{2}}=1\) or after solving for \(\vartheta_{k}\)

\[\vartheta_{k}^{2}=\frac{1+\frac{1}{2}K^{2}}{\gamma^{2}(kN_{\rm p}-1)}. \tag{26.62}\]

Assuming an undulator with many periods \(\left(kN_{\rm p}\gg 1\right)\) the rms opening angle of undulator radiation is finally

\[\sigma_{r}\approx\frac{1}{\sqrt{2}}\vartheta_{k}=\frac{1}{\gamma}\sqrt{\frac {1+\frac{1}{2}K^{2}}{2kN_{\rm p}}}. \tag{26.63}\]

Radiation emitted into a solid angle defined by this small opening angle

\[\Delta\Omega=\pi\sigma_{r}^{2} \tag{26.64}\]

is referred to as the forward radiation cone. The opening angle of undulator radiation becomes more collimated as the number of periods and the order of the harmonic increases. On the other hand, the radiation cone opens up as the undulator strength \(K\) is increased. We may use this opening angle to calculate the total photon flux of the \(k\)th harmonic within a bandwidth \(\frac{\Delta\omega}{\omega}\) into the forward cone

\[N_{\rm ph}(\omega_{k})\big{|}_{\vartheta=0}=\frac{1}{2}\pi\,\alpha N_{\rm p} \,\frac{\Delta\omega}{\omega_{k}}\,\frac{k\,K^{2}}{1+\frac{1}{2}K^{2}}JJ^{2}, \tag{26.65}\]where \(\omega_{k}=k\,\omega_{1}\). The radiation spectrum from an undulator magnet into the forward direction has been reduced to a simple form exhibiting the most important characteristic parameters. Utilizing (26.61) the number of photons emitted into a band width \(\frac{\Delta\omega_{2}}{\omega_{k}}\) from a single electron passing through an undulator in the \(k\)th harmonic is

\[N_{\rm ph}(\omega_{k})\big{|}_{\vartheta=0}=\tfrac{1}{2}\pi\alpha N_{\rm p}\, \frac{\Delta\omega}{\omega_{k}}\,\frac{1+\tfrac{1}{2}K^{2}}{k}A(K). \tag{26.66}\]

Equation (26.66) is to be multiplied by the number of particles in the electron beam to get the total photon intensity. In case of a storage ring, particles circulate with a high revolution frequency and we get from (26.66) by multiplication with \(I/e\), where \(I\) is the circulating beam current, the photon flux

\[\left.\frac{\mathrm{d}N_{\rm ph}(\omega_{k})}{\mathrm{d}t}\right|_{\vartheta=0 }=\tfrac{1}{2}\pi\,\alpha N_{\rm p}\,\frac{I}{e}\frac{\Delta\omega}{\omega_{k} }\,\frac{1+\tfrac{1}{2}K^{2}}{k}A(K). \tag{26.67}\]

The spectrum includes only odd harmonic since all even harmonics are suppressed through the cancellation of Bessel's functions. This photon flux represents fully spatial coherent radiation as long as the beam divergence does not significantly contribute to the photon divergence (26.63).

#### Spectral Undulator Brightness

Similar to Chap. 27 we define the spectral brightness of undulator radiation as the photon density in six-dimensional phase space. The actual photon brightness is reduced from the diffraction limit due to betatron motion of the particles, transverse beam oscillation in the undulator, apparent source size on axis and under an oblique angle. All of these effects tend to increase the source size and reduce brightness.

The particle beam cross section varies in general along the undulator. We assume here for simplicity that the beam size varies symmetrically along the undulator with a waist in its center. From beam dynamics it is then known that, for example, the horizontal beam size varies like \(\sigma_{\rm b}^{2}=\sigma_{\rm b0}^{2}+{\sigma^{\prime}}_{\rm b0}^{2}s^{2}\), where \(\sigma_{\rm b0}\) is the beam size at the waist, \(\sigma^{\prime}_{\rm b0}\) the divergence of the beam at the waist and \(-\tfrac{1}{2}L\leqq s\leqq\frac{1}{2}L\) the distance from the waist. The average beam size along the undulator length \(L\) is then

\[\langle\sigma_{\rm b}^{2}\rangle=\sigma_{\rm b0}^{2}+\tfrac{1}{12}{\sigma^{ \prime}}_{\rm b0}^{2}L^{2}. \tag{26.68}\]

Similarly, due to an oblique observation angle \(\vartheta\) with respect to the \((y,z)\)-plane or \(\psi\) with respect to the \((x,z)\)-plane we get a further additive contribution \(\tfrac{1}{6}\vartheta L\) to the apparent beam size. Finally, the apparent source size is widened by the transverse beam wiggle in the periodic undulator field. This oscillation amplitude is from (26.6) \(a=\lambda_{\rm p}K/(2\pi\gamma)\).

Collecting all contributions and adding them in quadrature, the total effective beam-size parameters are given by

\[\sigma_{t,x}^{2} = \tfrac{1}{2}\sigma_{r}^{2}+\sigma_{\text{b0},x}^{2}+\left(\frac{ \lambda_{\text{p}}K}{2\pi\gamma}\right)^{2}+\tfrac{1}{12}\sigma_{\text{b0},x^{ \prime}}^{2}L^{2}+\tfrac{1}{36}\vartheta^{2}L^{2}, \tag{26.69a}\] \[\sigma_{t,x^{\prime}}^{2} = \tfrac{1}{2}\sigma_{r^{\prime}}^{2}+\sigma_{\text{b0},x^{\prime}}^ {2}\,\] (26.69b) \[\sigma_{t,y}^{2} = \tfrac{1}{2}\sigma_{r}^{2}+\sigma_{\text{b0},y}^{2}+\left(\frac{ \lambda_{\text{p}}K}{2\pi\gamma}\right)^{2}+\tfrac{1}{12}\sigma_{\text{b0},y^{ \prime}}^{2}L^{2}+\tfrac{1}{36}\psi^{2}L^{2},\] (26.69c) \[\sigma_{t,y^{\prime}}^{2} = \tfrac{1}{2}\sigma_{r^{\prime}}^{2}+\sigma_{\text{b0},y^{\prime}} ^{2}\, \tag{26.69d}\]

where the particle beam sizes can be expressed by the beam emittance and betatron function as \(\sigma_{\text{b}}^{2}=\epsilon\beta\), \({\sigma_{\text{b}}^{\prime}}^{2}=\epsilon/\beta\), and the diffraction limited beam parameters are \(\sigma_{r^{\prime}}=\sqrt{\lambda/L}\), and \(\sigma_{r}=\sqrt{\lambda L}/(2\pi)\).

### Elliptical Polarization

During the discussion of bending magnet radiation in Chap. 25 and insertion radiation in this chapter, we noticed the appearance of two orthogonal components of the radiation field which we identified with the \(\sigma\)-mode and \(\pi\)-mode polarization. The \(\pi\)-mode radiation is observable only at a finite angle with the plane defined by the particle trajectory and the acceleration force vector, which is in general the horizontal plane. As we will see, both polarization modes can, under certain circumstances, be out of phase giving rise to elliptical polarization. In this section, we will shortly discuss such conditions.

#### 26.3.1 Elliptical Polarization from Bending Magnet Radiation

The direction of the electric component of the radiation field is parallel to the particle acceleration. Since radiation is the perturbation of electric field lines from the charge at the retarded time to the observer, we must take into account all apparent acceleration. To see this more clear, we assume an electron to travel counter clockwise on an orbit travelling from say a 12-o'clock position to 9-o'clock and then 6-o'clock. Watching the particle in the plane of deflection, the midplane, we notice only a horizontal acceleration which is maximum at 9-o'clock. Radiation observed in the midplane is therefore linearly polarized in the plane of deflection.

Now we observe the same electron at a small angle above the midplane. Apart from the horizontal motion, we notice now also an apparent vertical motion. Since the electron follows pieces of a circle this vertical motion is not uniform but exhibitsacceleration. Specifically, at 12-o'clock the particle seems to be accelerated only in the vertical direction (downward), horizontally it is in uniform motion; at 9-o'clock the acceleration is only horizontal (towards 3-o'clock) and the vertical motion is uniform; finally, at 6-o'clock the electron is accelerated only in the vertical plane again (upward). Because light travels faster than the electron, we observe radiation first coming from the 12-o'clock position, then from 9-o'clock and finally from 6-o'clock. The polarization of this radiation pulse changes from downward to horizontal (left-right) to upward which is what we call elliptical polarization where the polarization vector rotates with time. Of course, in reality we do not observe radiation from half the orbit, but only from a very short arc segment of angle \(\pm 1/\gamma\). However, if we consider Lorentz contraction the 9-o'clock trajectory in the particle system looks very close to a half circle radiation into \(\pm 180\) degrees which appears in the laboratory system within \(\pm 1/\gamma\). Therefore the short piece of arc from which we observe the radiation has all the features just used to explain elliptical polarization in a bending magnet.

If we observe the radiation at a small angle from below the midplane, the sequence of accelerations is opposite, upward-horizontal (left-right)-downward. The helicity of the polarization is therefore opposite for an observer below or above the midplane. This qualitative discussion of elliptical polarization must become obvious also in the formal derivation of the radiation field. Closer inspection of the radiation field (25.87) from a bending magnet

\[\mathbf{E}_{\rm r}(\omega)=-\frac{\sqrt{3}}{4\pi\epsilon_{0}}\frac{e}{cR}\frac{ \omega}{\omega_{\rm c}}\gamma(1+\gamma^{2}\vartheta^{2})\left[\,\mathrm{sign} \left(\frac{1}{\rho}\right)K_{2/3}(\xi)\,\mathbf{u}_{\sigma}-\mathrm{i}\,\frac{ \gamma\vartheta\,K_{1/3}(\xi)}{\sqrt{1+\gamma^{2}\vartheta^{2}}}\,\mathbf{u}_{ \pi}\,\right] \tag{26.70}\]

shows that both polarization terms are \(90^{\circ}\) out of phase. As a consequence, the combination of both terms does not just introduce a rotation of the polarization direction but generates a time dependent rotation of the polarization vector which we identify with circular or elliptical polarization. In this particular case, the polarization is elliptical since the \(\pi\)-mode radiation is always weaker than the \(\sigma\)-mode radiation. The field rotates in time just as expected from the qualitative discussion above. The linear dependence of the second term in (26.70) also defines the helicity proportional to the sign of \(\vartheta\).

We may quantify the polarization property considering that the electrical field is proportional to the acceleration vector \(\hat{\mathbf{\beta}}\). Observing radiation at an angle with the horizontal plane, we note that the acceleration being normal to the trajectory and in the midplane can be decomposed into two components \(\hat{\beta}_{x}\) and \(\hat{\beta}_{z}\) as shown in Fig. 26.10a.

The longitudinal acceleration component together with a finite observation angle \(\vartheta\) gives rise to an apparent vertical acceleration with respect to the observation direction and the associated vertical electric field component is

\[\mathbf{E}_{y}\propto\hat{\beta}_{y}=n_{y}\hat{\beta}_{z}+n_{x}n_{y}\hat{\beta}_{x }\,.\]An additional component appears, if we observe the radiation also at an angle with respect to the \((x,y)\)-plane which we, however, ignore here for this discussion. The components \(n_{x},n_{y}\) are components of the observation unit vector from the observer to the source with \(n_{y}=-\sin\vartheta\). We observe radiation first from an angle \(\vartheta>0\). The horizontal and vertical radiation field components as a function of time are shown in Fig. 26.10b. Both being proportional to the acceleration (Fig. 26.10a), we observe a symmetric horizontal field \(E_{x}\) and an antisymmetric vertical field \(E_{y}\). The polarization vector (Fig. 26.10c) therefore rotates with time in a counter clockwise direction giving rise to elliptical polarization with lefthanded helicity. Observing the radiation from below with \(\vartheta<0\), the antisymmetric field switches sign and the helicity becomes righthanded. The visual discussion of the origin of elliptical polarization of bending magnet radiation is in agreement with the mathematical result (26.70) displaying the sign dependence of the \(\pi\)-mode component with \(\vartheta\).

The intensities for both polarization modes are shown in Fig. 26.11 as a function of the vertical observation angle \(\vartheta\) for different photon energies. Both intensities are normalized to the forward intensity of the \(\sigma\)-mode radiation. From Fig. 26.11 it becomes obvious that circular polarization is approached for large observation angles. At high photon energies both radiation lobes are confined to very small angles but expand to larger angle distributions for photon energies much lower than the critical photon energy.

The elliptical polarization is left or right handed depending on whether we observe the radiation from above or below the horizontal mid plane. Furthermore, the helicity depends on the direction of deflection in the bending magnet or the sign of the curvature \(\mathrm{sign}(1/\rho)\). By changing the sign of the bending magnet field the helicity of the elliptical polarization can be reversed. This is of no importance for radiation from a bending magnet since we cannot change the field without loss of the particle beam but is of specific importance for elliptical polarization state of radiation from wiggler and undulator magnets.

Figure 26.10: Acceleration along an arc-segment of the particle trajectory in (**a**) a bending magnet, (**b**) polarization as a function of time, and (**c**) radiation field components as a function of time

#### Elliptical Polarization from Periodic Insertion Devices

We apply the visual picture for the formation of elliptically polarized radiation in a bending magnet to the periodic magnetic field of wiggler and undulator magnets. The acceleration vectors and associated field vectors are shown in Fig. 26.12a, b for one period and similar to the situation in bending magnets we do not expect any elliptical polarization in the mid plane where \(\vartheta=0\). Off the mid-plane, we observe now the radiation from a positive and a negative pole. From each pole we get elliptical polarization but the combination of lefthanded polarization from one pole with righthanded polarization from the next pole leads to a cancellation of elliptical polarization from periodic magnets (Fig. 26.12c). In bending magnets, this cancellation did not occur for lack of alternating deflection. Since there are generally an equal number of positive and negative poles in a wiggler or undulator magnet the elliptical polarization is completely suppressed. Ordinary wiggler and undulator magnets do not produce elliptically polarized radiation.

#### Asymmetric Wiggler Magnet

The elimination of elliptical polarization in periodic magnets results from a compensation of left and righthanded helicity and we may therefore look for an insertion device in which this symmetry is broken. Such an insertion device is the asymmetric wiggler magnet which is designed similar to a wavelength shifter with one strong central pole and two weaker poles on either side such that the total integrated field vanishes or \(\int B_{y}\,\mathrm{d}s=0\). A series of such magnets may be

Figure 26.11: Relative intensities of \(\sigma\)-mode and \(\pi\)-mode radiation as a function of vertical observation angle \(\theta\) for different photon energiesaligned to produce an insertion device with many poles to enhance the intensity. The compensation of both helicities does not work anymore since the radiation depends on the magnetic field and not on the total deflection angle. A permanent magnet rendition of an asymmetric wiggler magnet is shown schematically in Fig. 26.13

The degree of polarization from an asymmetric wiggler depends on the desired photon energy. The critical photon energy is high for radiation from the high field pole \(\left(\epsilon_{\mathrm{c}}^{+}\right)\) and lower for radiation from the low field pole \(\left(\epsilon_{\mathrm{c}}^{-}\right)\). For high photon energies \(\left(\epsilon_{\mathrm{ph}}\approx\epsilon_{\mathrm{c}}^{+}\right)\) the radiation from the low field poles is negligible and the radiation is essentially the same as from a series of bending magnets with its particular polarization characteristics. For lower photon energies \(\left(\epsilon_{\mathrm{c}}^{-}<\epsilon_{\mathrm{ph}}<\epsilon_{\mathrm{c}}^ {+}\right)\) the radiation intensity from high and low field pole become similar and cancellation of the elliptical polarization occurs. At low photon energies \(\left(\epsilon_{\mathrm{ph}}<\epsilon_{\mathrm{c}}^{-}\right)\) the intensity from the low field poles exceeds that from the high field poles and we observe again elliptical polarization although with reversed helicity.

##### Elliptically Polarizing Undulator

The creation of elliptically and circularly polarized radiation is important for a large class of experiments using synchrotron radiation and special insertion devices have therefore been developed to meet such needs in an optimal way. Different approaches have been suggested and realized as sources for elliptically

Figure 26.12: Acceleration vectors along one period of (**a**) a wiggler magnet, (**b**) associated polarization vectors, and (**c**) corresponding radiation fields

Figure 26.13: Asymmetric wiggler magnet

polarized radiation, among them for example, those described in [9, 10]. All methods are based on permanent magnet technology, sometimes combined with electromagnets, to produce vertical and horizontal fields shifted in phase such that elliptically polarized radiation can be produced. Utilizing four rows of permanent magnets which are movable with respect to each other and magnetized as shown in Fig. 26.14, elliptically polarized radiation can be obtained.

Figure 26.15 shows the arrangement in a three dimensional rendition to visualize the relative movement of the magnet rows [9, 11].

Figure 26.14: Permanent magnet arrangement to produce elliptically polarized undulator radiation [11]

Figure 26.15: 3-D view of an elliptically polarizing undulator, EPU [11]

The top as well as the bottom row of magnet poles are split into two rows, each of which can be shifted with respect to each other. This way, a continuous variation of elliptical polarization from left to linear to right handed helicity can be obtained. By shifting the top magnet arrays with respect to the bottom magnets the fundamental frequency of the undulator radiation can be varied as well. Figure 26.16 shows a photo of such a magnet [10].

## Problems

### 26.1 (S)

Consider an undulator magnet with a period length of \(\lambda_{\mathrm{p}}=5\,\mathrm{cm}\) in a \(7\,\mathrm{GeV}\) storage ring. The strength parameter be \(K=1\). What is the maximum oscillation amplitude of an electron passing through this undulator? What is the maximum longitudinal oscillation amplitude with respect to the reference system moving with velocity \(\tilde{\beta}\)?

### 26.2 (S)

An undulator with 50 poles, a period length of \(\lambda_{\mathrm{p}}=5\,\mathrm{cm}\) and a strength parameter of \(K=1\) is to be installed into a \(1\,\mathrm{GeV}\) storage ring. Calculate the focal length of the undulator magnet. Does the installation of this undulator require compensation of its focusing properties? How about a wiggler magnet with \(K=5\)?

Figure 26.16: Undulator for elliptically polarized radiation [10]

**26.3 (S).** Consider the expression (26.67) for the photon flux into the forward cone. We also know that the band width of undulator radiation scales like \(\Delta\omega/\omega\propto 1/N_{\rm p}\). With this, the photon flux (26.67) becomes independent of the number of undulator periods! Explain in words, why this expression for the photon flux is indeed a correct scaling law.

**26.4 (S).** A hybrid undulator is to be installed into a 7 GeV storage ring to produce undulator radiation in a photon energy range of 4 keV to 15 keV. The maximum undulator field shall not exceed a value of \(B_{0}\leq 2\) T at a gap aperture of 10 mm. The available photon flux in the forward cone shall be at least 10 % of the maximum flux within the whole spectral range. Specify the undulator parameters and show that the required photon energy range can be covered by changing the magnet gap only.

**26.5 (S).** Consider an electron colliding head-on with a laser beam. What is the wavelength of the laser as seen from the electron system. Derive from this the wavelength of the "undulator" radiation in the laboratory system.

**26.6 (S).** An electron of energy 2 GeV performs transverse oscillations in a wiggler magnet of strength \(K=1.5\) and period length \(\lambda_{\rm p}=7.5\) cm. Calculate the maximum transverse oscillation amplitude. What is the maximum transverse velocity in units of \(c\) during those oscillations. Define and calculate a transverse relativistic factor \(\gamma_{\perp}\). Note, that for \(K\gtrsim 1\) the transverse relativistic effect becomes significant in the generation of harmonic radiation.

**26.7 (S).** Calculate for a 3 GeV electron beam the fundamental photon energy for a 100 period-undulator with \(K=1\) and a period length of \(\lambda_{\rm p}=5\) cm. What is the maximum angular acceptance angle \(\vartheta\) (as determined by adjustable slits) of the beam line, if the radiation spectrum is to be restricted to a bandwidth of 10 %?

**26.8 (S).** Strong mechanical forces exist between the magnetic poles of an undulator when energized. Are these forces attracting or repelling the poles? Why? Consider a \(\ell=\)1 m long undulator with a pole width \(w=0.1\) m, 15 periods each \(\lambda_{\rm p}=7\) cm long and a maximum field of \(B_{0}=1.5\) T. Estimate the total force between the two magnet poles?

**26.9 (S).** In Chap. 23 we mentioned undulator radiation as a result of Compton scattering of the undulator field by electrons. Derive the fundamental undulator wavelength from the process of Compton scattering.

**26.10 (S).** The undulator radiation intensity is a function of the strength parameter \(K\). Find the strength parameter \(K\) for which the fundamental radiation intensity is a maximum. Determine the range of \(K\)-values for which the intensity of the fundamental radiation is within 10 % of the maximum.

**26.11 (S).** Show from (26.54) that along the axis (\(\vartheta=0\)) radiation is emitted only in odd harmonics.

**26.12 (S).** Show from (26.51) that undulator radiation does not produce elliptically polarized radiation in the forward direction (\(\vartheta=0\)).

**26.13 (S).**: Try to design a hybrid undulator for a 3 GeV storage ring to produce 4 keV to 15 keV photon radiation. Is it possible? Why not? Optimize the undulator parameters such that this photon energy range can be covered with the highest flux possible and utilizing lower order harmonics (order 7 or less). Plot the radiation spectrum that can be covered by changing the gap height of the undulator.

**26.14 (S).**: An undulator is constructed from hybrid permanent magnet material with a period length of \(\lambda_{,\mathrm{p}}=5.0\) cm. What is the fundamental wavelength range in a 800 MeV storage ring and in a 7 GeV storage ring if the undulator gap is to be at least 10 mm?

**26.15 (S).**: Determine the tuning range for a hybrid magnet undulator in a 2.5 GeV storage ring with an adjustable gap \(g\geq 10\) mm. Plot the fundamental wavelength as a function of magnet gap for two different period lengths, \(\lambda_{,\mathrm{p}}=15\) mm and \(\lambda_{,\mathrm{p}}=75\) mm. Why are the tuning ranges so different?

**26.16**.: Consider a 26-pole wiggler magnet with a field \(B_{y}\left(\mathrm{T}\right)=1.5\sin\left(\frac{2\pi}{\lambda_{,\mathrm{p}}}z\right)\) and a period length of \(\lambda_{,\mathrm{p}}=15\) cm as the radiation source for a straight through photon beam line and two side stations at an angle \(\vartheta=4\) mr and \(\vartheta=8\) mr in a storage ring with a beam energy of 2.0 GeV. What is the critical photon energy of the photon beam in the straight ahead beam line and in the two side stations?

**26.17**.: Verify the relative intensities of \(\sigma\)-mode and \(\pi\)-mode radiation in Fig.26.12 for two quantitatively different pairs of observation angles \(\vartheta\) and photon energies \(\varepsilon/\varepsilon_{\mathrm{c}}\).

**26.18**.: Design an asymmetric wiggler magnet assuming hard edge fields and optimized for the production of elliptical polarized radiation at a photon energy of your choice. Calculate and plot the photon flux of polarized radiation in the vicinity of the optimum photon energy.

**26.19**.: Calculate the total undulator \(\left(N_{\mathrm{p}}=50,\lambda_{\mathrm{p}}=4.5\) cm, \(K=1.0\right)\) radiation power from a 200 mA, 6 GeV electron beam. Pessimistically, assume all radiation to come from a point source and be contained within the central cone. This is a safe assumption for the design of the vacuum chamber or mask absorbers. Determine the power density at a distance of 15 m from the source. Compare this power density with the maximum acceptable of 10 W/mm\({}^{2}\). How can you reduce the power density, on say a mask, to the acceptable value or below?

**26.20**.: Use the beam and undulator from problem 26.19 and estimate the total radiation power into the forward cone alone. What percentage of all radiation falls within the forward cone? [hint: make reasonable approximations to simplify the math but keep the result reasonably close to the correct answer].

**26.21**.: Derive an expression for the average velocity component \(\tilde{\beta}=\tilde{v}/c\) of a particle traveling through an undulator magnet of strength \(K\).

## References

* [1] H. Motz, J. Appl. Phys. **22**, 527 (1951)
* [2] B.M. Kincaid, J. Appl. Phys. **48**, 2684 (1977)
* [3] S. Chunjarean, Supercond. Sci. Technol. **24**, 055013 (2011)
* [4] P. Elleaume, Synchrotron radiation and free electron laser, in _The CERN Accelerator School (CAS)_, ed. by D. Brandt, number CERN-2005-012 (CERN, Geneva, 2003)
* [5] D.F. Alferov, Y.A. Bashmakov, E.G. Bessonov, Sov. Phys. Tech. Phys. **18**, 1336 (1974)
* [6] S. Krinsky, IEEE Trans. Nucl. Sci. **30**, 3078 (1983)
* [7] A. Bienenstock, G. Brown, H. Winick, H. Wiedemann, Rev. Sci. Instrum. **60**, 1393 (1989)
* [8] W.M. Lavender, Observation and analysis of x-ray undulator radiation from PEP. PhD thesis, Stanford University (1988)
* [9] R. Carr, Nucl. Instrum. Methods A **306**, 391 (1991)
* [10] S. Sasaki, K. Kakuno, T. Takada, T. Shimada, K. Yanagida, Y. Miyahara, Nucl. Instrum. Methods A **331**, 763 (1993)
* [11] R. Carr, S. Lidia, The adjustable phase planar helical undulator. SPIE Proc., vol. 2013 (1993)

