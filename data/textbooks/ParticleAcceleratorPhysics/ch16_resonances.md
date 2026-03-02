## Chapter 16 Resonances

Particle resonances in circular accelerators occur as a result of perturbation terms involving particular Fourier harmonics. That approach is based on the common knowledge that periodic perturbations of a harmonic oscillator can cause a resonance when the perturbation frequency is equal to an eigenfrequency of the oscillator.

### 16.1 Lattice Resonances

Perturbation terms in the equation of motion can lead to a special class of beam instabilities called resonances, which occur if perturbations act on a particle in synchronism with its oscillatory motion. While such a situation is conceivable in a very long beam transport line composed of many periodic sections, the appearance of resonances is generally restricted to circular accelerators. There, perturbations occur periodically at every turn and we may Fourier analyze the perturbation with respect to the revolution frequency. If any of the harmonics of the perturbation terms coincides with the eigenfrequency of the particles a resonance can occur and particles may get lost. Such resonances caused by field imperfections of the magnet lattice are also called structural resonances or lattice resonances. We have already come across two such resonances, the integer and the half-integer resonances.

The characteristics of these two resonances is that the equilibrium orbit and the overall focusing is not defined. Any small dipole or quadrupole error would therefore lead to particle loss as would any deviation of the particle position and energy from ideal values. Since these resonances are caused by linear field errors, we call them also linear resonances. Higher order resonances are caused in a similar way by nonlinear fields which are considered to be field errors with respect to the ideal linear lattice even though we may choose to specifically include such magnets,like sextupoles or octupoles, into the lattice to compensate for particular beam dynamics problems.

#### Resonance Conditions

In this section the characteristics of resonances in circular accelerators will be derived starting from the equation of motion in normalized coordinates with only the \(n\)th order multipole perturbation term. This is no restriction of generality since in linear approximation each multipole perturbation having its own resonance structure will be superimposed to that of other multipole perturbations. On the other hand, the treatment of only a single multipole perturbation will reveal much clearer the nature of the resonance. The equation of motion in normalized horizontal coordinates for an \(n\)th-order perturbation is from (15.1)

\[\ddot{w}+v_{0x}^{2}w=v_{0x}^{2}\beta_{x}^{3/2}\beta_{x}^{r/2}\beta_{y}^{s/2}p _{rs}(\varphi)w^{r}v^{s}, \tag{16.1}\]

where \(v_{0x}\) is the unperturbed horizontal tune and \(r,s=0,1,2\ldots\)integers with \(r+s=n-1\). A similar equation holds for vertical oscillations by replacing \(w\left(\varphi\right),\,v_{0x},x\) with \(v(\varphi)\), etc. and vice versa. The perturbations can be any appropriate term in the equations of motion (6.95) and (6.96), however, we will consider primarily perturbation terms which occur most frequently in circular accelerators and are due to rotated quadrupole or nonlinear multipole fields. The general treatment of resonances for other perturbations is not fundamentally different and is left to the interested reader. From Chap. 9 we extract the dominant perturbation terms of order \(n=r+s+1\) in normalized coordinates and compile them in Table 16.1 ordered by perturbations of order \(r\) and \(s\). To keep track of all types of perturbations requires the use of many indices. We will keep the discussion more simple by ignoring coupling, which will be treated separately, and concentrate only on a single order

\begin{table}
\begin{tabular}{l|l|l|l} \hline Order & & \\ \hline \(r\) & \(s\) & \(v_{i0}^{2}\sqrt{\beta_{x}^{(r+3)}\beta_{y}^{s}p_{rs,x}(\varphi)w^{r}v^{s}}\) & \(v_{j0}^{2}\beta_{x}^{r/2}\beta_{y}^{(s+3)/2}p_{rs,y}(\varphi)\) \\ \hline
0 & 1 & \(-v_{i0}^{2}\beta_{x}^{3/2}\beta_{y}^{1/2}kv\) & \\ \hline
1 & 0 & & \(-v_{j0}^{2}\beta_{x}^{1/2}\beta_{y}^{3/2}kw\) \\ \hline
2 & 0 & \(-v_{i0}^{2}\beta_{x}^{5/2}\frac{1}{2}mw^{2}\) & \\ \hline
2 & 2 & & \(-v_{j0}^{2}\beta_{x}^{1/2}\beta_{y}^{2}m\,wv\) \\ \hline
0 & 2 & \(+v_{x0}^{2}\beta_{x}^{3/2}\beta_{y}\frac{1}{2}mv^{2}\) & \\ \hline
4 & 1 & \(-v_{i0}^{2}\beta_{x}^{3/2}\frac{1}{6}r\,w^{3}\) & \\ \hline
2 & 1 & & \(+v_{j0}^{2}\beta_{x}\beta_{y}^{2}\frac{1}{2}r\,w^{2}v\) \\ \hline
1 & 2 & \(+v_{x0}^{2}\beta_{x}^{2}\beta_{y}\frac{1}{2}r\,wv^{2}\) & \\ \hline
0 & 3 & & \(-v_{j0}^{2}\beta_{y}^{3}\frac{1}{6}r\,v^{3}\) \\ \hline \end{tabular}
\end{table}
Table 16.1: Lowest order perturbation termsof perturbation. For simplicity we drop from here on the index \({}_{x}\) and set \(\beta_{x}=\beta\). Equation (16.1) becomes then

\[\ddot{w}+v_{0}^{2}w=v_{0}^{2}\beta^{n/2+1}p_{n}(\varphi)w^{n-1}, \tag{16.2}\]

All perturbations are periodic in \(\varphi\) and can be expanded into a Fourier series

\[q_{n}\left(\varphi\right)=v_{0}^{2}\beta^{n/2+1}p_{n}(\varphi)=\sum_{m}q_{nm} \mathrm{e}^{\mathrm{i}m\varphi}. \tag{16.3}\]

Since the perturbation is supposed to be small, we will insert the unperturbed oscillation \(w_{0}\) on the right-hand side of (16.2). The general form of the unperturbed betatron oscillation can be written like

\[w_{0}(\varphi)=a\,\mathrm{e}^{\mathrm{i}v_{0}\varphi}+b\,\mathrm{e}^{- \mathrm{i}v_{0}\varphi}, \tag{16.4}\]

where \(a\) and \(b\) are arbitrary constants and we may now express the amplitude factor in the perturbation term \(w^{n-1}(\varphi)\) by a sum of exponential terms which we use on the right hand side of (16.2) as a first approximation

\[w^{n-1}(\varphi)\approx w_{0}^{n-1}(\varphi)=\sum_{|l|\leq n-1}W_{l}\,\mathrm{ e}^{-\mathrm{i}lv_{0}\varphi}. \tag{16.5}\]

Inserting both (16.3) and (16.5) into (16.2) we get for the equation of motion

\[\ddot{w}+v_{0}^{2}w=\sum_{l,m}W_{l}\,q_{nm}\,\mathrm{e}^{-\mathrm{i}(m+lv_{0}) \varphi}. \tag{16.6}\]

The solution of this equation includes resonant terms whenever there is a perturbation term with a frequency equal to the eigenfrequency \(v_{0}\) of the oscillator. The resonance condition is therefore \((m,n,l\) are integers)

\[m+lv_{0}=v_{0}\qquad\text{with}\qquad|l|\leq n-1. \tag{16.7}\]

From earlier discussions we expect to find here the integer resonance caused by dipole errors \(n=1\). In this case the index \(l\) can only be \(l=0\) and we get from (16.7)

\[v_{0}=m \tag{16.8}\]

which is indeed the condition for an integer resonance.

Magnetic gradient field errors (\(n=2\)) can cause both a half integer-resonance as well as an integer resonance. The index \(l\) can have the values \(l=0\) and \(l=\pm 1\). Note however that not all coefficients \(W_{l}\) necessarily are nonzero. In this particular case, the coefficient for \(l=0\) is indeed zero as becomes obvious by inspection of (16.4). The resonance conditions for these second order resonances are

\[\begin{array}{llll}m+v_{0}=v_{0}\ \to\ m=0&\to\ \mbox{tune shift at any tune,}\\ m-v_{0}=v_{0}\ \to\ m=2v_{0}&\to\ \mbox{integer and half integer resonance,}\\ m=v_{0}&\to\ \mbox{no resonance because $W_{0}=0$.}\end{array} \tag{16.9}\]

Among the resonance conditions (16.9) we notice that for \(m=0\) the effect of the perturbation on the particle motion exists independent of the particular choice of the tune \(v_{0}\). The perturbation includes a nonvanishing average value \(q_{20}\) which in this particular case represents the average gradient error of the perturbation. Like any other gradient field in the lattice, this gradient error also contributes to the tune and therefore causes a tune shift. From (16.2) we find the new tune to be determined by \(v^{2}=v_{0}^{2}\left[1-\left\langle\beta^{2}p_{2}\right\rangle_{\varphi}\right]\) and the tune shift is \(\delta v\approx-\frac{1}{2}v_{0}\left\langle\beta^{2}p_{2}\right\rangle_{\varphi}\) in agreement with our earlier result in Sect. 15.3.1.

Third order resonances (\(n=3\)) can be driven by sextupole fields and the index \(l\) can have values

\[l=-2,\ -1,\ 0,\ +1,\ +2\,. \tag{16.10}\]

Here we note that \(W_{1}=W_{-1}=0\) and therefore no resonances occur for \(l=\pm 1\). The resonance for sextupole field perturbations are then

\[\begin{array}{llll}m-2v_{0}=v_{0}\ \to\ m=3v_{0}&\to\ \mbox{third order resonance,}\\ m=v_{0}\ \to\ m=v_{0}&\to\ \mbox{integer resonance,}\\ m+2v_{0}=v_{0}\ \to\ m=-v_{0}&\to\ \mbox{integer resonance.}\end{array} \tag{16.11}\]

Sextupole fields can drive third order resonances at tunes of

\[v_{0}=r+\tfrac{1}{3}\qquad\quad\mbox{or}\qquad\quad v_{0}=r-\tfrac{1}{3}\,, \tag{16.12}\]

where \(r\) is an integer. They also can drive integer resonances.

Finally we derive resonance conditions for octupole fields (\(n=4\)) where

\[l=-3,\ -2,\ -1,\ 0,\ +1,\ +2,\ +3 \tag{16.13}\]

and again some values of \(l\) do not lead to a resonance since the amplitude coefficient \(W_{q}\) is zero. For octupole terms this is the case for \(l=0\) and \(l=\pm 2\). The remaining resonance terms are then

\[\begin{array}{llll}m-3v_{0}=v_{0}\ \to\ m=4v_{0}&\to\ \mbox{quarter integer resonance,}\\ m-v_{0}=v_{0}&\to\ m=2v_{0}&\to\ \mbox{half integer resonance,}\\ m+v_{0}=v_{0}&\to\ m=0&\to\ \mbox{tune spread at any tune,}\\ m+3v_{0}=v_{0}&\to\ m=-2v_{0}&\to\ \mbox{half integer resonance.}\end{array} \tag{16.14}\]The resonance condition for \(m=0\) leads to a shift in the oscillation frequency. Different from gradient errors, however, we find the tune shift generated by octupole fields to be amplitude dependent \(\upsilon^{2}=\upsilon_{0}^{2}\left[1-\left\langle\beta^{3}p_{4}w^{2}\right\rangle _{\varphi}\right]\). The amplitude dependence of the tune shift causes an asymmetric tune spread to higher or lower values depending on the sign of the perturbation term \(p_{4}\) while the magnitude of the shift is determined by the oscillation amplitude of the particle.

The general resonance condition for betatron oscillations in one plane can be expressed by

\[|m|=(|l|\pm 1)\upsilon_{0}, \tag{16.15}\]

where \(|l|\leq n-1\) and the value \(|l|+1\) is the order of resonance. The index \(m\) is equal to the order of the Fourier harmonic of the perturbation and we call therefore these resonances structural or lattice resonances to distinguish them from resonances caused, for example, by externally driven oscillating fields.

The maximum order of resonances in this approximation depends on the order of nonlinear fields present. An \(n\)th-order multipole field can drive all resonances up to \(n^{\text{th}}\)-order with driving amplitudes that depend on the actual multipole field strength and locations within the lattice. Generally, the higher the order \(n\) the weaker is the resonance. In electron circular accelerators radiation damping makes higher order resonances ineffective. This is not the case for proton or ion beams which accumulate any effect leading, if not to beam loss, then to beam dilution or emittance blow-up.

The term resonance is used very generally to include also effects which do not necessarily lead to a loss of the beam. Such "resonances" are characterized by \(m=0\) and are independent of the tune. In the case of gradient errors this condition was shown to lead to a stable shift in tune for the whole beam. Unless this tune shift moves the beam onto another resonance the beam stability is not affected. Similarly, octupole fields introduce a spread of tunes in the beam proportional to the square of the oscillation amplitude. Again no loss of particles occurs unless the tune spread reaches into the stop band of a resonance. By induction we conclude that all even perturbation terms, where \(n\) is an even integer, lead to some form of tune shift or spread. No such tune shifts occur for odd order perturbations in the approximation used here. Specifically we note that dipoles, sextupoles or decapoles etc. do not lead to a tune shift for weak perturbations. Later, however, we will discuss the Hamiltonian resonance theory and find, for example, that strong sextupole perturbations can indeed cause a tune spread in higher order.

In this derivation of resonance parameters we have expanded the perturbations into Fourier series and have assumed the full circular accelerator lattice as the expansion period. In general, however, a circular accelerator is composed of one or more equal superperiods. For a circular lattice composed of \(N\) superperiods the Fourier expansion has nonzero coefficients only every \(N\)th-harmonic and therefore the modified resonance conditions are

\[|j|N=(|l|\pm 1)v_{0}\,, \tag{16.16}\]

where \(j\) is an integer. A high super-periodicity actually eliminates many resonances and is therefore a desirable design feature for circular accelerator lattices. The integer and half-integer resonances, however, will always be present independent of the super-periodicity because the equilibrium orbits and the betatron functions respectively are not defined. On the other hand, integer and half-integer resonances driven by multipole perturbations may be eliminated in a high periodicity lattice with the overall effect of a reduced stop band width. It should be noted here, that the reduction of the number of resonances works only within the applied approximation. "Forbidden" resonances may be driven through field and alignment errors which compromise the high lattice periodicity or by strong non-linearities and coupling creating resonant driving terms in higher order approximation. Nevertheless, the forbidden resonances are weaker in a lattice of high periodicity compared to a low periodicity lattice.

#### Coupling Resonances

Betatron motion in a circular accelerator occurs in both the horizontal and vertical plane. Perturbations can be present which depend on the betatron oscillation amplitude in both planes. Such terms are called coupling terms. The lowest order coupling term is caused by a rotated quadrupole or by the rotational misalignment of regular quadrupoles. In general we have in the horizontal plane the equation of motion from (16.1)

\[\ddot{w}+v_{0x}^{2}w=v_{0x}^{2}\beta_{x}^{3/2}\beta_{x}^{r/2}\beta_{y}^{s/2}p_ {rs}(\varphi)w^{r}v^{s}, \tag{16.17}\]

where \(r,s\) are integers and \(w,v\) describe betatron oscillations in the horizontal and vertical plane, respectively. Again we use the unperturbed solutions \(w_{0}(\varphi)\) and \(v_{0x}(\varphi)\) of the equations of motion in the form (16.4) and express the higher order amplitude terms in the perturbation by the appropriate sums of trigonometric expressions:

\[q_{rs}(\varphi)=v_{0x}^{2}\beta_{x}^{3/2}\beta_{x}^{r/2}\beta_{y}^{s/2}p_{rs}( \varphi)=\sum_{m}q_{rsm}\mathrm{e}^{\mathrm{i}mp}, \tag{16.18}\]

and similar to (16.5)

\[w^{r-1}(\varphi) =\sum_{|l|\leq r-1}W_{l}\mathrm{e}^{\mathrm{i}lv_{0x}\varphi}, \tag{16.19a}\] \[v^{s-1}(\varphi) =\sum_{|\ell|\leq s-1}V_{\ell}\mathrm{e}^{\mathrm{i}fv_{0y}\varphi}. \tag{16.19b}\]Insertion into (16.17) gives after some sorting

\[\ddot{w}+v_{0x}^{2}w=\sum q_{\tau sm}W_{l}V_{\ell}\mathrm{e}^{\mathrm{i}[(m+lv_{0x }+\ell v_{0y})\varphi]}, \tag{16.20}\]

where \(m,l\) and \(\ell\) are integers. The resonance condition is

\[m+lv_{0x}+\ell v_{0y}=v_{0x}\,, \tag{16.21}\]

and the quantity

\[|l|+|\ell|+1\leq n \tag{16.22}\]

designates the order of the coupling resonances. Again, for a super-periodicity \(N\) we replace \(m\) by \(jN\), where \(j\) is an integer. As an example, we discuss a perturbation term caused by a rotated quadrupole for which the equation of motion is

\[\ddot{w}+v_{0}^{2}w=q_{01}(\varphi)v\,. \tag{16.23}\]

In this case we have \(n=2\) and \(r=0\) and the lowest order resonance condition with \(l=0\) and \(\ell=\pm 1\) is from (16.21)

\[m+\ell v_{0y}=v_{0x}. \tag{16.24}\]

Resonance occurs for

\[|m|=v_{0x}+v_{0y}\qquad\text{and}\qquad|m|=v_{0x}-v_{0y}. \tag{16.25}\]

There is no coupling resonance for \(\ell=0\) since \(V_{0}=0\). The resonances identified in (16.25) are called linear coupling resonances or linear sum resonance (left) and linear difference resonance (right), respectively.

Delaying proof for a later discussion we note at this point that the sum resonance can lead to a loss of beam while the difference does not cause a loss of beam but rather leads to an exchange of horizontal and vertical betatron oscillations. In circular accelerator design we therefore adjust the tunes such that a sum resonance is avoided.

##### Resonance Diagram

The resonance condition (16.15) has been derived for horizontal motion only, but a similar equation can be derived for the vertical motion. Both resonance conditions can be written in a more symmetric way

\[lv_{0x}+\ell v_{0y}=jN, \tag{16.26}\]where \(l,\ell,j\) are integers and \(|l|\,+\,|\ell|\) is the order of the resonance. Plotting all lines (16.26) for different values of \(l,\ell,j\) in a \((v_{y},v_{x})\) diagram produces what is called a resonance diagram. In Fig. 16.1 an example of a resonance diagram for \(N=1\) is shown displaying all resonances up to third order with \(|l|\,+\,|\ell|\,\leq 3\).

The operating points for a circular accelerator are chosen to be clear of any of these resonances. It should be noted here that the resonance lines are not mathematically thin lines in the resonance diagram but rather exhibit some "thickness" which is called the stop band width. This stop band width depends on the strength of the resonance as was discussed earlier.

Not all resonances are of the same strength and generally get weaker with increasing order. While a particle beam would not survive on an integer or a half-integer resonance all other resonances are basically survivable, at least for electron beams. For proton or ion beams higher order resonance must be avoided to prevent beam dilution. Only in particular cases, where strong multipole field perturbations cause a higher order resonance, may we observe beam loss. This is very likely to be the case for third order resonances in rings, where strong sextupole magnets are employed to correct for chromatic aberrations.

The beneficial effect of a high super-periodicity or symmetry \(N\) in a circular accelerator becomes apparent in such a resonance diagram because the density of resonance lines is reduced by the factor \(N\) and the area of stability between resonances to operate the accelerator becomes proportionately larger. In Fig. 16.2, the resonance diagram for a ring with super-periodicity four (\(N=4\)) is shown and the reduced number of resonances is obvious. Wherever possible a high symmetry in the design of a circular accelerator should be attempted. Conversely, breaking a high order of symmetry can lead to a reduction in stability if not otherwise compensated.

Figure 16.1: Resonance diagram for a ring with superperiodicity one, \(N=1\)

### 16.2 Hamiltonian Resonance Theory*

In the realm of Hamiltonian resonance theory we will be able to derive not only obvious resonant behavior but also resonant dynamics which does not necessarily lead to a loss of the beam but to a significant change of beam parameters. We also will be able to determine the strength of resonances, effectiveness, escape mechanisms and more.

#### Non-linear Hamiltonian

While simple Fourier expansions of perturbations around a circular accelerator allow us to derive the locations of lattice resonances in the tune diagram, we can obtain much deeper insight into the characteristics of resonances through the application of the Hamiltonian theory of linear and nonlinear oscillators. Soon after the discovery of strong focusing, particle dynamicists noticed the importance of perturbations with respect to beam stability and the possibility of beam instability even in the presence of very strong focusing.

Extensive simulations and development of theories were pursued in an effort to understand beam stability in high-energy proton synchrotrons then being designed at the Brookhaven National Laboratory and CERN. The first Hamiltonian theory of linear and non linear perturbations has been published by Schoch [1] which includes also references to early attempts to solve perturbation problems. A modern, consistent and complete theory of all resonances has been developed, for example,

Figure 16.2: Resonance diagram for a ring with superperiodicity four, \(N=4\)by Guignard [2]. In this text, we will concentrate on main features of resonance theory and point the interested reader for more details to these references.

Multipole perturbations have been discussed as the source of resonances and we will discuss in this chapter the Hamiltonian resonance theory. The equation of motion under the influence of an \(n\)th-order perturbation is in normalized coordinates and in the horizontal plane without coupling (see Table 16.1)

\[\ddot{w}+v_{0}^{2}w=q_{n}(\varphi)\,w^{n-1}, \tag{16.27}\]

which can be also derived from the nonlinear Hamiltonian

\[H_{w}=\,\tfrac{1}{2}\dot{w}^{2}+\,\tfrac{1}{2}v_{0}^{2}\,w^{2}+\overline{q}_{ n}(\varphi)\left(\frac{v_{0}}{2}\right)^{n/2}\,w^{n}. \tag{16.28}\]

Here we introduced

\[\overline{q}_{n}(\varphi)=-q_{n}(\varphi)\frac{1}{n}\left(\frac{v_{0}}{2} \right)^{-n/2} \tag{16.29}\]

for future convenience.

To discuss resonance phenomena it is useful to perform a canonical transformation from the coordinates \((w,\dot{w})\) to action-angle variables \((J,\psi)\) which can be derived from the generating function (5.52) and the new Hamiltonian expressed in action-angle variables is

\[H=v_{0}J+\overline{q}_{n}(\varphi)J^{n/2}\cos^{n}\left(\psi-\vartheta\right). \tag{16.30}\]

The action-angle variables take on the role of an "energy" and frequency of the oscillatory system. Due to the phase dependent perturbation \(p_{n}\left(\varphi\right)\) the oscillation amplitude \(J\) is no more a constant of motion and the circular motion in phase space becomes distorted as shown in Fig. 16.3 for a sextupolar perturbation. The oscillator frequency \(\dot{\psi}=\partial\psi/\partial\varphi=v\) is similarly perturbed and can be derived from the second Hamiltonian equation of motion

\[\frac{\partial H}{\partial J}=\dot{\psi}=v_{0}+\,\tfrac{n}{2}\overline{q}_{n} (\varphi)J^{n/2-1}\cos^{n}\left(\psi-\vartheta\right). \tag{16.31}\]

Perturbation terms seem to modify the oscillator frequency \(v_{0}\) but because of the oscillatory trigonometric factor it is not obvious if there is a net shift or spread in the tune. We therefore expand the perturbation \(\overline{q}_{n}(\varphi)\) as well as the trigonometric factor \(\cos^{n}\psi\) to determine its spectral content. The distribution of the multipole perturbations in a circular accelerator is periodic with a periodicity equal to the length of a superperiod or of the whole ring circumference and we are therefore able to expand the perturbation \(\overline{q}_{n}(\varphi)\) into a Fourier series

\[\overline{q}_{n}(\varphi)=\sum_{l}\overline{q}_{nl}\,\mathrm{e}^{- \mathrm{i}N\varphi}, \tag{16.32}\]

where \(N\) is the super-periodicity of the circular accelerator. We also expand the trigonometric factor in (16.31) into exponential functions, while dropping the arbitrary phase \(\vartheta\)

\[\cos^{n}\psi=\sum_{|m|\leq n}c_{nm}\,\mathrm{e}^{\mathrm{i}m\psi} \tag{16.33}\]

and get

\[\overline{q}_{n}(\varphi)\cos^{n}\psi = \sum_{l}\overline{q}_{nl}\,\mathrm{e}^{-\mathrm{i}N\varphi}\sum_ {|m|\leq n}c_{nm}\,\mathrm{e}^{\mathrm{i}m\psi}\] \[= \sum_{|m|\leq n}c_{nm}\overline{q}_{nl}\,\mathrm{e}^{\mathrm{i}( m\psi-\mathit{l}N\varphi)}\] \[= c_{n0}\overline{q}_{n0}+\sum_{\begin{subarray}{c}l\geq 0\\ 0<m\leq n\end{subarray}}2c_{nm}\overline{q}_{nl}\cos(m\psi-\mathit{l}N\varphi)\,.\]

In the last equation, the perturbation \(\overline{q}_{n}(\varphi)\) is expanded about a symmetry point merely to simplify the expressions of resonant terms. For asymmetric lattices the derivation is similar but includes extra terms. We have also separated the non-oscillatory term \(c_{n0}p_{n0}\) from the oscillating terms to distinguish between systematic frequency shifts and mere periodic variations of the tune. The Hamiltonian (16.30) now becomes with (16.34)

\[H=v_{0}J+c_{n0}\,\overline{q}_{n0}\,J^{n/2}+J^{n/2}\sum_{ \begin{subarray}{c}l\geq 0\\ 0<m\leq n\end{subarray}}2c_{nm}\,\overline{q}_{nl}\,\cos(m\psi-\mathit{l}N \varphi)\,. \tag{16.35}\]

Figure 16.3: Nonlinear perturbation of phase-space motion

The third term on the r.h.s. consists mostly of fast oscillating terms which in this approximation do not lead to any specific consequences. For the moment we will ignore these terms and remember to come back later in this chapter. The shift of the oscillator frequency due to the lowest-order perturbation becomes obvious and may be written as

\[\frac{\partial H}{\partial J}=\dot{\psi}=v_{0}+\,\tfrac{\pi}{2}c_{n0}\,\overline {q}_{n0}\,J^{n/2-1}+\text{oscillatory terms}. \tag{16.36}\]

Since \(c_{n0}\neq 0\) for even values of \(n\) only, we find earlier results confirmed, where we observed the appearance of amplitude-dependent tune shifts and tune spreads for even-order perturbations. Specifically we notice, that there is a coherent amplitude independent tune shift for all particles within a beam in case of a gradient field perturbation with \(n=2\) and an amplitude dependent tune spread within a finite beam size for all other higher- and even-order multipole perturbations.

To recapitulate the canonical transformation of the normalized variables to action-angle variables has indeed eliminated the angle coordinate as long as we neglect oscillatory terms. The angle variable therefore is in this approximation a cyclic variable and the Hamiltonian formalism tells us that the conjugate variable, in this case the amplitude \(J\) is a constant of motion or an invariant. This is an important result which we obtained by simple application of the Hamiltonian formalism confirming our earlier expectation to isolate constants of motion.

This has not been possible in a rigorous way since we had to obtain approximate invariants by neglecting summarily all oscillatory terms. In certain circumstances this approximation can lead to totally wrong results. To isolate these circumstances we pursue further canonical transformations to truly separate from the oscillating terms all non-oscillating terms of order \(n/2\) while the rest of the oscillating terms are transformed to a higher order in the amplitude \(J\).

#### Resonant Terms

Neglecting oscillating terms is justified only in such cases where these terms oscillate rapidly. Upon closer inspection of the arguments in the trigonometric functions we notice however that for each value of \(m\) in (16.35) there exists a value \(l\) which causes the phase

\[m_{r}\psi_{r}\approx mV\psi-lN\varphi \tag{16.37}\]

to vary only slowly possibly leading to a resonance. The condition for the occurrence of such a resonance is \(\psi_{r}\approx 0\) or with \(\psi\approx v_{0}\varphi\)

\[m_{r}v_{0}\approx rN, \tag{16.38}\]where we have set \(l=r\) to identify the index for which the resonance condition (16.38) is met. The index \(m_{r}\) is the order of the resonance and can be any integer \(1\leq m_{r}\leq n\).

The effects of resonances do not only appear when the resonance condition is exactly fulfilled for \(\psi_{\tau}=0\). Significant changes in the particle motion can be observed when the particle oscillation frequency approaches the resonance condition. We therefore keep all terms which vary slowly compared to the betatron frequency \(\dot{\psi}\).

After isolating resonant terms we may now neglect all remaining fast oscillating terms with \(m\neq m_{r}\). Later we will show that these terms can be transformed to higher order and are therefore of no consequence to the order of approximation of interest. Keeping only resonant terms defined by (16.38) we get from (16.35) the \(n\)th-order Hamiltonian in normalized coordinates

\[H=v_{0}J+c_{n0}\,\overline{q}_{n0}J^{n/2}+J^{n/2}\sum_{\begin{subarray}{c}r\\ 0<m_{r}\leq n\end{subarray}}2c_{nn_{r}}\,\overline{q}_{nr}\cos(m_{r}\psi_{r}). \tag{16.39}\]

The value of \(m_{r}\) indicates the order of the resonance and we note that the maximum order of resonance driven by a multipole of order \(n\) is not higher than \(n\). A dipole field therefore can drive only an integer resonance, a quadrupole field up to a half-integer resonance, a sextupole up to a third-order resonance, an octupole up to a quarter resonance and so forth although not all allowed resonances become real. We know for example already that a sextupole does not drive a tune shift or a quarter integer resonance in the approximation used so far. As we have noticed before, whenever we derive mathematical results we should keep in mind that such results are valid only within the approximation under consideration. It is, for example, known [3] that sextupoles can also drive quarter integer resonances through higher-order terms. In nonlinear particle beam dynamics any statement about stability or instability must be accompanied by a statement defining the order of approximation made to allow independent judgement for the validity of a result to a particular problem.

The interpretation of the Hamiltonian (16.39) becomes greatly simplified after another canonical transformation to eliminate the appearance of the independent variable \(\varphi\). We thereby transform to a coordinate system that moves with the reference particle, thus eliminating the linear motion that we already know. The new coordinates rotate once per revolution and thereby eliminate the linear rotation in phase space that we know already. This can be achieved by a canonical similarity transformation from the coordinates \((J,\psi)\) to \((J_{1},\psi_{1})\) which we derive from the generating function

\[G_{1}=J_{1}\left(\psi-\frac{rN\varphi}{m_{r}}\right). \tag{16.40}\]From this we get the relations between the old and new coordinates

\[\frac{\partial G_{1}}{\partial J_{1}}=\psi_{1}=\psi-\frac{rN}{m_{r}}\varphi \tag{16.41}\]

and

\[\frac{\partial G_{1}}{\partial\psi}=J=J_{1}\,. \tag{16.42}\]

The quantity \(\psi_{1}\) now describes the phase deviation of a particle from that of the reference particle. Since the generating function depends on the independent variable \(\varphi\) we get for the new Hamiltonian \(H_{1}=H+\partial G_{1}/\partial\varphi\) or

\[H_{1}=\left(v_{0}-\frac{rN}{m_{r}}\right)J_{1}+c_{n0}\,\overline{q}_{n0}J_{1} ^{n/2}+2c_{nm_{r}}\,\overline{q}_{nr}J_{1}^{n/2}\cos(m_{r}\psi_{1}+rN\varphi), \tag{16.43}\]

where we have retained for simplicity only the highest-order resonant term \((m_{r}=n)\). With \(\dot{\psi}=(\mathrm{d}\psi/\mathrm{d}\varphi)=v\) and (16.38) a resonance condition occurs whenever

\[v_{0}\approx\frac{rN}{m_{r}}=v_{r}. \tag{16.44}\]

Setting \(\Delta v_{r}=v_{0}-v_{r}\) for the distance of the tune \(v_{0}\) from the resonance tune \(v_{r}\) the Hamiltonian becomes with all perturbation terms

\[H=\Delta v_{r}J_{1}+\sum_{n}c_{n0}\,\overline{q}_{n0}\,J_{1}^{n/2}+\sum_{n}J_ {1}^{n/2}\sum_{\begin{subarray}{c}r\\ 0<m_{r}\leq n\end{subarray}}2c_{nm_{r}}\overline{q}_{nr}\cos(m_{r}\psi_{1}). \tag{16.45}\]

The coefficients \(c_{n0}\) are defined by (16.33) and the harmonic amplitudes of the perturbations are defined by the Fourier expansion (16.32). The resonance order \(r\) and integer \(m_{r}\) depend on the ring tune and are selected such that (16.44) is approximately met. A selection of most common multipole perturbations are compiled in Table 16.1 and picking an \(n\)th-order term we get from (16.29) the expression for \(\overline{q}_{nr}\).

In the course of the mathematical derivation we started out in (16.28) with only one multipole perturbation of order \(n\). For reasons of generality, however, all orders of perturbation \(n\) have been included again in (16.45). We will, however, not deal with the complexity of this multi-resonance Hamiltonian nor do we need to in order to investigate the character of individual resonances. Whatever the number of actual resonances may be in a real system the effects are superpositions of individual resonances. We will therefore investigate in more detail single resonances and discuss superpositions of more than one resonance later in this chapter.

#### Resonance Patterns and Stop-Band Width

Equation (16.45) can be used to calculate the stop band width of resonances and to explore resonance patterns which are a superposition of particle trajectories \(H=\)const in \((\psi_{1},J_{1})\) phase-space. Depending on the nature of the problem under study, we may use selective terms from both sums in (16.45). Specifically to illustrate characteristic properties of resonances, we will use from the first sum the term \(c_{40}\overline{q}_{40}\), which originates from an octupole field. This is the lowest order term that provides some beam stability as we will see. From the second sum we choose a single \(n\)th-order term driving the \(r\)th-order resonance and get the simplified Hamiltonian

\[H_{1}=\Delta v_{r}J_{1}+c_{40}\,\overline{q}_{40}J_{1}^{2}+2c_{nm_{r}}\overline {q}_{nr}\,J_{1}^{n/2}\mbox{cos}(m_{r}\psi_{1})=\mbox{const}. \tag{16.46}\]

To further simplify the writing of equations and the discussion of results we divide (16.46) by \(2c_{nm_{r}}\overline{q}_{nr}\,J_{10}^{n/2}\), where the amplitude \(J_{10}\) is an arbitrary reference amplitude of a particle at the starting point. Defining an amplitude ratio or beat factor

\[R=\frac{J_{1}}{J_{10}}, \tag{16.47}\]

and considering only resonances of order \(m_{r}\approx n\), (16.46) becomes

\[R\Delta+R^{2}\Omega+R^{n/2}\cos n\psi_{1}=\mbox{const} \tag{16.48}\]

where the detuning from the resonance is

\[\Delta=\frac{\Delta v_{r}}{2c_{nm_{r}}\overline{q}_{nr}\,J_{10}^{n/2-1}} \tag{16.49}\]

and the tune-spread parameter

\[\Omega=\frac{c_{40}\,\overline{q}_{40}}{2c_{nm_{r}}\overline{q}_{nr}\,J_{10}^ {n/2-2}}\,. \tag{16.50}\]

This expression has been derived first by Schoch [1] for particle beam dynamics. Because the ratio \(R\) describes the variation of the oscillation amplitude in units of the starting amplitude \(J_{0}\) we call the quantity \(R\) the beat factor of the oscillation.

Before we discuss stop-bands and resonance patterns we make some general observations concerning particle stability. The stability of particle motion in the vicinity of resonances depends strongly on the distance of the tune from the nearest \(n\)th-order resonance and on the tune-spread parameter \(\Omega\). When both parameters \(\Delta\) and \(\Omega\) vanish we have no stability for any finite oscillation amplitude, since (16.48)can be solved for all values of \(\psi_{1}\) only if \(R=0\). For a finite tune-spread parameter \(\Omega\neq 0\) while \(\Delta=0\) (16.48) becomes \(R^{2}\left(\Omega+R^{n/2-2}\cos n\psi_{1}\right)=\text{const}\) and resonances of order \(n>4\) exhibit some range of stability for amplitudes \(R^{n/2-2}<|\Omega|\). Oscillations in the vicinity of, for example, a quarter resonance are all stable for \(|\Omega|>1\) and all unstable for smaller values of the tune-spread parameter \(|\Omega|<1\). A finite tune-spread parameter \(\Omega\) appears in this case due to an octupolar field and has a stabilizing effect at least for small amplitudes.

For very small oscillation amplitudes (\(R\to 0\)) the oscillating term in (16.48) becomes negligible for \(n>4\) compared to the detuning term and the particle trajectory approaches the form of a circle with radius \(R\). This well behaved character of particle motion at small amplitudes becomes distorted for resonances of order \(n=2\) and \(n=3\) in case of small detuning and a finite tune spread parameter. We consider \(\Delta=0\) and have

\[R^{2}\Omega+R^{n/2}\cos n\psi=\text{const}, \tag{16.51}\]

where \(n=2\) or \(n=3\). For very small amplitudes the quadratic term is negligible and the dominant oscillating term alone is unstable. The amplitude for a particle starting at \(R\approx 0\) and \(\psi_{1}=0\) grows to large amplitudes as \(\psi_{1}\) increases, reaching values which make the quadratic tune-spread term dominant before the trigonometric term becomes negative. The resulting trajectory in phase space becomes a figure of eight for the half-integer resonance as shown in Fig. 16.4.

In the case of a third-order resonance small-amplitude oscillations behave similarly and follow the outline of a clover leave as shown in Fig. 16.5.

#### Half-Integer Stop-Band

A more detailed discussion of (16.45) will reveal that instability due to resonances does not only happen exactly at resonant tunes. Particle oscillations become unstable within a finite vicinity of resonance lines in the resonance diagram and such areas of instability are known as stop-bands. The most simple case occurs for \(\Omega=0\) and a half-integer resonance, where \(n=2\) and

\[R(\Delta+\cos 2\psi_{1})=\text{const}\,. \tag{16.52}\]

For this equation to be true for all values of the angle variable \(\psi_{1}\) we require that the quantity in the brackets does not change sign while \(\psi_{1}\) varies from 0 to 2 \(\pi\). This condition cannot be met if \(|\Delta|\,\leq 1\). To quantify this we observe a particle starting with an amplitude \(J\,=\,J_{0}\) at \(\psi_{1}=0\) and (16.52) becomes

\[R\Delta+R\cos 2\psi_{1}=\Delta+1. \tag{16.53}\]

Now we calculate the variation of the oscillation amplitude \(R\) as the angle variable \(\psi_{1}\) increases. The beat factor \(R\) reaches its maximum value at \(2\psi_{1}=\pi\) and is

\[R_{\text{max}}=\frac{\Delta+1}{\Delta-1}>0. \tag{16.54}\]The variation of the amplitude \(R\) is finite as long as \(\Delta>1\). If \(\Delta<0\), we get a similar stability condition

\[R_{\rm max}=\frac{|\Delta|-1}{|\Delta|+1}>0 \tag{16.55}\]

and stability occurs for \(\Delta<-1\). The complete resonance stability criterion for the half-integer resonance is therefore

\[|\Delta|>1. \tag{16.56}\]

Beam instability due to a half-integer resonance (\(n=2\)) occurs within a finite vicinity \(\Delta v_{r}=\pm 2c_{2r}\overline{q}_{2r}\), as defined by (16.49) and the total stop-band width for a half-integer resonance becomes

\[\Delta v_{\rm stop}^{(2)}=\pm 2c_{2m}\overline{q}_{2r}\,. \tag{16.57}\]

The width of the stop-band increases with the strength of the perturbation but does not depend on the oscillation amplitude \(J_{0}\). However, for higher-order resonances the stop band width does depend on the oscillation amplitudes as will be discussed later.

To observe the particle trajectories in phase space, we calculate the contour lines for (16.48) setting \(n=2\) and obtain patterns as shown in Fig. 16.4. Here the particle trajectories are plotted in the \((\psi,J)\) phase space for a variety of detuning parameters \(\Delta\) and tune-spread parameters \(\Omega\). Such diagrams are called resonance patterns. The first row of Fig. 16.4 shows particle trajectories for the case of a half-integer resonance with a vanishing tune-spread parameter \(\Omega=0\). As the detuning \(\Delta\) is increased we observe a deformation of particle trajectories but no appearance of a stable island as long as \(|\Delta|<1\). Although we show mostly resonance patterns for negative values of the detuning \(\Delta<0\) the patterns look exactly the same for \(\Delta>0\) except that they are rotated by \(90^{\circ}\). For \(|\Delta|>1\) the unstable trajectories part vertically from the origin and allow the appearance of a stable island that grows as the detuning grows. In the second row of resonance patterns, we have included a finite tune-spread parameter of \(\Omega=1\) which leads to a stabilization of all large amplitude particle trajectories. Only for small amplitudes do we still recognize the irregularity of a figure of eight trajectory as mentioned above.

#### Separatrices

The appearance of island structures as noticeable from the resonance patterns is a common phenomenon and is due to tune-spread terms of even order like that of an octupole field. In Fig. 16.6 common features of resonance patterns are shown and we note specifically the existence of a central stable part and islands surrounding the central part.

The boundaries of the areas of stable motion towards the islands are called separatrices. These separatrices also separate the area of stable motion from that for unstable motion. The crossing points of these separatrices, as well as the center of the islands, are called fixed points of the dynamic system and are defined by the conditions

\[\frac{\partial H_{1}}{\partial\psi_{1}}=0\qquad\text{and}\qquad\frac{\partial H _{1}}{\partial J_{1}}=0\,. \tag{16.58}\]

Application of these conditions to (16.46) defines the location of the fixed points and we find from the first equation (16.58) the azimuthal positions \(\psi_{1}=\psi_{\text{f}}\) of the fixed points from

\[\sin\left(m_{r}\psi_{1\text{f}}\right)=0 \tag{16.59}\]

or

\[m_{r}\psi_{1\text{f}\,k}=k\pi, \tag{16.60}\]

where \(k\) is an integer number in the range \(0<k<2m_{r}\). From the second equation (16.58) we get an expression for the radial location of the fixed points \(J_{\text{f}\,k}\)

\[\Delta v_{t}+2c_{40}q_{40}J_{\text{f}\,k}+\tfrac{n}{2}2c_{nm_{r}}\overline{q}_ {nr}J_{\text{f}\,k}^{n/2-1}\cos(\pi k)=0. \tag{16.61}\]

There are in principle \(2m_{r}\) separate fixed points in each resonance diagram. Closer inspections shows that alternately every second fixed point is a stable fixed point or an unstable fixed point, respectively. The unstable fixed points coincide with the crossing points of separatrices and exist even in the absence of octupole terms.

Figure 16.6: Common features of resonance patterns

Stable fixed points define the center of stable islands and, except for the primary stable fixed point at the origin of the phase diagram, exist only in the presence of a tune spread caused by octupole like terms \(c_{n0}\,q_{n0}\,J^{n/2}\) in (16.43), which contribute to beam stability. Trajectories that were unstable without the octupole term become closed trajectories within an island area centered at stable fixed points. This island structure is characteristic for resonances since the degree of symmetry is equal to the order of the resonance (see Fig. 16.7).

#### General Stop-Band Width

From the discussion of the half-integer resonance, it became apparent that certain conditions must be met to obtain stability for particle motion. Specifically we expect instability in the vicinity of resonances and we will try to determine quantitatively the area of instability or stop-band width for general resonances. Similar to (16.53) we look for stable solutions from

\[R\,\,\Delta+R^{n/2}\cos n\psi_{1}=\,\Delta\pm 1, \tag{16.62}\]

which describes a particle starting with an amplitude \(R=1\). Equation (16.62) must be true along all points of the trajectory and for reasons of symmetry the particle oscillation amplitude approaches again the starting amplitude for \(\psi_{1}=0\) as \(\psi_{1}\to 2\pi/n\). Solving for \(\Delta\) we get real solutions for \(R\) only if

\[\Delta^{+}\geq-\,\frac{R^{n/2}-1}{R-1}\Longrightarrow-\tfrac{1}{2}n\qquad \text{for}\qquad R\approx 1, \tag{16.63}\]where the index \({}^{+}\) indicates the sign to be used on the r.h.s. of (16.62). Similarly, following a particle starting with \(R=1\) at \(\psi_{1}=\pi/n\) to \(\psi_{1}=3\pi/n\) we get the condition

\[\Delta^{-}\leq\tfrac{1}{2}n. \tag{16.64}\]

The total \(n\)th-order stop-band width is therefore with (16.49)

\[\Delta v_{\text{stop}}^{(n)}=n\left|c_{mn_{r}}\overline{q}_{nr}\right|J_{0}^{n /2-1} \tag{16.65}\]

indicating that stable particle motion is possible only for tunes outside this stop-band. The stop-band width of nonlinear resonances (\(n>2\)) is strongly amplitude dependent and special effort must be exercised to minimize higher-order perturbations. Where higher-order magnetic fields cannot be eliminated it is prudent to minimize the value of the betatron functions at those locations.

Where higher-order magnetic fields cannot be eliminated it is prudent to minimize the value of the betatron functions at those locations. Such a case occurs, for example, in colliding-beam storage rings, where the strongly nonlinear field of one beam perturbs the trajectories of particles in the other beam. This effect is well known as the beam-beam effect.

Through a series of canonical transformations and redefinitions of parameters we seem to have parted significantly from convenient laboratory variables and parameters. We will therefore convert (16.65) back to variables we are accustomed to use. We set \(l=r\) and \(m_{r}=n\) where \(r\approx\tfrac{n}{N}v_{0}\) and tacitly ignored lower-order resonances \(m_{r}<n\). From (16.32) we find the Fourier components

\[\overline{q}_{nr}=\frac{1}{2\pi}\int_{0}^{2\pi}\overline{q}_{n}(\varphi) \mathrm{e}^{\mathrm{i}rN\varphi}\mathrm{d}\varphi\;, \tag{16.66}\]

and from (16.33) we have \(c_{nn}=\tfrac{1}{2^{n}}\). The amplitude factor \(J_{0}^{n/2-1}\) is replaced by (8.95) which becomes with (5.54a), (5.54b) and \(\psi_{1}=0\)

\[J_{0}=\tfrac{1}{2}v_{0}w_{0}^{2}=\tfrac{1}{2}v_{0}\frac{x_{0}^{2}}{\beta}. \tag{16.67}\]

Finally, we recall the definition (16.29) \(q_{n}(\varphi)=-\tfrac{1}{n}\overline{q}_{n}\left(\varphi\right)\left(\tfrac {v_{0}}{2}\right)^{-n/2}\)and get for the \(n\)th-order stop-band width

\[\Delta v_{\text{stop}}^{(n)}=\frac{w_{0}^{n-2}}{2^{n-1}\pi v_{0}}\left|\int_{ 0}^{2\pi}\overline{q}_{n}(\varphi)\mathrm{e}^{\mathrm{i}rN\varphi}\mathrm{d }\varphi\right|\;, \tag{16.68}\]

where \(\overline{q}_{n}\) is the \(n\)th-order perturbation from Table 16.1. This result is general and includes our earlier finding for the half-integer resonance. For resonances of order \(n>2\) the stop-band width increases with amplitude limiting the stability of particle beams to the vicinity of the axis (Fig. 16.8). The introduction of sufficiently strong octupole terms can lead to a stabilization of resonances and we found, for example, that the quarter resonance is completely stabilized if \(\Omega\geq 1\). For resonances of order \(n>4\), however, the term \(R^{n/2}\cos n\psi_{1}\) becomes dominant for large values of the amplitude and resonance therefore cannot be avoided.

Figure 16.9 shows, for example, a stable area for small amplitudes at the fifth-order resonance, as we would expect, but at larger amplitudes the motion becomes unstable.

### Third-Order Resonance

The third-order resonance plays a special role in accelerator physics and we will therefore discuss this resonance in more detail. The special role is generated by the need to use sextupoles for chromaticity correction. While such magnets are beneficial in one respect, they may introduce third-order resonances that need to be avoided or at least kept under control. Sometimes the properties of a third-order

Figure 16.8: Stop-band width as a function of the amplitude \(J_{0}\) for resonances of order \(n=2\), \(3\), \(4\), \(5\) and detuning parameter \(\Omega=0\)

resonance are also used constructively to eject particles at the end of a synchrotron acceleration cycle slowly over many turns.

In the absence of octupole fields the Hamiltonian for the third-order resonance is from (16.46) for \(n=3\) and \(q_{40}=0\)

\[H_{1} = \Delta v_{1/3}J_{1}+\overline{q}_{3r}J_{1}^{3/2}\text{cos}\,3\psi_{ 1}. \tag{16.69}\]

We expand \(\text{cos}\,3\psi_{1}\!=\!\text{cos}^{3}\,\psi_{1}-3\,\text{cos}\,\psi_{1}\, \text{sin}^{2}\,\psi_{1}\) and return to normalized coordinates

\[w=\sqrt{\frac{2J_{1}}{v_{0}}}\,\text{cos}\,\psi_{1},\quad\text{and}\quad\dot {w}=\sqrt{2v_{0}J_{1}}\,\text{sin}\,\psi_{1}. \tag{16.70}\]

In these coordinates the Hamiltonian reveals the boundaries of the stable region from the unstable resonant region. Introducing the normalized coordinates into (16.69), we get the Hamiltonian

\[H_{1} = \Delta v_{1/3}\frac{v_{0}}{2}\left(w^{2}+\frac{\dot{w}^{2}}{v_{0} ^{2}}\right)+\overline{q}_{3r}\,\frac{v_{0}^{3/2}}{2^{3/2}}\left(w^{3}-3w \frac{\dot{w}^{2}}{v_{0}^{2}}\right)\,. \tag{16.71}\]

Dividing by \(\overline{q}_{3r}\left(\frac{v_{0}}{2}\right)^{3/2}\) and subtracting a constant term \(\frac{1}{2}W_{0}^{3}\), where

\[W_{0} = \frac{4}{3}\frac{\Delta v_{1/3}}{\overline{q}_{3r}\sqrt{2v_{0}}}, \tag{16.72}\]

the Hamiltonian assumes a convenient form to exhibit the boundaries between the stable and unstable area

\[\tilde{H}_{1} = \tfrac{3}{2}W_{0}\left(w^{2}+\frac{\dot{w}^{2}}{v_{0}^{2}} \right)+\left(w^{3}-3w\frac{\dot{w}^{2}}{v_{0}^{2}}\right)-\tfrac{1}{2}W_{0} ^{3} \tag{16.73}\] \[= \left(w-\tfrac{1}{2}W_{0}\right)\left(w-\sqrt{3}\frac{\dot{w}}{v _{0}}+W_{0}\right)\left(w+\sqrt{3}\frac{\dot{w}}{v_{0}}+W_{0}\right).\]

This Hamiltonian has three linear solutions for \(\tilde{H}_{1}=0\) defining the separatrices. The resonance plot for (16.73) is shown in Fig. 16.10 where we have assumed that \(W_{0}\) is positive. For a given distribution of the sextupoles \(\overline{q}_{3r}\) the resonance pattern rotates by \(180^{\circ}\) while moving the tune from one side of the resonance to the other. Clearly, there is a stable central part bounded by separatrices. The area of the central part depends on the strength and distribution of the sextupole fields summarized by \(\overline{q}_{3r}\) and the distance \(\Delta v_{1/3}\) of the tune from the third-order resonance.

The higher-order field perturbation \(\overline{q}_{3r}\) depends on the distribution of the sextupoles around the circular accelerator. In the horizontal plane

\[\overline{q}_{3}(\varphi)=-v_{0}^{2}\beta^{5/2}\frac{1}{2}m\,. \tag{16.74}\]

or with (16.29)

\[q_{3}(\varphi)=\tfrac{1}{3}\sqrt{2v_{0}}\beta^{5/2}m\,. \tag{16.75}\]

The Fourier components of this perturbation are given by

\[q_{3r}=\frac{1}{2\pi}\int_{0}^{2\pi}q_{3}(\varphi)\mathrm{e}^{\mathrm{i}rN \varphi}\mathrm{d}\varphi \tag{16.76}\]

and the perturbation term becomes finally with \(m_{r}=3\) and \(c_{33}=\frac{1}{8}\) from (16.33)

\[\overline{q}_{3r}=\frac{\sqrt{2v_{0}}}{24\pi}\int_{0}^{2\pi}\beta^{5/2}m \mathrm{e}^{\mathrm{i}rN\varphi}\mathrm{d}\varphi \tag{16.77}\]

where \(\varphi=\int_{0}^{z}\frac{\mathrm{d}\xi}{v_{0}\beta}\), \(m=m(\varphi)\) is the sextupole distribution and \(\beta=\beta(\varphi)\) the horizontal betatron function. From this expression, it becomes clear that the perturbation and with it the stable area in phase space depends greatly on the distribution of the sextupoles around the ring. Minimizing the \(r\)th Fourier component obviously benefits beam stability.

Figure 16.10: Third-order resonance

#### Particle Motion in Phase Space

It is interesting to study particle motion close to a resonance in some more detail by deriving the equations of motion from the Hamiltonian (16.69). The phase variation is

\[\frac{\partial H_{1}}{\partial J_{1}}=\frac{\partial\psi_{1}}{\partial\varphi}= \Delta v_{1/3}+\tfrac{3}{2}\overline{q}_{3r}J_{1}^{1/2}\cos 3\psi_{1}\,. \tag{16.78}\]

Now, we follow a particle as it orbits the ring and observe its coordinates every time it passes by the point with phase \(\varphi_{0}\) or \(\psi_{0}\), which we assume for convenience to be zero. Actually, we observe the particle only every third turn, since we are not interested in the rotation of the resonance pattern in phase space by \(120^{\circ}\) every turn.

For small amplitudes the first term is dominant and we note that the particles move in phase space clockwise or counter clockwise depending on \(\Delta v_{1/3}\) being negative or positive, respectively. The motion becomes more complicated in the vicinity and outside the separatrices, where the second term is dominant. For a particle starting at \(\psi_{1}=0\) the phase \(\psi_{1}\)increases or decreases from turn to turn and asymptotically approaches \(\psi_{1}=\pm 30^{\circ}\) depending on the perturbation \(\overline{q}_{3r}\) being positive or negative, respectively. The particles therefore move clockwise or counter clockwise and the direction of this motion is reversed, whenever we move into an adjacent area separated by separatrices because the trigonometric term has changed sign.

To determine exactly the position of a particle after \(3q\) turns we have with \(\psi(q)=3q\cdot 2\pi\,v_{0}\)

\[\psi_{1}(q)=2\pi\,\left(3v_{0}-rN\right)q \tag{16.79}\]

With this phase expression we derive the associated amplitude \(J_{1q}\) from the Hamiltonian (16.69) and may plot the particle positions for successive triple turns \(3q=0,3,6,9,\ldots\) in a figure similar to Fig. 16.10. The change in the oscillation amplitude is from the second Hamiltonian equation of motion

\[\frac{\partial H_{1}}{\partial\psi_{1}}=-\frac{\partial J_{1}}{\partial\varphi }=-3\overline{q}_{3r}J_{1}^{3/2}\sin 3\psi_{1} \tag{16.80}\]

and is very small in the vicinity of \(\psi_{1}\approx 0\) or even multiples of \(30^{\circ}\) (for \(w>\) separatrix in Fig. 16.10). For \(\psi_{1}\) being equal to odd multiples of \(30^{\circ}\), on the other hand, the oscillation amplitude changes rapidly as shown in Fig. 16.10 on the left side beyond the crossing point of the separatrixes or beyond the unstable point.

## Problems

### 16.1 (S)

Consider a simple optimized FODO lattice forming a circular ring. Calculate the natural chromaticity (ignore focusing in bending magnets) and correct the chromaticities to zero by placing thin sextupoles in the center of the quadrupoles. Calculate and plot the horizontal third-order stop-band width as a function of the horizontal tune.

### 16.2 (S)

Show that in (16.33) the coefficients \(c_{n0}\) are non-zero only for even values of \(n\).

### 16.3 (S)

Show that in (16.33) the coefficients \(c_{nn}=\frac{1}{2^{n}}\).

Plot a resonance diagram up to fourth order for the PEP lattice with tunes \(v_{x}=21.28\) and \(v_{y}=18.16\) and a super-periodicity of \(N=6\) or any other circular accelerator lattice with multiple super-periodicity. Choose the parameters of the diagram such that a resonance environment for the above tunes of at least \(\pm 3\) (\(\pm\) half the number of superperiods) integers is covered.

Choose numerical values for parameters of a single multipole in the Hamiltonian (16.45) and plot a resonance diagram \(\mathcal{H}\left(J,\psi\right)=\)const. Determine the stability limit for your choice of parameters. What would the tolerance on the multipole field perturbation be if you require a stability for an emittance as large as \(\epsilon=100\,\)mmrad?

Take the lattice of Problem 16.1 and adjust its tune to the third-order resonance so that the unstable fixed point on the symmetry axis are \(5\,\)cm from the beam center. Determine the equations for the separatrices. Choose a point \(P\) just outside the stable area and close to the crossing of two separatrices along the symmetry axis. Where in the diagram would a particle starting at \(P\) be after 3, 6, and 9 turns? At what amplitude could you place a \(5\,\)mm thin septum magnet to eject the beam from the accelerator?

## References

* [1] A. Schoch, Theory of linear and nonlinear perturbations of betatron oscillations in alternating gradient synchrotrons. Technical Report CERN 57-23, CERN, CERN, Geneva (1957)
* [2] G. Guignard, The general theory of all sum and difference resonances in a three dimensional magnetic field in a synchrotron. Technical Report CERN 76-06, CERN, CERN, Geneva (1976)
* [3] S.Ohnuma, Quarter integer resonance by sextupoles. Technical Report TM-448, FERMI Lab, Batavia, IL (1973)

