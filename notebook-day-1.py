import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, np, plt, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    (mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell
def _():
    g=1
    M=1
    l=1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The reactor force has a magnitude \( f \) and makes an angle \( \phi \) with respect to the booster axis.  
    The booster axis makes an angle \( \theta \) with the vertical.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    f_x = -f \cdot \sin(\theta + \phi)
    $$

    $$
    f_y = f \cdot \cos(\theta + \phi)
    $$
    """
    )
    return


@app.cell
def _(np):
    def compute_reactor_force_components(f, theta, phi):
        fx = -f * np.sin(theta +phi)
        fy = f * np.cos(theta+phi)

        return fx, fy
    return (compute_reactor_force_components,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    These equations are obtained by applying Newton's second law to the system.  
    For the horizontal motion (\(x\)), the acceleration is proportional to the horizontal component of the reactor force.  
    This component depends on the angle of the booster (\( \theta \)) and the angle of the force relative to the booster axis (\( \phi \)).  
    For the vertical motion (\(y\)), the acceleration results from the difference between the vertical component of the reactor force and the gravitational force.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Forces

    #### Gravité :
    $$
    \vec{f}_g =
    \begin{bmatrix}
    0 \\
    - M g
    \end{bmatrix}
    $$

    #### Force du réacteur :
    $$
    \vec{f} =
    \begin{bmatrix}
    - f \cdot \sin(\theta + \phi) \\
    f \cdot \cos(\theta + \phi)
    \end{bmatrix}
    $$

    ---

    ### Somme des forces :

    $$
    f_x^{\text{total}} = -f \cdot \sin(\theta + \phi)
    $$

    $$
    f_y^{\text{total}} = f \cdot \cos(\theta + \phi) - M g
    $$

    ---

    ### Accélérations linéaires :

    $$
    \ddot{x} = \frac{f_x^{\text{total}}}{M} = -\frac{f}{M} \cdot \sin(\theta + \phi)
    $$

    $$
    \ddot{y} = \frac{f_y^{\text{total}}}{M} = \frac{f \cdot \cos(\theta + \phi)}{M} - g
    $$
    """
    )
    return


@app.cell
def _(M, compute_reactor_force_components, g, phi):
    def force_gravite():
        return 0, -M * g
    def equations_mouvement(t, y):
        x, x_dot, y, y_dot, theta = y

        # Forces
        fx_r, fy_r = compute_reactor_force_components(theta, phi)
        fx_g, fy_g = force_gravite()

        # Somme des forces
        fx_total = fx_r + fx_g
        fy_total = fy_r + fy_g

        # Accélérations linéaires
        x_ddot = fx_total / M
        y_ddot = fy_total / M


        return [x_dot, x_ddot, y_dot, y_ddot]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The booster is modeled as a uniform rod of length 2l, with total mass M, and rotation about its center""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    J = \frac{1}{3} M \ell^2
    $$
    """
    )
    return


@app.cell
def _(M, l):
    J = (1/3) * M * l**2
    print(f"Moment of inertia J = {J:.3f} kg·m²")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The torque \( \tau \) generated by the reactor force (applied at the base of the booster) about the center of mass is:

    \[
    \tau = -\ell f \sin(\phi)
    \]

    According to Newton's second law for rotation:

    \[
    J \ddot{\theta} = \tau = -\ell f \sin(\phi)
    \]

    We get:

    \[
    \ddot{\theta} = \frac{-\ell f \sin(\phi)}{J} = \frac{-3f \sin(\phi)}{M \ell}
    \]
    """
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell
def _(M, g, l, np):
    from scipy.integrate import solve_ivp

    def redstart_solve(t_span, y0, f_phi):
        J = (1/3) * M * l**2

        def ode(t, y):
            x, dx, y_pos, dy, theta, dtheta = y
            f, phi = f_phi(t, y)

            ddx = (-f / M) * np.sin(theta + phi)
            ddy = (f / M) * np.cos(theta + phi) - g
            ddtheta = (l * f * np.sin(phi)) / J

            return [dx, ddx, dy, ddy, dtheta, ddtheta]

        sol_ivp = solve_ivp(ode, t_span, y0, dense_output=True)

        def sol(t):
            return sol_ivp.sol(t)

        return sol
    return (redstart_solve,)


@app.cell
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Given that the force is applied along the booster axis and that \( \theta = 0 \), the vertical motion is governed by:

    \[
    \ddot{y} = \frac{f(t)}{M} - g
    \]

    The system is a 1D controlled motion problem.


    To satisfy the boundary conditions on position and velocity at both \( t = 0 \) and \( t = 5 \), we define a cubic polynomial trajectory:

    \[
    y(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3
    \]

    Using the initial conditions, we solve for the coefficients \( a_0, a_1, a_2, a_3 \), and then compute the second derivative:

    \[
    \ddot{y}(t) = 2a_2 + 6a_3 t
    \]

    We plug this into the force equation:

    \[
    f(t) = M \left( \ddot{y}(t) + g \right)
    \]
    """
    )
    return


@app.cell
def _(M, g, np, plt, redstart_solve):
    # Parameters
    t0, tf = 0, 5
    y_init = 10
    y_final = 1  # l = 1
    v0 = 0  # initial vertical velocity


    # Define cubic polynomial for y(t)
    T = tf
    A = np.array([
        [1, 0,     0,     0],
        [0, 1,     0,     0],
        [1, T,   T*2, T*3],
        [0, 1, 2*T,   3*T**2]
    ])
    b = np.array([y_init, v0, y_final, 0])
    a0, a1, a2, a3 = np.linalg.solve(A, b)

    # Define f(t) = M (d²y/dt² + g)
    def y_ddot(t):
        return 2*a2 + 6*a3*t

    def f_phi(t, y):
        f = M * (y_ddot(t) + g)
        return [f, 0]  # phi = 0 (aligned with booster)


    y0 = [0, 0, y_init, v0, 0, 0]

    sol = redstart_solve((t0, tf), y0, f_phi)

    # Plotting
    times = np.linspace(0, 5, 1000)          
    states = np.array([sol(t) for t in times])                      
    y_vals = states[:, 2]                    
    dy_vals = states[:, 3]                 
    f_vals = [f_phi(t, None)[0] for t in times]  
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(times, y_vals, label='y(t)')
    plt.plot(times, dy_vals, label="y'(t)")
    plt.title("Vertical Position and Speed")
    plt.xlabel("Time [s]")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times, f_vals, label='f(t)', color='orange')
    plt.title("Control Force")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell
def _(M, g, l, np, plt):
    from matplotlib.patches import Rectangle
    def draw_booster_scene(x, y, theta, f, phi, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.clear()
    

        # Compute flame length relative to f
        flame_length = (f / (M * g)) * l

        # Booster coordinates
        dx = (l / 2) * np.sin(theta)
        dy = (l / 2) * np.cos(theta)
        x1, y1 = x - dx, y - dy  # bottom
        x2, y2 = x + dx, y + dy  # top

        # Draw booster body
        ax.plot([x1, x2], [y1, y2], color='black', lw=4)

        # Compute flame direction and base position
        flame_angle = theta + phi
        fx = -flame_length * np.sin(flame_angle)
        fy = -flame_length * np.cos(flame_angle)

        # Draw flame (as a colored arrow or rectangle)
        ax.arrow(x1, y1, fx, fy, width=0.05, color='orange', length_includes_head=True)

        # Draw target zone
        ax.plot(0, 0, 'ro', label='Target landing zone')
        ax.plot([-0.5, 0.5], [0, 0], color='red', linestyle='--')

        # Visual setup
        ax.set_aspect('equal')
        ax.set_xlim(x - l, x + l)
        ax.set_ylim(0, y + l)
        ax.set_title("Booster Scene")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.legend()
        return
    return (draw_booster_scene,)


@app.cell
def _(draw_booster_scene, plt):
    draw_booster_scene(ax=None, x=0, y=10, theta=0, f=0, phi=0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell
def _(FuncAnimation, draw_booster_scene, mo, np, plt):
    from matplotlib import rc
    import io
    import base64
    from matplotlib.animation import PillowWriter, HTMLWriter
    def make_booster_animation():
        fig, ax = plt.subplots(figsize=(4, 8))
        duration = 5  # seconds
        fps = 10
        frames = duration * fps

        def animate(i):
            t = i / fps
            if t < 1.5:
                x, y, theta, f, phi = 0, 10, 0, 0, 0
            elif t < 3.0:
                x, y, theta, f, phi = 0, 10, 0, 9.81, 0
            else:
                x, y, theta, f, phi = 0, 10, 0, 9.81, np.pi/8
            draw_booster_scene(ax, x, y, theta, f, phi)

        anim = FuncAnimation(fig, animate, frames=frames, interval=1000 / fps)

    
        html = anim.to_jshtml()  
        plt.close(fig)  
        return mo.html(html)

    make_booster_animation()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
