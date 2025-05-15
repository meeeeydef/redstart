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


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
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
    return FFMpegWriter, FuncAnimation, mpl, np, plt, scipy, tqdm


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


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


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


@app.cell(hide_code=True)
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
    mo.show_code(mo.video(src=_filename))
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


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
    return


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
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
    """
    )
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


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


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
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
    """
    )
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


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
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

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
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
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
     System Equations

    $M\ddot{x} = -f \sin(\theta + \phi)$
    $M\ddot{y} = +f \cos(\theta + \phi) - Mg$
    $J\ddot{\theta} = -\ell(\sin \phi)f$

     Assumptions
    - $|\theta| < \pi/2$
    - $|\phi| < \pi/2$
    - $f > 0$

     Finding Equilibrium Points

    At equilibrium, all accelerations are zero:
    $\ddot{x} = \ddot{y} = \ddot{\theta} = 0$

     From the equation $J\ddot{\theta} = -\ell(\sin \phi)f$

    If $\ddot{\theta} = 0$, then:
    $-\ell(\sin \phi)f = 0$

    Given that $f > 0$ and $\ell \neq 0$, we deduce:
    $\sin \phi = 0$

    Since $|\phi| < \pi/2$, the only solution is:
    $\phi = 0$

     From the equation $M\ddot{x} = -f \sin(\theta + \phi)$

    If $\ddot{x} = 0$, then:
    $-f \sin(\theta + \phi) = 0$

    Substituting $\phi = 0$:
    $-f \sin(\theta) = 0$

    Given that $f > 0$, we deduce:
    $\sin \theta = 0$

    Since $|\theta| < \pi/2$, the only solution is:
    $\theta = 0$

     From the equation $M\ddot{y} = +f \cos(\theta + \phi) - Mg$

    If $\ddot{y} = 0$, then:
    $f \cos(\theta + \phi) - Mg = 0$

    Substituting $\theta = 0$ and $\phi = 0$:
    $f \cos(0) - Mg = 0$
    $f - Mg = 0$

    We deduce:
    $f = Mg$

     Regarding positions $x$ and $y$

    The differential equations of the system determine accelerations, but not directly positions.

    At equilibrium, we have:
    - $\ddot{x} = 0$ → $x$ has constant velocity
    - $\ddot{y} = 0$ → $y$ has constant velocity

    For true equilibrium, velocities must also be zero:
    - $\dot{x} = 0$ → $x$ is constant
    - $\dot{y} = 0$ → $y$ is constant

    However, the specific values of $x$ and $y$ can be arbitrary.

     Conclusion:

    The system has an equilibrium point characterized by:
    - $\phi = 0$ (angle $\phi$ is zero)
    - $\theta = 0$ (angle $\theta$ is zero)
    - $f = Mg$ (applied force exactly balances weight)
    - $x = \text{constant}$ (any horizontal position)
    - $y = \text{constant}$ (any vertical position)

    Therefore, there are infinitely many equilibrium positions in space (for different values of $x$ and $y$), but all with the same angles ($\phi = 0$ and $\theta = 0$) and the same applied force ($f = Mg$).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As defined:
    - *Equilibrium state*: $(x = x_0, y = y_0, \theta = 0)$
    - *Equilibrium input*: $(f = Mg, \phi = 0)$

    Error variables:
    - $\Delta x = x - x_0$
    - $\Delta y = y - y_0$
    - $\Delta \theta = \theta$ (since $\theta = 0$ at equilibrium)
    - $\Delta f = f - Mg$
    - $\Delta \phi = \phi$ (since $\phi = 0$ at equilibrium)

    Original nonlinear equations:
    1. $M\ddot{x} = -f \sin(\theta + \phi)$
    2. $M\ddot{y} = +f \cos(\theta + \phi) - Mg$
    3. $J\ddot{\theta} = -\ell(\sin\phi)f$


    For small angles $\theta$ and $\phi$:
    - $\sin(\theta + \phi) \approx \theta + \phi$
    - $\cos(\theta + \phi) \approx 1 - \frac{(\theta + \phi)^2}{2} \approx 1$ (ignoring second-order terms)
    - $\sin(\phi) \approx \phi$

    Substituting $f = Mg + \Delta f$ and the small-angle approximations:

    #### Horizontal Dynamics (Equation 1):
    $M\ddot{x} = -f \sin(\theta + \phi)$
    $M\Delta\ddot{x} = -(Mg + \Delta f)(\Delta\theta + \Delta\phi)$
    $M\Delta\ddot{x} \approx -Mg(\Delta\theta + \Delta\phi) - \Delta f(\Delta\theta + \Delta\phi)$

    Ignoring higher-order terms (product of small quantities):
    $M\Delta\ddot{x} \approx -Mg(\Delta\theta + \Delta\phi)$

    #### Vertical Dynamics (Equation 2):
    $M\ddot{y} = +f \cos(\theta + \phi) - Mg$
    $M\Delta\ddot{y} = +(Mg + \Delta f) \cdot 1 - Mg$ (approximating $\cos(\theta + \phi) \approx 1$)
    $M\Delta\ddot{y} = \Delta f$

    #### Angular Dynamics (Equation 3):
    $J\ddot{\theta} = -\ell(\sin\phi)f$
    $J\Delta\ddot{\theta} = -\ell(\Delta\phi)(Mg + \Delta f)$
    $J\Delta\ddot{\theta} \approx -\ell Mg \Delta\phi - \ell \Delta\phi \Delta f$

    Ignoring higher-order terms:
    $J\Delta\ddot{\theta} \approx -\ell Mg \Delta\phi$

    The linearized system can be written as:

    1. $\Delta\ddot{x} = -g(\Delta\theta + \Delta\phi)$
 
    2. $\Delta\ddot{y} =\frac{\Delta f} {M}$
 
    3. $\Delta\ddot{\theta} = -\frac {3g} {\ell} \Delta\phi$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We express the linearized equations in the form:

    \[
    \dot{X} = A X + B U
    \]

    Let the state vector be:

    \[
    X = \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta y \\
    \Delta \dot{y} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    \in \mathbb{R}^6
    \]

    And the input vector be:

    \[
    U = \begin{bmatrix}
    \Delta f \\
    \Delta \phi
    \end{bmatrix}
    \in \mathbb{R}^2
    \]


    We rewrite the second-order equations as a first-order system:

    \[
    \begin{aligned}
    \Delta x &= \Delta \dot{x} \\
    \Delta \dot{x} &= g(\Delta \theta + \Delta \phi) \\
    \Delta y &= \Delta \dot{y} \\
    \Delta \dot{y} &= \frac{1}{M} \Delta f - g (\Delta \theta + \Delta \phi) \\
    {\Delta \theta} &= \Delta \dot{\theta} \\
    \Delta \dot{\theta} &= -\frac{3g}{\ell} \Delta \phi
    \end{aligned}
    \]

    We identify the matrices A and B such that:

    \[
    \dot{X} = A X + B U
    \]

    This linearized model is used for control and stability analysis near the hovering equilibrium.
    """
    )
    return


@app.cell
def _(M, g, l, np):

    B = np.array([
        [0,     0],
        [0,     -g],
        [0,     0],
        [1/M,   0],
        [0,     0],
        [0, -3*g/ℓ]
    ])
    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, - g, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""


    We previously linearized the system around the *hovering equilibrium*, with:

    - \( f = Mg \), \( \phi = 0 \)
    - \( \theta = 0 \), \( \dot{\theta} = 0 \)
    - and all velocities zero.

    The *linearized matrix* \( A \) is:

    \[
    A = \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    \]

    This matrix has *eigenvalues on the imaginary axis or zero. Since there's no damping or negative feedback in the natural dynamics, the system is **marginally stable, but **not asymptotically stable*.

    >  *Conclusion: The equilibrium is **not asymptotically stable*. It requires external control to stabilize.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We analyze the *controllability* of the linearized system by checking the *controllability matrix*:

    \[
    \mathcal{C} = [B, AB, A^2B, A^3B, A^4B, A^5B] \in \mathbb{R}^{6 \times 12}
    \]

    If \( \text{rank}(\mathcal{C}) = 6 \), the system is *controllable*.

    Although computing this directly is possible, we can reason structurally:

    - The inputs \( \Delta f \) and \( \Delta \phi \) influence all subsystems:
      - \( \Delta f \): affects vertical dynamics
      - \( \Delta \phi \): affects both angular and horizontal dynamics
    - The matrix \( A \) couples these effects over time

    Hence, all states become reachable through input influence.

    > *Conclusion: The linearized model is **controllable*.
    """
    )
    return


@app.cell
def _(A, B, np):
    # Construction de la matrice de contrôlabilité
    C = B
    for i in range(1, 6):
        C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))

    # Calcul du rang
    rang_C = np.linalg.matrix_rank(C)

    # Affichage
    print("Matrice de contrôlabilité C :")
    print(C)
    print("\nRang de C :", rang_C)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We focus on the lateral dynamics and tilt, ignoring vertical motion \( y \) and \( \dot{y} \), and fix the thrust amplitude \( f = Mg \). The only control input is now the tilt angle of the force, \( \phi \).


    We define the reduced state vector and input

    \[
    X_{red} = \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}, \quad
    U_{red} = \Delta \phi
    \]


    We write the reduced linearized equations

    From the full linearized model:

    \[
    \begin{cases}
    \dot{\Delta x} = \Delta \dot{x} \\
    \dot{\Delta \dot{x}} = g(\Delta \theta + \Delta \phi) \\
    \dot{\Delta \theta} = \Delta \dot{\theta} \\
    \dot{\Delta \dot{\theta}} = -\frac{3g}{\ell} \Delta \phi
    \end{cases}
    \]

    Expressed in matrix form:

    \[
    \dot{X}{red} = A{red} X_{red} + B_{red} U_{red}
    \]

    where

    \[
    A_{red} =
    \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}
    \quad , \quad
    B_{red} =
    \begin{bmatrix}
    0 \\
    -g \\
    0 \\
    -\frac{3g}{\ell}
    \end{bmatrix}
    \]

    """
    )
    return


@app.cell
def _(np):
    from numpy.linalg import matrix_rank

    def reduced_matrices_and_controllability(g, l):
        A_red = np.array([
            [0, 1, 0, 0],
            [0, 0, -g, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])

        B_red = np.array([
            [0],
            [-g],
            [0],
            [-3*g/l]
        ])
     # Controllability matrix
        controllability_matrix = np.hstack([
            B_red,
            A_red @ B_red,
            A_red @ A_red @ B_red,
            A_red @ A_red @ A_red @ B_red
        ])

        rank = matrix_rank(controllability_matrix)
        controllable = (rank == A_red.shape[0])

        return A_red, B_red, controllability_matrix, controllable


    A_red, B_red, C_matrix, is_controllable = reduced_matrices_and_controllability(1.0, 1.0)
    print(f"Controllability matrix rank: {C_matrix.shape[0]} expected, got {np.linalg.matrix_rank(C_matrix)}")
    print(f"Controllability? {is_controllable}")
    return A_red, B_red


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Since \( \phi(t) = 0 \), the input vector is zero:
    U(t) = 0

    The system reduces to homogeneous equations:

    \[
    \dot{X} = A X
    \]

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We begin with the equation of motion in the horizontal direction:

    \[
    \ddot{x} = \frac{f \sin (\theta + \phi)}{M} 
    \]


    If we assume small angles for \( \theta \) 

    Then:

    \[
    \sin(\theta) \approx   \theta
    \Rightarrow \ddot{x} \approx g  \theta  
    \]

    So even after linearizing, the horizontal acceleration is linear in \( \theta \).  


    This means vertical motion is governed only by gravity (free fall):


    $x(t) = x(0) - \frac{1}{2}g \theta t^2$

    Since $\theta = \frac{\pi}{4}$

    $x(t)=-g \frac{\pi}{8} t^2$
 

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - θ(t) stays constant (no torque, no change in angle).  
    - x(t) is a parabola due to free-fall acceleration under gravity.

    Interpretation
    - With no input (ϕ(t) = 0), the system undergoes pure vertical fall.  
    - Tilt remains at 45° due to zero angular acceleration.  

    """
    )
    return


@app.cell
def _(g, np, plt):
    theta0 = np.pi/4  
    t = np.linspace(0, 2, 500)
    theta = np.full_like(t, theta0)


    x = - (g * np.pi / 8) * t**2
    plt.figure(figsize=(12, 6))
    # θ(t)
    plt.subplot(1, 2, 1)
    plt.plot(t, theta, 'r', label=r'$\theta(t)$')
    plt.xlabel('Time (s)')
    plt.ylabel('θ (rad)')
    plt.title('θ(t)')
    plt.grid()
    plt.legend()


    # x(t)
    plt.subplot(1, 2, 2)
    plt.plot(t, x, 'g', label=r'$x(t)$')
    plt.xlabel('Time (s)')
    plt.ylabel('x (m)')
    plt.title('Side motion x(t)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We want to design a state feedback controller for the reduced system

    \[
    \dot{X}{red} = A{red} X_{red} + B_{red} \Delta \phi,
    \]

    with control law

    \[
    \Delta \phi(t) = - k_3 \Delta \theta - k_4 \Delta \dot{\theta},
    \]

    $$
    \dot{X}_{\text{red}} =
    \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g(1-k_3) & gk_4 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & \frac{3g}{\ell}k_3 & \frac{3g}{\ell}k_4
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    $$

    We note the matrix with dimension 4x4 as $\tilde{A}$

    det($\lambda$I-$\tilde{A}$) = $\lambda^2$.($\lambda^2$-3$\lambda$$k_4$ -3$k_3$)

    We have double pole -1 and -2 

    ($\lambda$+2).($\lambda$+1) = $\lambda^2$ + 3$\lambda$ +2 = $\lambda^2$-3$\lambda$$k_4$ -3$k_3$

    We conclude that k_4 = -1  and  k_3 = -$\frac{2}{3}$
     Convergence Rate Analysis

    If we want $\Delta \theta(t)$ to converge to zero in approximately 20 seconds, then the eigenvalue $\lambda$ of $A - BK$ must satisfy:

    $$\mathrm{Re}(\lambda) \lesssim -\frac{3}{T_c} = -\frac{3}{20} = -0.15$$

    This coefficient $-0.15$ represents a lower bound on the exponential decay rate. The more negative the real part, the faster the convergence.
    """
    )
    return


@app.cell
def _(X0, g, l, np, plt, solve_ivp):

    k32= -2/3
    k42 = -1


    A_1 = np.array([
        [0, 1, 0, 0],
        [0, 0, -g*(1 - k32), g*k42],
        [0, 0, 0, 1],
        [0, 0, (3*g/l)*k32, (3*g/l)*k42]
    ])


    def closed_loop_dynamics(t, X):
        return A_1 @ X

    X0_1 = np.array([0, 0, 0.1, 0])


    t_span1 = (0, 10)
    t_eval1 = np.linspace(t_span1[0], t_span1[1], 500)

    # Solve ODE
    sol2 = solve_ivp(closed_loop_dynamics, t_span1, X0, t_eval=t_eval1)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sol2.t, sol2.y[2], label='Delta theta (rad)')
    plt.plot(sol2.t, sol2.y[3], label='Delta theta dot (rad/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('States')
    plt.title('Closed-loop response with manual tuning k3 = -2/3, k4 = -1')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(A_red, B_red, np):
    from scipy.signal import place_poles


    desired_poles = np.array([-1, -2, -3, -4])

    place_obj = place_poles(A_red, B_red, desired_poles)
    K_pp = place_obj.gain_matrix

    print("State feedback gain K_pp:")
    print(K_pp)
    return (K_pp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(A_red, B_red, np):
    from scipy.linalg import solve_continuous_are
    Q = np.diag([1, 0.1, 100, 10])  
    R1 = np.array([[1]])
    P = solve_continuous_are(A_red, B_red, Q, R1)
    K_oc = np.linalg.inv(R1) @ B_red.T @ P
    print("K_oc =", K_oc)
    return (K_oc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


@app.cell
def _(A_red, B_red, K_oc, K_pp, np, plt):

    from scipy.integrate import solve_ivp

    K_manual = np.array([0, 0, -2/3, -1])
    A_cl_manual = A_red - B_red @ K_manual.reshape(1, -1)
    A_cl_pp = A_red - B_red @ K_pp.reshape(1, -1)
    A_cl_oc = A_red - B_red @ K_oc.reshape(1, -1)

    X0 = np.array([0, 0, 0.1, 0])

    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 500)

    def simulate_response(A_cl, X0, t_span, t_eval):
        def dyn(t, x):
            return A_cl @ x
        sol = solve_ivp(dyn, t_span, X0, t_eval=t_eval)
        return sol

    sol_manual = simulate_response(A_cl_manual, X0, t_span, t_eval)
    sol_pp = simulate_response(A_cl_pp, X0, t_span, t_eval)
    sol_oc = simulate_response(A_cl_oc, X0, t_span, t_eval)

    plt.figure(figsize=(10, 6))
    plt.plot(sol_manual.t, sol_manual.y[2], label='Manual tuning')
    plt.plot(sol_pp.t, sol_pp.y[2], label='Pole placement')
    plt.plot(sol_oc.t, sol_oc.y[2], label='Optimal (LQR)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle $\Delta \\theta$ (rad)')
    plt.title('Closed-loop response comparison')
    plt.legend()
    plt.grid()
    plt.show()
    return X0, solve_ivp


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
