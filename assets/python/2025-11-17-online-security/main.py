# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
a = -1
b = 1


def ec(x, y, a=-1, b=1):
    """Elliptic curve equation: y^2 = x^3 + a*x + b"""
    return pow(y, 2) - pow(x, 3) - x * a - b


def solve_ec_for_x(y, a=-1, b=1):
    coeffs = [1, 0, a, b - y**2]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    return real_roots


def solve_ec_for_y(x, a=-1, b=1):
    return np.sqrt(x**3 + a * x + b)


def animate(frame):
    ax.clear()
    state = states[frame]

    # Detect addition vs multiplication
    if (state["x1"] - state["x2"]) < 0.01 or state["i"] > 0:
        # multiplication
        x1_name = "P"
        x2_name = f"{'' if state['i'] < 1 else state['i'] + 1}P"
        xnew_name = f"{state['i'] + 2}P"
        label_shift = 0.3
    else:
        # addition
        x1_name = "P"
        x2_name = "Q"
        xnew_name = "P+Q"
        label_shift = 0.1

    # Plot the elliptic curve
    ax.contour(
        x.ravel(), y.ravel(), ec(x, y, a, b), [0], colors="#2C3E50", linewidths=2.5
    )

    # Only show tangent line if step is not 'points'
    if state["step"] != "points":
        ax.plot(
            state["x_tangent"],
            state["y_tangent"],
            "--",
            color="#3498DB",
            linewidth=2,
            alpha=0.8,
        )

    # Plot the current two points
    ax.plot(
        state["x1"],
        state["y1"],
        "o",
        color="#E74C3C",
        markersize=10,
        markeredgewidth=2,
        markeredgecolor="#C0392B",
        zorder=5,
    )
    ax.text(
        state["x1"],
        state["y1"] + label_shift,
        x1_name,
        fontsize=18,
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="right",
        color="#C0392B",
    )

    ax.plot(
        state["x2"],
        state["y2"],
        "o",
        color="#E74C3C",
        markersize=10,
        markeredgewidth=2,
        markeredgecolor="#C0392B",
        zorder=5,
    )
    ax.text(
        state["x2"],
        state["y2"] + label_shift,
        x2_name,
        fontsize=18,
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="right",
        color="#C0392B",
    )

    # Show intersection and reflection based on step
    if state["step"] in ["intersection", "reflection"]:
        if state["step"] == "intersection":  # unreflected
            ax.plot(
                state["x_new"],
                state["y_new"],
                "o",
                color="#2ECC71",
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="#27AE60",
                zorder=5,
            )
            ax.text(
                state["x_new"],
                state["y_new"] + label_shift,
                f"-{xnew_name}",
                fontsize=18,
                fontweight="bold",
                verticalalignment="bottom",
                horizontalalignment="right",
                color="#27AE60",
            )
        else:  # reflection
            ax.plot(
                state["x_new"],
                -state["y_new"],
                "o",
                color="#2ECC71",
                markersize=5,
                markeredgewidth=1,
                markeredgecolor="#27AE60",
                zorder=5,
            )
            ax.plot(
                state["x_new"],
                state["y_new"],
                "o",
                color="#2ECC71",
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="#27AE60",
                zorder=5,
            )
            ax.text(
                state["x_new"],
                state["y_new"] + label_shift,
                f"{xnew_name}",
                fontsize=18,
                fontweight="bold",
                verticalalignment="bottom",
                horizontalalignment="right",
                color="#27AE60",
            )

            ax.vlines(
                state["x_new"],
                -state["y_new"],
                state["y_new"],
                linestyles="--",
                color="#3498DB",
                linewidth=2,
                alpha=0.8,
            )

    # Set dynamic limits and aspect ratio
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Remove grid and improve axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=15)

    # Set background color
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("white")


# %% [markdown]
# # EC Addition

# %%
x1 = 1
y1 = solve_ec_for_y(x1, a, b)
print(f"Point on the curve: ({x1}, {y1})")
x2 = 0
y2 = solve_ec_for_y(x2, a, b)
print(f"Point on the curve: ({x2}, {y2})")

# Store all states for animation (4 frames per iteration)
states = []
all_x = [x1, x2]  # Track all x coordinates for dynamic limits
all_y = [y1, y2]  # Track all y coordinates for dynamic limits

slope = (y2 - y1) / (x2 - x1)
intercept = y2 - slope * x2

# as per Vieta's formulas
x_new = slope**2 - x1 - x2
y_new = slope * x_new + intercept

x_tangent = np.linspace(min(x1, x2, x_new), max(x1, x2, x_new), 100)
y_tangent = slope * x_tangent + intercept

# Track coordinates
all_x.extend([x_new])
all_y.extend([y_new, -y_new])

# Frame 1: Show just the two points (no tangent)
states.append({"step": "points", "i": 0, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

# Frame 2: Show tangent line
states.append(
    {
        "step": "tangent",
        "i": 0,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x_tangent": x_tangent,
        "y_tangent": y_tangent,
    }
)

# Frame 3: Show intersection point (before reflection)
states.append(
    {
        "step": "intersection",
        "i": 0,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x_new": x_new,
        "y_new": y_new,
        "x_tangent": x_tangent,
        "y_tangent": y_tangent,
    }
)

# Frame 4: Show reflection
states.append(
    {
        "step": "reflection",
        "i": 0,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x_new": x_new,
        "y_new": -y_new,
        "x_tangent": x_tangent,
        "y_tangent": y_tangent,
    }
)

x2, y2 = x_new, -y_new

# Calculate dynamic limits with minimal padding
x_min, x_max = -2, 3
y_min, y_max = -2, 2
x_padding = (x_max - x_min) * 0.08
y_padding = (y_max - y_min) * 0.08

xlim = [x_min - x_padding, x_max + x_padding]
ylim = [y_min - y_padding, y_max + y_padding]

print(f"Using x limits: {xlim}")
print(f"Using y limits: {ylim}")

# Create animation
fig, ax = plt.subplots(figsize=(10, 6))
y, x = np.ogrid[-12:12:120j, -12:12:100j]
anim = FuncAnimation(fig, animate, frames=len(states), interval=1000, repeat=True)
anim.save(
    "img/elliptic_curve_addition.gif",
    writer="pillow",
    fps=1,
    dpi=300,
    savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0.1},
)
plt.close()
print("Animation saved as 'img/elliptic_curve_addition.gif'")

# %% [markdown]
# # EC Multiplication

# %%
x1 = 1
y1 = solve_ec_for_y(x1, a, b)
print(f"Point on the curve: ({x1}, {y1})")
x2 = x1 + 0.0001
y2 = solve_ec_for_y(x2, a, b)
print(f"Point on the curve: ({x2}, {y2})")

# Store all states for animation (4 frames per iteration)
states = []
all_x = [x1, x2]  # Track all x coordinates for dynamic limits
all_y = [y1, y2]  # Track all y coordinates for dynamic limits

for i in range(0, 5):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y2 - slope * x2

    # as per Vieta's formulas
    x_new = slope**2 - x1 - x2
    y_new = slope * x_new + intercept

    x_tangent = np.linspace(min(x1, x2, x_new), max(x1, x2, x_new), 100)
    y_tangent = slope * x_tangent + intercept

    # Track coordinates
    all_x.extend([x_new])
    all_y.extend([y_new, -y_new])

    # Frame 1: Show just the two points (no tangent)
    states.append({"step": "points", "i": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    # Frame 2: Show tangent line
    states.append(
        {
            "step": "tangent",
            "i": i,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x_tangent": x_tangent,
            "y_tangent": y_tangent,
        }
    )

    # Frame 3: Show intersection point (before reflection)
    states.append(
        {
            "step": "intersection",
            "i": i,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x_new": x_new,
            "y_new": y_new,
            "x_tangent": x_tangent,
            "y_tangent": y_tangent,
        }
    )

    # Frame 4: Show reflection
    states.append(
        {
            "step": "reflection",
            "i": i,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x_new": x_new,
            "y_new": -y_new,
            "x_tangent": x_tangent,
            "y_tangent": y_tangent,
        }
    )

    x2, y2 = x_new, -y_new

# Calculate dynamic limits with minimal padding
x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
x_padding = (x_max - x_min) * 0.08
y_padding = (y_max - y_min) * 0.08

xlim = [x_min - x_padding, x_max + x_padding]
ylim = [y_min - y_padding, y_max + y_padding]

print(f"Using x limits: {xlim}")
print(f"Using y limits: {ylim}")

# Create animation
fig, ax = plt.subplots(figsize=(10, 6))
y, x = np.ogrid[-12:12:120j, -12:12:100j]
anim = FuncAnimation(fig, animate, frames=len(states), interval=1000, repeat=True)
anim.save(
    "img/elliptic_curve_multiplication.gif",
    writer="pillow",
    fps=1,
    dpi=300,
    savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0.1},
)
plt.close()
print("Animation saved as 'img/elliptic_curve_multiplication.gif'")


# %%
def ec_mod(x, y, a=-1, b=1, p=47):
    """Check if point (x, y) is on the elliptic curve over finite field F_p
    Equation: y^2 ≡ x^3 + ax + b (mod p)
    """
    left = (y * y) % p
    right = (x**3 + a * x + b) % p
    return left == right


def find_all_ec_points(a=-1, b=1, p=47):
    """Find all points on elliptic curve over finite field F_p"""
    points = []

    # Check all possible (x, y) pairs
    for x in range(p):
        for y in range(p):
            if ec_mod(x, y, a, b, p):
                points.append((x, y))

    return points


# Find all points on the curve
p = 47
a = -1
b = 1
points = find_all_ec_points(a, b, p)

print(f"Elliptic curve: y^2 ≡ x^3 + {a}x + {b} (mod {p})")
print(f"Found {len(points)} points on the curve")

# Plot the points
fig, ax = plt.subplots(figsize=(10, 6))

if points:
    x_coords, y_coords = zip(*points)
    ax.scatter(x_coords, y_coords, s=50, alpha=0.6, color="#2C3E50")
    ax.axhline(p / 2, color="black", linewidth=0.5, ls="--")

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
ax.set_title(
    f"Elliptic Curve: $y^2 \\equiv x^3 + {a}x + {b} (\\operatorname{{mod}} {p})$",
    fontsize=16,
)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, p)
ax.set_ylim(-1, p)
ax.set_aspect("equal")

# Add grid lines at integer positions
ax.set_xticks(range(0, p, 5))
ax.set_yticks(range(0, p, 5))

plt.tight_layout()
plt.savefig("img/elliptic_curve_finite_field.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nPlot saved as 'img/elliptic_curve_finite_field.png'")
print(f"\nFirst 10 points: {points[:10]}")

# %%
