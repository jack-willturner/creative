import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


@dataclass
class PlotParams:
    figsize: tuple[int, int]
    background_colour: str


@dataclass
class LineStyleConfig:
    variance: float = 1e3
    num_scatter_points: int = 4000
    line_colour: str = "#2F1E1E"


class TexturedLine:
    def __init__(self, points: list[Point], linestyle: LineStyleConfig = None) -> None:
        self.points = points

        if linestyle is None:
            linestyle = LineStyleConfig()

        self.variance = linestyle.variance
        self.num_scatter_points = linestyle.num_scatter_points
        self.line_colour = linestyle.line_colour

        self.compute_points_as_lists_x_y()

    def compute_points_as_lists_x_y(self) -> None:
        self.xs = [point.x for point in self.points]
        self.ys = [point.y for point in self.points]

    def sample_points_along_line(self) -> tuple[list[float], list[float]]:
        from scipy import interpolate

        f = interpolate.interp1d(self.xs, self.ys)

        xs_to_interpolate = np.linspace(
            min(self.xs), max(self.xs), self.num_scatter_points
        )

        return xs_to_interpolate, f(xs_to_interpolate)

    def draw(self, ax) -> None:
        ax.plot(self.xs, self.ys, color=self.line_colour)

        x_dots, y_dots = self.sample_points_along_line()

        ## TODO should these be drawn from the same distribution?
        horizontal_noise = np.random.normal(0, self.variance, self.num_scatter_points)
        vertical_noise = np.random.normal(0, self.variance, self.num_scatter_points)

        ax.scatter(
            x_dots + horizontal_noise,
            y_dots + vertical_noise,
            s=1,
            color=self.line_colour,
        )


def get_fig_ax(plot_params: PlotParams):
    fig, ax = plt.subplots(figsize=plot_params.figsize)
    fig.set_facecolor(plot_params.background_colour)
    ax.set_facecolor(plot_params.background_colour)

    # remove all spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def straight_lines(plot_params):
    ax = get_fig_ax(plot_params)

    lines = [
        LineStyleConfig(0.001, 1000),
        LineStyleConfig(0.001, 2000),
        LineStyleConfig(0.001, 3000),
        LineStyleConfig(0.001, 4000),
    ]

    for i, linestyle in enumerate(lines):
        line = TexturedLine([Point(0, i + 1), Point(1, i + 1)], linestyle)
        line.draw(ax)

    ax.set_ylim(0, len(lines) + 1)
    plt.savefig("outputs/straight_lines.png")


def curved_line(plot_params):
    ax = get_fig_ax(plot_params)

    linestyle = LineStyleConfig(0.001, 4000)

    line = TexturedLine([Point(x, x ** 2) for x in range(25)], linestyle)
    line.draw(ax)

    ax.set_ylim(0, max(line.ys))

    plt.savefig("outputs/curved_line.png")


def main():
    basic_plot_params = PlotParams((10, 5), "#E7DACB")

    straight_lines(basic_plot_params)
    curved_line(basic_plot_params)


if __name__ == "__main__":
    main()
