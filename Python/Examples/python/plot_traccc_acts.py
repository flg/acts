import argparse
import pathlib
import logging
import os.path
import pandas
import matplotlib.pyplot
import scipy
import numpy

log = logging.getLogger("plot_traccc_acts")

PLOT_NAMES = [
    "trackeff_vs_eta",
    "trackeff_vs_pT",
    "fakeRatio_vs_eta",
    "fakeRatio_vs_pT",
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info("plotting traccc acts performance")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "actsdir",
        type=pathlib.Path,
        help="where acts csv files are",
    )
    parser.add_argument(
        "tracccdir",
        type=pathlib.Path,
        help="where traccc csv files are",
    )
    parser.add_argument(
        "outdir",
        type=pathlib.Path,
        help="where output files are written",
    )

    args = parser.parse_args()

    if os.path.isfile(args.outdir):
        raise RuntimeError(f"{args.outdir} already exists as a file")
    if not os.path.isdir(args.outdir):
        log.info(f"creating output directory: {args.outdir}")
        os.mkdir(args.outdir)

    for plot in PLOT_NAMES:
        acts_file = args.actsdir / f"{plot}.csv"
        traccc_file = args.tracccdir / f"{plot}.csv"
        if not os.path.isfile(acts_file):
            raise RuntimeError(f"{acts_file} does not exist")
        if not os.path.isfile(traccc_file):
            raise RuntimeError(f"{traccc_file} does not exist")

        acts_data = read_data(acts_file)
        traccc_data = read_data(traccc_file)

        fig = matplotlib.pyplot.figure()
        (ax_data, ax_ratio) = fig.subplots(
            nrows=2,
            height_ratios=(3, 1)
        )

        # Plot data
        draw_data(ax_data, traccc_data, "Traccc")
        draw_data(ax_data, acts_data, "ACTS")

        draw_ratio(ax_ratio, traccc_data, acts_data)

        # Axes labels
        if "trackeff" in plot:
            ax_data.set_ylabel("Tracking Efficiency")
        elif "fakeRatio" in plot:
            ax_data.set_ylabel("Fake Track Ratio")
        elif "duplicationRatio" in plot:
            ax_data.set_ylabel("Duplicate Track Ratio")

        if "vs_eta" in plot:
            ax_data.set_xlabel("$\\eta$")
        elif "vs_pT" in plot:
            ax_data.set_xlabel("$p_T$ (GeV)")

        ax_ratio.set_ylabel("Traccc / ACTS")

        # Limits
        # if "Ratio" in plot:
        #     axes.set_ylim([0, 0.1])
        if "vs_pT" in plot:
            ax_data.set_xlim([0, 100])

        # Ticks and grid
        for ax in (ax_data, ax_ratio):
            major_xticks = ax.get_xticks()
            minor_xticks = []
            for i in range(1, len(major_xticks)):
                minor_xticks.append((major_xticks[i] + major_xticks[i-1]) / 2)
            ax.set_xticks(minor_xticks, minor=True)

            ax.grid(which="major", linestyle="-")
            ax.grid(which="minor", linestyle="--")

        # Export
        if "fakeRatio" in plot:
            legend_pos = "upper left"
        else:
            legend_pos = "lower left"
        ax_data.legend(loc=legend_pos)
        fig.tight_layout()
        fig.savefig(args.outdir / f"{plot}.png")

        matplotlib.pyplot.close()

    log.info("plotting complete")

def draw_data(axes, data, label):
    alpha = 1
    if label == "ACTS":
        alpha = 0.6
    axes.errorbar(
        data["center"],
        data["efficiency"],
        yerr=(
            data["err_low"],
            data["err_high"],
        ),
        xerr=(
            data["center"] - data["bin_left"],
            data["bin_right"] - data["center"],
        ),
        # linewidth=1,
        linestyle="",
        # capsize=0,
        label=label,
        alpha=alpha,
    )

def draw_ratio(axes, traccc_data, acts_data):
    # print(f"x={traccc_data["center"]}")
    # print(f"y={traccc_data["efficiency"] / acts_data["efficiency"]}")
    axes.plot(
        traccc_data["center"],
        traccc_data["efficiency"] / acts_data["efficiency"],
        "+",
        color="c"
    )
    axes.axhline(y=1, linestyle="--", color="black", linewidth=1)

def draw_data_filled_error(axes, data, label):
    axes.plot(
        data["center"],
        data["efficiency"],
        "+",
        label=label,
    )
    color = axes.get_lines()[-1].get_color()
    axes.plot(
        data["center"],
        data["efficiency"] - data["err_low"],
        linestyle=":",
        linewidth=0.5,
        color=color,
    )
    axes.plot(
        data["center"],
        data["efficiency"] + data["err_high"],
        linestyle=":",
        linewidth=0.5,
        color=color,
    )
    axes.fill_between(
        data["center"],
        data["efficiency"] - data["err_low"],
        data["efficiency"] + data["err_high"],
        alpha=0.2,
        color=color,
    )

def draw_data_spline(axes, data, label):
    axes.plot(
        data["center"],
        data["efficiency"],
        "+",
        # xerr=data["width"],
        # yerr=(data["err_low"], data["err_high"]),
        label=label,
    )

    smooth_x = numpy.linspace(data["center"].min(), data["center"].max(), 200)

    ylow_spline = scipy.interpolate.make_interp_spline(
        data["center"],
        data["efficiency"] - data["err_low"],
        k=3)
    smooth_ylow = ylow_spline(smooth_x)

    yhigh_spline = scipy.interpolate.make_interp_spline(
        data["center"],
        data["efficiency"] + data["err_high"],
        k=3)
    smooth_yhigh = yhigh_spline(smooth_x)

    # axes.fill_between(
    #     data["center"],
    #     data["efficiency"] - data["err_low"],
    #     data["efficiency"] + data["err_high"],
    #     alpha=0.2
    # )
    axes.fill_between(
        smooth_x,
        smooth_ylow,
        smooth_yhigh,
        alpha=0.2
    )


def read_data(filepath: pathlib.Path):
    data = pandas.read_csv(filepath)

    data["center"] = (data["bin_left"] + data["bin_right"]) / 2
    data["width"] = data["center"] - data["bin_left"]
    # return data[data["ntotal"] > 0]

    if "vs_eta" in str(filepath):
        data = data[data["bin_left"] >= -3]
        data = data[data["bin_right"] <= 3]
    elif "vs_pT" in str(filepath):
        data = data[data["bin_left"] >= 0]
        data = data[data["bin_right"] <= 100]

    return data

if __name__ == "__main__":
    main()