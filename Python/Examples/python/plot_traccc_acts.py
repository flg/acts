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
        axes = fig.subplots()

        # Plot data
        draw_data(axes, acts_data, "ACTS")
        draw_data(axes, traccc_data, "Traccc")

        # Axes labels
        if "trackeff" in plot:
            axes.set_ylabel("Tracking Efficiency")
        elif "fakeRatio" in plot:
            axes.set_ylabel("Fake Track Ratio")
        elif "duplicationRatio" in plot:
            axes.set_ylabel("Duplicate Track Ratio")

        if "vs_eta" in plot:
            axes.set_xlabel("$\\eta$")
        elif "vs_pT" in plot:
            axes.set_xlabel("$p_T$ (GeV)")

        # Limits
        if "Ratio" in plot:
            axes.set_ylim([0, 0.1])

        # Ticks and grid
        major_xticks = axes.get_xticks()
        minor_xticks = []
        for i in range(1, len(major_xticks)):
            minor_xticks.append((major_xticks[i] + major_xticks[i-1]) / 2)
        axes.set_xticks(minor_xticks, minor=True)

        axes.grid(which="major", linestyle="-")
        axes.grid(which="minor", linestyle="--")

        # Export
        axes.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / f"{plot}.png")

        matplotlib.pyplot.close()

    log.info("plotting complete")

def draw_data(axes, data, label):
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
    return data[data["ntotal"] > 0]

if __name__ == "__main__":
    main()