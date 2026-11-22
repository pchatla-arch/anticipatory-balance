import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser=argparse.ArgumentParser(description="Plot the predeclared development sweep.")
    parser.add_argument("directory",type=Path)
    folder=parser.parse_args().directory
    with (folder/"joined.csv").open(newline="") as f:
        rows=[r for r in csv.DictReader(f) if (r["family"],r["channel"],r["window"],r["envelope"])
              ==("gain_070","declared","first_second","speed_magnitude")]
    fig,axes=plt.subplots(1,2,figsize=(9,4.1),sharey=True)
    for ax,group in zip(axes,("hips","knees")):
        rr=sorted((r for r in rows if r["group"]==group),key=lambda r:int(r["acceleration_rad_s2"]))
        x=[float(r["acceleration_rad_s2"]) for r in rr]
        for metric,color,label in [("applied_over_oracle","#2f5983","Oracle correction supplied"),
                                   ("two_endpoint_coverage","#a04920","Endpoint pairs covered")]:
            ax.plot(x,[100*float(r[metric]) for r in rr],"o-",color=color,label=label,ms=4)
        ax.axhline(50,color="#2f5983",ls=":",lw=1)
        ax.axhline(99,color="#a04920",ls=":",lw=1)
        ax.set(xscale="log",xticks=x,xticklabels=[str(int(k)) for k in x],ylim=(-4,105),title=group.capitalize())
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.16)
        ax.set_xlabel("Assumed acceleration limit (rad/s²)")
    axes[0].set_ylabel("Percent")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper left",bbox_to_anchor=(.073,.925),ncol=2,frameon=False)
    fig.suptitle("First second after a 30% leg gain loss",x=.085,y=.98,ha="left",fontweight="bold",fontsize=15)
    fig.text(.085,.025,"Eight development walks, declared sensor delay. Dotted lines mark the 50% correction and 99% coverage targets.",fontsize=8)
    fig.subplots_adjust(left=.085,right=.98,bottom=.20,top=.74,wspace=.13)
    fig.savefig(folder/"velocity_sweep.png",dpi=145)
    plt.close(fig)


if __name__=="__main__":
    main()
