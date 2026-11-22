"""Plot time-aligned development authority from replay_authority.py."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    with (args.directory/"curves.csv").open(newline="") as f:
        curves=list(csv.DictReader(f))
    fig,axes=plt.subplots(3,2,figsize=(9,8.2),sharex=True,sharey="row")
    colors={"gain_085":"#12646b","gain_070":"#a54919","gain_055":"#4558a0"}
    for i,(measure,title) in enumerate([("mean_width","Mean gain interval width"),
                                        ("fraction_excluding_one","Share excluding gain 1"),
                                        ("oracle_fraction","Oracle offset permitted (fraction)")]):
        axes[i,0].set_ylabel(title)
        for j,group in enumerate(("hips","knees")):
            ax=axes[i,j]
            for family,color in colors.items():
                for channel,style in (("exact","--"),("declared","-")):
                    rows=[r for r in curves if (r["family"],r["channel"],r["group"])==(family,channel,group)]
                    x=np.array([float(r["seconds"]) for r in rows])
                    y=np.array([float(r[measure]) if r[measure] else np.nan for r in rows])
                    ax.plot(x,y,linestyle=style,color=color,lw=1.3,alpha=0.94)
            ax.axvline(1.0,color="#686d71",lw=0.8,alpha=.8)
            if i==2:
                ax.axhline(.5,color="#959b9d",lw=0.8)
            if i>0:
                ax.set_ylim(-.04,1.04)
            ax.set_xlim(left=0)
            ax.grid(axis="y",color="#d8dddf",lw=.5)
            ax.spines[["right","top"]].set_visible(False)
            if i==0:
                ax.set_title(group.capitalize(),loc="left",fontweight="bold")
            if i==2:
                ax.set_xlabel("Seconds after gain loss")
    from matplotlib.lines import Line2D
    items=[Line2D([0],[0],color=color,label=f"Gain {int(family[-3:])/100:.2f}") for family,color in colors.items()]
    items.extend([Line2D([0],[0],color="#454a4b",linestyle=style,label=channel) for channel,style in [("Exact","--"),("Declared","-")]])
    fig.legend(handles=items,loc="upper left",bbox_to_anchor=(.105,.934),ncol=5,frameon=False)
    fig.suptitle("How much correction do the measurements permit?",x=.11,y=.975,ha="left",fontsize=17,fontweight="bold")
    fig.text(.11,.028,"Uncorrected development walks. Values use surviving episodes at each time. See curves.csv for exposure counts.",fontsize=8,color="#434849")
    fig.subplots_adjust(left=.11,right=.98,top=.85,bottom=.11,hspace=.26,wspace=.12)
    fig.savefig(args.directory/"authority.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    main()
