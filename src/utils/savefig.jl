# Author: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

export _wsave

using PyPlot: Figure
import DrWatson: _wsave

_wsave(s, fig::Figure; dpi::Int=200) = fig.savefig(s, bbox_inches="tight", dpi=dpi)