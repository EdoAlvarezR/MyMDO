"""
  Methods for multidisciplinary design optimization.

  # AUTHORSHIP
    * Author    : Eduardo J Alvarez
    * Email     : Edo.AlvarezR@gmail.com
    * Created   : Jan 2018
    * License   : MIT License
"""

module MyMDO


# ------------ GENERIC MODULES -------------------------------------------------
using PyPlot

# ------------ GLOBAL VARIABLES ------------------------------------------------
global module_path; module_path,_ = splitdir(@__FILE__);   # Path to this module
global data_path = module_path*"/../data/"       # Path to data folder


# ------------ HEADERS ---------------------------------------------------------
for header_name in ["gradientbased", "ga", "visualization"]
  include("MyMDO_"*header_name*".jl")
end

include("tools/ad.jl")



end # END OF MODULE
