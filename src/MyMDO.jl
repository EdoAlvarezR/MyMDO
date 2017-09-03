
module MyMDO

include("tools/ad.jl")

"""
Algorithm 2 in Martin's *Multidisciplinary design optimization*. Given a
multivariate function `f`: R^n->R, its gradient `g`, a direction `p` where the
function decreases at point `x`, it returns the step size that satisfies strong
Wolf conditions. `a1` is an initial guess, `amax` is the maximum search range,
`mu1` and `mu2` are decrease parameters on `f` and `g`, respectively, such that
0<=`mu1`<=1 and `mu1`<=`mu2`<=1.
"""
function linesearch(f, g, p::Array{Float64,1}, x::Array{Float64,1},
                    a1::Float64, amax::Float64, mu1::Float64, mu2::Float64;
                    max_ite::Int64=10^3, linestage=nothing, zoomstage=nothing)
  phi(a) = Float64(f(x+a*p)[1])
  phip(a) = Float64(((g(x+a*p))' * p)[1])

  phi_0 = phi(0)
  phip_0 = phip(0)

  prev_a = 0.0
  prev_phi = nothing
  this_a = a1
  for i in 1:max_ite
    if linestage!=nothing; push!(linestage, this_a); end;
    this_phi = phi(this_a)

    # Cases of too big of a step
    if ( this_phi > (phi_0+mu1*this_a*phip_0) ) || (i!=1 && this_phi>prev_phi)
      return _zoom(phi, phip, prev_a, this_a, mu1, mu2, phi_0, phip_0,
                    prev_phi==nothing ? phi(prev_a) : prev_phi;
                    max_ite=max_ite, zoomstage=zoomstage)
    end

    this_phip = phip(this_a)

    # Case curvature is satisfied
    if abs(this_phip) <= -mu2*phip_0
      return this_a

    # Another case of too big of a step (function value is increasing)
    elseif this_phip>=0
      return _zoom(phi, phip, this_a, prev_a, mu1, mu2, phi_0, phip_0, this_phi;
                    max_ite=max_ite, zoomstage=zoomstage)
    end

    prev_a = this_a
    prev_phi = this_phi
    # Choose a new `a` such that this_a<new_a<amax
    this_a = (amax + prev_a)/2
  end
  error("Limit of iterations ($max_ite) reached in first stage without results")
end
function _zoom(phi, phip, alow_::Float64, ahigh_::Float64,
              mu1::Float64, mu2::Float64,
              phi_0::Float64, phip_0::Float64, phi_alow_::Float64;
              max_ite::Int64=10^3, zoomstage=nothing)

  alow, ahigh, phi_alow = alow_, ahigh_, phi_alow_

  for i in 1:max_ite
    this_a = (alow + ahigh)/2
    this_phi = phi(this_a)
    if zoomstage!=nothing; push!(zoomstage, this_a); end;
    # Case of not sufficient decrease or lower than low point
    if (this_phi > phi_0+mu1*this_a*phi_0) || (this_phi > phi_alow)
      ahigh = this_a
    else
      this_phip = phip(this_a)
      # Case of sufficient curvature
      if abs(this_phip) <= -mu2*phip_0
        return this_a
      elseif this_phip*(ahigh-alow) >= 0
        ahigh = alow
      end
      alow = this_a
      phi_alow = this_phi
    end
  end

  error("Limit of iterations ($max_ite) reached in zoom without results")
end



end # END OF MODULE
