
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
  phip(a) = Float64(((g(x+a*p)) * p)[1])

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

    if this_a==amax
      return this_a
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
              max_ite::Int64=10^3, zoomstage=nothing, zero=1/10^7)

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

    crit = nothing
    den = abs(alow)>zero ? alow : ahigh
    if abs(den)<=zero
      crit = abs( (1+abs(ahigh-alow)) / (1+abs(alow)) - 1)
    else
      crit = abs(ahigh-alow)/abs(den)
    end
    if crit<=zero
      return this_a
    end
    # println("high: $ahigh\tlow:$alow\tcrit:$crit")
  end

  error("Limit of iterations ($max_ite) reached in zoom without results")
end

"""
Optimization using Steepest Descent method

# Arguments
  * f     : scalar-valued function
  * g     : gradient function of f
  * X0    : initial guess
  * eps_a : absolute tolerance on change of function value (rec=10^-6)
  * eps_r : relative tolerance on change of function value (rec=0.01)
  * eps_g : gradient convergence tolerance
  * mu1   : decrease parameter on f for line search
  * mu2   : decrease parameter on g for line search
  * amax  : maximum search range for line search
  * max_ite     : maximum loop iterations
"""
function steepest_descent(f, g, X0::Array{Float64,1},
                          eps_a::Float64, eps_r::Float64, eps_g::Float64,
                          mu1::Float64, mu2::Float64, amax::Float64,
                          max_ite::Int64; Xs=nothing, fs=nothing, gs=nothing,
                          linestage=nothing, zoomstage=nothing)
  prev_X = X0
  prev_f = f(prev_X)[1]
  for i in 1:max_ite
    prev_g = g(prev_X)
    p = vec(-prev_g/norm(prev_g)) # Direction of search
    step = linesearch(f, g, p, prev_X, amax/2, amax, mu1, mu2;
                      max_ite=max_ite, linestage=linestage, zoomstage=zoomstage)

    this_X = prev_X + step*p
    this_f = f(this_X)[1]

    if Xs!=nothing; push!(Xs, this_X); end;
    if fs!=nothing; push!(fs, this_f); end;
    if gs!=nothing; push!(gs, prev_g); end;
    if abs(prev_f - this_f) <= eps_a + eps_r*abs(prev_f) && norm(prev_g)<=eps_g
      return this_X, this_f
    end

    prev_X, prev_f = this_X, this_f
  end
end

"""
Optimization using Steepest Descent method on inequalities constrains

# Arguments
  * f     : scalar-valued function
  * cons  : Array of constrain functions such that cons[i](x) >= 0
  * gtes  : cons[i](x) >= 0 if gtes[i]==true, cons[i](x) <= 0 if gtes[i]==false
  * barrs : Barrier parameter of each constrain
  * rs    : Radius around cons[i](x)=0 where the penalty kicks in at 0.01 the value of barrs[i]
  * And all other arguments in `steepest_descent()`
"""
function steepest_descent_constrained(f, cons, gtes::Array{Bool, 1},
                      barrs::Array{Float64, 1}, rs::Array{Float64, 1}, args...;
                      key_args...)

  # Constrained objective function
  new_f = _gen_constrained_f(f, cons, gtes, barrs, rs)

  # Gradient of constrained objective
  new_g(X) = GradEval.fad(new_f, X)[2]

  # Calls steepest descent method on constrained objective
  return steepest_descent(new_f, new_g, args...; key_args...)
end


# --------- INTERNAL FUNCTIONS -------------------------------------------------
function _gen_constrained_f(f, cons, gtes::Array{Bool, 1},
                      barrs::Array{Float64, 1}, rs::Array{Float64, 1})

  # ERROR CASES
  ncons = size(cons)[1]
  if size(gtes)[1]!=ncons
    error("Expected $ncons elements in `gtes`, found $(size(gtes)[1]).")
  elseif size(barrs)[1]!=ncons
    error("Expected $ncons elements in `barrs`, found $(size(barrs)[1]).")
  elseif size(rs)[1]!=ncons
    error("Expected $ncons elements in `rs`, found $(size(rs)[1]).")
  end

  # Constrained objective function
  function new_f(X)
    # Original objective value
    val = f(X)

    # Constrain penalties
    for i in 1:ncons
      con_val = cons[i](X)                            # Evaluates the constrain
      aux1 = (-1)^!gtes[i] * con_val/rs[i]*log(0.01)  # Penalty criteria
      val += barrs[i]*exp.(aux1)                      # Adds the penalty
    end

    return val
  end

  return new_f
end

end # END OF MODULE
