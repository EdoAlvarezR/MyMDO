
module MyMDO

include("tools/ad.jl")
using PyPlot


################################################################################
# STEEPEST DESCENT METHODS
################################################################################
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

  println("Maximum iterations reached!")
  return prev_X, prev_f
end

"""
Unequality-constrained optimization using Steepest Descent method. Constraints
are imposed as penalty functions and are in the format cons[i](x)>=0.

# Arguments
  * cons  : Array of constraint functions such that cons[i](x) >= 0
  * barrs : Barrier parameter of each constrain
  * rs    : Radius around cons[i](x)=0 where the penalty kicks in at 0.01 the value of barrs[i]
  * And all other arguments in `steepest_descent()`
"""
function steepest_descent_cons(cons, barrs::Array{Float64, 1},
                                rs::Array{Float64, 1}, f, g,
                                args...; key_args...)

  # Penalty function and its gradient
  this_penalty = _gen_penalty(cons, barrs, rs)
  penalty_grad(X) = GradEval.fad(this_penalty, X)[2]

  # Constrained objective and its gradient
  cons_f(X) = f(X) + this_penalty(X)
  cons_g(X) = g(X) + penalty_grad(X)

  # Calls unconstrained quasi-newton on constrained objective
  return steepest_descent(cons_f, cons_g, args...; key_args...)
end
# END OF STEEPEST DESCENT ######################################################













################################################################################
# QUASI-NEWTON METHODS
################################################################################
"""
Unconstrained optimization using Quasi-Newton method with BFGS update as found
in Martin's Sec. 3.7.2.

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
function quasinewton_bfgs(f, g, X0::Array{Float64,1},
                          eps_a::Float64, eps_r::Float64, eps_g::Float64,
                          mu1::Float64, mu2::Float64, amax::Float64,
                          max_ite::Int64; Xs=nothing, fs=nothing, gs=nothing,
                          linestage=nothing, zoomstage=nothing, debug=true)

  eye_X =  eye(size(X0)[1])       # Identity matrix dimensioned for X
  prev_X = X0                     # Initial/previous X
  prev_f = f(prev_X)[1]           # Initial/previous f value
  prev_g = vec(g(prev_X))         # Initial/previous gradient
  prev_V = eye_X                  # Inverse Hessian approximation at previous X
  if Xs!=nothing; push!(Xs, prev_X); end;
  if fs!=nothing; push!(fs, prev_f); end;
  if gs!=nothing; push!(gs, prev_g); end;


  for k in 1:max_ite              # k-th iteration

    # Takes a step
    p = -prev_V*prev_g            # Step direction
                                  # Step length
    step = linesearch(f, g, p, prev_X, amax/2, amax, mu1, mu2;
                      max_ite=max_ite, linestage=linestage, zoomstage=zoomstage)
    this_s = step*p               # Step
    this_X = prev_X + this_s      # New X

    # Calculates values at new X
    this_f = f(this_X)[1]         # Objective
    this_g = vec(g(this_X))       # Gradient
    this_y = this_g - prev_g      # Change in gradient
    str_y  = this_s'*this_y
    this_V = ( eye_X - (this_s*this_y')/str_y ) * prev_V * (  # Hessian aprox
                  eye_X - (this_y*this_s')/str_y ) + this_s*this_s'/str_y

    # Stores path
    if Xs!=nothing; push!(Xs, this_X); end;
    if fs!=nothing; push!(fs, this_f); end;
    if gs!=nothing; push!(gs, this_g); end;

    # Checks if optimum was reached
    if abs(prev_f - this_f) <= eps_a + eps_r*abs(prev_f) && norm(prev_g)<=eps_g
      return this_X, this_f
    end

    prev_X, prev_f, prev_g, prev_V = this_X, this_f, this_g, this_V
  end

  println("Maximum iterations reached!")
  return prev_X, prev_f
end


"""
Unequality-constrained optimization using Quasi-Newton method with BFGS update
as found in Martin's Sec. 3.7.2.  Constraints are imposed as penalty functions
and are in the format cons[i](x)>=0.

# Arguments
  * cons  : Array of constraint functions such that cons[i](x) >= 0
  * barrs : Barrier parameter of each constrain
  * rs    : Radius around cons[i](x)=0 where the penalty kicks in at 0.01 the value of barrs[i]
  * And all other arguments in `quasinewton_bfgs()`
"""
function quasinewton_bfgs_cons(cons, barrs::Array{Float64, 1},
                                rs::Array{Float64, 1}, f, g,
                                args...; key_args...)

  # Penalty function and its gradient
  this_penalty = _gen_penalty(cons, barrs, rs)
  penalty_grad(X) = GradEval.fad(this_penalty, X)[2]

  # Constrained objective and its gradient
  cons_f(X) = f(X) + this_penalty(X)
  cons_g(X) = g(X) + penalty_grad(X)

  # Calls unconstrained quasi-newton on constrained objective
  return quasinewton_bfgs(cons_f, cons_g, args...; key_args...)
end
# END OF QUASI-NEWTON ##########################################################















################################################################################
# MISCELLANEOUS
################################################################################
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
    if linestage!=nothing; push!(linestage, x+this_a*p); end;
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


"Give the path of the optimizer and it will plot it"
function plot_opt(fs, gs; fig_name="opt_path")

    fig2 = figure(fig_name, figsize=(7*2,5*1))
    subplot(121)
    title("Convergence on f(X)")
    plot([i for i in 1:size(fs)[1]], fs, "--ok")
    ylabel("f(x)")
    xlabel("Iteration")
    grid(true, color="0.8", linestyle="--")
    subplot(122)
    title("Convergence on grad(f(X))")
    plot([i for i in 1:size(gs)[1]], [norm(this_g) for this_g in gs], "--ok")
    ylabel("|grad(f(x))|")
    xlabel("Iteration")
    grid(true, color="0.8", linestyle="--")
end

"Evaluates and prints each constraint at X"
function print_constraints(X, cons, barrs::Array{Float64, 1},
                              rs::Array{Float64, 1})

  ncons = size(cons)[1]
  println("Cons(x) \t Feasability ")
  for i in 1:ncons
    cons_val = cons[i](X)
    penalty = _penalty(X, cons[i], barrs[i], rs[i])
    if cons_val<0
      feas = abs(cons_val)
    else
      feas = 0.0
    end
    println("$(@sprintf("%.5f", cons_val)) \t $(@sprintf("%.5f", feas))")
  end
end

# --------- INTERNAL FUNCTIONS -------------------------------------------------
"Generates and returns the total penalty function"
function _gen_penalty(cons, barrs::Array{Float64, 1}, rs::Array{Float64, 1})

  # ERROR CASES
  ncons = size(cons)[1]
  if size(barrs)[1]!=ncons
    error("Expected $ncons elements in `barrs`, found $(size(barrs)[1]).")
  elseif size(rs)[1]!=ncons
    error("Expected $ncons elements in `rs`, found $(size(rs)[1]).")
  end

  function this_penalty(X)
    val = 0.0
    for i in 1:ncons # Adds the penalty of each constraint
      val += _penalty(X, cons[i], barrs[i], rs[i])
    end
    return [val]
  end

  return this_penalty
end

"Exponential penalty function"
function _penalty(X, cons, barr, r)
  con_val = cons(X)                            # Evaluates the constraint
  aux1 = con_val/r*log(0.01)                   # Penalty criteria
  val = barr*exp.(aux1)                        # Penalty value

  return val
end
# END OF MISC ##################################################################


end # END OF MODULE
