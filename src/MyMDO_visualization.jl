


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


"Plots a design space of only two dimensions"
function plot_space(f, xmin,xmax, ymin,ymax, n;
                    Xs=nothing, xopt=nothing, x_i=1, y_i=2,
                    xlbl=L"x", ylbl=L"y", zlbl=L"f(x,y)",
                    title_str="Objective function", ppltn=nothing, con=nothing)
  x = linspace(xmin, xmax, n)
  y = linspace(ymin, ymax, n)

  xgrid = repmat(x',n,1)
  ygrid = repmat(y,1,n)

  z = zeros(n,n)
  for i in 1:n
      for j in 1:n
          z[j,i] = f([x[i],y[j]])[1]
      end
  end

  if con!=nothing
    zcon = zeros(n,n)
    for i in 1:n
        for j in 1:n
            zcon[j,i] = con([x[i],y[j]])[1]
        end
    end
  end


  # ----------------- Objective plots --------------------
  fig = figure("opt1", figsize=(7*2,5))

  # Surface plot
  subplot(121)
  ax1 = fig[:add_subplot](1,2,1, projection = "3d")
  title(title_str)
  ## Surface plot of objective
  ax1[:plot_surface](xgrid, ygrid, z,
              rstride=2,edgecolors="k", cstride=2,
              cmap=ColorMap("coolwarm"), alpha=0.5,
              linewidth=0.05)
  ## Optimization path
  if Xs!=nothing
      ax1[:plot]([x[x_i] for x in Xs], [x[y_i] for x in Xs],
                                      [f(x)[1] for x in Xs], "--.k")
      # ax1[:scatter3D]([x[x_i] for x in Xs], [x[y_i] for x in Xs],
      #                 [f(x)[1] for x in Xs], marker="o", c="k", label="Optimum")
  end
  ## Population
  if ppltn!=nothing
      npop = size(ppltn,1)
      ax1[:plot]([ppltn[i,x_i] for i in 1:npop], [ppltn[i,y_i] for i in 1:npop],
                                      [f(ppltn[i,:])[1] for i in 1:npop], ".k")
  end
  if xopt!=nothing
    ax1[:scatter3D]([xopt[x_i]], [xopt[y_i]], [f(xopt)[1]],
                                  s=100, marker="*", c="r", label="Optimum")
  end
  xlabel(xlbl)
  ylabel(ylbl)
  zlabel(zlbl)
  xlim([xmin, xmax])
  ylim([ymin, ymax])

  # Contour plot
  subplot(122)
  ax2 = fig[:add_subplot](1,2,2)
  cp = ax2[:contour](xgrid, ygrid, z, 15,
                  colors="black", linewidth=2.0)
  ax2[:clabel](cp, inline=1, fontsize=10)
  if con!=nothing
    cp2 = ax2[:contour](xgrid, ygrid, zcon, 15, levels=[0.0],
                  colors="red", linewidth=2.0)
  end
  if Xs!=nothing
      ax2[:plot]([x[x_i] for x in Xs], [x[y_i] for x in Xs], "-ok")
  end
  if ppltn!=nothing
      npop = size(ppltn,1)
      ax2[:plot]([ppltn[i,x_i] for i in 1:npop], [ppltn[i,y_i] for i in 1:npop], ".k")
  end
  if xopt!=nothing
    ax2[:plot]([xopt[x_i]], [xopt[y_i]], "*r", label="Optimum")
  end
  xlabel(xlbl)
  ylabel(ylbl)
  xlim([xmin, xmax])
  ylim([ymin, ymax])
  tight_layout()
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

"""
Receives the computation function `compfun`, optimization path `Xs`, the
optimum `xopt`, and linear constraints `lb`,`ub`, and generates a three
dimensional plot on the two variables of maximum variation showing the
optimization path.

NOTE: It expects to find the objective in the first output of `compfun`.
"""
function design_space(compfun, Xs,
                        xopt::AbstractArray, lb::AbstractArray,
                        ub::AbstractArray; x_i::Int64=-1, y_i::Int64=-1,
                        ndiscr::Int64=15, lbl="\nxopt", lbl_add_val=true,
                        alive=false, saved_gen=nothing,
                        compfun_args...)

  # Chooses what variables to put in axes x and y
  if saved_gen!=nothing
    _Xs = []
    for (ppltn, ftnss) in saved_gen
      push!(_Xs, ppltn[ftnss[1][2], :])
    end
    _xopt = saved_gen[end][1][saved_gen[end][2][1][2], :]
  else
    _Xs = Xs
    _xopt = xopt
  end
  _x_i, _y_i = most_variation(_xopt, _Xs, lb, ub; x_i=x_i, y_i=y_i)

  # Ranges for surface plotting
  xmin, ymin = lb[_x_i], lb[_y_i]
  xmax, ymax = ub[_x_i], ub[_y_i]

  # Wraps compfun as a 2D function
  fncalls = 0
  function wrap_compfun(x)
    fncalls += 1
    prev_t = time()
    if size(x,1)==2 # Case of generating surface and contour plots
      compfun_x = deepcopy(_xopt)
      compfun_x[_x_i] = x[1]
      compfun_x[_y_i] = x[2]
      out = compfun(compfun_x; compfun_args...)[1]
    else  # Case of plotting the path
      compfun_x = deepcopy(_xopt)
      compfun_x[_x_i] = x[_x_i]
      compfun_x[_y_i] = x[_y_i]
      out = compfun(compfun_x; compfun_args...)[1]
    end
    if alive; println("\tFunction call #$fncalls: $(time()-prev_t) (s)"); end;
    return out
  end

  ttl = lbl * (lbl_add_val ? "=$(signif.(_xopt,3))" : "")

  plot_space(wrap_compfun, xmin, xmax, ymin, ymax, ndiscr;
                      Xs=saved_gen!=nothing ? nothing : Xs,
                      # xopt=saved_gen!=nothing ? nothing : xopt,
                      xopt=saved_gen!=nothing ? _xopt : xopt,
                      xlbl="x$(_x_i)", ylbl="x$(_y_i)",
                      zlbl="f(x$(_x_i),y$(_y_i))",
                      title_str="Design space at "*ttl,
                      x_i=_x_i, y_i=_y_i,
                      ppltn=saved_gen!=nothing ? saved_gen[end][1] : nothing
                      )
end

"Determines the two variables with most variation`"
function most_variation(xopt, Xs, lb, ub; x_i=-1, y_i=-1)
  x_is = [x_i, y_i]
  # Case of automatic choice
  if -1 in x_is
    n = size(xopt, 1) # Number of variables
    needed = size([i for i in x_is if i==-1],1) # Number of variables needed

    # Calculates variation range of each variable
    variations = [
    ( maximum([X[i] for X in Xs]) - minimum([X[i] for X in Xs]) )/(ub[i]-lb[i])
                                          for i in 1:n ]

    # Finds the two variables with max variation
    max1_i = indmax(variations)
    max2_i = indmax(vcat( variations[1:max1_i-1], variations[max1_i+1:end] ))

    # Saves selection
    if needed==1
      if x_i==-1; x_is[1]=max1_i; else; x_is[2]=max1_i; end;
    else
      x_is[1] = max1_i
      x_is[2] = max2_i
    end
  end

  return x_is[1], x_is[2]
end


function design_space_animation_ga(save_path::String, run_name::String,
                                compfun, gens, lb, ub; verbose=true,
                                first=1, last=-1, ext=".png", x_i=-1, y_i=-1,
                                optargs...)

  # Points to iterate through
  _gens = gens[first : (last==-1 ? size(gens,1) : last) ]
  xopt = _gens[end][1][_gens[end][2][1][2], :]
  Xs = []
  for (ppltn, ftnss) in _gens
    push!(Xs, ppltn[ftnss[1][2], :])
  end

  # Chooses what variables to put in axes x and y
  _x_i, _y_i = most_variation(xopt, Xs, lb, ub; x_i=x_i, y_i=y_i)

  # Iterates through the optimization path
  for (i,(ppltn, ftnss)) in enumerate(_gens)
    if verbose; println("Plotting iteration $i of $(size(_gens,1))..."); end;
    this_gens = _gens[1:i]
    design_space(compfun, [], [], lb, ub;
                        lbl="generation #$i", x_i=_x_i, y_i=_y_i, saved_gen=this_gens,
                        lbl_add_val=false,
                        optargs...)
    this_num = ( i<10 ? "000" : (i<100 ? "00" : (i<1000 ? "0" : "")) )*"$i"
    tight_layout()
    savefig(joinpath(save_path, run_name)*"."*this_num*ext)
    clf()
  end

  if verbose; println("\tDone plotting"); end;
end


function design_space_animation(save_path::String, run_name::String,
                                compfun, Xs, xopt, args...; verbose=true,
                                first=1, last=-1, ext=".png", x_i=-1, y_i=-1,
                                optargs...)

  # Points to iterate through
  _Xs = Xs[ first : (last==-1 ? size(Xs,1) : last) ]
  if last==-1 && Xs[end]!=xopt
    push!(_Xs, xopt)
  end

  # Chooses what variables to put in axes x and y
  _x_i, _y_i = most_variation(xopt, Xs, lb, ub; x_i=x_i, y_i=y_i)

  # Iterates through the optimization path
  for (i, this_x) in enumerate(_Xs)
    if verbose; println("Plotting iteration $i of $(size(_Xs,1))..."); end;
    this_Xs = _Xs[1:i]
    design_space(compfun, this_Xs, this_x, args...;
                        lbl="ite $i", x_i=_x_i, y_i=_y_i, optargs...)

    this_num = ( i<10 ? "000" : (i<100 ? "00" : (i<1000 ? "0" : "")) )*"$i"
    tight_layout()
    savefig(joinpath(save_path, run_name)*"."*this_num*ext)
    clf()
  end

  if verbose; println("\tDone plotting"); end;
end


"Plots the pareto front on the population `ppltn` of a multiobjective function
`f`"
function plot_multiobjective(f, ppltn; f_x=1, f_y=2, f_z=3, labels=true, rnd=2,
                                scaling=nothing,
                                title_str="Multiobjective population fitness",
                                ftnss=nothing, fvals=nothing, dlegend=true,
                                ylims=nothing, crv_label=nothing,
                                plot_population=true, new_fig=true,
                                annotate_num=1,
                                title_y=nothing)
  npop = size(ppltn, 1)                # Population size

  if fvals!=nothing
    nfun = size(fvals[1],1)
  else
    nfun = size(f(ppltn[1,:]), 1)      # Number of objectives
  end

  _labels = ("abcdefghijklmnopqrstuvwxyz"^Int(ceil(npop/27 + 1)))[1:npop]

  if scaling!=nothing
    _s = scaling
  else
    _s = ones(nfun)
  end

  if new_fig; fig = figure("pareto"); end;

  if nfun<=1
    error("Invalid multiobjective function of dimensions $nfun")
  elseif nfun==2
    if title_y!=nothing
      title(title_str, y=title_y)
    else
      title(title_str)
    end
    xlabel("$(scaling!=nothing ? "Scaled " : "")f$f_x")
    ylabel("$(scaling!=nothing ? "Scaled " : "")f$f_y")
    grid(true, color="0.8", linestyle="--")

    # Evaluates fitness
    if ftnss!=nothing
      _ftnss = ftnss
    else
      _ftnss = _eval_f(f, nfun, ppltn; scaling=scaling)
    end

    # Builds the pareto front
    pareto = [ this_ftnss for this_ftnss in _ftnss if this_ftnss[1]<0]

    # Evaluates function
    if fvals!=nothing
      _fvals = [_s.*fval for fval in fvals]
    else
      _fvals = [_s.*f(ppltn[i,:]) for i in 1:npop]
    end

    minftnss = minimum([val[1] for val in _ftnss])
    maxftnss = maximum([val[1] for val in _ftnss])
    xmax, xmin = -Inf, Inf
    ymax, ymin = -Inf, Inf

    # Plots the population colored by fitness
    for (ftnssval, ind) in _ftnss
        clr = (maxftnss - ftnssval)/(maxftnss - minftnss)
        xval = _fvals[ind][f_x]
        yval = _fvals[ind][f_y]
        if plot_population
          scatter([xval], [yval], c=[[clr,0,0]],
                                label="$(_labels[ind]) -> $(round(ftnssval,rnd)) ")
        end

        if xval>xmax; xmax=xval; end;
        if xval<xmin; xmin=xval; end;
        if yval>ymax; ymax=yval; end;
        if yval<ymin; ymin=yval; end;
    end
    if labels
      x_disp = (xmax-xmin)*0.01
      y_disp = (ymax-ymin)*0.01
      disp = [x_disp, y_disp]
      for i in 1:npop
        j = -1
        if i in [prt[2] for prt in pareto]
          j+=1
          if j%annotate_num==0
            annotate("$(_labels[i])=$(round.(ppltn[i,:], rnd))",
                                        xy=[_fvals[i][f_x],_fvals[i][f_y]]+disp)
          end
        end
      end
      if dlegend*labels; legend(loc="best"); end;
    end

    # Plots the pareto front
    prt_fvals = [[_fvals[ind][f_x], _fvals[ind][f_y]] for (ftnssval, ind) in pareto]
    sort!(prt_fvals, by=x -> x[1])
    prt_x = [val[1] for val in prt_fvals]
    prt_y = [val[2] for val in prt_fvals]


    if crv_label!=nothing
      # annotate(crv_label, xy=[xmax,ymin]-[20,40].*disp, size=20)
      plot(prt_x, prt_y, "--o", label=crv_label)
    else
      plot(prt_x, prt_y, "--k", label="Pareto Front")
    end

    if ylims=="maxmin"; ylim([ymin, ymax]); end;

  else
    pareto = nothing
  end

  return pareto
end

function multiobjective_animation(save_path::String, run_name::String,
                                f, saved_gens; verbose=true,
                                first=1, last=-1, ext=".png",
                                xlims=nothing, ylims=nothing,
                                optargs...)

  gens = saved_gens[first : (last==-1 ? size(saved_gens,1) : last ) ]

  for (i, (ppltn, ftnss)) in enumerate(gens)
    if verbose; println("Plotting iteration $i of $(size(saved_gens,1))..."); end;

    title_str = "Population fitness at generation #$i"
    plot_multiobjective(f, ppltn; title_str=title_str, optargs...)

    if xlim!=nothing; xlim(xlims); end;
    if ylim!=nothing; ylim(ylims); end;

    this_num = ( i<10 ? "000" : (i<100 ? "00" : (i<1000 ? "0" : "")) )*"$i"
    tight_layout()
    savefig(joinpath(save_path, run_name)*"."*this_num*ext)
    clf()
  end

  if verbose; println("\tDone plotting"); end;
end
