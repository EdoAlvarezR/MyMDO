
################################################################################
# GENETIC ALGORITHM
################################################################################
"""

  **Arguments**
  * `f`                     : Function to be minimized.
  * `xl::Array{Float64,1}`  : Lower bounds of design variables.
  * `xu::Arrya{Float64,1}`  : Upper bounds of design variables.
  * `npop::Int64`           : Population size in each generation.
  * `ngen::Int64`           : Number of generations to simulate.

  **Optional Arguments**
  * `mutP::Float64=0.05`    : Mutation probability.
  * `beta::Float64=2.0`     : Dynamic mutation parameter.
  * `mutate_parents::Bool=false`    : If true, it will mutate parents along
                                        with offspring.
  * `save_gen::Any[]`       : Give it an empty array and it will push
                              `[ppltn, ftnss]` on it at every generation.
  * `scaling::Array{Float64,1}`     : If multiobjective, use this array to
                                        define the scaling of each objective.

NOTE: If the codomain of `f` is multidimensional, it switches automatically
      to multibojective optimization evaluating fitness in a maximin criteria
      and building a pareto front.
"""
function genetic_algorithm(f, xl::Array{T,1} where {T<:Real},
                              xu::Array{T,1} where {T<:Real},
                              npop::Int64, ngen::Int64;
                              mutP::Real=0.05, beta::Real=2.0,
                              mutate_parents::Bool=false,
                              save_gen=nothing,
                              scaling=nothing,
                              verbose::Bool=false, v_lvl::Int64=0,
                              graph::Bool=true, init_meth="grid")

  # Error cases
  if size(xl,1)!=size(xu,1)
    error("Number of lower bounds do not match number of upper bounds!")
  elseif npop<=0
    error("Invalid population size $npop.")
  elseif npop%2!=0
    error("Population size must be an even number; got $npop.")
  elseif ngen<=0
    error("Invalid number of generations $ngen.")
  elseif mutP<=0 && mutP>1
    error("Invalid mutation probability $mutP; provide a value in range [0,1].")
  elseif beta<0
    error("Invalid mutation parameter $beta; provide a positive number.")
  end
  for i in 1:size(xl,1)
    if xl[i]>xu[i]
      error("Found invalid boundary: ($(xl[i])>$(xu[i]))")
    end
  end

  nfun = size(f((xl+xu)/2),1)       # Number of objective functions
  nvars = size(xl,1)                # Number of design variables
  ppltn = zeros(npop, nvars)        # Population (ppltn[i,j] j-th variable value of i-th candidate)
  ftnss = nothing                   # Fitness of candidates in increasing order

  line = collect(1:npop)            # Line up of candidates for tournament selection
  pool = zeros(Int64, npop)         # Mating pool of current generation
  ffsprng = zeros(npop, nvars)      # Offspring at current generation
  this_ppltn = zeros(2*npop, nvars) # Total population at current generation
  this_ftnss = zeros(2*npop)        # Fitness of total population

  multiobjective = nfun!=1

  # Initial population
  if init_meth=="random"
    for i in 1:npop
      ppltn[i,:] = xl + (xu-xl).*rand(nvars)
    end

  elseif init_meth=="grid"
    if npop<2^nvars
      error("Grid initiation method requires a population size of at"*
            " least $(2^nvars); got $npop.")
    end
    nsecs = Int(floor(npop^(1/nvars)))
    secs = collect(linspace(0,1,nsecs))
    for i in 1:npop
      if i<=nsecs^nvars
        subs = ind2sub( Tuple([nsecs for j in 1:nvars]), i)
        ppltn[i,:] = xl + (xu-xl).*[secs[sub] for sub in subs]
      else
        ppltn[i,:] = xl + (xu-xl).*rand(nvars)
      end
    end

  else
      error("Invalid initiation method $init_meth")
  end
  fvals = save_gen!=nothing ? [] : nothing
  ftnss = _eval_f(f, nfun, ppltn; scaling=scaling, save_f=fvals) # Current fitness

  if save_gen!=nothing; push!(save_gen, deepcopy.([ppltn, ftnss, fvals])); end;


  # Iterates over generations
  for gen in 1:ngen-1
    if verbose; println("\t"^v_lvl*"Generation #$gen"); end;

    # Performs two tournament selections to define mating pool
    mate_i = 1
    for ntournament in 1:2

      shuffle!(line)
      for i in 1:2:npop

        p1, p2 = line[i], line[i+1] # Players
        if ftnss[p1][1]<ftnss[p2][1]
          pool[mate_i] = p1
        else
          pool[mate_i] = p2
        end

        mate_i += 1
      end

    end

    # Generates offspring
    for i in 1:2:npop   # Iterates over mates
      p1, p2 = pool[i], pool[i+1]

      # Blend crossover
      rands = rand(nvars)
      ffsprng[i, :] = ppltn[p1,:] + (ppltn[p2,:]-ppltn[p1,:]).*rands
      ffsprng[i+1, :] = ppltn[p1,:] + (ppltn[p2,:]-ppltn[p1,:]).*(1-rands)
    end

    # Creates total population
    this_ppltn[1:npop, :] = ppltn
    this_ppltn[npop+1:2*npop, :] = ffsprng

    # Dynamic mutation
    alpha = (1-(gen-1)/ngen)^beta   # Uniformity exponent

    for i in (mutate_parents ? 1 : npop+1) : 2*npop # Iterates over candidates
      for j in 1:nvars                              # Iterates over variables

        if rand(Float64)<=mutP # Case of mutation
          x = this_ppltn[i,j]
          r = xl[j] + (xu[j]-xl[j])*rand(Float64)

          if r<=x
            this_ppltn[i,j] = xl[j] + (r-xl[j])^alpha*(x-xl[j])^(1-alpha)
          else
            this_ppltn[i,j] = xu[j] - (xu[j]-r)^alpha*(xu[j]-x)^(1-alpha)
          end

        end

      end
    end

    # Evaluates fitness
    this_fvals = save_gen!=nothing ? [] : nothing
    if mutate_parents || nfun>1
      this_ftnss = _eval_f(f, nfun, this_ppltn; scaling=scaling, save_f=this_fvals)
    else
      this_ftnss = ftnss
      this_ftnss = vcat(ftnss, _eval_f(f, nfun, this_ppltn;
                              imin=npop+1, scaling=scaling, save_f=this_fvals))
      sort!(this_ftnss, by=x -> x[1])

      if save_gen!=nothing; this_fvals = vcat(deepcopy.(fvals), this_fvals); end;
    end

    # Elitism selection
    ftnss = this_ftnss[1:npop]
    for (i,(fval, p)) in enumerate(deepcopy.(ftnss))
      ppltn[i, :] = this_ppltn[p, :]
      ftnss[i][2] = i

      if save_gen!=nothing; fvals[i] = this_fvals[p]; end;
    end

    if save_gen!=nothing; push!(save_gen, deepcopy.([ppltn, ftnss, fvals])); end;

    if graph && nfun>1
      if gen!=1; clf(); end;
      plot_multiobjective(f, ppltn; labels=true, scaling=scaling,
                                    title_str="Population Generation #$(gen+1)",
                                    ftnss=ftnss, fvals=fvals, dlegend=false)
    end
  end

  xopt = ppltn[ftnss[1][2], :]
  ftnssopt = ftnss[1][1]

  return ppltn, ftnss, xopt, ftnssopt
end

"""
  Evaluates the fitness of a population and returns the array of elements
`ftnss[i]=[fval, p]` containing the i-th best fitness `fval` of the total
population, corresponding to the p-th candidate. `ftnss` is ordered in
increasing order according to `fval`. If single objective, `fval` is the
objective function; if multiobjective, `fval` is the maximin dominance.
If multiobjective, scaling factors for each objective can be given through
`scaling`.
"""
function _eval_f(f, nfun::Int64, ppltn::Array{T,2} where {T<:Real};
                                                imin::Int64=1, scaling=nothing,
                                                save_f=nothing)
  npop = size(ppltn, 1)
  effnpop = npop-(imin-1)

  if scaling!=nothing && nfun!=1 && size(scaling,1)!=nfun
    error("Expected $nfun scaling factors; got $(size(scaling,1)).")
  end

  # ------- Evaluates fitness value --------------------------
  # Case of single objective: Fitness is the objective function
  if nfun==1
    ftnss = [ Any[ f(ppltn[i, :]), i ] for i in imin:npop ]

  # Case of multiojbective: Fitness is the maximin dominance function
  else

    if scaling==nothing
      _s = ones(nfun)
    else
      _s = scaling
    end

    # Evaluates objectives
    fvals = [ Any[ f(ppltn[i, :]), i ] for i in imin:npop ]

    if save_f!=nothing; for fval in fvals; push!(save_f, fval[1]); end; end;

    scld_fvals = [ Any[ _s.*fval, i] for (fval,i) in fvals]

    # Dominance matrix: dominance[i,j] = min(f(xi)-f(xj)), j dominates i if >=0
    dominance = zeros(effnpop, effnpop)

    for i in 1:effnpop
      for j in i:effnpop

        if i==j
          dominance[i,j] = -Inf

        else # Dominance criteria min(f(xi)-f(xj))
          fdiff = scld_fvals[i][1] - scld_fvals[j][1]
          mindiff = minimum(fdiff)
          maxdiff = maximum(fdiff)
          dominance[i,j] = mindiff
          dominance[j,i] = -maxdiff

        end

      end
    end

    # Makes the maximum dominance the fitness value
    ftnss = [ Any[ maximum(dominance[i,:]), scld_fvals[i][2] ] for i in 1:effnpop ]
    # ftnss = [ Any[ sign(val)*abs(val)^(1/2), ind] for (val,ind) in ftnss]

  end

  # Sorts fitness in increasing order
  sort!(ftnss, by=x -> x[1])

  return ftnss
end
