
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

"""
function genetic_algorithm(f, xl::Array{T,1} where {T<:Real},
                              xu::Array{T,1} where {T<:Real},
                              npop::Int64, ngen::Int64;
                              mutP::Real=0.05, beta::Real=2.0,
                              mutate_parents::Bool=false,
                              save_gen=nothing)

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

  nvars = size(xl,1)                # Number of design variables
  ppltn = zeros(npop, nvars)        # Population (ppltn[i,j] j-th variable value of i-th candidate)
  ftnss = nothing                   # Fitness of candidates in increasing order

  line = collect(1:npop)            # Line up of candidates for tournament selection
  pool = zeros(Int64, npop)         # Mating pool of current generation
  ffsprng = zeros(npop, nvars)      # Offspring at current generation
  this_ppltn = zeros(2*npop, nvars) # Total population at current generation
  this_ftnss = zeros(2*npop)        # Fitness of total population

  # Initial population
  for i in 1:npop
    ppltn[i,:] = xl + (xu-xl).*rand(nvars)
  end
  ftnss = _eval_f(f, ppltn)         # Current fitness

  if save_gen!=nothing; push!(save_gen, deepcopy.([ppltn, ftnss])); end;


  # Iterates over generations
  for gen in 1:ngen-1

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
    if mutate_parents
      this_ftnss = _eval_f(f, this_ppltn)
    else
      this_ftnss = ftnss
      this_ftnss = vcat(ftnss, _eval_f(f, this_ppltn; imin=npop+1))
      sort!(this_ftnss, by=x -> x[1])
    end

    # Elitism selection
    ftnss = this_ftnss[1:npop]
    for (i,(fval, p)) in enumerate(ftnss)
      ppltn[i, :] = this_ppltn[p, :]
      ftnss[i][2] = i
    end

    if save_gen!=nothing; push!(save_gen, deepcopy.([ppltn, ftnss])); end;
  end

  xopt = ppltn[ftnss[1][2], :]
  ftnssopt = ftnss[1][1]

  return ppltn, ftnss, xopt, ftnssopt
end

"""
  Evaluates the fitness of a population and returns the array of elements
`ftnss[i]=[fval, p]` containing the i-th best fitness `fval` of the total
population, corresponding to the p-th candidate. `ftnss` is ordered in
increasing order according to `fval`.
"""
function _eval_f(f, ppltn::Array{T,2} where {T<:Real}; imin::Int64=1)

  # Evaluates fitness value
  ftnss = [Any[f(ppltn[i, :]), i] for i in imin:size(ppltn,1)]

  # Sorts fitness in decreasing order
  sort!(ftnss, by=x -> x[1])

  return ftnss
end
