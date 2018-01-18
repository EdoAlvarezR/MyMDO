module GradEval

using ForwardDiff
using ReverseDiff

# export fad, radinit, rad, centraldiff



function fad(fun, x)

    f = fun(x)
    Nin = length(x)
    Nout = length(f)

    dfdx = zeros(Nout, Nin)
    cfg = ForwardDiff.JacobianConfig(nothing, x,  ForwardDiff.Chunk{Nin}())  # TODO: revisit chunk size
    ForwardDiff.jacobian!(dfdx, fun, x, cfg)

    # TODO: this wastes a funciton call but has convenience of not specifying Nout.
    # The below saves one function call.  May be worth switching later depending on
    # how this is used and if we can use that function call regardless.

    # xc = copy(x)
    # Jac = zeros(Nout, Nin)
    # out =  DiffBase.DiffResult(xc, Jac)
    # cfg = ForwardDiff.JacobianConfig{Nin}(xc)  # TODO: revisit chunk size
    # ForwardDiff.jacobian!(out, fun, xc, cfg)
    #
    # f = DiffBase.value(out)
    # dfdx = DiffBase.jacobian(out)

    return f, dfdx
end


function radinit(fun, x)

    f_tape = ReverseDiff.JacobianTape(fun, x)
    const compiled_f_tape = ReverseDiff.compile(f_tape)  # TODO: need to save this outside of function scope if we will be reusing

    return compiled_f_tape
end

function rad(fun, x, ftape)

    f = fun(x)
    Nin = length(x)
    Nout = length(f)

    dfdx = zeros(Nout, Nin)
    ReverseDiff.jacobian!(dfdx, ftape, x)

    return f, dfdx
end


function centraldiff(fun, x, h=1e-6)

    f = fun(x)
    Nin = length(x)
    Nout = length(f)

    dfdx = zeros(Nout, Nin)
    for i = 1:Nin
        step = h*x[i]
        step = max(step, h)  # make sure step size isn't too small (or 0)
        x[i] += step
        fp = fun(x)
        x[i] -= 2*step
        fm = fun(x)
        x[i] += step

        dfdx[:, i] = (fp - fm)/(2*step)
    end

    return f, dfdx
end


function check(func, x, gcheck, gtol=1e-6)
    """check against centraldiff"""

    f, gfd = centraldiff(func, x)

    nf = length(f)
    nx = length(x)
    gerror = zeros(nf, nx)

    for j = 1:nx
        for i = 1:nf
            if gcheck[i, j] <= gtol
                gerror[i, j] = abs(gcheck[i, j] - gfd[i, j])
            else
                gerror[i, j] = abs(1 - gfd[i, j]/gcheck[i, j])
            end
            if gerror[i, j] > gtol
                println("**** gerror(", i, ", ", j, ") = ", gerror[i, j])
            end
        end
    end

    return gerror
end


function manualderiv{T<:ForwardDiff.Dual}(f, x::Array{T, 1}, J)  # J is df_i/dx_j

    nf, nx = size(J)
    fdual = Array{T}(nf)  # an array of dual numbers

    for i = 1:nf
        p = ForwardDiff.partials(x[1]) * 0  # initialize to zeros
        for j = 1:nx  # chain rule
            p += J[i, j] * ForwardDiff.partials(x[j])
        end
        fdual[i] = ForwardDiff.Dual(f[i], p)
    end

    return fdual
end


function values(x)

    nx = length(x)
    xv = zeros(nx)
    for i = 1:nx
        xv[i] = ForwardDiff.value(x[i])
    end

    return xv
end


end
