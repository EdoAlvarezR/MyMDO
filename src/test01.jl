include("MyMDO.jl")
mdo = MyMDO

using PyPlot
pygui(true)

ign_error = true

# f(x) = [x[1]^2 + x[2]^2]
# xmin, xmax = -3.0, 3.0
# ymin, ymax = -3.0, 3.0
# f(x) = [(x[1]-3).*(x[1].^3).*((x[1]-6).^4) + x[2]^4]
# f(x) = [(x[1]*x[2]-3).*(x[1]*x[2].^3).*((x[1]*x[2]-6).^4)]
# xmin, xmax = -0.5, 7.0
# ymin, ymax = -10.0, 10.0
# X0 = [3.75, 10.0]


# f(x) = [1.5*x[1].^2 + x[2].^2 - 2*x[1].*x[2] + 2*x[1].^3 + 0.5*x[1].^4]
# xmin, xmax = -4.0, 2.0
# ymin, ymax = -4.0, 2.0
# X0 = [-3.75, 1.9]

f(x) = [1-exp(-(10*x[1].^2 + x[2].^2))]
xmin, xmax = -0.6, 0.6
ymin, ymax = -0.6, 0.6
X0 = [-0.55, 0.55]

g(x::Array{Float64,1}) = mdo.GradEval.fad(f, x)[2]

# X0 = [xmin + (xmax-xmin)*0.75, ymin + (ymax-ymin)*0.25]
# X0 = [3.5, ymin + (ymax-ymin)*0.25]

eps_a, eps_r, eps_g = 1/10^6, 0.01, 0.01
mu1, mu2, amax = 0.001, 0.002, 0.05
max_ite = 10^3
Xs, fs, gs, zoom, line = [], [], [], [], []

if ign_error
  try
    xopt, fopt = mdo.steepest_descent(f, g, X0, eps_a, eps_r, eps_g, mu1, mu2, amax,
                    max_ite; Xs=Xs, fs=fs, gs=gs, linestage=line, zoomstage=zoom)
  catch e
    println(e)
    println("Initial guess: $X0")
  end
else
  xopt, fopt = mdo.steepest_descent(f, g, X0, eps_a, eps_r, eps_g, mu1, mu2, amax,
                  max_ite; Xs=Xs, fs=fs, gs=gs, linestage=line, zoomstage=zoom)
end

n = 100
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

plot_surface(xgrid, ygrid, z,
            rstride=2,edgecolors="k", cstride=2,
            cmap=ColorMap("coolwarm"), alpha=0.75, linewidth=0.05)
scatter3D([this_X[1] for this_X in Xs], [this_X[2] for this_X in Xs],
            fs, c="k")

xlabel("x")
ylabel("y")
zlabel("f(x,y)")
