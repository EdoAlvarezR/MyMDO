include("MyMDO.jl")
mdo = MyMDO

using PyPlot

f(x) = (x-3).*(x.^3).*((x-6).^4)
g(x::Array{Float64,1}) = mdo.GradEval.fad(f, x)[2]
xmin, xmax = [-0.5, 7]
n = 100
x = [xmin+i*(xmax-xmin)/n for i in 1:n]

x0 = 4.0
amax = 2.5
a1 = 0.01
p = 1.0
mu1 = 0.001
mu2 = 0.0015

line, zoom = [], []
a = MyMDO.linesearch(f, g, [p], [x0], a1, amax, mu1, mu2;
            linestage=line, zoomstage=zoom)


plot(x,f(x), "k", label="f(x) = (x-3).*(x.^3).*((x-6).^4)")
plot(x0+p*line,[f([x0+p*line_i]) for line_i in line], "o", label="line stage")
plot(x0+p*zoom,[f([x0+p*zoom_i]) for zoom_i in zoom], "x", label="zoom stage")
xlim([-1, 7.25])
ylim([-2250, 1500])
xlabel("x")
ylabel("f(x)")
legend(loc="best")
grid(true, color="0.8", linestyle="--")

println("Suggested step: $a")
println("f(x0+a*p) = f($(x0+a*p)) = $(f([x0+a*p]))")
