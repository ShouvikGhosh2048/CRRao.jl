mtcars = dataset("datasets", "mtcars")

tests = [
    (Prior_Ridge(), 20.080877893580514),
    (Prior_Laplace(), 20.070783434589128),
    (Prior_Cauchy(), 20.019759144845644),
    (Prior_TDist(), 20.042614331921428),
    (Prior_HorseShoe(), 20.042984550677183),
]

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_Ridge(), 0.01, 10000)
@test mean(predict(model, mtcars)) ≈ 20.080877893580514

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_Laplace(), 0.01, 10000)
@test mean(predict(model, mtcars)) ≈ 20.070783434589128

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_Cauchy(), 10000)
@test mean(predict(model, mtcars)) ≈ 20.019759144845644

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_TDist(), 2.0, 10000)
@test mean(predict(model, mtcars)) ≈ 20.042614331921428

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_HorseShoe(), 10000)
@test mean(predict(model, mtcars)) ≈ 20.042984550677183

gauss_test = 20.0796026428345

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_Gauss(), 30.0, [0.0,-3.0,1.0], 10000)

@test mean(predict(model, mtcars)) ≈ gauss_test